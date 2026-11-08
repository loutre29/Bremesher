//! DESIGN 6.11 tests (mesh-free, testkit TMeshes).
#![allow(clippy::needless_range_loop)]

use std::time::{Duration, Instant};

use aq_core::{ArcKind, CtlState, ErrKind, NONE, NodeKind, QComb, TArc, TMesh, TNode, TPatch};
use aq_quantize::*;
use aq_testkit::*;

fn params() -> QuantParams {
    QuantParams::default()
}

fn run(t: &TMesh, target: f64, p: &QuantParams) -> QuantResult {
    let st = CtlState::new();
    quantize(t, target, p, &st.unbounded()).unwrap()
}

/// sum sigma n == 0 on every (patch, axis) node.
fn balanced(t: &TMesh, n: &[u32]) -> bool {
    t.patches.iter().all(|p| {
        (0..2).all(|axis| {
            let plus: i64 = p.sides[axis].iter().map(|&(a, _)| n[a as usize] as i64).sum();
            let minus: i64 = p.sides[axis + 2].iter().map(|&(a, _)| n[a as usize] as i64).sum();
            plus == minus
        })
    })
}

fn side_sum(n: &[u32], side: &[(u32, bool)]) -> u32 {
    side.iter().map(|&(a, _)| n[a as usize]).sum()
}

/// Valence (number of distinct incident edges) of every QComb vertex.
fn valences(c: &QComb) -> Vec<u32> {
    let mut e: Vec<(u32, u32)> = Vec::new();
    for q in &c.quads {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.sort_unstable();
    e.dedup();
    let mut v = vec![0u32; c.n_verts as usize];
    for (a, b) in e {
        v[a as usize] += 1;
        v[b as usize] += 1;
    }
    v
}

#[test]
fn rectangle_rounds_each_side() {
    let t = tmesh_rect(3.4, 5.6).unwrap();
    let p = QuantParams { max_secant: 0, ..params() };
    let r = run(&t, 1.0, &p);
    let s: Vec<u32> = t.patches[0].sides.iter().map(|s| side_sum(&r.n, s)).collect();
    assert_eq!(s, vec![3, 6, 3, 6]);
    assert_eq!(r.count, 18);
    assert_eq!(r.solves, 1);
    assert_eq!(r.sigma, 1.0);
    r.comb.validate(&t).unwrap();
    assert_eq!(r.comb.patch_dims[0], [3, 6]);
}

#[test]
fn t_offsets_collapse() {
    let t = tmesh_grid(2, 2, &[0.0, 0.3]).unwrap();
    let short = t.arcs.iter().position(|a| (a.len - 0.3).abs() < 1e-12).unwrap();
    let r = run(&t, 4.0, &QuantParams { allow_collapse: true, ..params() });
    assert_eq!(r.n[short], 0, "{:?}", r.n);
    r.comb.validate(&t).unwrap();
    assert!(balanced(&t, &r.n));
    // The two T-nodes merged into one regular vertex.
    let e = t.arcs[short].ends;
    assert_eq!(r.comb.node_id[e[0] as usize], r.comb.node_id[e[1] as usize]);
    let r = run(&t, 4.0, &QuantParams { allow_collapse: false, ..params() });
    assert_eq!(r.n[short], 1, "{:?}", r.n);
    r.comb.validate(&t).unwrap();
}

/// Exact minimum of the objective at scale `sigma` over integer solutions with `n >= lb`
/// (branch and bound). `incumbent` is any feasible solution: its objective bounds the search,
/// since an arc can only take a value whose own cost does not exceed it. Constraint nodes are
/// checked as soon as their last arc is assigned.
fn brute_force(t: &TMesh, sigma: f64, lb: &[u32], incumbent: &[u32]) -> f64 {
    assert!(balanced(t, incumbent));
    let c0 = objective(t, sigma, incumbent);
    let na = t.arcs.len();
    let tgt = |a: usize| arc_target(&t.arcs[a], sigma);
    let den = |a: usize| (sigma * t.arcs[a].len).max(0.5);
    let mut ranges = Vec::new();
    for a in 0..na {
        let w = arc_weight(&t.arcs[a]);
        let r = (c0 * den(a) / w).sqrt() + 1e-9;
        let lo = ((tgt(a) - r).ceil().max(0.0) as u32).max(lb[a]);
        let hi = (tgt(a) + r).floor().max(0.0) as u32;
        ranges.push((lo, hi));
    }
    // Node terms (arc, sign) and the depth after which each node is complete.
    let mut terms: Vec<Vec<(usize, i64)>> = Vec::new();
    for p in &t.patches {
        for axis in 0..2 {
            let mut v: Vec<(usize, i64)> = p.sides[axis].iter().map(|&(a, _)| (a as usize, 1)).collect();
            v.extend(p.sides[axis + 2].iter().map(|&(a, _)| (a as usize, -1)));
            terms.push(v);
        }
    }
    let mut check_at: Vec<Vec<usize>> = vec![Vec::new(); na];
    for (q, v) in terms.iter().enumerate() {
        if let Some(last) = v.iter().map(|&(a, _)| a).max() {
            check_at[last].push(q);
        }
    }
    let cost1 = |a: usize, x: u32| {
        let d = x as f64 - tgt(a);
        arc_weight(&t.arcs[a]) * d * d / den(a)
    };
    let mut best = c0;
    let mut n = vec![0u32; na];
    let mut stack: Vec<(usize, u32, f64)> = vec![(0, ranges[0].0, 0.0)];
    while let Some((a, x, acc)) = stack.pop() {
        if x > ranges[a].1 {
            continue;
        }
        stack.push((a, x + 1, acc));
        let c = acc + cost1(a, x);
        if c > best + 1e-12 {
            continue;
        }
        n[a] = x;
        let ok = check_at[a].iter().all(|&q| terms[q].iter().map(|&(b, s)| s * n[b] as i64).sum::<i64>() == 0);
        if !ok {
            continue;
        }
        if a + 1 == na {
            if c < best {
                best = c;
            }
        } else {
            stack.push((a + 1, ranges[a + 1].0, c));
        }
    }
    best
}

/// Solver vs brute force on 200 random bi-directed instances at scale `sigma`: (instances,
/// optimal ones, worst ratio, instances whose relaxation was not integral at the start).
fn bidirected_stats(sigma: f64, collapse: bool) -> (usize, usize, f64) {
    let (mut total, mut exact, mut worst) = (0usize, 0usize, 1.0f64);
    'outer: for seed in 0..100u64 {
        for n_arcs in 2..=8u32 {
            if total == 200 {
                break 'outer;
            }
            let t = tmesh_random_bidirected(n_arcs, seed * 7 + n_arcs as u64).unwrap();
            let lb: Vec<u32> = t.arcs.iter().map(|a| table_lb(a, sigma, collapse)).collect();
            let n = solve_bimdf(&t, sigma, &lb)
                .unwrap_or_else(|e| panic!("infeasible sigma {sigma} seed {seed} arcs {n_arcs}: {e}"));
            assert!(balanced(&t, &n), "unbalanced seed {seed}");
            assert!(n.iter().zip(&lb).all(|(x, l)| x >= l), "lb violated seed {seed}");
            let obj = objective(&t, sigma, &n);
            let opt = brute_force(&t, sigma, &lb, &n);
            assert!(obj + 1e-9 >= opt);
            if obj <= opt + 1e-9 {
                exact += 1;
            }
            let ratio = if opt > 1e-12 { obj / opt } else if obj <= 1e-12 { 1.0 } else { f64::INFINITY };
            worst = worst.max(ratio);
            total += 1;
        }
    }
    (total, exact, worst)
}

#[test]
fn random_bidirected_vs_brute_force() {
    // At sigma = 1 the planted solution round(len) is each arc's own minimizer (the noise is
    // below 0.45), so the other scales are what exercise the parity and integer repairs.
    for sigma in [1.0, 0.73, 1.37, 1.5, 2.3] {
        for collapse in [false, true] {
            let (total, exact, worst) = bidirected_stats(sigma, collapse);
            eprintln!("sigma {sigma} collapse {collapse}: {exact}/{total} optimal, worst ratio {worst:.4}");
            assert_eq!(total, 200);
            assert!(exact * 100 >= 95 * total, "only {exact}/{total} optimal");
            assert!(worst <= 1.5 + 1e-9, "worst ratio {worst}");
        }
    }
}

#[test]
fn random_500_patch_grids_hit_the_count() {
    // Random dissections (a uniform grid is not used: all its columns share one width, so its
    // count is a step function of sigma and most targets are unreachable).
    let mut cases: Vec<(TMesh, f64)> = Vec::new();
    for (seed, target) in [(0u64, 5000.0), (1, 3000.0), (2, 12000.0), (3, 2600.0), (4, 8000.0), (5, 1500.0)] {
        cases.push((tmesh_random_dissection(500, seed).unwrap(), target));
    }
    // The secant stops at the first count within count_tol, so the 3% gate of DESIGN 6.11 is
    // run with count_tol = 0.03; the default (0.04) must meet its own tolerance.
    for (i, (t, target)) in cases.iter().enumerate() {
        assert_eq!(t.patches.len(), 500);
        let d = run(t, *target, &params());
        let err_d = (d.count as f64 - target).abs() / target;
        assert!(err_d < 0.04, "case {i} (default tol): count {} vs {target}", d.count);
        let r = run(t, *target, &QuantParams { count_tol: 0.03, ..params() });
        let err = (r.count as f64 - target).abs() / target;
        eprintln!(
            "case {i}: count {} target {target} err {:.4} (default tol {:.4}) sigma {:.4} solves {} cuts {}",
            r.count, err, err_d, r.sigma, r.solves, r.cuts
        );
        assert!(err < 0.03, "case {i}: count {} vs {target}", r.count);
        assert!(r.solves <= MAX_SOLVES);
        r.comb.validate(t).unwrap();
        assert!(build_comb(t, &r.n).is_ok());
        assert!(comb_violations(t, &r.n, 8).unwrap().is_empty());
        assert!(balanced(t, &r.n));
        assert!(r.n.iter().zip(&r.lb).all(|(x, l)| x >= l));
    }
}

#[test]
fn planted_cone_merge_is_rejected() {
    // A box with short vertical arcs: collapsing them merges pairs of +1 cones.
    let t = tmesh_box(4.0, 4.0, 0.3).unwrap();
    let short: Vec<usize> = (0..t.arcs.len()).filter(|&a| t.arcs[a].len == 0.3).collect();
    assert_eq!(short.len(), 4);
    let mut n: Vec<u32> = t.arcs.iter().map(|a| a.len.round() as u32).collect();
    for &a in &short {
        n[a] = 0;
    }
    let v = comb_violations(&t, &n, 8).unwrap();
    assert!(v.iter().any(|(x, _)| matches!(x, Violation::BadValence(_))), "{v:?}");
    for (x, c) in &v {
        if matches!(x, Violation::BadValence(_)) {
            assert!(!c.is_empty() && c.iter().all(|a| short.contains(&(*a as usize))), "{x:?} {c:?}");
        }
    }
    assert!(build_comb(&t, &n).is_err());
    let r = run(&t, 40.0, &params());
    for &a in &short {
        assert!(r.n[a] >= 1, "{:?}", r.n);
        assert_eq!(r.lb[a], 1);
    }
    assert!(r.cuts >= 1);
    assert!(r.solves >= 2);
    r.comb.validate(&t).unwrap();
}

/// An annulus: one patch between the bottom rim (chain 0) and the top rim (chain 1), with a
/// seam trace T of length `h` on sides 1 and 3.
fn annulus(circ: f64, h: f64) -> TMesh {
    let nodes = vec![
        TNode { kind: NodeKind::Regular, index4: 0, chain: 0 },
        TNode { kind: NodeKind::Regular, index4: 0, chain: 1 },
    ];
    let arc = |ends: [u32; 2], len: f64, kind: ArcKind, chain: u32| TArc {
        ends,
        len,
        kind,
        lb: 0,
        weight: 1.0,
        wrap: false,
        chain,
    };
    let arcs = vec![
        arc([0, 0], circ, ArcKind::Boundary, 0),
        arc([1, 1], circ, ArcKind::Boundary, 1),
        arc([0, 1], h, ArcKind::SeedTrace, NONE),
    ];
    let patches = vec![TPatch { sides: [vec![(0, false)], vec![(2, false)], vec![(1, true)], vec![(2, true)]] }];
    let t = TMesh { nodes, arcs, patches, rings: vec![vec![(0, false)], vec![(1, false)]], chi: 0 };
    t.validate().unwrap();
    t
}

#[test]
fn zero_height_patch_between_boundary_loops_is_rejected() {
    let t = annulus(8.0, 0.3);
    let v = comb_violations(&t, &[8, 8, 0], 8).unwrap();
    let pinch = v.iter().find(|(x, _)| *x == Violation::Pinch(0)).unwrap_or_else(|| panic!("{v:?}"));
    assert_eq!(pinch.1, vec![2]);
    assert!(build_comb(&t, &[8, 8, 0]).is_err());
    let r = run(&t, 8.0, &params());
    assert_eq!(r.lb[2], 1);
    assert!(r.n[2] >= 1);
    assert!(r.cuts >= 1);
    r.comb.validate(&t).unwrap();
    assert_eq!(r.comb.patch_dims[0], [8, r.n[2]]);
}

#[test]
fn helix_wrap_arcs_collapse_to_rings() {
    let (circ, turns) = (8.0, 5u32);
    let t = tmesh_helix(circ, 6.0, 0.6, turns).unwrap();
    let r = run(&t, 32.0, &params());
    for (a, arc) in t.arcs.iter().enumerate() {
        if arc.wrap {
            assert_eq!(r.n[a], 0, "wrap arc {a}");
        }
    }
    let c = &r.comb;
    c.validate(&t).unwrap();
    let np = t.patches.len();
    let w = c.patch_dims[0][0];
    assert_eq!(w, 8);
    // Every row of every patch is a closed ring of w distinct vertices.
    let mut rows: Vec<Vec<u32>> = Vec::new();
    for p in 0..np {
        let [pw, ph] = c.patch_dims[p];
        assert_eq!(pw, w);
        let ids = &c.patch_ids[p];
        for j in 0..=ph {
            let row: Vec<u32> = (0..=pw).map(|i| ids[(j * (pw + 1) + i) as usize]).collect();
            assert_eq!(row[0], row[pw as usize], "patch {p} row {j} is not closed");
            let mut s = row[..pw as usize].to_vec();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), pw as usize);
            rows.push(row);
        }
    }
    // The helical bands are glued: the top row of the bottom region is the bottom row of the
    // top region.
    let h0 = c.patch_dims[0][1] as usize;
    assert_eq!(rows[h0], rows[h0 + 1]);
    for p in 1..np - 1 {
        assert_eq!(c.patch_dims[p][1], 0);
    }
    // All quads are on closed rings around the axis: count = w * total height.
    assert_eq!(c.quads.len() as u32 % w, 0);
}

#[test]
fn qcomb_euler_and_cube() {
    let t = tmesh_cube().unwrap();
    for k in 1..=3u32 {
        let c = build_comb(&t, &[k; 12]).unwrap();
        c.validate(&t).unwrap();
        assert_eq!(c.quads.len() as u32, 6 * k * k);
        let val = valences(&c);
        assert_eq!(val.iter().filter(|&&v| v == 3).count(), 8);
        assert!(val.iter().all(|&v| v == 3 || v == 4));
        for i in 0..8 {
            assert_eq!(val[c.node_id[i] as usize], 3);
        }
    }
    let r = run(&t, 6.0, &params());
    assert_eq!(r.n, vec![1; 12]);
    r.comb.validate(&t).unwrap();
    // Euler on a disk, a cylinder and a sphere.
    for t in [tmesh_grid(3, 4, &[0.0, 0.5]).unwrap(), annulus(6.0, 3.0), tmesh_box(2.0, 3.0, 4.0).unwrap()] {
        let lb: Vec<u32> = t.arcs.iter().map(|a| table_lb(a, 1.0, false)).collect();
        let n = solve_bimdf(&t, 1.0, &lb).unwrap();
        let c = build_comb(&t, &n).unwrap();
        c.validate(&t).unwrap();
        let mut e: Vec<(u32, u32)> = c.quads.iter().flat_map(|q| (0..4).map(move |k| (q[k].min(q[(k + 1) % 4]), q[k].max(q[(k + 1) % 4])))).collect();
        e.sort_unstable();
        e.dedup();
        assert_eq!(c.n_verts as i64 - e.len() as i64 + c.quads.len() as i64, t.chi as i64);
    }
}

#[test]
fn unbalanced_lengths_are_reported() {
    let t = tmesh_rect(3.0, 2.0).unwrap();
    let e = build_comb(&t, &[3, 2, 4, 2]).unwrap_err();
    assert_eq!(e, vec![Violation::Unbalanced(0)]);
    assert_eq!(build_comb(&t, &[3, 2]).unwrap_err(), vec![Violation::Unbalanced(NONE)]);
}

#[test]
fn short_ring_is_cut() {
    // A 1 x 1 square: its boundary ring of 4 arcs must keep max(3, 8/2) = 4 segments, which
    // holds; a ring_min of 12 needs 6 and is reported.
    let t = tmesh_rect(1.0, 1.0).unwrap();
    assert!(build_comb_with(&t, &[1, 1, 1, 1], 8).is_ok());
    let v = build_comb_with(&t, &[1, 1, 1, 1], 12).unwrap_err();
    assert_eq!(v, vec![Violation::ShortRing(0)]);
    let r = run(&t, 1.0, &QuantParams { ring_min: 12, max_secant: 0, ..params() });
    let ring: u32 = t.rings[0].iter().map(|&(a, _)| r.n[a as usize]).sum();
    assert!(ring >= 6, "{:?}", r.n);
    assert!(r.cuts >= 1);
}

#[test]
fn capacity_scaling_matches_the_optimum() {
    // One patch whose opposite sides disagree a lot: total excess far above 2000.
    let mut t = tmesh_rect(3000.0, 2.0).unwrap();
    let bottom = t.patches[0].sides[0][0].0 as usize;
    t.arcs[bottom].len = 100.0;
    let lb = vec![1u32; 4];
    let n = solve_bimdf(&t, 1.0, &lb).unwrap();
    assert!(balanced(&t, &n));
    // Brute force over the common width.
    let mut best = (f64::INFINITY, 0u32);
    for w in 1..4000u32 {
        let mut m = n.clone();
        m[bottom] = w;
        m[t.patches[0].sides[2][0].0 as usize] = w;
        let o = objective(&t, 1.0, &m);
        if o < best.0 {
            best = (o, w);
        }
    }
    assert_eq!(n[bottom], best.1, "{n:?}");
    assert!((objective(&t, 1.0, &n) - best.0).abs() < 1e-9);
}

#[test]
fn infeasible_and_invalid_inputs() {
    let t = tmesh_helix(8.0, 6.0, 0.6, 5).unwrap();
    let lb = vec![1u32; t.arcs.len()];
    let e = solve_bimdf(&t, 1.0, &lb).unwrap_err();
    assert_eq!(e.kind, ErrKind::Infeasible);
    assert_eq!(solve_bimdf(&t, 1.0, &[1]).unwrap_err().kind, ErrKind::InvalidInput);
    assert_eq!(solve_bimdf(&t, f64::NAN, &lb).unwrap_err().kind, ErrKind::InvalidInput);
    let st = CtlState::new();
    assert_eq!(quantize(&t, -1.0, &params(), &st.unbounded()).unwrap_err().kind, ErrKind::InvalidInput);
    let mut bad = t.clone();
    bad.chi = 5;
    assert_eq!(quantize(&bad, 10.0, &params(), &st.unbounded()).unwrap_err().kind, ErrKind::InvalidInput);
    st.cancel();
    assert_eq!(quantize(&t, 10.0, &params(), &st.unbounded()).unwrap_err().kind, ErrKind::Cancelled);
}

#[test]
fn deterministic() {
    let t = tmesh_random_dissection(300, 11).unwrap();
    let a = run(&t, 2000.0, &params());
    for _ in 0..3 {
        assert_eq!(a, run(&t, 2000.0, &params()));
    }
    let pool = on_fresh_thread(move || run(&tmesh_random_dissection(300, 11).unwrap(), 2000.0, &params()));
    assert_eq!(a, pool);
}

/// Runs `f` on a fresh thread (the crate is single-threaded; this only checks that nothing
/// depends on thread-local state).
fn on_fresh_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::spawn(f).join().unwrap()
}

#[test]
fn performance_5k_arcs_24_solves() {
    if cfg!(debug_assertions) {
        eprintln!("skipped in debug builds");
        return;
    }
    let offs: Vec<f64> = (0..40).map(|r| [0.0, 0.3, 0.7, 0.45][r % 4]).collect();
    let t = tmesh_grid_sized(40, 60, 1.5, 1.5, &offs).unwrap();
    assert!(t.arcs.len() >= 5000, "{} arcs", t.arcs.len());
    let p = QuantParams { count_tol: 0.0, max_secant: 40, ..params() };
    let target = 4000.0;
    let st = CtlState::new();
    let ctl = st.ctl_for(Duration::from_secs(60));
    let _ = quantize(&t, target, &p, &ctl).unwrap();
    let mut best = Duration::MAX;
    let mut r = None;
    for _ in 0..3 {
        let t0 = Instant::now();
        let x = quantize(&t, target, &p, &ctl).unwrap();
        best = best.min(t0.elapsed());
        r = Some(x);
    }
    let r = r.unwrap();
    eprintln!("{} arcs, {} patches: {} solves, {} cuts, count {}, best {:?}", t.arcs.len(), t.patches.len(), r.solves, r.cuts, r.count, best);
    assert_eq!(r.solves, MAX_SOLVES);
    assert!(best < Duration::from_millis(300), "{best:?}");
}

/// Hexagonal disk: 6 patches around a centre node, each with corners centre, spoke i, rim i,
/// spoke i+1 (spokes and rim arcs of length 2). The centre has index4 `centre_index` and the 6
/// rim corners +1 (chi 1 needs a sum of 4, so a -2 centre is the consistent one).
fn tmesh_hexagon(centre_index: i8) -> TMesh {
    let mut nodes = vec![TNode { kind: NodeKind::Cone, index4: centre_index, chain: NONE }];
    nodes.extend((0..6).map(|_| TNode { kind: NodeKind::Regular, index4: 0, chain: NONE })); // spoke ends 1..=6
    nodes.extend((0..6).map(|_| TNode { kind: NodeKind::BoundaryCorner, index4: 1, chain: NONE })); // rim corners 7..=12
    let arc = |ends: [u32; 2], kind: ArcKind| TArc { ends, len: 2.0, kind, lb: 1, weight: 1.0, wrap: false, chain: NONE };
    let mut arcs = Vec::new();
    for i in 0..6u32 {
        arcs.push(arc([0, 1 + i], ArcKind::Separatrix)); // spoke i: 0..=5
    }
    for i in 0..6u32 {
        arcs.push(arc([1 + i, 7 + i], ArcKind::Boundary)); // rim a_i: 6..=11
        arcs.push(arc([7 + i, 1 + (i + 1) % 6], ArcKind::Boundary)); // rim b_i: 12.. (interleaved)
    }
    let rim_a = |i: u32| 6 + 2 * i;
    let rim_b = |i: u32| 7 + 2 * i;
    let patches = (0..6u32)
        .map(|i| TPatch { sides: [vec![(i, false)], vec![(rim_a(i), false)], vec![(rim_b(i), false)], vec![((i + 1) % 6, true)]] })
        .collect();
    TMesh { nodes, arcs, patches, rings: Vec::new(), chi: 1 }
}

/// WP-23: a lone cone of index4 -2 (two -1 cones merged by S4) is an interior valence 6 pole,
/// which the QComb rule (ii) accepts; valence 6 anywhere else stays a violation.
#[test]
fn valence_six_is_valid_only_at_a_minus_two_cone() {
    let n = vec![2u32; 18];
    let ok = tmesh_hexagon(-2);
    ok.validate().unwrap();
    let v = comb_violations(&ok, &n, 8).unwrap();
    assert!(v.is_empty(), "{v:?}");
    let c = build_comb(&ok, &n).unwrap();
    assert_eq!(valences(&c).iter().filter(|&&k| k == 6).count(), 1);
    for idx in [0i8, -1] {
        let bad = tmesh_hexagon(idx);
        let v = comb_violations(&bad, &n, 8).unwrap();
        assert!(v.iter().any(|(x, _)| matches!(x, Violation::BadValence(_))), "index4 {idx}: {v:?}");
    }
}

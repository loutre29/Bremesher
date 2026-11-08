//! DESIGN 6.10 tests on testkit analytic params.

mod common;

use aq_core::{ArcKind, ErrKind, NodeKind, Stage};
use aq_layout::{LayoutParams, Seed};
use aq_testkit::*;
use common::*;

#[test]
fn planar_square_one_patch() {
    for (alpha, s) in [(0.0, 1.0), (0.37, 2.0), (-1.1, 0.5)] {
        let c = plane_param(alpha, s).unwrap();
        let (l, st) = build(&c, false, &[]);
        check_layout(&c, &l);
        assert_eq!(st.runs, 1);
        let t = &l.tmesh;
        assert_eq!(t.patches.len(), 1);
        assert_eq!(t.nodes.len(), 4);
        assert_eq!(t.arcs.len(), 4);
        for n in &t.nodes {
            assert_eq!(n.kind, NodeKind::BoundaryCorner);
        }
        for a in &t.arcs {
            assert_eq!(a.kind, ArcKind::Boundary);
            assert!((a.len - 8.0).abs() <= 1e-12, "arc length {} for alpha {alpha}", a.len);
        }
        let corners = t.patch_corners(0);
        let cs: std::collections::BTreeSet<u32> = corners.iter().copied().collect();
        assert_eq!(cs.len(), 4);
        assert_eq!(l.health.max, 0.0);
        assert_eq!(l.wmesh.tris.len(), c.surface.mesh.tris.len());
    }
}

fn forced_seed(c: &ParamCase) {
    let (v, f) = vertex_near(c, [3.3, 4.6]);
    let seeds = [Seed { vert: v, face: f, dir: 255 }];
    let (l, st) = build(c, false, &seeds);
    check_layout(c, &l);
    assert_eq!(st.runs, 1);
    let t = &l.tmesh;
    assert_eq!(t.patches.len(), 4);
    assert_eq!(t.nodes.len(), 9);
    assert_eq!(t.arcs.len(), 12);
    assert_eq!(t.nodes.iter().filter(|n| n.kind == NodeKind::Seed).count(), 1);
    assert_eq!(t.arcs.iter().filter(|a| a.kind == ArcKind::SeedTrace).count(), 4);
    assert!(l.health.max < 1e-6);
    let sum: f64 = t.arcs.iter().filter(|a| a.kind == ArcKind::Boundary).map(|a| a.len).sum();
    assert!((sum - 32.0).abs() < 1e-9);
}

#[test]
fn planar_square_forced_seed_four_patches() {
    forced_seed(&plane_param(0.37, 2.0).unwrap());
    let c = plane_param_with(0.0, 1.0, 8.0, ParamOpts::aligned(2)).unwrap();
    forced_seed(&c);
}

#[test]
fn cube_sphere_six_patches() {
    for n in 2..=4u32 {
        let c = cube_sphere_param(n).unwrap();
        let (l, st) = build(&c, false, &[]);
        check_layout(&c, &l);
        assert_eq!(st.runs, 1);
        let t = &l.tmesh;
        assert_eq!(t.patches.len(), 6, "n = {n}");
        assert_eq!(t.nodes.len(), 8);
        for nd in &t.nodes {
            assert_eq!(nd.kind, NodeKind::Cone);
            assert_eq!(nd.index4, 1);
        }
        assert_eq!(t.arcs.len(), 12);
        for a in &t.arcs {
            assert_eq!(a.len, n as f64, "arc lengths are exact integers");
            assert_eq!(a.kind, ArcKind::Separatrix);
        }
        assert!(occurrences(&l).iter().all(|&o| o == 2));
        assert_eq!(l.health.max, 0.0);
        check_no_duplicates(&l);
    }
    // Finer mesh: the separatrices run along mesh edges and through vertices (SoS).
    let c = cube_sphere_param_with(3, CubeSphereOpts { res: 2, ..Default::default() }).unwrap();
    let (l, _) = build(&c, false, &[]);
    check_layout(&c, &l);
    assert_eq!(l.tmesh.patches.len(), 6);
    assert!(l.tmesh.arcs.iter().all(|a| a.len == 3.0));
    check_no_duplicates(&l);
}

fn check_closed(c: &ParamCase, l: &aq_core::Layout) {
    check_layout(c, l);
    let t = &l.tmesh;
    assert!(occurrences(l).iter().all(|&o| o == 2), "every arc of a closed surface lies on 2 sides");
    assert_eq!(t.nodes.len() as i64 - t.arcs.len() as i64 + t.patches.len() as i64, 2);
    assert!(l.health.max < 1e-6, "health {}", l.health.max);
}

#[test]
fn perturbed_cube_sphere_twenty_seeds() {
    for seed in 0..20u64 {
        let c = perturbed_cube_sphere_param(4, 0.3, seed).unwrap();
        let (l, _) = build(&c, false, &[]);
        check_closed(&c, &l);
        assert_eq!(l.tmesh.nodes.iter().filter(|n| n.kind == NodeKind::Cone).count(), 8);
    }
    for seed in 0..4u64 {
        let c = perturbed_cube_sphere_param(6, 0.4, seed).unwrap();
        let (l, _) = build(&c, false, &[]);
        check_closed(&c, &l);
    }
}

#[test]
fn torus_with_exact_loops() {
    for (n, m) in [(6u32, 4u32), (4, 3)] {
        let c = torus_param(n, m).unwrap();
        let (l, st) = build(&c, true, &[]);
        check_layout(&c, &l);
        assert_eq!(st.runs, 1);
        assert_eq!(st.n_loops, (n + m) as usize);
        assert_eq!(l.tmesh.patches.len(), (n * m) as usize);
        assert!(occurrences(&l).iter().all(|&o| o == 2));
        assert_eq!(l.tmesh.rings.len(), (n + m) as usize);
        assert!(l.tmesh.arcs.iter().all(|a| a.kind == ArcKind::Loop));
        assert!(l.health.max < 1e-6);
    }
}

#[test]
fn bare_torus_gets_a_seed() {
    let c = torus_param(6, 4).unwrap();
    let (l, st) = build(&c, false, &[]);
    check_layout(&c, &l);
    assert_eq!(st.runs, 2);
    assert_eq!(st.seeds.len(), 1);
    assert_eq!(st.seeds[0].dir, 255);
    assert!(!l.tmesh.patches.is_empty());
    assert_eq!(l.tmesh.chi, 0);
}

#[test]
fn open_cylinder_one_seed_round() {
    let c = cylinder_param(8).unwrap();
    let (l, st) = build(&c, false, &[]);
    check_layout(&c, &l);
    assert_eq!(st.runs, 2);
    assert_eq!(l.tmesh.patches.len(), 1);
    assert_eq!(st.seeds.len(), 1);
    assert!(st.seeds[0].dir < 4);
    let (l2, _) = build(&c, true, &[]);
    check_layout(&c, &l2);
    assert!(l2.tmesh.patches.len() > 1);
    assert!(!l2.tmesh.rings.is_empty());
}

#[test]
fn grid_aligned_variants() {
    let cases: Vec<(ParamCase, bool, Vec<Seed>)> = vec![
        (plane_param_with(0.0, 1.0, 6.0, ParamOpts::aligned(1)).unwrap(), false, vec![]),
        (torus_param_with(6, 4, 3.0, 1.0, ParamOpts::aligned(1)).unwrap(), true, vec![]),
        (torus_param_with(4, 4, 3.0, 1.0, ParamOpts::aligned(2)).unwrap(), false, vec![]),
        (cylinder_param_with(6, 3.0, ParamOpts::aligned(1)).unwrap(), false, vec![]),
        (cylinder_param_with(6, 3.0, ParamOpts::aligned(2)).unwrap(), true, vec![]),
        (cube_sphere_param_with(2, CubeSphereOpts { res: 3, ..Default::default() }).unwrap(), false, vec![]),
    ];
    for (i, (c, loops, seeds)) in cases.iter().enumerate() {
        let (l, _) = build(c, *loops, seeds);
        check_layout(c, &l);
        check_no_duplicates(&l);
        let (l2, _) = build(c, *loops, seeds);
        assert!(l == l2, "case {i} not deterministic");
    }
    // A forced seed on the aligned plane: its rays run along mesh edges.
    let c = plane_param_with(0.0, 1.0, 6.0, ParamOpts::aligned(1)).unwrap();
    let (v, f) = vertex_near(&c, [2.0, 3.0]);
    let (l, _) = build(&c, false, &[Seed { vert: v, face: f, dir: 255 }]);
    check_layout(&c, &l);
    check_no_duplicates(&l);
    assert_eq!(l.tmesh.patches.len(), 4);
    let mut lens: Vec<f64> = l.tmesh.arcs.iter().filter(|a| a.kind == ArcKind::SeedTrace).map(|a| a.len).collect();
    lens.sort_by(f64::total_cmp);
    assert_eq!(lens, vec![2.0, 3.0, 3.0, 4.0]);
}

#[test]
fn determinism_runs_and_threads() {
    let c = perturbed_cube_sphere_param(4, 0.3, 7).unwrap();
    let (l0, _) = build(&c, false, &[]);
    for _ in 0..10 {
        let (l, _) = build(&c, false, &[]);
        assert!(l == l0, "run differs");
    }
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let l = pool.install(|| build(&c, false, &[]).0);
        assert!(l == l0, "{threads} threads differ");
    }
    let t = torus_param(5, 3).unwrap();
    let (a, _) = build(&t, true, &[]);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let b = pool.install(|| build(&t, true, &[]).0);
    assert!(a == b);
}

#[test]
fn runaway_with_tiny_trace_cap() {
    let c = cube_sphere_param(4).unwrap();
    let p = LayoutParams { max_trace_len: 0.5, ..Default::default() };
    let e = build_p(&c, false, &[], &p).unwrap_err();
    assert_eq!(e.kind, ErrKind::Infeasible);
    assert_eq!(e.stage, Stage::Layout);
}

#[test]
fn cancelled_ctl() {
    let c = cube_sphere_param(2).unwrap();
    let st = aq_core::CtlState::new();
    st.cancel();
    let ctl = st.unbounded();
    let e = aq_layout::build_layout(&px(&c), &c.param, &[], &LayoutParams::default(), &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
}

#[test]
fn bad_input_sizes() {
    let c = cube_sphere_param(2).unwrap();
    let mut prm = c.param.clone();
    prm.uv.pop();
    let st = aq_core::CtlState::new();
    let ctl = st.unbounded();
    let e = aq_layout::build_layout(&px(&c), &prm, &[], &LayoutParams::default(), &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(e.stage, Stage::Layout);
}

#[test]
fn performance_100k_faces() {
    let c = perturbed_cube_sphere_param_with(30, 0.4, 1, CubeSphereOpts { res: 3, ..Default::default() }).unwrap();
    let nf = c.surface.mesh.tris.len();
    assert!(nf >= 95_000, "{nf} faces");
    let t0 = std::time::Instant::now();
    let (l, _) = build(&c, false, &[]);
    let dt = t0.elapsed();
    println!("perturbed cube sphere, {nf} faces: layout in {dt:?}, {} patches", l.tmesh.patches.len());
    if !cfg!(debug_assertions) {
        assert!(dt.as_secs_f64() < 1.0, "layout took {dt:?}");
    }
    assert!(l.health.max < 1e-6);
}

#[test]
fn voided_stop_events_keep_the_layout_exact() {
    // On this instance some Stop events are voided at processing time because the curve
    // they stop on was truncated before the point in the meantime.
    let c = perturbed_cube_sphere_param(3, 0.1, 0).unwrap();
    let (l, st) = build(&c, false, &[]);
    assert!(st.n_voided > 0);
    check_closed(&c, &l);
}

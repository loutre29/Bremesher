//! DESIGN 6.6 planted singularity tests: +- pairs at 0.5h and 1.0h (50 random placements each
//! on a plane and on a sphere) and a planted +2 vertex.
#![allow(clippy::needless_range_loop)]

mod common;

use aq_core::{CrossField, CurvField, SplitMix64};
use aq_field::{FieldParams, MAX_EDIT_DELTA, edit_singularities, field_energy, resolve_fixed_matching, solve_field};
use aq_testkit::{CurvKind, FieldKind, PlantChart, Surface, analytic_curv, analytic_field};
use common::*;

/// A gently rotating target on the plane (conf 0.5), so that the clean field has a positive
/// energy and the cancellation distance is the normal one (1.2 h).
fn plane_curv(s: &Surface, angle: f64) -> CurvField {
    let mut c = zero_curv(s);
    for f in 0..s.n_faces() {
        let x = centroid(&s.mesh, f as u32);
        let g = angle + 0.15 * (x[0] / 3.0).sin();
        c.phi[f] = angle_in_face(&s.frames, f as u32, [g.cos(), g.sin(), 0.0]);
        c.conf[f] = 0.5;
        c.k1[f] = 0.0;
    }
    c
}

struct Case {
    s: Surface,
    c: CurvField,
    base: CrossField,
    e_ref: f64,
    chart: PlantChart,
    /// Vertices allowed as the first planted vertex.
    seeds: Vec<u32>,
}

fn plane_case() -> Case {
    let angle = 0.2;
    let s = Surface::new(aq_testkit::plane_grid(42, 42, 1.0 / 3.0, angle, 0.15, 9).unwrap()).unwrap();
    let c = plane_curv(&s, angle);
    let sz = size(&s, 1.0);
    let st = ctl_state();
    let (base, _) = solve_field(&px(&s), &c, &sz, &FieldParams::default(), &st.unbounded()).unwrap();
    assert!(interior_cones(&s, &base).is_empty());
    let e_ref = field_energy(&px(&s), &c, &sz, &base, &FieldParams::default()).unwrap();
    assert!(e_ref > 0.0);
    // At least 3.5 h from the boundary.
    let bnd: Vec<u32> = (0..s.n_verts() as u32).filter(|&v| s.topo.is_bnd_vert[v as usize]).collect();
    let db = edge_dijkstra(&s, &bnd);
    let seeds = (0..s.n_verts() as u32).filter(|&v| db[v as usize] > 3.5).collect();
    Case { s, c, base, e_ref, chart: PlantChart::Plane, seeds }
}

fn sphere_case() -> Case {
    let r = 5.0;
    let s = Surface::new(aq_testkit::icosphere(4, r).unwrap()).unwrap();
    let c = analytic_curv(&s, &CurvKind::Sphere { center: [0.0; 3], radius: r }).unwrap();
    let sz = size(&s, 1.0);
    let st = ctl_state();
    let (base, rep) = solve_field(&px(&s), &c, &sz, &FieldParams::default(), &st.unbounded()).unwrap();
    assert!(rep.eigen_path);
    assert_eq!(base.cones().len(), 8);
    let e_ref = field_energy(&px(&s), &c, &sz, &base, &FieldParams::default()).unwrap();
    // At least 4 h from the 8 cones.
    let src: Vec<u32> = base.cones().iter().map(|c| c.0).collect();
    let dc = edge_dijkstra(&s, &src);
    let seeds = (0..s.n_verts() as u32).filter(|&v| dc[v as usize] > 4.0).collect();
    Case { s, c, base, e_ref, chart: PlantChart::Sphere { center: [0.0; 3] }, seeds }
}

/// Plants a pair at Dijkstra distance about `d` (in h) and checks that S4 removes it.
fn run_pairs(cs: &Case, d: f64, n: usize, seed: u64) -> (usize, f64, f64) {
    let s = &cs.s;
    let sz = size(s, 1.0);
    let p = FieldParams::default();
    let mut rng = SplitMix64::new(seed);
    let base_cones = cs.base.cones();
    let mut removed = 0usize;
    let mut worst_ratio = 0.0f64;
    let mut worst_delta = 0.0f64;
    for _ in 0..n {
        let a = cs.seeds[rng.below(cs.seeds.len() as u64) as usize];
        let da = edge_dijkstra(s, &[a]);
        let b = (0..s.n_verts() as u32)
            .filter(|&v| v != a && !s.topo.is_bnd_vert[v as usize])
            .min_by(|&x, &y| (da[x as usize] - d).abs().total_cmp(&(da[y as usize] - d).abs()).then(x.cmp(&y)))
            .unwrap();
        let (plus, minus) = if rng.below(2) == 0 { (a, b) } else { (b, a) };
        let kind = FieldKind::planted_pair(FieldKind::Theta(cs.base.theta.clone()), plus, minus, cs.chart.clone());
        let mut f = analytic_field(s, &kind).unwrap();
        // The planted field carries exactly the base cones plus the pair.
        let mut want = base_cones.clone();
        want.push((plus, 1));
        want.push((minus, -1));
        want.sort_unstable();
        assert_eq!(f.cones(), want, "planting at {plus}/{minus} (d = {})", da[b as usize]);
        resolve_fixed_matching(&px(s), &cs.c, &sz, &mut f, &p).unwrap();
        let st = ctl_state();
        let rep = edit_singularities(&px(s), &cs.c, &sz, &mut f, &p, &st.unbounded()).unwrap();
        f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
        assert_eq!(f.cones(), base_cones, "pair {plus}/{minus} at d = {} not removed: {rep:?}", da[b as usize]);
        removed += 1;
        let md = f.delta.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(md <= MAX_EDIT_DELTA, "|delta| {md} > pi/3 after the cancellation");
        worst_delta = worst_delta.max(md);
        let e = field_energy(&px(s), &cs.c, &sz, &f, &p).unwrap();
        let ratio = e / cs.e_ref;
        assert!(ratio < 1.05, "energy increase {ratio} >= 5% (pair {plus}/{minus})");
        worst_ratio = worst_ratio.max(ratio);
    }
    (removed, worst_ratio, worst_delta)
}

#[test]
fn planted_pairs_on_plane() {
    let cs = plane_case();
    for (d, seed) in [(0.5, 1u64), (1.0, 2)] {
        let (n, ratio, md) = run_pairs(&cs, d, 50, seed);
        assert_eq!(n, 50);
        println!("plane d = {d}h: 50/50 removed, worst energy ratio {ratio:.6}, max |delta| {md:.3}");
    }
}

#[test]
fn planted_pairs_on_sphere() {
    let cs = sphere_case();
    for (d, seed) in [(0.5, 3u64), (1.0, 4)] {
        let (n, ratio, md) = run_pairs(&cs, d, 50, seed);
        assert_eq!(n, 50);
        println!("sphere d = {d}h: 50/50 removed, worst energy ratio {ratio:.6}, max |delta| {md:.3}");
    }
}

#[test]
fn planted_plus_two_is_split() {
    let cs = sphere_case();
    let s = &cs.s;
    let sz = size(s, 1.0);
    let p = FieldParams::default();
    let mut rng = SplitMix64::new(17);
    for _ in 0..5 {
        let v = cs.seeds[rng.below(cs.seeds.len() as u64) as usize];
        let kind = FieldKind::Planted {
            base: Box::new(FieldKind::Theta(cs.base.theta.clone())),
            sings: vec![(v, 2)],
            chart: cs.chart.clone(),
        };
        let mut f = analytic_field(s, &kind).unwrap();
        assert_eq!(f.index4[v as usize], 2);
        let st = ctl_state();
        let rep = edit_singularities(&px(s), &cs.c, &sz, &mut f, &p, &st.unbounded()).unwrap();
        assert!(rep.split >= 1);
        // WP-23: a -2 cone (a merged pair of -1 cones) is allowed; no +2 may remain.
        assert!(f.index4.iter().all(|&i| aq_field::interior_index_ok(i)), "a high index remains");
        let dv = edge_dijkstra(s, &[v]);
        let near: Vec<(u32, i8)> = f.cones().into_iter().filter(|&(u, _)| dv[u as usize] < 2.0).collect();
        assert_eq!(near.len(), 2, "cones near the planted +2 at {v}: {near:?}");
        assert!(near.iter().all(|&(_, i)| i == 1));
    }
}

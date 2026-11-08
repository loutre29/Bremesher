//! DESIGN 6.5 M2 thickness cap tests (WP-20): shape diameter, plate filter, the cap with its
//! budget floor, and the global plan across components.

mod common;

use std::f64::consts::TAU;

use common::*;

use aq_core::{CurvField, TriMesh, math};
use aq_curv::{G_MIN, SizingParams, shape_diameter, sizing_diam, sizing_with, thickness_plan};
use aq_testkit::Surface;

const EDGE: f64 = 1.0 / 3.0;

fn closed_cylinder(r: f64, len: f64, seed: u64) -> TriMesh {
    let na = ((TAU * r / EDGE).round() as u32).max(6);
    let nl = (len / EDGE).round() as u32;
    aq_testkit::cylinder_closed(r, len, na, nl, 0.1, seed).unwrap()
}

/// A curvature field with no tube anywhere, so only the thickness cap acts.
fn flat_curv(s: &Surface) -> CurvField {
    let (nf, nv) = (s.n_faces(), s.n_verts());
    CurvField { k1: vec![0.0; nf], k2: vec![0.0; nf], phi: vec![0.0; nf], conf: vec![0.0; nf], kappa_hat: vec![0.0; nv] }
}

/// Thin closed box: `w x w x t`, cells of about EDGE.
fn slab(w: f64, t: f64) -> TriMesh {
    let n = (w / EDGE).round() as u32;
    let mut m = aq_testkit::cube_sharp(n, 1.0, 0.0, 1).unwrap().mesh;
    for p in m.pos.iter_mut() {
        *p = [p[0] * w, p[1] * w, p[2] * t];
    }
    m
}

#[test]
fn diameter_of_a_thin_cylinder() {
    let (r, len) = (0.5, 8.0);
    let s = surf(closed_cylinder(r, len, 1));
    let st = ctl_state();
    let d = shape_diameter(&view(&s), &st.unbounded()).unwrap();
    let mut n = 0;
    for (v, p) in s.mesh.pos.iter().enumerate() {
        // Side vertices more than one diameter from the caps.
        if p[2] > 1.0 && p[2] < len - 1.0 && (p[0].hypot(p[1]) - r).abs() < 1e-6 {
            n += 1;
            assert!(d[v] > 0.9 * 2.0 * r && d[v] < 1.05 * 2.0 * r, "vertex {v}: d {} for 2R {}", d[v], 2.0 * r);
        }
    }
    assert!(n > 100);
}

#[test]
fn plate_interior_is_free_and_rim_is_thin() {
    // A 6 x 6 x 0.3 slab: the middle of the big faces is a plate (no cap), the rim is thin.
    let s = surf(slab(6.0, 0.3));
    let st = ctl_state();
    let d = shape_diameter(&view(&s), &st.unbounded()).unwrap();
    let (mut mid, mut rim) = (0, 0);
    for (v, p) in s.mesh.pos.iter().enumerate() {
        let edge_dist = (3.0 - p[0].abs()).min(3.0 - p[1].abs());
        if edge_dist > 1.0 {
            mid += 1;
            assert!(d[v].is_infinite(), "plate vertex {v} at {p:?}: d {}", d[v]);
        } else if edge_dist < 1e-9 && (p[2].abs() - 0.15).abs() < 1e-9 {
            // The rim edges: their rays cross the slab and the surface turns within d. (Mid-wall
            // vertices look across the whole slab; the gradation covers them.)
            rim += 1;
            assert!(d[v] < 1.0, "rim vertex {v}: d {}", d[v]);
        }
    }
    assert!(mid > 50 && rim > 20, "mid {mid}, rim {rim}");
}

#[test]
fn layered_sheets_are_not_thin() {
    // Two parallel open sheets 0.2 apart, both facing +z: a ray from the upper one meets the
    // front of the lower one, which the back-face test rejects.
    let a = aq_testkit::plane_grid(20, 20, EDGE, 0.0, 0.1, 2).unwrap();
    let mut b = aq_testkit::plane_grid(20, 20, EDGE, 0.0, 0.1, 3).unwrap();
    translate(&mut b, [0.0, 0.0, -0.2]);
    let s = surf(aq_testkit::merge(&[a, b]));
    let st = ctl_state();
    let d = shape_diameter(&view(&s), &st.unbounded()).unwrap();
    assert!(d.iter().all(|x| x.is_infinite()));
}

/// DESIGN 6.5 M2: thin cylinder R = 0.5 next to a sheet: 2 pi R / h >= 8 from the thickness cap
/// alone (no curvature tube), when the budget allows it.
#[test]
fn thin_cylinder_thickness_cap() {
    let r = 0.5;
    let tube = closed_cylinder(r, 8.0, 3);
    let n_tube = tube.pos.len();
    let mut sheet = aq_testkit::plane_grid(60, 60, EDGE, 0.0, 0.2, 4).unwrap();
    translate(&mut sheet, [5.0, -10.0, 0.0]);
    let s = surf(aq_testkit::merge(&[tube, sheet]));
    let n: f64 = s.frames.area.iter().sum();
    let st = ctl_state();
    let ctl = st.unbounded();
    let c = flat_curv(&s);
    let sp = SizingParams { thickness: true, thick_budget: 1.0, thick_floor_min: G_MIN, ..SizingParams::new(n) };
    let (sf, rep) = sizing_with(&view(&s), &c, &sp, None, &ctl).unwrap();
    println!("thickness cap: {rep:?}");
    assert_eq!(rep.tube_verts, 0);
    assert!(rep.thick_capped > 0 && !rep.cap_infeasible);
    let mut worst = f64::INFINITY;
    for v in 0..n_tube {
        let p = s.mesh.pos[v];
        if p[2] > 1.0 && p[2] < 7.0 && (p[0].hypot(p[1]) - r).abs() < 1e-6 {
            worst = worst.min(TAU * r / sf.h_vert[v]);
        }
    }
    println!("min 2 pi R / h = {worst}");
    assert!(worst >= 8.0, "2 pi R / h = {worst}");
    assert!((sf.predicted_count(&s.frames) - n).abs() <= 1e-9 * n);
    // Without the thickness cap (and no curvature) the size is uniform: 2 pi R / h is about 3.
    let (sf0, _) = sizing_with(&view(&s), &c, &SizingParams::new(n), None, &ctl).unwrap();
    assert!(TAU * r / sf0.h_vert[0] < 4.0);
}

/// The budget floor: with the default 10% budget the thickness caps spend at most about 10% of
/// N (estimate without gradation), the floor is above G_MIN, the count is exact and the
/// gradation holds.
#[test]
fn thickness_budget_floor() {
    let tube = closed_cylinder(0.3, 8.0, 5);
    let mut sheet = aq_testkit::plane_grid(45, 45, EDGE, 0.0, 0.2, 6).unwrap();
    translate(&mut sheet, [5.0, -7.0, 0.0]);
    let s = surf(aq_testkit::merge(&[tube, sheet]));
    let n: f64 = s.frames.area.iter().sum();
    let st = ctl_state();
    let ctl = st.unbounded();
    let c = flat_curv(&s);
    let sp = SizingParams { thickness: true, thick_floor_min: G_MIN, ..SizingParams::new(n) };
    let (sf, rep) = sizing_with(&view(&s), &c, &sp, None, &ctl).unwrap();
    println!("budget floor: {rep:?}");
    assert!(rep.thick_floor > G_MIN && rep.thick_floor.is_finite());
    assert!(rep.thick_share <= 0.1 + 1e-6, "share {}", rep.thick_share);
    assert!((sf.predicted_count(&s.frames) - n).abs() <= 1e-9 * n);
    let hmin = sf.h_vert.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(hmin >= rep.thick_floor * (1.0 - 1e-9), "h min {hmin} below the floor {}", rep.thick_floor);
    // Gradation (Persson, grad 0.25) on every edge.
    for e in 0..3 * s.n_faces() as u32 {
        if !s.topo.is_canonical(e) {
            continue;
        }
        let a = s.topo.tail(&s.mesh, e) as usize;
        let b = s.topo.head(&s.mesh, e) as usize;
        let l = math::dist(s.mesh.pos[a], s.mesh.pos[b]);
        for (x, y) in [(a, b), (b, a)] {
            let h = &sf.h_vert;
            assert!(h[y].ln() - h[x].ln() - 0.25 * l / h[x] <= 1e-9);
        }
    }
    // The default floor minimum (0.7 h) holds too.
    let (_, rep2) = sizing_with(&view(&s), &c, &SizingParams { thickness: true, ..SizingParams::new(n) }, None, &ctl).unwrap();
    assert!(rep2.thick_floor >= 0.7);
    // A global floor from the engine only raises the floor solved in the component, and
    // precomputed diameters give the same field as computing them inside.
    let d = shape_diameter(&view(&s), &ctl).unwrap();
    let low = SizingParams { thickness: true, thick_floor: Some(0.6), ..SizingParams::new(n) };
    let (_, rl) = sizing_with(&view(&s), &c, &low, None, &ctl).unwrap();
    assert_eq!(rl.thick_floor, rep2.thick_floor);
    let high = SizingParams { thickness: true, thick_floor: Some(2.0 * rep2.thick_floor), ..SizingParams::new(n) };
    let (a, ra) = sizing_with(&view(&s), &c, &high, None, &ctl).unwrap();
    let (b, _) = sizing_diam(&view(&s), &c, &high, None, Some(&d), &ctl).unwrap();
    assert_eq!(ra.thick_floor, 2.0 * rep2.thick_floor);
    assert_eq!(a.h_vert, b.h_vert);
}

/// Global plan (engine, WP-20): a thin tube component gets more than its area share.
#[test]
fn plan_gives_thin_components_more() {
    let tube = closed_cylinder(0.3, 8.0, 7);
    let sphere = aq_testkit::icosphere(4, 6.0).unwrap();
    let st = ctl_state();
    let ctl = st.unbounded();
    let area = |m: &TriMesh| (0..m.tris.len() as u32).map(|f| m.face_area(f)).sum::<f64>();
    let (at, asph) = (area(&tube), area(&sphere));
    let n = at + asph; // h0 = 1
    let plan = thickness_plan(&[&tube, &sphere], &[true, true], 8, n, 0.25, 0.5, &ctl).unwrap();
    println!("plan: {plan:?}, tube area {at}, sphere area {asph}");
    assert!((plan.demand.iter().sum::<f64>() - n).abs() < 1e-6 * n);
    assert!(plan.demand[0] > 1.5 * at, "tube demand {} for area {at}", plan.demand[0]);
    assert!(plan.floor >= 0.5);
    assert!(plan.thin_area[1] == 0.0 && plan.thin_area[0] > 0.5 * at);
    // Deterministic across thread counts.
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| {
        thickness_plan(&[&tube, &sphere], &[true, true], 8, n, 0.25, 0.5, &ctl).unwrap()
    });
    assert_eq!(p1, plan);
}

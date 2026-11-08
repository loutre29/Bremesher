//! DESIGN 6.6 tests (M1 part) on testkit surfaces.
#![allow(clippy::needless_range_loop)]

mod common;

use aq_core::math;
use aq_core::{CrossField, ErrKind, NONE};
use aq_field::{FieldParams, edit_path, resolve_fixed_matching, solve_field, try_edit_path};
use aq_testkit::{CurvKind, FieldKind, Surface, analytic_curv, analytic_field};
use common::*;

fn run(s: &Surface, c: &aq_core::CurvField, h: f64, p: &FieldParams) -> (CrossField, aq_field::FieldReport) {
    let st = ctl_state();
    let ctl = st.unbounded();
    let sz = size(s, h);
    let (f, rep) = solve_field(&px(s), c, &sz, p, &ctl).unwrap();
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
    (f, rep)
}

#[test]
fn planar_square_with_boundary() {
    for (jit, seed) in [(0.0, 1u64), (0.2, 2), (0.3, 3)] {
        let s = Surface::new(aq_testkit::square_disk(30, 10.0, jit, seed).unwrap()).unwrap();
        let (f, rep) = run(&s, &zero_curv(&s), 1.0, &FieldParams::default());
        assert!(interior_cones(&s, &f).is_empty(), "interior cones {:?}", interior_cones(&s, &f));
        // The 4 corners have index4 +1, nothing else on the boundary.
        let bc: Vec<_> = f.cones().into_iter().filter(|&(v, _)| s.topo.is_bnd_vert[v as usize]).collect();
        assert_eq!(bc.len(), 4, "boundary cones {bc:?}");
        assert!(bc.iter().all(|&(_, i)| i == 1));
        // Constrained faces exact.
        let mut nh = 0;
        for fc in 0..s.n_faces() {
            for k in 0..3 {
                let h = (3 * fc + k) as u32;
                if s.topo.twin[h as usize] == NONE {
                    assert!(f.hard[fc]);
                    let a = s.frames.edge_angle(&s.mesh, h);
                    assert!(cross_dev(f.theta[fc], a) < 1e-12, "face {fc}: {}", cross_dev(f.theta[fc], a));
                    nh += 1;
                }
            }
        }
        assert!(nh > 0);
        assert!(!rep.eigen_path);
        assert!(rep.max_abs_delta < 0.5);
    }
}

/// Fraction of faces whose field is within `tol` of the circumferential direction about z.
fn circ_fraction(s: &Surface, f: &CrossField, tol: f64) -> f64 {
    let mut ok = 0usize;
    for fc in 0..s.n_faces() {
        let c = centroid(&s.mesh, fc as u32);
        let d = [-c[1], c[0], 0.0];
        let a = angle_in_face(&s.frames, fc as u32, d);
        if cross_dev(f.theta[fc], a) < tol {
            ok += 1;
        }
    }
    ok as f64 / s.n_faces() as f64
}

#[test]
fn open_cylinder_zero_cones_circumferential() {
    // R = 3h, proxy edge about h/3.
    let s = Surface::new(aq_testkit::cylinder_open(3.0, 12.0, 56, 36, 0.2, 7).unwrap()).unwrap();
    let c = analytic_curv(&s, &CurvKind::Cylinder { radius: 3.0 }).unwrap();
    let (f, _) = run(&s, &c, 1.0, &FieldParams::default());
    assert!(f.cones().is_empty(), "cones {:?}", f.cones());
    let frac = circ_fraction(&s, &f, 2.0 * DEG);
    println!("within 2 degrees on {frac:.4} of faces");
    assert!(frac >= 0.95, "circumferential within 2 degrees on {frac}");
    // Also with the test-local curvature estimate.
    let ce = estimate_curv(&s, 1.0);
    let (f, _) = run(&s, &ce, 1.0, &FieldParams::default());
    assert!(f.cones().is_empty(), "cones {:?}", f.cones());
    let frac = circ_fraction(&s, &f, 2.0 * DEG);
    println!("within 2 degrees on {frac:.4} of faces");
    assert!(frac >= 0.95, "estimated curvature: circumferential within 2 degrees on {frac}");
}

#[test]
fn torus_zero_cones_along_circles() {
    let (big_r, r) = (4.0, 1.5);
    let s = Surface::new(aq_testkit::torus(big_r, r, 104, 30, 0.2, 11).unwrap()).unwrap();
    for c in [analytic_curv(&s, &CurvKind::Torus { big_r, r }).unwrap(), estimate_curv(&s, 1.0)] {
        let (f, rep) = run(&s, &c, 1.0, &FieldParams::default());
        assert!(f.cones().is_empty(), "cones {:?} ({rep:?})", f.cones());
        let frac = circ_fraction(&s, &f, 2.0 * DEG);
        println!("within 2 degrees on {frac:.4} of faces");
        assert!(frac >= 0.95, "torus: within 2 degrees of the circles on {frac}");
    }
}

#[test]
fn cube_with_hard_chains_has_8_corner_cones() {
    for (n, jit, seed) in [(12u32, 0.0, 1u64), (12, 0.15, 2), (9, 0.25, 3)] {
        let sm = aq_testkit::cube_sharp(n, 4.0, jit, seed).unwrap();
        let s = Surface::from_synth(&sm).unwrap();
        let (f, _) = run(&s, &zero_curv(&s), 1.0, &FieldParams::default());
        let cones = f.cones();
        assert_eq!(cones.len(), 8, "cones {cones:?}");
        for &(v, i) in &cones {
            assert_eq!(i, 1);
            assert!(s.feats.vert_corner[v as usize]);
            let p = s.mesh.pos[v as usize];
            assert!(p.iter().all(|x| (x.abs() - 2.0).abs() < 1e-12), "cone {v} at {p:?} is not a cube corner");
        }
        // Hard chain faces exact.
        for fc in 0..s.n_faces() {
            for k in 0..3 {
                let h = (3 * fc + k) as u32;
                if s.feats.he_chain[h as usize] != NONE {
                    assert!(f.hard[fc]);
                    assert!(cross_dev(f.theta[fc], s.frames.edge_angle(&s.mesh, h)) < 1e-12);
                }
            }
        }
    }
}

#[test]
fn icosphere_eigen_path_8_cones() {
    let r = 5.0;
    let s = Surface::new(aq_testkit::icosphere(4, r).unwrap()).unwrap();
    for seed in [0u64, 1, 7] {
        let p = FieldParams { seed, ..FieldParams::default() };
        let c = analytic_curv(&s, &CurvKind::Sphere { center: [0.0; 3], radius: r }).unwrap();
        let (f, rep) = run(&s, &c, 1.0, &p);
        assert!(rep.eigen_path);
        let cones = f.cones();
        assert_eq!(cones.len(), 8, "cones {cones:?} {rep:?}");
        assert!(cones.iter().all(|&(_, i)| i == 1));
        let mut dmin = f64::INFINITY;
        for i in 0..8 {
            for j in i + 1..8 {
                let a = s.mesh.pos[cones[i].0 as usize];
                let b = s.mesh.pos[cones[j].0 as usize];
                let cosang = (math::dot(a, b) / (math::norm(a) * math::norm(b))).clamp(-1.0, 1.0);
                dmin = dmin.min(r * cosang.acos());
            }
        }
        assert!(dmin >= 0.85 * r, "min pairwise geodesic distance {dmin} < 0.85 R");
    }
}

#[test]
fn hemisphere_symmetry_loop_4_cones() {
    let r = 4.0;
    let sm = aq_testkit::hemisphere(20, r).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    assert!(s.feats.chains.iter().any(|c| c.kind == aq_core::FeatureKind::Symmetry && c.corners.is_empty()));
    let c = analytic_curv(&s, &CurvKind::Sphere { center: [0.0; 3], radius: r }).unwrap();
    let (f, _) = run(&s, &c, 1.0, &FieldParams::default());
    let cones = f.cones();
    assert_eq!(cones.len(), 4, "cones {cones:?}");
    for &(v, i) in &cones {
        assert_eq!(i, 1);
        let p = s.mesh.pos[v as usize];
        let dist = r * (p[2] / r).clamp(-1.0, 1.0).asin();
        assert!(dist >= 0.5, "cone {v} only {dist} from the Symmetry chain");
    }
}

#[test]
fn edit_sign_and_index_stability() {
    // Plane with a constant field: no interior cones.
    let s = Surface::new(aq_testkit::plane_grid(24, 24, 1.0 / 3.0, 0.3, 0.2, 5).unwrap()).unwrap();
    let c = zero_curv(&s);
    let sz = size(&s, 1.0);
    let p = FieldParams::default();
    let mut f = analytic_field(&s, &FieldKind::Constant { angle: 0.3 }).unwrap();
    resolve_fixed_matching(&px(&s), &c, &sz, &mut f, &p).unwrap();
    // Pick an interior path a -> b of 3 edges.
    let a = (12 * 25 + 12) as u32;
    let path = [a, a + 1, a + 2, a + 27];
    let path: Vec<u32> = {
        // Keep only consecutive mesh edges.
        let mut v = vec![path[0]];
        for &w in &path[1..] {
            if s.topo.find_he(&s.mesh, *v.last().unwrap(), w).is_some() {
                v.push(w);
            }
        }
        v
    };
    assert!(path.len() >= 3);
    let (a, b) = (path[0], *path.last().unwrap());
    let before = f.clone();
    edit_path(&px(&s), &mut f, &path, 1);
    assert_eq!(f.index4[a as usize], before.index4[a as usize] + 1);
    assert_eq!(f.index4[b as usize], before.index4[b as usize] - 1);
    // The incremental index equals its recomputation.
    let idx = aq_core::index4_all(&s.mesh, &s.topo, &s.frames, &f.delta).unwrap();
    assert_eq!(idx, f.index4);
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
    // Moving a quarter of positive index with s = -1.
    let mut g = before.clone();
    edit_path(&px(&s), &mut g, &path, -1);
    assert_eq!(g.index4[a as usize], -1);
    assert_eq!(g.index4[b as usize], 1);
    // Index stability: the re-solve leaves p and index4 bitwise unchanged.
    let (p0, i0) = (f.p.clone(), f.index4.clone());
    resolve_fixed_matching(&px(&s), &c, &sz, &mut f, &p).unwrap();
    assert_eq!(f.p, p0);
    assert_eq!(f.index4, i0);
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
    // An invalid path leaves the field unchanged.
    let snap = f.clone();
    edit_path(&px(&s), &mut f, &[a, a + 300], 1);
    assert_eq!(f, snap);
    let e = try_edit_path(&px(&s), &mut f, &[0, 1], 1).unwrap_err(); // boundary edge
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(f, snap);
}

#[test]
fn index_stability_on_solved_fields() {
    let s = Surface::new(aq_testkit::torus(4.0, 1.5, 80, 24, 0.25, 3).unwrap()).unwrap();
    let c = estimate_curv(&s, 1.0);
    let sz = size(&s, 1.0);
    let p = FieldParams::default();
    let (mut f, _) = run(&s, &c, 1.0, &p);
    let (p0, i0) = (f.p.clone(), f.index4.clone());
    for _ in 0..3 {
        resolve_fixed_matching(&px(&s), &c, &sz, &mut f, &p).unwrap();
        assert_eq!(f.p, p0);
        assert_eq!(f.index4, i0);
    }
    // A field with a +-1 pair planted by an edit keeps its indices through the re-solve too.
    let v = (0..s.n_verts() as u32).find(|&v| f.index4[v as usize] == 0).unwrap();
    let w = s.topo.head(&s.mesh, s.topo.vert_he[v as usize]);
    let x = s.topo.fan_out(w).map(|h| s.topo.head(&s.mesh, h)).find(|&x| x != v).unwrap();
    edit_path(&px(&s), &mut f, &[v, w, x], 1);
    assert_eq!((f.index4[v as usize], f.index4[x as usize]), (1, -1));
    let (p1, i1) = (f.p.clone(), f.index4.clone());
    resolve_fixed_matching(&px(&s), &c, &sz, &mut f, &p).unwrap();
    assert_eq!(f.p, p1);
    assert_eq!(f.index4, i1);
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
}

#[test]
fn rejects_bad_inputs_and_cancel() {
    let s = Surface::new(aq_testkit::square_disk(6, 2.0, 0.0, 1).unwrap()).unwrap();
    let st = ctl_state();
    let ctl = st.unbounded();
    let mut c = zero_curv(&s);
    let sz = size(&s, 1.0);
    c.conf.pop();
    let e = solve_field(&px(&s), &c, &sz, &FieldParams::default(), &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(e.stage, aq_core::Stage::Field);
    let c = zero_curv(&s);
    let mut bad = sz.clone();
    bad.h_face[0] = f64::NAN;
    assert_eq!(solve_field(&px(&s), &c, &bad, &FieldParams::default(), &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    st.cancel();
    let e = solve_field(&px(&s), &c, &sz, &FieldParams::default(), &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
    // Two non-orthogonal constraint edges in one face (no feature isolation): Invariant.
    let m = aq_core::TriMesh { pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.0, 0.0]], tris: vec![[0, 1, 2]] };
    let s = Surface::new(m).unwrap();
    let st = ctl_state();
    let e = solve_field(&px(&s), &zero_curv(&s), &size(&s, 1.0), &FieldParams::default(), &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::Invariant);
}

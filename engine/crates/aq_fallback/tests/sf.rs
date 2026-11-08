//! DESIGN 6.14 tests: sphere, torus, cube, noisy blob, open cylinder and a 20-triangle piece
//! each give 100% quads, manifold output, 0 flips and a count within 25% of the target.
//! Targets cover both remesh paths: the local remesher (expected triangle count below
//! aq_prep::MIN_PROXY_TRIS, about 4000 quads) and aq_prep::build_proxy (above).

mod common;

use aq_core::{CtlState, ErrKind, Projector, TriMesh};
use aq_fallback::{SfParams, VFLAG_BOUNDARY, VFLAG_CORNER, sf_batch, sf_quadrangulate, vflag_chain};
use aq_testkit::{add_noise, cube_sharp, cylinder_open, icosphere, plane_grid, torus};
use common::*;

const TARGETS: [u32; 3] = [400, 2000, 6000];

fn suite(name: &str, m: TriMesh, pairs: &[[u32; 2]], kinds: &[u8]) {
    for &t in &TARGETS {
        let c = single(m.clone(), pairs, kinds, t as f64);
        let q = run(&c, t);
        check(&format!("{name} @{t}"), &q, &c, t);
    }
}

#[test]
fn sphere() {
    suite("sphere", icosphere(5, 1.0).unwrap(), &[], &[]);
}

#[test]
fn torus_mesh() {
    suite("torus", torus(3.0, 1.0, 96, 32, 0.2, 7).unwrap(), &[], &[]);
}

#[test]
fn cube() {
    let s = cube_sharp(16, 2.0, 0.2, 3).unwrap();
    suite("cube", s.mesh.clone(), &s.feat_pairs, &s.feat_kind);
    // Corners stay exact and are flagged; chain vertices carry their chain.
    let c = single(s.mesh.clone(), &s.feat_pairs, &s.feat_kind, 2000.0);
    let q = run(&c, 2000);
    let corners = q.vflag.iter().filter(|&&f| f & VFLAG_CORNER != 0).count();
    assert_eq!(corners, 8, "cube corners");
    let half = (c.mesh.pos.iter().map(|p| p[0].abs()).fold(0.0, f64::max) * 1e12).round() / 1e12;
    for (v, &f) in q.vflag.iter().enumerate() {
        if f & VFLAG_CORNER != 0 {
            for k in 0..3 {
                assert!((q.pos[v][k].abs() - half).abs() < 1e-9, "corner off the cube corner");
            }
        }
    }
    // No quad straddles a cube edge: every quad lies in one face plane.
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        let on_plane = (0..3).any(|k| {
            let s0 = q.pos[fv[0] as usize][k];
            s0.abs() > half - 1e-6 && fv.iter().all(|&v| (q.pos[v as usize][k] - s0).abs() < 1e-6)
        });
        assert!(on_plane, "quad {f} crosses a cube edge");
    }
    assert!(q.vflag.iter().any(|&f| vflag_chain(f) != aq_core::NONE));
}

#[test]
fn noisy_blob() {
    let mut m = icosphere(5, 1.0).unwrap();
    add_noise(&mut m, 0.08, 0.5, 11).unwrap();
    suite("noisy blob", m, &[], &[]);
}

#[test]
fn open_cylinder() {
    let m = cylinder_open(1.0, 4.0, 64, 40, 0.2, 5).unwrap();
    suite("open cylinder", m.clone(), &[], &[]);
    let c = single(m, &[], &[], 2000.0);
    let q = run(&c, 2000);
    assert!(q.vflag.iter().any(|&f| f & VFLAG_BOUNDARY != 0));
}

#[test]
fn twenty_triangle_pieces() {
    // An open 5 x 2 grid (20 triangles) and a closed icosahedron (20 triangles).
    let open = plane_grid(5, 2, 1.0, 0.3, 0.2, 1).unwrap();
    assert_eq!(open.tris.len(), 20);
    let closed = icosphere(0, 1.0).unwrap();
    assert_eq!(closed.tris.len(), 20);
    for (name, m) in [("open piece", open), ("icosahedron", closed)] {
        for t in [24u32, 40, 120] {
            let c = single(m.clone(), &[], &[], t as f64);
            let q = run(&c, t);
            check(&format!("{name} @{t}"), &q, &c, t);
            // The debris batch path gives the same guarantees.
            let st = CtlState::new();
            let b = sf_batch(&[&c], &[SfParams { target: t, relax_iters: 5 }], &st.unbounded()).unwrap();
            check(&format!("{name} batch @{t}"), &b[0], &c, t);
        }
    }
}

#[test]
fn batch_matches_order_and_threads() {
    let a = single(icosphere(3, 1.0).unwrap(), &[], &[], 300.0);
    let b = single(torus(3.0, 1.0, 48, 16, 0.1, 2).unwrap(), &[], &[], 500.0);
    let c = single(plane_grid(6, 6, 1.0, 0.0, 0.1, 3).unwrap(), &[], &[], 60.0);
    let srcs = [&a, &b, &c];
    let ps = [300u32, 500, 60].map(|t| SfParams { target: t, relax_iters: 5 });
    let st = CtlState::new();
    let run_with = |n: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
        pool.install(|| sf_batch(&srcs, &ps, &st.unbounded()).unwrap())
    };
    let r1 = run_with(1);
    let r4 = run_with(4);
    assert_eq!(r1, r4, "sf_batch depends on the thread count");
    for (i, q) in r1.iter().enumerate() {
        check(&format!("batch {i}"), q, srcs[i], ps[i].target);
    }
}

#[test]
fn determinism_threads() {
    let mut m = icosphere(5, 1.0).unwrap();
    add_noise(&mut m, 0.05, 0.5, 4).unwrap();
    for t in [1500u32, 6000] {
        let c = single(m.clone(), &[], &[], t as f64);
        let go = |n: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| run(&c, t))
        };
        let q1 = go(1);
        let q4 = go(4);
        let q16 = go(16);
        assert!(q1 == q4 && q4 == q16, "output depends on the thread count (target {t})");
    }
}

#[test]
fn errors() {
    let c = single(icosphere(3, 1.0).unwrap(), &[], &[], 500.0);
    let (fr, bvh) = c.source_geometry().unwrap();
    let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
    let st = CtlState::new();
    let e = sf_quadrangulate(&c, &proj, &SfParams { target: 0, relax_iters: 5 }, &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    let st = CtlState::new();
    st.cancel();
    let e = sf_quadrangulate(&c, &proj, &SfParams { target: 500, relax_iters: 5 }, &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
    let e = sf_batch(&[&c], &[], &CtlState::new().unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
}

#[test]
fn relaxation_improves_angles() {
    let mut m = icosphere(5, 1.0).unwrap();
    add_noise(&mut m, 0.03, 0.6, 2).unwrap();
    for t in [800u32, 6000] {
        let c = single(m.clone(), &[], &[], t as f64);
        let (fr, bvh) = c.source_geometry().unwrap();
        let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
        let st = CtlState::new();
        let raw = sf_quadrangulate(&c, &proj, &SfParams { target: t, relax_iters: 0 }, &st.unbounded()).unwrap();
        let rel = sf_quadrangulate(&c, &proj, &SfParams { target: t, relax_iters: 5 }, &st.unbounded()).unwrap();
        let (a0, a1) = (angle_dev_deg(&raw), angle_dev_deg(&rel));
        eprintln!("target {t}: angle deviation {a0:.2} deg without relaxation, {a1:.2} deg with");
        check("unrelaxed", &raw, &c, t);
        assert!(a1 < a0, "relaxation did not improve the angles");
    }
}

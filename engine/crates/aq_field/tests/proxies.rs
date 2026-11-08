//! WP-08 definition of done: Poincare-Hopf and the DESIGN 7.3 S4 invariants on all 12 testdata
//! meshes through real WP-03 proxies (aq_prep is a dev-dependency only; the library still
//! depends on aq_core and aq_linalg alone). Features as the CLI's `--features auto` at 30
//! degrees (Hard chains, hard_mode 2), 5000 target quads, proxy edge h/3, uniform h = 1, and
//! the test-local quad-scale curvature of `tests/common`.
#![allow(clippy::needless_range_loop)]

mod common;

use aq_core::math;
use aq_core::{FeatureKind, Projector, TriMesh};
use aq_field::{FieldParams, solve_field};
use aq_prep::{ProxyParams, build_proxy, classify_chains, ingest, isolate_features_with, split_components};
use aq_testkit::Surface;
use common::*;

/// Edges whose dihedral angle exceeds `deg`, as Hard feature pairs.
fn auto_features(m: &TriMesh, deg: f64) -> (Vec<[u32; 2]>, Vec<u8>) {
    let mut e: Vec<(u32, u32, usize)> = Vec::new();
    for (f, t) in m.tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            e.push((a.min(b), a.max(b), f));
        }
    }
    e.sort_unstable();
    let mut pairs = Vec::new();
    let lim = deg.to_radians().cos();
    let mut i = 0;
    while i < e.len() {
        let mut j = i;
        while j < e.len() && e[j].0 == e[i].0 && e[j].1 == e[i].1 {
            j += 1;
        }
        if j - i == 2
            && let (Some(n1), Some(n2)) = (m.face_normal(e[i].2 as u32), m.face_normal(e[i + 1].2 as u32))
            && math::dot(n1, n2) < lim
        {
            pairs.push([e[i].0, e[i].1]);
        }
        i = j;
    }
    let kinds = vec![FeatureKind::Hard as u8; pairs.len()];
    (pairs, kinds)
}

#[test]
fn poincare_hopf_and_s4_invariants_on_testdata_proxies() {
    let names = aq_testkit::testdata_names();
    assert_eq!(names.len(), 12);
    let st = ctl_state();
    for name in names {
        let t0 = std::time::Instant::now();
        let m = aq_testkit::load_testdata(name).unwrap();
        let (pairs, kinds) = auto_features(&m, 30.0);
        let ing = ingest(m, &pairs, &kinds, &st.unbounded()).unwrap();
        let comps = split_components(&ing);
        let area: f64 = comps.iter().map(|c| c.area).sum();
        let h0 = (area / 5000.0).sqrt();
        let (mut faces, mut cones, mut hard, mut chains) = (0usize, 0usize, 0usize, 0usize);
        for c in comps.iter().map(|c| c.scaled(1.0 / h0)) {
            let share = (150_000.0 * c.area / area * h0 * h0).max(1.0) as u32;
            let p = ProxyParams { max_tris: share, ..ProxyParams::default() };
            let (fr, bvh) = c.source_geometry().unwrap();
            let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
            let mut pr = build_proxy(&c, &proj, &|_| 1.0 / 3.0, &p, &st.unbounded()).unwrap();
            isolate_features_with(&mut pr, Some(&proj)).unwrap();
            let hv = vec![1.0; pr.mesh.pos.len()];
            classify_chains(&mut pr, &hv, &p);
            pr.validate().unwrap();
            let s = Surface { mesh: pr.mesh, topo: pr.topo, frames: pr.frames, feats: pr.feats };
            let curv = estimate_curv(&s, 1.0);
            let (f, rep) = solve_field(&px(&s), &curv, &size(&s, 1.0), &FieldParams::default(), &st.unbounded())
                .unwrap_or_else(|e| panic!("{name}: solve_field: {e}"));
            // DESIGN 7.3 after S4 (solve_field also checks all of these before returning).
            f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
            aq_core::cross::poincare_hopf(&s.mesh, &s.topo, &f.index4).unwrap();
            let sum: i64 = f.index4.iter().map(|&i| i as i64).sum();
            assert_eq!(sum, 4 * s.chi_total(), "{name}: Poincare-Hopf");
            for h in 0..3 * s.n_faces() {
                let tw = s.topo.twin[h];
                if tw != aq_core::NONE {
                    assert_eq!(f.p[tw as usize], (4 - f.p[h]) % 4, "{name}: p twin rule at {h}");
                }
            }
            for v in 0..s.n_verts() {
                if !s.topo.is_bnd_vert[v] && !s.feats.vert_corner[v] {
                    assert!(aq_field::interior_index_ok(f.index4[v]), "{name}: vertex {v} has index4 {}", f.index4[v]);
                }
            }
            // Hard faces exact: every hard face lies on a branch of one of its constraint edges.
            for fc in 0..s.n_faces() {
                if !f.hard[fc] {
                    continue;
                }
                hard += 1;
                let best = (0..3)
                    .map(|k| 3 * fc + k)
                    .filter(|&h| {
                        s.topo.twin[h] == aq_core::NONE
                            || matches!(
                                s.feats.he_kind[h],
                                FeatureKind::Hard | FeatureKind::GuideHard | FeatureKind::Boundary | FeatureKind::Symmetry
                            )
                    })
                    .map(|h| cross_dev(f.theta[fc], s.frames.edge_angle(&s.mesh, h as u32)))
                    .fold(f64::INFINITY, f64::min);
                assert!(best <= 1e-9, "{name}: hard face {fc} off its constraint edge by {best:e}");
            }
            faces += s.n_faces();
            cones += rep.cones_final as usize;
            chains += s.feats.chains.len();
        }
        println!(
            "{name:22} {:2} comps, proxy {faces:6} faces, {chains:4} chains, {hard:5} hard faces, {cones:4} cones, {:.2} s",
            comps.len(),
            t0.elapsed().as_secs_f64()
        );
    }
}

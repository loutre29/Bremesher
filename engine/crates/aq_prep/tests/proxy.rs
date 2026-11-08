//! DESIGN 6.4 tests: proxy quality, cube chains, parallel sheets, hard_mode, density, determinism.

mod common;

use aq_core::{Bvh, FeatureKind, Frames, NONE, Projector, Topology};
use aq_prep::{
    ProxyParams, build_proxy, classify_chains, count_bridge_edges, count_multi_chain_faces, ingest,
    isolate_features, isolate_features_with, split_components, transfer_density,
};
use aq_testkit::{add_noise, cube_sharp, hemisphere, icosphere, merge, plane_grid};
use common::*;

fn no_budget() -> ProxyParams {
    ProxyParams { max_tris: 0, ..ProxyParams::default() }
}

#[test]
fn proxy_quality_noisy_sphere() {
    for seed in [11, 12, 13] {
        proxy_quality_case(seed);
    }
}

fn proxy_quality_case(seed: u64) {
    let mut m = icosphere(5, 1.0).unwrap();
    add_noise(&mut m, 0.01, 0.6, seed).unwrap();
    let ing = ingest_plain(m);
    let comps = split_components(&ing);
    assert_eq!(comps.len(), 1);
    let c = &comps[0];
    let len = 0.1;
    let t0 = std::time::Instant::now();
    let px = proxy_const(c, len, &no_budget()).unwrap();
    println!("noisy sphere proxy: {} tris in {:.3} s", px.mesh.tris.len(), t0.elapsed().as_secs_f64());
    px.validate().unwrap();
    assert_eq!(px.topo.euler_char(&px.mesh, 0), 2, "chi preserved");
    let el = edge_lengths(&px.mesh);
    let lo = el.iter().filter(|&&l| l < 0.8 * len).count() as f64 / el.len() as f64;
    let hi = el.iter().filter(|&&l| l > 1.33 * len).count() as f64 / el.len() as f64;
    let mean = el.iter().sum::<f64>() / el.len() as f64;
    println!("edges below {lo:.4} above {hi:.4} mean {mean:.4}");
    let inside = el.iter().filter(|&&l| l >= 0.8 * len && l <= 1.33 * len).count() as f64 / el.len() as f64;
    let nf = px.mesh.tris.len();
    let angs: Vec<f64> = (0..nf as u32).map(|f| min_angle_deg(&px.mesh, f)).collect();
    let good = angs.iter().filter(|&&a| a >= 25.0).count() as f64 / nf as f64;
    let worst = angs.iter().cloned().fold(180.0, f64::min);
    let hd = hausdorff(&px.mesh, &c.mesh);
    println!("edges in range {:.4}, faces >= 25 deg {:.4}, min angle {:.2}, hausdorff {:.5}", inside, good, worst, hd);
    assert!(inside >= 0.95, "edges in [0.8, 1.33] len: {inside}");
    assert!(good >= 0.98, "faces with min angle >= 25: {good}");
    assert!(worst >= 5.0, "min angle {worst}");
    assert!(hd <= 0.1 * len, "Hausdorff {hd}");
}

/// Every chain vertex of a cube of half size 1 lies on a cube edge: two coordinates at +-1.
fn on_cube_edge(p: [f64; 3], tol: f64) -> bool {
    let n = p.iter().filter(|x| (x.abs() - 1.0).abs() <= tol).count();
    n >= 2 && p.iter().all(|x| x.abs() <= 1.0 + tol)
}

#[test]
fn cube_with_hard_chains() {
    let s = cube_sharp(8, 2.0, 0.2, 5).unwrap();
    let st = ctl();
    let ing = ingest(s.mesh.clone(), &s.feat_pairs, &s.feat_kind, &st.unbounded()).unwrap();
    assert_eq!(ing.feats.chains.iter().filter(|c| c.kind == FeatureKind::Hard).count(), 12);
    let comps = split_components(&ing);
    let c = &comps[0];
    let len = 0.1;
    let mut px = proxy_const(c, len, &no_budget()).unwrap();
    px.validate().unwrap();
    let hv = vec![3.0 * len; px.mesh.pos.len()];
    classify_chains(&mut px, &hv, &ProxyParams::default());
    px.validate().unwrap();
    assert_eq!(px.feats.chains.len(), 12);
    for ch in &px.feats.chains {
        assert_eq!(ch.kind, FeatureKind::Hard);
        assert!(!ch.closed);
        assert_eq!(ch.corners, vec![0], "a cube edge has no interior corner");
    }
    // The 8 corners are exact.
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let hit = px.mesh.pos.iter().any(|p| {
                    (p[0] - sx).abs() <= 1e-12 && (p[1] - sy).abs() <= 1e-12 && (p[2] - sz).abs() <= 1e-12
                });
                assert!(hit, "corner ({sx}, {sy}, {sz}) missing");
            }
        }
    }
    let n_corner = px.feats.vert_corner.iter().filter(|&&x| x).count();
    assert_eq!(n_corner, 8);
    // Every chain vertex lies on its source edge.
    let mut n_chain_verts = 0;
    for ch in &px.feats.chains {
        for &h in &ch.hes {
            for v in [px.topo.tail(&px.mesh, h), px.topo.head(&px.mesh, h)] {
                n_chain_verts += 1;
                assert!(on_cube_edge(px.mesh.pos[v as usize], 1e-9), "chain vertex {v} off its edge");
            }
        }
    }
    assert!(n_chain_verts > 12 * 15);
    // Faces stay planar: every vertex lies on the cube surface.
    for p in &px.mesh.pos {
        assert!(p.iter().any(|x| (x.abs() - 1.0).abs() <= 1e-9), "vertex off the cube: {p:?}");
    }
    // Isolation on the cube.
    let (fr, bvh) = c.source_geometry().unwrap();
    let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
    let n = isolate_features_with(&mut px, Some(&proj)).unwrap();
    println!("cube isolation inserted {n} vertices");
    px.validate().unwrap();
    assert_eq!(count_bridge_edges(&px.mesh, &px.topo, &px.feats), 0);
    assert_eq!(count_multi_chain_faces(&px.mesh, &px.feats), 0);
    assert_eq!(px.feats.chains.len(), 12);
    assert_eq!(px.topo.euler_char(&px.mesh, 0), 2);
}

#[test]
fn hard_mode_drops_or_softens() {
    let s = cube_sharp(6, 2.0, 0.0, 1).unwrap();
    let st = ctl();
    let ing = ingest(s.mesh.clone(), &s.feat_pairs, &s.feat_kind, &st.unbounded()).unwrap();
    let c = &split_components(&ing)[0];
    let p0 = ProxyParams { hard_mode: 0, ..no_budget() };
    let px = proxy_const(c, 0.15, &p0).unwrap();
    assert!(px.feats.chains.is_empty());
    let p1 = ProxyParams { hard_mode: 1, ..no_budget() };
    let mut px = proxy_const(c, 0.15, &p1).unwrap();
    assert_eq!(px.feats.chains.len(), 12);
    assert!(px.feats.chains.iter().all(|c| c.kind == FeatureKind::Guide));
    // classify_chains with hard_mode 0 on a Hard proxy drops the chains.
    let mut px2 = proxy_const(c, 0.15, &no_budget()).unwrap();
    classify_chains(&mut px2, &[], &p0);
    assert!(px2.feats.chains.is_empty());
    classify_chains(&mut px, &[], &p1);
    assert_eq!(px.feats.chains.len(), 12);
}

#[test]
fn parallel_planes_do_not_mix() {
    let a = plane_grid(30, 30, 1.0 / 15.0, 0.3, 0.2, 1).unwrap();
    let mut b = plane_grid(30, 30, 1.0 / 15.0, 0.3, 0.2, 2).unwrap();
    for p in b.pos.iter_mut() {
        p[2] += 0.05;
    }
    let ing = ingest_plain(merge(&[a, b]));
    assert_eq!(ing.report.components, 2);
    let comps = split_components(&ing);
    // One projector over both sheets (as a global source BVH would be).
    let fr = Frames::new(&ing.mesh, &ing.topo).unwrap();
    let bvh = Bvh::build(&ing.mesh);
    let proj = Projector::new(&ing.mesh, &ing.topo, &fr, &bvh);
    let st = ctl();
    for (ci, c) in comps.iter().enumerate() {
        let z = if ci == 0 { 0.0 } else { 0.05 };
        let px = build_proxy(c, &proj, &|_| 0.1, &no_budget(), &st.unbounded()).unwrap();
        px.validate().unwrap();
        let mut crossed = 0;
        for (v, p) in px.mesh.pos.iter().enumerate() {
            let f = px.src_face[v];
            let on_sheet = ing.topo.face_comp[f as usize] == ci as u32;
            if (p[2] - z).abs() > 1e-9 || !on_sheet {
                crossed += 1;
            }
        }
        assert_eq!(crossed, 0, "sheet {ci}: vertices crossed");
        // And with the component's own projector.
        let px2 = proxy_const(c, 0.1, &no_budget()).unwrap();
        assert!(px2.mesh.pos.iter().all(|p| (p[2] - z).abs() <= 1e-9));
    }
}

#[test]
fn hemisphere_symmetry_loop() {
    let s = hemisphere(16, 4.0).unwrap();
    let st = ctl();
    let ing = ingest(s.mesh.clone(), &s.feat_pairs, &s.feat_kind, &st.unbounded()).unwrap();
    let c = &split_components(&ing)[0];
    let mut px = proxy_const(c, 1.0 / 3.0, &ProxyParams::default()).unwrap();
    let hv = vec![1.0; px.mesh.pos.len()];
    classify_chains(&mut px, &hv, &ProxyParams::default());
    px.validate().unwrap();
    assert_eq!(px.feats.chains.len(), 1);
    let ch = &px.feats.chains[0];
    assert_eq!(ch.kind, FeatureKind::Symmetry);
    assert!(ch.closed);
    assert_eq!(ch.corners.len(), 0);
    assert_eq!(px.feats.vert_corner.iter().filter(|&&x| x).count(), 0);
    for &h in &ch.hes {
        let p = px.mesh.pos[px.topo.tail(&px.mesh, h) as usize];
        assert_eq!(p[2], 0.0, "symmetry vertex off the plane");
    }
    assert_eq!(px.topo.euler_char(&px.mesh, 0), 1);
    assert!(px.mesh.tris.len() >= 2000);
}

#[test]
fn budget_scales_lengths() {
    let m = icosphere(4, 1.0).unwrap();
    let c = &split_components(&ingest_plain(m))[0];
    // len 0.02 asks for about 36k triangles; the budget allows 5000.
    let p = ProxyParams { max_tris: 5000, ..ProxyParams::default() };
    let px = proxy_const(c, 0.02, &p).unwrap();
    let n = px.mesh.tris.len() as f64;
    assert!(n > 3500.0 && n < 6500.0, "{n} triangles for a 5000 budget");
    // The 2000 minimum.
    let px = proxy_const(c, 1.0, &ProxyParams::default()).unwrap();
    let n = px.mesh.tris.len() as f64;
    assert!(n > 1400.0 && n < 2800.0, "{n} triangles for the 2000 minimum");
}

#[test]
fn density_is_transferred() {
    let m = plane_grid(10, 10, 0.1, 0.0, 0.0, 0).unwrap();
    let nv = m.pos.len();
    let dens: Vec<f32> = m.pos.iter().map(|p| if p[0] < 0.5 { 1.0 } else { 4.0 }).collect();
    let ing = ingest_plain(m);
    let c = &split_components(&ing)[0];
    let mut px = proxy_const(c, 0.05, &no_budget()).unwrap();
    transfer_density(&mut px, c, Some(&dens), 0.5);
    assert_eq!(px.density.len(), px.mesh.pos.len());
    for (v, p) in px.mesh.pos.iter().enumerate() {
        let d = px.density[v];
        if p[0] < 0.39 {
            assert!((d - 1.0).abs() < 1e-9, "{d} at {p:?}");
        } else if p[0] > 0.61 {
            assert!((d - 2.0).abs() < 1e-9, "{d} at {p:?}");
        }
    }
    transfer_density(&mut px, c, None, 1.0);
    assert!(px.density.iter().all(|&d| d == 1.0));
    assert_eq!(nv, c.mesh.pos.len());
}

#[test]
fn proxy_src_tracking_is_consistent() {
    let mut m = icosphere(4, 1.0).unwrap();
    add_noise(&mut m, 0.01, 0.5, 3).unwrap();
    let c = &split_components(&ingest_plain(m))[0];
    let px = proxy_const(c, 0.08, &no_budget()).unwrap();
    for v in 0..px.mesh.pos.len() {
        let f = px.src_face[v];
        assert!(f != NONE && (f as usize) < c.mesh.tris.len());
        let sp = aq_core::SurfacePoint { face: f, bary: px.src_bary[v] };
        sp.validate(c.mesh.tris.len()).unwrap();
        let q = sp.position(&c.mesh);
        assert!(aq_core::math::dist(q, px.mesh.pos[v]) < 1e-9, "vertex {v} src mismatch");
    }
}

#[test]
fn isolation_on_proxy_without_projector() {
    let s = cube_sharp(4, 2.0, 0.1, 9).unwrap();
    let st = ctl();
    let ing = ingest(s.mesh.clone(), &s.feat_pairs, &s.feat_kind, &st.unbounded()).unwrap();
    let c = &split_components(&ing)[0];
    let mut px = proxy_const(c, 0.2, &no_budget()).unwrap();
    let nv0 = px.mesh.pos.len();
    isolate_features(&mut px).unwrap();
    px.validate().unwrap();
    assert_eq!(count_bridge_edges(&px.mesh, &px.topo, &px.feats), 0);
    assert_eq!(count_multi_chain_faces(&px.mesh, &px.feats), 0);
    assert!(px.mesh.pos.len() >= nv0);
    let t = Topology::build(&px.mesh).unwrap();
    assert_eq!(t.euler_char(&px.mesh, 0), 2);
    // Running it again changes nothing.
    assert_eq!(isolate_features(&mut px).unwrap(), 0);
}

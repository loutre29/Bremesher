//! DESIGN 6.4 tests: performance (noisy_blob x4 to l for 5k quads, and the coarse torus and
//! cylinder_open sources refined to the same proxy size) and determinism across thread counts.

mod common;

use aq_core::Projector;
use aq_prep::{ComponentMesh, Proxy, ProxyParams, build_proxy, classify_chains, ingest, isolate_features_with};
use common::*;

fn run(c: &ComponentMesh, p: &ProxyParams) -> Proxy {
    let (fr, bvh) = c.source_geometry().unwrap();
    let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
    let st = ctl();
    let mut px = build_proxy(c, &proj, &|_| 1.0 / 3.0, p, &st.unbounded()).unwrap();
    isolate_features_with(&mut px, Some(&proj)).unwrap();
    let h = vec![1.0; px.mesh.pos.len()];
    classify_chains(&mut px, &h, p);
    px
}

#[test]
fn performance_noisy_blob_x4() {
    if cfg!(debug_assertions) {
        return;
    }
    let m = subdivide(&aq_testkit::load_testdata("noisy_blob").unwrap());
    assert!(m.tris.len() >= 320_000);
    let st = ctl();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    pool.install(|| {
        let ing = ingest(m.clone(), &[], &[], &st.unbounded()).unwrap();
        let (comps, _) = quad_units(&ing, 5000.0);
        assert_eq!(comps.len(), 1);
        let c = &comps[0];
        let p = ProxyParams::default();
        let (fr, bvh) = c.source_geometry().unwrap();
        let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
        let mut best = f64::INFINITY;
        let mut nf = 0;
        for _ in 0..2 {
            let t0 = std::time::Instant::now();
            let px = build_proxy(c, &proj, &|_| 1.0 / 3.0, &p, &st.unbounded()).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
            nf = px.mesh.tris.len();
        }
        println!("noisy_blob x4 ({} tris) to {nf} proxy tris: {best:.3} s", c.mesh.tris.len());
        assert!(best < 4.0, "proxy took {best} s");
    });
}

/// Coarse sources (6k and 8k triangles, cylinder_open with faces of aspect 18 at the target
/// length) refined to a proxy of about 110k triangles stay within the 4 s proxy budget of
/// DESIGN 7.4 (which a 300k-triangle input meets).
#[test]
fn performance_coarse_sources() {
    if cfg!(debug_assertions) {
        return;
    }
    let st = ctl();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    pool.install(|| {
        for name in ["torus", "cylinder_open"] {
            let m = aq_testkit::load_testdata(name).unwrap();
            let ing = ingest(m, &[], &[], &st.unbounded()).unwrap();
            let (comps, _) = quad_units(&ing, 5000.0);
            assert_eq!(comps.len(), 1);
            let c = &comps[0];
            let p = ProxyParams::default();
            let (fr, bvh) = c.source_geometry().unwrap();
            let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
            let mut best = f64::INFINITY;
            let mut nf = 0;
            for _ in 0..2 {
                let t0 = std::time::Instant::now();
                let px = build_proxy(c, &proj, &|_| 1.0 / 3.0, &p, &st.unbounded()).unwrap();
                best = best.min(t0.elapsed().as_secs_f64());
                nf = px.mesh.tris.len();
            }
            println!("{name} ({} tris) to {nf} proxy tris: {best:.3} s", c.mesh.tris.len());
            assert!(nf < 130_000, "{name}: {nf} proxy tris");
            assert!(best < 4.0, "{name}: proxy took {best} s");
        }
    });
}

#[test]
fn deterministic_across_thread_counts() {
    let s = aq_testkit::cube_sharp(10, 2.0, 0.25, 3).unwrap();
    let mut m = s.mesh.clone();
    aq_testkit::add_noise(&mut m, 0.0, 1.0, 1).unwrap();
    let st = ctl();
    let ing = ingest(m, &s.feat_pairs, &s.feat_kind, &st.unbounded()).unwrap();
    let (comps, _) = quad_units(&ing, 1500.0);
    let blob = aq_testkit::load_testdata("retopo_b").unwrap();
    let (pairs, kinds) = auto_features(&blob, 30.0);
    let ing2 = ingest(blob, &pairs, &kinds, &st.unbounded()).unwrap();
    let (comps2, _) = quad_units(&ing2, 2000.0);
    let p = ProxyParams::default();
    for c in [&comps[0], &comps2[0]] {
        let mut runs: Vec<Proxy> = Vec::new();
        for threads in [1, 4, 16] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            runs.push(pool.install(|| run(c, &p)));
        }
        for r in &runs[1..] {
            assert_eq!(r.mesh.tris, runs[0].mesh.tris);
            assert!(r.mesh.pos.iter().zip(&runs[0].mesh.pos).all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits)));
            assert_eq!(r.feats, runs[0].feats);
            assert_eq!(r.src_face, runs[0].src_face);
            assert!(r.src_bary.iter().zip(&runs[0].src_bary).all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits)));
        }
    }
}

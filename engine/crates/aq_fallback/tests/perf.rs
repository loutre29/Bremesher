//! DESIGN 6.14: 100k triangles in < 2 s.

mod common;

use std::time::Instant;

use aq_core::{CtlState, Projector};
use aq_fallback::{SfParams, sf_quadrangulate};
use aq_testkit::{add_noise, torus};
use common::*;

#[test]
fn hundred_k_triangles_under_two_seconds() {
    let mut m = torus(3.0, 1.0, 400, 125, 0.2, 9).unwrap();
    add_noise(&mut m, 0.02, 0.4, 3).unwrap();
    assert_eq!(m.tris.len(), 100_000);
    for target in [5000u32, 2000] {
        let c = single(m.clone(), &[], &[], target as f64);
        let st = CtlState::new();
        let mut best = f64::INFINITY;
        let mut q = None;
        for _ in 0..2 {
            let t0 = Instant::now();
            let (fr, bvh) = c.source_geometry().unwrap();
            let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
            let r = sf_quadrangulate(&c, &proj, &SfParams { target, relax_iters: 5 }, &st.unbounded()).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
            q = Some(r);
        }
        let q = q.unwrap();
        eprintln!("100k triangles -> {} quads (target {target}): {best:.3} s", q.n_faces());
        check(&format!("perf @{target}"), &q, &c, target);
        assert!(best < 2.0, "SF took {best:.3} s");
    }
}

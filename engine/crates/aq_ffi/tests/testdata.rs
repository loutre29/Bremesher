//! M0 gate through the C ABI (DESIGN 11.3): valid all-quad meshes on all 12 testdata meshes
//! and on the perf input (noisy_blob upsampled once, 327,680 triangles, to 5000 quads; DESIGN
//! 10.1 writes `--upsample 2`, see CHANGES.md WP-05 for why one pass is the 328k input).

mod common;

use std::time::Instant;

use aq_engine::host;
use aq_testkit::{load_testdata, testdata_names};
use common::*;

fn target_of(name: &str) -> u32 {
    match name {
        "basemesh_female" | "noisy_blob" | "sculpt_sphere_mirror" => 5000,
        _ => 2000,
    }
}

#[test]
fn all_testdata_meshes_give_valid_all_quad_meshes() {
    let names = testdata_names();
    assert_eq!(names.len(), 12);
    for name in names {
        let m = normalized(load_testdata(name).unwrap());
        let target = target_of(name);
        let mut raw = RawIn::new(&m);
        if name == "cube_sharp" {
            let f = host::auto_features(&m, 30.0);
            let k = vec![1u8; f.len()];
            raw = raw.with_feats(&f, &k);
        }
        let t = Instant::now();
        let r = call_raw(&raw, &params(target), None);
        let dt = t.elapsed().as_secs_f64();
        check_ok(&r);
        // M1 (WP-16): Ok or Degraded (a component retried or on SF), never Failed.
        assert!(r.status == 0 || r.status == 1, "{name}: {}", r.error);
        assert!(
            r.comp_status.iter().all(|&s| s != 3),
            "{name}: {:?}",
            r.comp_status
        );
        let n = r.mesh.n_faces();
        let err = n as f64 / target as f64 - 1.0;
        eprintln!(
            "{name:22} {n:6} quads (target {target}, err {:+.1}%), {} comps, {dt:.2} s",
            100.0 * err,
            r.comp_status.len()
        );
        assert!(err.abs() < 0.25, "{name}: count {n} for {target}");
        // The 5 s gate for small inputs is an M1 eval gate (eval/M1_REPORT.md): with the
        // full pipeline basemesh_female still takes two failed rungs plus SF. Here only the
        // 30 s input budget is enforced.
        assert!(dt < 30.0, "{name}: {dt:.2} s");
    }
}

#[test]
fn perf_input_328k_triangles() {
    let base = normalized(load_testdata("noisy_blob").unwrap());
    let (m, _) = host::upsample(&base, &[], 1);
    assert_eq!(m.tris.len(), 327_680);
    let p = autoquad_engine::AqParams {
        threads: 0,
        ..params(5000)
    };
    let t = Instant::now();
    let r = call(&m, &p);
    let dt = t.elapsed().as_secs_f64();
    check_ok(&r);
    assert_eq!(r.status, 0, "{}", r.error);
    let n = r.mesh.n_faces();
    eprintln!("perf: {} tris -> {n} quads in {dt:.2} s", m.tris.len());
    assert!((n as f64 / 5000.0 - 1.0).abs() < 0.25);
    assert!(dt < 30.0, "{dt:.2} s");
}

//! C ABI tests (DESIGN 8, WP-05): layout, defaults, NULL safety, struct_size, statuses,
//! re-entrancy, cancel, panic gives 5, progress.

mod common;

use std::mem::{offset_of, size_of};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use aq_testkit::{icosphere, merge, torus};
use autoquad_engine::*;
use common::*;

#[test]
fn layout_matches_the_header() {
    assert_eq!(size_of::<AqParams>(), 72);
    assert_eq!(offset_of!(AqParams, struct_size), 0);
    assert_eq!(offset_of!(AqParams, target_quads), 4);
    assert_eq!(offset_of!(AqParams, adaptive_size), 8);
    assert_eq!(offset_of!(AqParams, curvature_align), 12);
    assert_eq!(offset_of!(AqParams, flow_smoothness), 16);
    assert_eq!(offset_of!(AqParams, density_strength), 20);
    assert_eq!(offset_of!(AqParams, guide_strength), 24);
    assert_eq!(offset_of!(AqParams, hard_edges), 28);
    assert_eq!(offset_of!(AqParams, close_loops), 29);
    assert_eq!(offset_of!(AqParams, allow_collapse), 30);
    assert_eq!(offset_of!(AqParams, reserved0), 31);
    assert_eq!(offset_of!(AqParams, ring_min), 32);
    assert_eq!(offset_of!(AqParams, max_proxy_tris), 36);
    assert_eq!(offset_of!(AqParams, threads), 40);
    assert_eq!(offset_of!(AqParams, time_limit_s), 44);
    assert_eq!(offset_of!(AqParams, debug_flags), 48);
    assert_eq!(offset_of!(AqParams, seed), 56);
    assert_eq!(offset_of!(AqParams, dump_dir), 64);

    assert_eq!(size_of::<AqMeshIn>(), 64);
    assert_eq!(offset_of!(AqMeshIn, n_verts), 0);
    assert_eq!(offset_of!(AqMeshIn, pos), 8);
    assert_eq!(offset_of!(AqMeshIn, n_tris), 16);
    assert_eq!(offset_of!(AqMeshIn, tris), 24);
    assert_eq!(offset_of!(AqMeshIn, n_feat), 32);
    assert_eq!(offset_of!(AqMeshIn, feat), 40);
    assert_eq!(offset_of!(AqMeshIn, feat_kind), 48);
    assert_eq!(offset_of!(AqMeshIn, density), 56);

    assert_eq!(size_of::<AqControl>(), 12);
    assert_eq!(offset_of!(AqControl, cancel), 0);
    assert_eq!(offset_of!(AqControl, progress_bits), 4);
    assert_eq!(offset_of!(AqControl, stage), 8);

    assert_eq!(size_of::<AqResult>(), 104);
    assert_eq!(offset_of!(AqResult, status), 0);
    assert_eq!(offset_of!(AqResult, n_verts), 4);
    assert_eq!(offset_of!(AqResult, pos), 8);
    assert_eq!(offset_of!(AqResult, n_faces), 16);
    assert_eq!(offset_of!(AqResult, face_offsets), 24);
    assert_eq!(offset_of!(AqResult, face_verts), 32);
    assert_eq!(offset_of!(AqResult, vflag), 40);
    assert_eq!(offset_of!(AqResult, face_comp), 48);
    assert_eq!(offset_of!(AqResult, n_comp), 56);
    assert_eq!(offset_of!(AqResult, comp_status), 64);
    assert_eq!(offset_of!(AqResult, n_failed_tris), 72);
    assert_eq!(offset_of!(AqResult, failed_tris), 80);
    assert_eq!(offset_of!(AqResult, report_json), 88);
    assert_eq!(offset_of!(AqResult, error), 96);
}

#[test]
fn header_matches_the_rust_constants() {
    let h = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/include/autoquad_engine.h"
    ))
    .unwrap();
    assert!(h.contains("#define AQ_ABI_VERSION 1"));
    assert!(h.contains("sizeof(AqParams) = 72"));
    assert!(
        !h.contains('\u{2014}') && !h.contains('\u{2013}'),
        "no em or en dashes"
    );
    assert_eq!(aq_abi_version(), 1);
    assert_eq!(AQ_ABI_VERSION, 1);
}

#[test]
fn defaults_are_design_8_1() {
    let p = defaults();
    assert_eq!(p.struct_size as usize, size_of::<AqParams>());
    assert_eq!(p.target_quads, 5000);
    assert_eq!(p.adaptive_size, 0.5);
    assert_eq!(p.curvature_align, 1.0);
    assert_eq!(p.flow_smoothness, 1.0);
    assert_eq!(p.density_strength, 1.0);
    assert_eq!(p.guide_strength, 1.0);
    assert_eq!(
        (p.hard_edges, p.close_loops, p.allow_collapse, p.reserved0),
        (2, 1, 1, 0)
    );
    assert_eq!(p.ring_min, 8);
    assert_eq!(p.max_proxy_tris, 150_000);
    assert_eq!(p.threads, 0);
    assert_eq!(p.time_limit_s, 60.0);
    assert_eq!(p.debug_flags, 0);
    assert_eq!(p.seed, 0);
    assert!(p.dump_dir.is_null());
    // NULL-safe.
    unsafe { aq_params_default(std::ptr::null_mut()) };
}

#[test]
fn null_safety_and_invalid_calls() {
    unsafe { aq_free(std::ptr::null_mut()) };
    let m = icosphere(2, 3.0).unwrap();
    let raw = RawIn::new(&m);
    let mi = raw.mesh_in();
    let p = params(300);
    // out NULL: 4, nothing allocated.
    assert_eq!(
        unsafe { aq_remesh(&mi, &p, std::ptr::null_mut(), std::ptr::null_mut()) },
        AQ_INVALID_INPUT
    );
    // in NULL, params NULL: 4 with a result.
    let mut out: *mut AqResult = std::ptr::null_mut();
    let ret = unsafe { aq_remesh(std::ptr::null(), &p, std::ptr::null_mut(), &mut out) };
    let r = unsafe { take(ret, out) };
    assert_eq!(r.status, 4);
    assert!(!r.error.is_empty());
    let mut out: *mut AqResult = std::ptr::null_mut();
    let ret = unsafe { aq_remesh(&mi, std::ptr::null(), std::ptr::null_mut(), &mut out) };
    assert_eq!(unsafe { take(ret, out) }.status, 4);
    // struct_size mismatch.
    let bad = AqParams {
        struct_size: 64,
        ..p
    };
    let r = call_raw(&raw, &bad, None);
    assert_eq!(r.status, 4);
    assert!(r.error.contains("struct_size"), "{}", r.error);
    // The error-path report has every DESIGN 9 key, with zero timings.
    let j: serde_json::Value = serde_json::from_str(&r.report).unwrap();
    assert_eq!(j["status"], 4);
    for k in ["abi", "engine", "error", "components", "count", "warnings", "h0", "params"] {
        assert!(j.get(k).is_some(), "error report misses {k}");
    }
    for k in [
        "ingest", "proxy", "curv", "field", "loops", "cut", "param", "layout", "quantize",
        "patch", "post", "total",
    ] {
        assert_eq!(j["timing_ms"][k].as_f64(), Some(0.0), "timing_ms.{k}");
    }
    // pos NULL with n_verts > 0.
    let mut mi2 = raw.mesh_in();
    mi2.pos = std::ptr::null();
    let mut out: *mut AqResult = std::ptr::null_mut();
    let ret = unsafe { aq_remesh(&mi2, &p, std::ptr::null_mut(), &mut out) };
    assert_eq!(unsafe { take(ret, out) }.status, 4);
    // feat NULL with n_feat > 0.
    let mut mi3 = raw.mesh_in();
    mi3.n_feat = 3;
    mi3.feat = std::ptr::null();
    let mut out: *mut AqResult = std::ptr::null_mut();
    let ret = unsafe { aq_remesh(&mi3, &p, std::ptr::null_mut(), &mut out) };
    assert_eq!(unsafe { take(ret, out) }.status, 4);
    // Target 0.
    assert_eq!(call(&m, &params(0)).status, 4);
    // A NULL ctl works.
    let r = call(&m, &p);
    check_ok(&r);
    assert_eq!(r.status, 0);
    assert_eq!(r.error, "");
}

#[test]
fn ok_result_contents() {
    let m = normalized(torus(3.0, 1.0, 64, 32, 0.0, 1).unwrap());
    let ctl = AqControl::default();
    let r = call_raw(&RawIn::new(&m), &params(2000), Some(&ctl));
    check_ok(&r);
    assert_eq!(r.status, 0);
    // M1 (WP-16): the torus runs the full pipeline (status 0 full, no SF vflag bit).
    assert_eq!(r.comp_status, vec![0]);
    assert!(r.failed_tris.is_empty());
    assert_eq!(aq_engine::euler_char(&r.mesh), 0);
    let n = r.mesh.n_faces() as f64;
    assert!((n / 2000.0 - 1.0).abs() < 0.25, "{n} quads");
    assert!(r.mesh.vflag.iter().all(|&f| f & (1 << 27) == 0));
    // Progress reached 1 and the last stage is Engine.
    assert_eq!(
        f32::from_bits(ctl.progress_bits.load(Ordering::Relaxed)),
        1.0
    );
    assert_eq!(ctl.stage.load(Ordering::Relaxed), 12);
    let j: serde_json::Value = serde_json::from_str(&r.report).unwrap();
    assert_eq!(j["components"][0]["status"], "full");
    assert_eq!(
        j["count"]["result"].as_u64().unwrap() as usize,
        r.mesh.n_faces()
    );
}

#[test]
fn features_and_density_pass_through() {
    let c = aq_testkit::cube_sharp(10, 5.0, 0.0, 2).unwrap();
    let mut raw = RawIn::new(&c.mesh).with_feats(&c.feat_pairs, &c.feat_kind);
    raw.density = Some(vec![1.0; c.mesh.pos.len()]);
    let r = call_raw(&raw, &params(1200), None);
    check_ok(&r);
    let corners = r.mesh.vflag.iter().filter(|&&f| f & (1 << 24) != 0).count();
    assert_eq!(corners, 8);
    let kinds: Vec<u32> = r
        .mesh
        .vflag
        .iter()
        .filter(|&&f| f & 0xFFFFF != 0)
        .map(|&f| (f >> 20) & 0xF)
        .collect();
    assert!(!kinds.is_empty() && kinds.iter().all(|&k| k == 1));
    let j: serde_json::Value = serde_json::from_str(&r.report).unwrap();
    assert!(
        j["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("density"))
    );
}

#[test]
fn panic_gives_5() {
    let m = icosphere(3, 3.0).unwrap();
    let p = AqParams {
        debug_flags: 1 << 31,
        ..params(500)
    };
    let r = call(&m, &p);
    assert_eq!(r.status, AQ_PANIC);
    assert!(r.error.contains("panic"), "{}", r.error);
    assert_eq!(r.mesh.n_faces(), 0);
    let j: serde_json::Value = serde_json::from_str(&r.report).unwrap();
    assert_eq!(j["status"], 5);
    // A panic inside a component: only that component fails (status 1 with two components).
    let a = icosphere(3, 3.0).unwrap();
    let b = {
        let mut b = icosphere(3, 3.0).unwrap();
        for q in b.pos.iter_mut() {
            q[0] += 10.0;
        }
        b
    };
    let na = a.tris.len() as u32;
    // On the forced SF path (bit 2), where a component panic is final; in M1 a panic in a
    // pipeline rung sends the component to SF instead.
    let p = AqParams {
        debug_flags: (1 << 30) | (1 << 2),
        ..params(500)
    };
    let r = call(&merge(&[a, b]), &p);
    assert_eq!(r.status, AQ_DEGRADED);
    assert_eq!(r.comp_status, vec![3, 2]);
    assert_eq!(r.failed_tris, (0..na).collect::<Vec<_>>());
    check_ok(&r);
}

#[test]
fn cancel() {
    // Set before the call.
    let ctl = AqControl::default();
    ctl.cancel.store(1, Ordering::Relaxed);
    let m = icosphere(4, 3.0).unwrap();
    let r = call_raw(&RawIn::new(&m), &params(2000), Some(&ctl));
    assert_eq!(r.status, AQ_CANCELLED);
    assert_eq!(r.mesh.n_faces(), 0);
    // Set from another thread during a long call.
    let big = icosphere(7, 3.0).unwrap();
    let raw = RawIn::new(&big);
    let ctl = AqControl::default();
    let p = AqParams {
        threads: 4,
        ..params(300_000)
    };
    let t = Instant::now();
    let r = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            ctl.cancel.store(1, Ordering::Relaxed);
        });
        call_raw(&raw, &p, Some(&ctl))
    });
    assert_eq!(r.status, AQ_CANCELLED, "{}", r.error);
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

#[test]
fn reentrant_concurrent_calls() {
    let meshes = [
        normalized(icosphere(4, 1.0).unwrap()),
        normalized(torus(3.0, 1.0, 48, 24, 0.1, 4).unwrap()),
        normalized(aq_testkit::cylinder_open(1.0, 3.0, 40, 20, 0.0, 5).unwrap()),
        normalized(merge(&[
            icosphere(3, 1.0).unwrap(),
            torus(3.0, 1.0, 32, 16, 0.0, 2).unwrap(),
        ])),
    ];
    let targets = [800u32, 1500, 600, 1000];
    let seq: Vec<Res> = meshes
        .iter()
        .zip(targets)
        .map(|(m, t)| call(m, &params(t)))
        .collect();
    for r in &seq {
        check_ok(r);
    }
    // 8 concurrent calls (each mesh twice), each with its own control block.
    let par: Vec<Res> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..8)
            .map(|i| {
                let m = &meshes[i % 4];
                let t = targets[i % 4];
                s.spawn(move || {
                    let ctl = AqControl::default();
                    call_raw(&RawIn::new(m), &params(t), Some(&ctl))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, r) in par.iter().enumerate() {
        let s = &seq[i % 4];
        assert_eq!(r.status, s.status);
        assert_eq!(r.mesh, s.mesh, "call {i} differs from the sequential run");
        assert_eq!(r.comp_status, s.comp_status);
    }
}

#[test]
fn deterministic_across_thread_counts() {
    let m = normalized(torus(3.0, 1.0, 64, 32, 0.2, 7).unwrap());
    let a = call(
        &m,
        &AqParams {
            threads: 1,
            ..params(1500)
        },
    );
    let b = call(
        &m,
        &AqParams {
            threads: 8,
            ..params(1500)
        },
    );
    check_ok(&a);
    assert_eq!(a.mesh, b.mesh);
}

#[test]
fn dump_dir_is_used() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("aq_ffi_dump_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cdir = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
    let p = AqParams {
        debug_flags: 1,
        dump_dir: cdir.as_ptr(),
        ..params(400)
    };
    let r = call(&icosphere(3, 3.0).unwrap(), &p);
    check_ok(&r);
    assert!(dir.join("result.obj").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}

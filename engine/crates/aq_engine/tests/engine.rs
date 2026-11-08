//! aq_engine M0 tests (DESIGN 7, 11.3 M0): ladder to SF, budgets, merge, statuses, report,
//! cancel, panics, determinism.

use std::time::Duration;

use aq_core::{CtlState, QuadMesh, TriMesh};
use aq_engine::merge::{VFLAG_CHAIN_MASK, VFLAG_CORNER, VFLAG_SF};
use aq_engine::{CompStatus, Input, Output, Params, Status, debug, euler_char, remesh};
use aq_testkit::{cube_sharp, cylinder_open, icosphere, merge, torus};

fn run(mesh: TriMesh, p: &Params) -> Output {
    let st = CtlState::new();
    remesh(
        Input {
            mesh,
            ..Input::default()
        },
        p,
        &st.unbounded(),
    )
}

fn params(target: u32) -> Params {
    Params {
        target_quads: target,
        threads: 4,
        ..Params::default()
    }
}

fn check_valid(o: &Output, chi: i64) {
    o.mesh.validate().unwrap();
    assert!(o.mesh.is_all_quads());
    assert!(o.mesh.n_faces() > 0);
    assert_eq!(euler_char(&o.mesh), chi);
    assert_eq!(o.mesh.face_comp.len(), o.mesh.n_faces());
    for &c in &o.mesh.face_comp {
        assert!((c as usize) < o.comp_status.len());
    }
}

fn count_err(q: &QuadMesh, target: u32) -> f64 {
    (q.n_faces() as f64 / target as f64 - 1.0).abs()
}

fn scaled(mut m: TriMesh, s: f64) -> TriMesh {
    for p in m.pos.iter_mut() {
        *p = [p[0] * s, p[1] * s, p[2] * s];
    }
    m
}

fn shifted(mut m: TriMesh, d: [f64; 3]) -> TriMesh {
    for p in m.pos.iter_mut() {
        *p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    }
    m
}

#[test]
fn sphere_goes_through_sf() {
    // M1 runs the full pipeline by default; the SF path is forced (WP-16).
    let p = Params {
        debug_flags: debug::FORCE_SF,
        ..params(2000)
    };
    let o = run(icosphere(4, 3.0).unwrap(), &p);
    assert_eq!(o.status, Status::Ok, "{}", o.error);
    assert_eq!(o.error, "");
    check_valid(&o, 2);
    assert_eq!(o.comp_status, vec![CompStatus::Sf]);
    assert!(o.failed_tris.is_empty());
    assert!(
        count_err(&o.mesh, 2000) < 0.25,
        "{} quads",
        o.mesh.n_faces()
    );
    assert!(o.mesh.vflag.iter().all(|&f| f & VFLAG_SF != 0));
    // Output is back in input units: every vertex near the sphere of radius 3.
    for p in &o.mesh.pos {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((r - 3.0).abs() < 0.05, "r = {r}");
    }
}

#[test]
fn report_has_the_design_schema() {
    let o = run(torus(3.0, 1.0, 48, 24, 0.0, 1).unwrap(), &params(1000));
    assert_eq!(o.status, Status::Ok);
    let r: serde_json::Value = serde_json::from_str(&o.report_json()).unwrap();
    assert_eq!(r["abi"], 1);
    assert_eq!(r["status"], 0);
    assert_eq!(r["engine"], aq_engine::ENGINE_PATH);
    for k in [
        "ingest", "proxy", "curv", "field", "loops", "cut", "param", "layout", "quantize", "patch",
        "post", "total",
    ] {
        assert!(r["timing_ms"][k].is_number(), "timing_ms.{k}");
    }
    let c = &r["components"][0];
    for k in [
        "id", "status", "rung", "target", "quads", "cones", "field", "param", "layout", "quant",
        "patch", "post", "warnings",
    ] {
        assert!(!c[k].is_null(), "components[0].{k}");
    }
    for k in ["v3", "v5", "other"] {
        assert!(c["cones"][k].is_u64());
    }
    for k in [
        "patches",
        "arcs",
        "health_mean",
        "health_max",
        "seed_rounds",
    ] {
        assert!(c["layout"][k].is_number());
    }
    for k in ["sigma", "solves", "cuts", "count_err"] {
        assert!(c["quant"][k].is_number());
    }
    // M1 (WP-16): the torus runs the full pipeline on rung R0.
    assert_eq!(c["status"], "full");
    assert_eq!(c["rung"], 0);
    assert_eq!(c["target"], 1000);
    assert_eq!(c["quads"].as_u64().unwrap() as usize, o.mesh.n_faces());
    assert_eq!(r["count"]["target"], 1000);
    assert_eq!(
        r["count"]["result"].as_u64().unwrap() as usize,
        o.mesh.n_faces()
    );
    assert!(r["count"]["err"].is_number());
    check_valid(&o, 0);
}

#[test]
fn components_debris_and_merge() {
    // A big sphere, a tiny closed sphere (debris), an open cylinder.
    let big = icosphere(4, 3.0).unwrap();
    let tiny = shifted(icosphere(2, 0.05).unwrap(), [10.0, 0.0, 0.0]);
    let cyl = shifted(
        cylinder_open(1.0, 4.0, 32, 16, 0.0, 3).unwrap(),
        [0.0, 10.0, 0.0],
    );
    let m = merge(&[big, tiny, cyl]);
    let o = run(m, &params(2000));
    assert_eq!(o.status, Status::Ok, "{}", o.error);
    check_valid(&o, 2 + 2);
    assert_eq!(o.comp_status.len(), 3);
    assert_eq!(
        o.comp_status
            .iter()
            .filter(|&&s| s == CompStatus::DebrisSf)
            .count(),
        1
    );
    // M1 (WP-16): the other two run the full pipeline.
    assert_eq!(
        o.comp_status
            .iter()
            .filter(|&&s| s == CompStatus::Full || s == CompStatus::Retried)
            .count(),
        2,
        "{:?}",
        o.comp_status
    );
    let r = &o.report;
    let comps = r["components"].as_array().unwrap();
    let debris: Vec<_> = comps.iter().filter(|c| c["debris"] == true).collect();
    assert_eq!(debris.len(), 1);
    assert_eq!(debris[0]["target"], 24);
    assert_eq!(debris[0]["status"], "debris_sf");
    // Every component's faces carry its id, and the debris component has some quads.
    for (i, c) in comps.iter().enumerate() {
        let n = o
            .mesh
            .face_comp
            .iter()
            .filter(|&&f| f as usize == i)
            .count();
        assert_eq!(n as u64, c["quads"].as_u64().unwrap());
        assert!(n > 0);
    }
    let sum: u64 = comps.iter().map(|c| c["target"].as_u64().unwrap()).sum();
    assert_eq!(sum, 2000);
    // S0 chi per component matches the split components (none dropped).
    let chis: Vec<i64> = comps.iter().map(|c| c["chi"].as_i64().unwrap()).collect();
    assert_eq!(chis, vec![2, 2, 0]);
    assert!(o.failed_tris.is_empty());
}

#[test]
fn cube_features_and_hard_edges() {
    let c = cube_sharp(12, 4.0, 0.0, 5).unwrap();
    let input = Input {
        mesh: c.mesh.clone(),
        feat_pairs: &c.feat_pairs,
        feat_kind: &c.feat_kind,
        density: None,
    };
    let st = CtlState::new();
    let o = remesh(input, &params(1500), &st.unbounded());
    assert_eq!(o.status, Status::Ok, "{}", o.error);
    check_valid(&o, 2);
    let corners = o
        .mesh
        .vflag
        .iter()
        .filter(|&&f| f & VFLAG_CORNER != 0)
        .count();
    assert_eq!(corners, 8);
    let chains = o
        .mesh
        .vflag
        .iter()
        .filter(|&&f| f & VFLAG_CHAIN_MASK != 0)
        .count();
    assert!(chains > 0);
    // hard_edges = 0: no chain survives.
    let input = Input {
        mesh: c.mesh.clone(),
        feat_pairs: &c.feat_pairs,
        feat_kind: &c.feat_kind,
        density: None,
    };
    let p0 = Params {
        hard_edges: 0,
        ..params(1500)
    };
    let o0 = remesh(input, &p0, &st.unbounded());
    assert_eq!(o0.status, Status::Ok);
    assert!(
        o0.mesh
            .vflag
            .iter()
            .all(|&f| f & (VFLAG_CHAIN_MASK | VFLAG_CORNER) == 0)
    );
}

#[test]
fn invalid_inputs_give_status_4() {
    let mut m = icosphere(2, 1.0).unwrap();
    m.pos[3][1] = f64::NAN;
    assert_eq!(run(m, &params(500)).status, Status::InvalidInput);
    let mut m = icosphere(2, 1.0).unwrap();
    m.tris[7][2] = 1_000_000;
    assert_eq!(run(m, &params(500)).status, Status::InvalidInput);
    let o = run(TriMesh::default(), &params(500));
    assert_eq!(o.status, Status::InvalidInput);
    assert!(!o.error.is_empty());
    assert_eq!(o.mesh.n_faces(), 0);
    assert_eq!(
        run(icosphere(2, 1.0).unwrap(), &params(0)).status,
        Status::InvalidInput
    );
    let p = Params {
        time_limit_s: -1.0,
        ..params(500)
    };
    assert_eq!(
        run(icosphere(2, 1.0).unwrap(), &p).status,
        Status::InvalidInput
    );
    // Density with the wrong length.
    let m = icosphere(2, 1.0).unwrap();
    let d = vec![1.0f32; 3];
    let st = CtlState::new();
    let o = remesh(
        Input {
            mesh: m,
            density: Some(&d),
            ..Input::default()
        },
        &params(500),
        &st.unbounded(),
    );
    assert_eq!(o.status, Status::InvalidInput);
    // A feature kind above 5.
    let m = icosphere(2, 1.0).unwrap();
    let pairs = [[m.tris[0][0], m.tris[0][1]]];
    let o = remesh(
        Input {
            mesh: m,
            feat_pairs: &pairs,
            feat_kind: &[9],
            density: None,
        },
        &params(500),
        &st.unbounded(),
    );
    assert_eq!(o.status, Status::InvalidInput);
    // The report is still valid JSON with the status.
    let r: serde_json::Value = serde_json::from_str(&o.report_json()).unwrap();
    assert_eq!(r["status"], 4);
}

#[test]
fn cancel_before_and_during() {
    let st = CtlState::new();
    st.cancel();
    let o = remesh(
        Input {
            mesh: icosphere(4, 1.0).unwrap(),
            ..Input::default()
        },
        &params(2000),
        &st.unbounded(),
    );
    assert_eq!(o.status, Status::Cancelled);
    assert_eq!(o.mesh.n_faces(), 0);
    // During: a large run cancelled after 100 ms returns quickly with status 3.
    let st = CtlState::new();
    let m = icosphere(7, 1.0).unwrap();
    let t = std::time::Instant::now();
    let o = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(100));
            st.cancel();
        });
        remesh(
            Input {
                mesh: m,
                ..Input::default()
            },
            &params(200_000),
            &st.unbounded(),
        )
    });
    assert_eq!(o.status, Status::Cancelled, "{}", o.error);
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

#[test]
fn panics_are_caught() {
    let p = Params {
        debug_flags: debug::PANIC_TOP,
        ..params(500)
    };
    let o = run(icosphere(3, 1.0).unwrap(), &p);
    assert_eq!(o.status, Status::Panic);
    assert!(o.error.contains("injected panic"), "{}", o.error);
    // A panic in the only component: that component fails, its triangles go to failed_tris.
    let m = icosphere(3, 1.0).unwrap();
    let nt = m.tris.len();
    let p = Params {
        // The SF path, where a component panic is final (in M1 a rung panic goes to SF).
        debug_flags: debug::PANIC_COMPONENT | debug::FORCE_SF,
        ..params(500)
    };
    let o = run(m, &p);
    assert_eq!(o.status, Status::Failed);
    assert_eq!(o.comp_status, vec![CompStatus::Failed]);
    assert_eq!(o.failed_tris, (0..nt as u32).collect::<Vec<_>>());
    assert!(o.error.contains("injected panic"), "{}", o.error);
    // With two components only component 0 fails.
    let a = icosphere(3, 1.0).unwrap();
    let na = a.tris.len();
    let b = shifted(icosphere(3, 1.0).unwrap(), [5.0, 0.0, 0.0]);
    let o = run(merge(&[a, b]), &p);
    assert_eq!(o.status, Status::Degraded);
    assert_eq!(o.comp_status, vec![CompStatus::Failed, CompStatus::Sf]);
    assert_eq!(o.failed_tris, (0..na as u32).collect::<Vec<_>>());
    check_valid(&o, 2);
    assert!(o.mesh.face_comp.iter().all(|&c| c == 1));
}

#[test]
fn deterministic_for_any_thread_count() {
    let m = scaled(torus(3.0, 1.0, 64, 32, 0.2, 9).unwrap(), 1.3);
    let a = run(
        m.clone(),
        &Params {
            threads: 1,
            ..params(1200)
        },
    );
    let b = run(
        m.clone(),
        &Params {
            threads: 4,
            ..params(1200)
        },
    );
    let c = run(
        m,
        &Params {
            threads: 16,
            ..params(1200)
        },
    );
    assert_eq!(a.status, Status::Ok);
    assert_eq!(a.mesh, b.mesh);
    assert_eq!(a.mesh, c.mesh);
}

#[test]
fn huge_time_limit_is_no_limit() {
    for t in [1e20f32, f32::MAX, f32::INFINITY] {
        let p = Params {
            time_limit_s: t,
            ..params(500)
        };
        let o = run(icosphere(2, 1.0).unwrap(), &p);
        assert_eq!(o.status, Status::Ok, "time limit {t}: {}", o.error);
        check_valid(&o, 2);
    }
}

#[test]
fn timeout_never_panics() {
    let p = Params {
        time_limit_s: 0.001,
        ..params(50_000)
    };
    let m = icosphere(6, 1.0).unwrap();
    let nt = m.tris.len();
    let o = run(m, &p);
    assert!(
        matches!(o.status, Status::Failed | Status::Degraded | Status::Ok),
        "{:?}",
        o.status
    );
    if o.status == Status::Failed {
        assert_eq!(o.failed_tris.len(), nt);
    }
}

#[test]
fn stop_after_s0_returns_the_ingested_mesh() {
    let m = icosphere(2, 1.0).unwrap();
    let nt = m.tris.len();
    let p = Params {
        stop_after: Some(0),
        ..params(500)
    };
    let o = run(m, &p);
    assert_eq!(o.status, Status::Ok);
    assert_eq!(o.mesh.n_faces(), nt);
    let p = Params {
        stop_after: Some(12),
        ..params(500)
    };
    assert_eq!(
        run(icosphere(2, 1.0).unwrap(), &p).status,
        Status::InvalidInput
    );
}

#[test]
fn dumps_are_written() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("aq_engine_dump_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = Params {
        debug_flags: debug::DUMP,
        dump_dir: Some(dir.clone()),
        ..params(500)
    };
    let o = run(icosphere(3, 1.0).unwrap(), &p);
    assert_eq!(o.status, Status::Ok);
    for f in ["ingest.obj", "comp_0.obj", "result.obj"] {
        assert!(dir.join(f).is_file(), "{f}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn degenerate_and_repaired_inputs() {
    // Every face degenerate: status 2 or 4, never a panic.
    let m = TriMesh {
        pos: vec![[0.0; 3]; 3],
        tris: vec![[0, 1, 2]; 4],
    };
    let o = run(m, &params(100));
    assert!(
        matches!(o.status, Status::Failed | Status::InvalidInput),
        "{:?}",
        o.status
    );
    // A single triangle is an open component: all quads.
    let m = TriMesh {
        pos: vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        tris: vec![[0, 1, 2]],
    };
    let o = run(m, &params(100));
    assert_eq!(o.status, Status::Ok, "{}", o.error);
    check_valid(&o, 1);
}

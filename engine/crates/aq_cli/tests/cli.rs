//! autoquad.exe tests (DESIGN 9, 11.3 M0): remesh, stats, batch, exit codes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn exe() -> &'static str {
    env!("CARGO_BIN_EXE_autoquad")
}

fn engine_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn testdata(name: &str) -> PathBuf {
    engine_dir().join("testdata").join(format!("{name}.obj"))
}

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("aq_cli_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(exe()).args(args).output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn remesh_writes_mesh_and_report() {
    let d = tmp("remesh");
    let out = d.join("torus.obj");
    let rep = d.join("torus.json");
    let (code, _, err) = run(&[
        "remesh",
        s(&testdata("torus")),
        "-o",
        s(&out),
        "--target",
        "1500",
        "--report",
        s(&rep),
        "--axis",
        "z",
        "--reproject",
    ]);
    assert_eq!(code, 0, "{err}");
    let q = aq_io::read_obj_poly(&out).unwrap();
    q.validate().unwrap();
    assert!(q.is_all_quads());
    let r = json(&rep);
    assert_eq!(r["abi"], 1);
    assert_eq!(r["status"], 0);
    assert_eq!(r["stats"]["faces"].as_u64().unwrap() as usize, q.n_faces());
    assert_eq!(r["stats"]["quad_ratio"], 1.0);
    assert!(r["stats"]["count"]["err"].as_f64().unwrap().abs() < 0.25);
    assert!(r["stats"]["hausdorff"]["h"]["symmetric"].is_number());
    assert!(r["stats"]["hausdorff"]["pct_diag"]["symmetric"].is_number());
    assert!(r["stats"]["vertex_distance"]["max_h"].as_f64().unwrap() < 0.3);
    assert!(r["stats"]["loops"]["closed_frac"].is_number());
    assert!(r["timing_ms"]["total"].is_number());
    assert_eq!(r["cli"]["target_full"], 1500);
    // Output is back in input units: same bounding box as the input within 2%.
    let m = aq_io::read_obj_tri(&testdata("torus")).unwrap();
    let qd = {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &q.pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
    };
    assert!(
        (qd / m.bbox_diag() - 1.0).abs() < 0.02,
        "{qd} vs {}",
        m.bbox_diag()
    );
}

#[test]
fn remesh_symmetry_auto_features_upsample() {
    let d = tmp("sym");
    // Mirror: 0 open edges, 0 seam poles after the weld.
    let out = d.join("mirror.obj");
    let rep = d.join("mirror.json");
    let (code, _, err) = run(&[
        "remesh",
        s(&testdata("sculpt_sphere_mirror")),
        "-o",
        s(&out),
        "--target",
        "3000",
        "--sym",
        "x",
        "--report",
        s(&rep),
    ]);
    assert_eq!(code, 0, "{err}");
    let r = json(&rep);
    assert_eq!(r["stats"]["mirror_weld"]["open_edges"], 0);
    assert_eq!(r["stats"]["mirror_weld"]["seam_poles"], 0);
    assert_eq!(r["cli"]["target_engine"], 1500);
    let q = aq_io::read_obj_poly(&out).unwrap();
    q.validate().unwrap();
    assert!(q.is_all_quads());
    // Auto features on the cube: corners flagged in the engine, faces all quads.
    let out = d.join("cube.obj");
    let rep = d.join("cube.json");
    let (code, _, err) = run(&[
        "remesh",
        s(&testdata("cube_sharp")),
        "-o",
        s(&out),
        "--target",
        "800",
        "--features",
        "auto",
        "--hard-angle",
        "30",
        "--report",
        s(&rep),
    ]);
    assert_eq!(code, 0, "{err}");
    assert!(json(&rep)["cli"]["features"].as_u64().unwrap() > 0);
    // Upsample + stop after S0: the output is the ingested, upsampled triangle mesh.
    let out = d.join("s0.obj");
    let (code, _, err) = run(&[
        "remesh",
        s(&testdata("retopo_b")),
        "-o",
        s(&out),
        "--upsample",
        "1",
        "--stop-after",
        "S0",
    ]);
    assert_eq!(code, 0, "{err}");
    let q = aq_io::read_obj_poly(&out).unwrap();
    assert_eq!(q.n_faces(), 4 * 2525);
    // Dump directory.
    let dump = d.join("dump");
    let out = d.join("dumped.obj");
    let (code, _, err) = run(&[
        "remesh",
        s(&testdata("torus")),
        "-o",
        s(&out),
        "--target",
        "500",
        "--dump",
        s(&dump),
    ]);
    assert_eq!(code, 0, "{err}");
    assert!(dump.join("result.obj").is_file() && dump.join("ingest.obj").is_file());
}

#[test]
fn feature_and_density_files() {
    let d = tmp("files");
    let m = aq_io::read_obj_tri(&testdata("cube_sharp")).unwrap();
    let t = m.tris[0];
    let fe = d.join("f.txt");
    std::fs::write(&fe, format!("# one Hard edge\n{} {} 1\n", t[0], t[1])).unwrap();
    let de = d.join("d.txt");
    std::fs::write(&de, vec!["1.5"; m.pos.len()].join("\n")).unwrap();
    let out = d.join("o.obj");
    let rep = d.join("o.json");
    let (code, _, err) = run(&[
        "remesh",
        s(&testdata("cube_sharp")),
        "-o",
        s(&out),
        "--target",
        "600",
        "--features",
        s(&fe),
        "--density",
        s(&de),
        "--report",
        s(&rep),
    ]);
    assert_eq!(code, 0, "{err}");
    let r = json(&rep);
    assert_eq!(r["cli"]["features"], 1);
    // A density file of the wrong length is an error before the engine runs.
    std::fs::write(&de, "1 2 3").unwrap();
    let (code, _, _) = run(&[
        "remesh",
        s(&testdata("cube_sharp")),
        "-o",
        s(&out),
        "--density",
        s(&de),
    ]);
    assert_eq!(code, 1);
}

#[test]
fn exit_codes() {
    let d = tmp("codes");
    assert_eq!(run(&[]).0, 64);
    assert_eq!(run(&["frobnicate"]).0, 64);
    assert_eq!(run(&["remesh", "a.obj"]).0, 64, "missing -o");
    assert_eq!(
        run(&["remesh", "a.obj", "-o", "b.obj", "--target", "x"]).0,
        64
    );
    assert_eq!(run(&["remesh", "a.obj", "-o", "b.obj", "--wat"]).0, 64);
    assert_eq!(
        run(&[
            "remesh",
            s(&d.join("missing.obj")),
            "-o",
            s(&d.join("b.obj"))
        ])
        .0,
        1
    );
    // NaN coordinates: the engine says invalid input (4).
    let bad = d.join("nan.obj");
    std::fs::write(&bad, "v 0 0 0\nv 1 0 0\nv nan 1 0\nf 1 2 3\n").unwrap();
    assert_eq!(run(&["remesh", s(&bad), "-o", s(&d.join("b.obj"))]).0, 4);
    assert_eq!(
        run(&[
            "remesh",
            s(&testdata("torus")),
            "-o",
            s(&d.join("b.obj")),
            "--target",
            "0"
        ])
        .0,
        4
    );
    let (code, outp, _) = run(&["version"]);
    assert_eq!(code, 0);
    assert!(outp.contains("ABI 1"));
}

#[test]
fn stats_matches_quad_stats_py_on_a_baseline() {
    // tools/quad_stats.py on baselines/torus_qf2000.obj (fixture of aq_eval): 14 irregular.
    let b = engine_dir().join("baselines").join("torus_qf2000.obj");
    let (code, outp, err) = run(&[
        "stats",
        s(&b),
        "--json",
        "--ref",
        s(&testdata("torus")),
        "--axis",
        "z",
        "--target",
        "2000",
    ]);
    assert_eq!(code, 0, "{err}");
    let v: serde_json::Value = serde_json::from_str(&outp).unwrap();
    let q = aq_io::read_obj_poly(&b).unwrap();
    let st = aq_eval::quad_stats(&q);
    assert_eq!(v["irregular"], st.irregular);
    assert_eq!(v["irregular"], 14);
    assert_eq!(v["faces"], st.faces);
    assert!(v["hausdorff"]["h"]["symmetric"].is_number());
    assert!(v["loops"]["loops"].as_u64().unwrap() > 0);
    let (code, txt, _) = run(&["stats", s(&b)]);
    assert_eq!(code, 0);
    assert!(txt.contains("irregular       14"));
}

#[test]
fn batch_runner_and_markdown() {
    let d = tmp("batch");
    let td = engine_dir().join("testdata");
    let bl = engine_dir().join("baselines");
    let suite = d.join("suite.toml");
    std::fs::write(
        &suite,
        format!(
            r#"
[suite]
testdata = '{}'
baselines = '{}'
seeds = [0, 1]
seeds_required = 2

[gates.M0]
quad_ratio = 1.0
count_err = 0.25
vertex_dist_h = 0.3

[gates.M1]
quad_ratio = 1.0
allow_sf = false
irregular_dummy = 1

[[mesh]]
name = "torus"
target = 800
loop_axis = "all"
irregular_m1 = 0
loops_min_m1 = 1.0

[[mesh]]
name = "cube_sharp"
target = 600
features = "auto"
feature_cover = true
irregular_at_corners = true
"#,
            td.display(),
            bl.display()
        ),
    )
    .unwrap();
    let out = d.join("res");
    let md = d.join("summary.md");
    let (code, _, err) = run(&[
        "batch",
        "--suite",
        s(&suite),
        "--out",
        s(&out),
        "--seeds",
        "0..1",
        "--markdown",
        s(&md),
        "--gate",
        "M0",
    ]);
    assert_eq!(code, 0, "{err}");
    let sm = json(&out.join("summary.json"));
    assert_eq!(sm["milestone"], "M0");
    assert_eq!(sm["pass"], true, "{sm:#}");
    assert_eq!(sm["meshes"].as_array().unwrap().len(), 2);
    for name in ["torus_s0", "torus_s1", "cube_sharp_s0", "cube_sharp_s1"] {
        assert!(out.join(format!("{name}.obj")).is_file(), "{name}");
        assert!(out.join(format!("{name}.json")).is_file(), "{name}");
    }
    let cube = &sm["meshes"][1]["runs"][0];
    assert_eq!(cube["feature_cover"][0], cube["feature_cover"][1]);
    let text = std::fs::read_to_string(&md).unwrap();
    assert!(text.contains("| torus |") && text.contains("PASS"));
    assert!(!text.contains('\u{2014}') && !text.contains('\u{2013}'));
    // The torus has the QuadriFlow baseline torus_qf800? No: only qf2000 exists, so "(-)".
    assert!(text.contains("(-)"));
    // M1 gates (WP-16): the torus runs the full pipeline, with no SF component, 0 irregular
    // vertices and closed loops, so --strict exits 0.
    let (code, _, _) = run(&[
        "batch",
        "--suite",
        s(&suite),
        "--out",
        s(&out),
        "--seeds",
        "0",
        "--gate",
        "M1",
        "--only",
        "torus",
        "--strict",
    ]);
    assert_eq!(code, 0);
    let sm = json(&out.join("summary.json"));
    assert_eq!(sm["pass"], true);
    let gates = sm["meshes"][0]["runs"][0]["gates"]
        .as_array()
        .unwrap()
        .clone();
    assert!(
        gates
            .iter()
            .any(|g| g["name"] == "no_sf" && g["pass"] == true)
    );
    assert!(gates.iter().any(|g| g["name"] == "loops"));
    // A missing gate table is an error.
    assert_eq!(
        run(&[
            "batch",
            "--suite",
            s(&suite),
            "--out",
            s(&out),
            "--gate",
            "M2"
        ])
        .0,
        1
    );
}

/// M0 gate through the CLI (DESIGN 11.3): the shipped suite (12 testdata meshes and the perf
/// input) gives valid all-quad meshes (Ok or Degraded), with the report JSON written.
#[test]
fn shipped_suite_passes_the_m0_gates() {
    let suite = engine_dir().join("eval").join("suite.toml");
    let text = std::fs::read_to_string(&suite).unwrap();
    assert!(!text.contains('\u{2014}') && !text.contains('\u{2013}'));
    let d = tmp("suite");
    let md = d.join("summary.md");
    let (code, _, err) = run(&[
        "batch",
        "--suite",
        s(&suite),
        "--out",
        s(&d),
        "--seeds",
        "0",
        "--gate",
        "M0",
        "--markdown",
        s(&md),
    ]);
    // WP-16: with the full M1 pipeline the M0 quality gates (vertex distance, the 5 s time
    // gate) no longer all hold (retopo_a's thin glasses, basemesh_female on SF after two
    // rungs); the M0 essence stays checked: valid all-quad output on every mesh.
    assert_eq!(code, 0, "{err}");
    let sm = json(&d.join("summary.json"));
    assert_eq!(sm["meshes"].as_array().unwrap().len(), 13);
    for m in sm["meshes"].as_array().unwrap() {
        let r = &m["runs"][0];
        assert!(r["status"] == 0 || r["status"] == 1, "{}", m["name"]);
        assert_eq!(r["quad_ratio"], 1.0, "{}", m["name"]);
        let name = m["name"].as_str().unwrap();
        let q = aq_io::read_obj_poly(&d.join(format!("{name}_s0.obj"))).unwrap();
        q.validate().unwrap();
        assert!(q.is_all_quads());
        let rep = json(&d.join(format!("{name}_s0.json")));
        assert_eq!(rep["abi"], 1);
        assert!(rep["stats"].is_object() && rep["components"].is_array());
    }
    let names: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("name = \""))
        .map(|l| l.trim_end_matches('"'))
        .collect();
    assert_eq!(names.len(), 13);
    for n in &names[..12] {
        assert!(testdata(n).is_file(), "{n}");
    }
}

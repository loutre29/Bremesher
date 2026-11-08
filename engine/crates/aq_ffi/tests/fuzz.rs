//! Fuzz suite (DESIGN 10.3, WP-05): NaN, Inf, out-of-range indices, an empty mesh, a single
//! triangle, all faces degenerate, duplicate vertices, a bowtie, 500 floating components, a
//! mesh with 3-face edges, and random triangle soup (1000 cases). The only allowed results are
//! statuses 0, 1, 2 or 4, and a status 0 or 1 result must be a valid all-quad mesh.

mod common;

use aq_core::{SplitMix64, TriMesh};
use aq_testkit::{icosphere, merge};
use autoquad_engine::AqParams;
use common::*;

fn fuzz_params(target: u32) -> AqParams {
    AqParams {
        target_quads: target,
        threads: 2,
        time_limit_s: 20.0,
        ..defaults()
    }
}

/// Runs one case and checks the allowed statuses.
fn case(name: &str, raw: &RawIn, p: &AqParams) -> i32 {
    let r = call_raw(raw, p, None);
    assert!(
        matches!(r.status, 0 | 1 | 2 | 4),
        "{name}: status {} ({})",
        r.status,
        r.error
    );
    if r.status <= 1 {
        check_ok(&r);
    }
    if r.status == 2 {
        assert!(r.mesh.n_faces() == 0 || r.mesh.validate().is_ok(), "{name}");
    }
    let j: serde_json::Value = serde_json::from_str(&r.report).unwrap();
    assert_eq!(j["status"], r.status, "{name}");
    r.status
}

fn run_mesh(name: &str, m: &TriMesh, target: u32) -> i32 {
    case(name, &RawIn::new(m), &fuzz_params(target))
}

#[test]
fn named_cases() {
    let sphere = icosphere(3, 5.0).unwrap();
    // NaN and Inf.
    let mut m = sphere.clone();
    m.pos[5][0] = f64::NAN;
    assert_eq!(run_mesh("nan", &m, 500), 4);
    let mut m = sphere.clone();
    m.pos[9][2] = f64::INFINITY;
    assert_eq!(run_mesh("inf", &m, 500), 4);
    let mut m = sphere.clone();
    m.pos[1][1] = f64::NEG_INFINITY;
    assert_eq!(run_mesh("-inf", &m, 500), 4);
    // Out-of-range indices.
    let mut m = sphere.clone();
    m.tris[3][1] = m.pos.len() as u32;
    assert_eq!(run_mesh("index = n_verts", &m, 500), 4);
    let mut m = sphere.clone();
    m.tris[0][0] = u32::MAX;
    assert_eq!(run_mesh("index = u32::MAX", &m, 500), 4);
    // Empty meshes.
    assert_eq!(run_mesh("empty", &TriMesh::default(), 500), 4);
    let m = TriMesh {
        pos: sphere.pos.clone(),
        tris: Vec::new(),
    };
    assert_eq!(run_mesh("vertices only", &m, 500), 4);
    // A single triangle.
    let tri = TriMesh {
        pos: vec![[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
        tris: vec![[0, 1, 2]],
    };
    assert!(run_mesh("single triangle", &tri, 500) <= 1);
    assert!(run_mesh("single triangle, target 1", &tri, 1) <= 2);
    // All faces degenerate.
    let m = TriMesh {
        pos: vec![[1.0, 2.0, 3.0]; 4],
        tris: vec![[0, 1, 2], [1, 2, 3], [0, 2, 3]],
    };
    run_mesh("all degenerate (coincident)", &m, 500);
    let m = TriMesh {
        pos: (0..10).map(|i| [i as f64, 0.0, 0.0]).collect(),
        tris: (0..8).map(|i| [i, i + 1, i + 2]).collect(),
    };
    run_mesh("all degenerate (collinear)", &m, 500);
    let m = TriMesh {
        pos: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        tris: vec![[0, 0, 1], [1, 2, 2]],
    };
    run_mesh("repeated indices", &m, 500);
    // Duplicate vertices: every triangle has its own three vertices (a triangle soup sphere).
    let mut m = TriMesh::default();
    for t in &sphere.tris {
        let b = m.pos.len() as u32;
        for &v in t {
            m.pos.push(sphere.pos[v as usize]);
        }
        m.tris.push([b, b + 1, b + 2]);
    }
    assert!(run_mesh("duplicate vertices", &m, 500) <= 2);
    // A bowtie: two fans sharing one vertex.
    let m = TriMesh {
        pos: vec![
            [0.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, -1.0, 0.0],
            [-3.0, 1.0, 0.0],
            [-3.0, -1.0, 0.0],
        ],
        tris: vec![[0, 2, 1], [0, 3, 4]],
    };
    assert!(run_mesh("bowtie", &m, 200) <= 2);
    // A mesh with 3-face edges: three fins on one edge, plus a sphere next to it.
    let mut fins = TriMesh {
        pos: vec![[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]],
        tris: Vec::new(),
    };
    for k in 0..3 {
        let a = k as f64 * 2.0 * std::f64::consts::PI / 3.0;
        fins.pos.push([3.0 * a.cos(), 3.0 * a.sin(), 0.0]);
        fins.tris.push([0, 1, 2 + k]);
    }
    let m = merge(&[fins.clone(), sphere.clone()]);
    assert!(run_mesh("3-face edge", &m, 500) <= 2);
    // 500 floating components.
    let tiny = icosphere(1, 0.05).unwrap();
    let mut parts = Vec::new();
    for i in 0..500 {
        let mut t = tiny.clone();
        let (x, y, z) = (
            (i % 10) as f64 * 0.6,
            ((i / 10) % 10) as f64 * 0.6,
            (i / 100) as f64 * 0.6,
        );
        for p in t.pos.iter_mut() {
            *p = [p[0] + x, p[1] + y, p[2] + z];
        }
        parts.push(t);
    }
    let m = merge(&parts);
    let t = std::time::Instant::now();
    let s = run_mesh("500 components", &m, 5000);
    assert!(s <= 1, "500 components: status {s}");
    assert!(t.elapsed().as_secs_f64() < 20.0, "{:?}", t.elapsed());
    // Bad feature data.
    let raw = RawIn::new(&sphere).with_feats(&[[0, 999_999]], &[1]);
    assert_eq!(
        case("feature index out of range", &raw, &fuzz_params(500)),
        4
    );
    let raw = RawIn::new(&sphere).with_feats(&[[sphere.tris[0][0], sphere.tris[0][1]]], &[77]);
    assert_eq!(case("feature kind 77", &raw, &fuzz_params(500)), 4);
    let far = [sphere.tris[0][0], sphere.tris[sphere.tris.len() / 2][2]];
    let raw = RawIn::new(&sphere).with_feats(&[far], &[1]);
    assert!(case("feature pair far apart", &raw, &fuzz_params(500)) <= 1);
    // Bad density values (ignored at M0, never fatal).
    let mut raw = RawIn::new(&sphere);
    raw.density = Some(
        (0..sphere.pos.len())
            .map(|i| if i % 3 == 0 { f32::NAN } else { -1.0 })
            .collect(),
    );
    assert!(case("bad density", &raw, &fuzz_params(500)) <= 1);
    // Bad parameters.
    let raw = RawIn::new(&sphere);
    assert_eq!(
        case(
            "NaN align",
            &raw,
            &AqParams {
                curvature_align: f32::NAN,
                ..fuzz_params(500)
            }
        ),
        4
    );
    assert_eq!(case("huge target", &raw, &fuzz_params(u32::MAX)), 4);
    // A finite time limit too large for a Duration means no limit: status 0, no panic.
    for t in [1e20f32, f32::MAX, f32::INFINITY] {
        let p = AqParams {
            time_limit_s: t,
            ..fuzz_params(500)
        };
        assert_eq!(case(&format!("time limit {t}"), &raw, &p), 0);
    }
    assert!(
        case(
            "out-of-range sliders",
            &raw,
            &AqParams {
                adaptive_size: 9.0,
                hard_edges: 7,
                ..fuzz_params(500)
            }
        ) <= 1
    );
    // A huge coordinate scale and a tiny one.
    let mut m = sphere.clone();
    for p in m.pos.iter_mut() {
        *p = [p[0] * 1e12, p[1] * 1e12, p[2] * 1e12];
    }
    assert!(run_mesh("scale 1e12", &m, 300) <= 2);
    let mut m = sphere.clone();
    for p in m.pos.iter_mut() {
        *p = [p[0] * 1e-12, p[1] * 1e-12, p[2] * 1e-12];
    }
    assert!(run_mesh("scale 1e-12", &m, 300) <= 2);
}

/// One random soup: random vertices (sometimes on a coarse grid, which makes duplicates and
/// degenerate faces), random triangles over them, occasionally an out-of-range index.
fn soup(rng: &mut SplitMix64) -> (TriMesh, u32) {
    let nv = 3 + rng.below(60) as u32;
    let nt = 1 + rng.below(90) as u32;
    let grid = rng.below(3) == 0;
    let pos = (0..nv)
        .map(|_| {
            let mut p = [
                rng.range_f64(-5.0, 5.0),
                rng.range_f64(-5.0, 5.0),
                rng.range_f64(-5.0, 5.0),
            ];
            if grid {
                for c in p.iter_mut() {
                    *c = c.round();
                }
            }
            p
        })
        .collect();
    let oob = rng.below(50) == 0;
    let tris = (0..nt)
        .map(|_| {
            let mut t = [
                rng.below(nv as u64) as u32,
                rng.below(nv as u64) as u32,
                rng.below(nv as u64) as u32,
            ];
            if oob && rng.below(10) == 0 {
                t[1] = nv + rng.below(5) as u32;
            }
            t
        })
        .collect();
    let target = 8 + rng.below(300) as u32;
    (TriMesh { pos, tris }, target)
}

#[test]
fn random_triangle_soup_1000_cases() {
    let n = 1000u64;
    let workers = 4u64;
    let counts = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|w| {
                s.spawn(move || {
                    let mut c = [0u32; 6];
                    let mut i = w;
                    while i < n {
                        let mut rng = SplitMix64::new(0xF022 + i);
                        let (m, target) = soup(&mut rng);
                        let st = run_mesh(&format!("soup {i}"), &m, target);
                        c[st as usize] += 1;
                        i += workers;
                    }
                    c
                })
            })
            .collect();
        let mut tot = [0u32; 6];
        for h in hs {
            let c = h.join().unwrap();
            for k in 0..6 {
                tot[k] += c[k];
            }
        }
        tot
    });
    eprintln!("soup statuses 0..5: {counts:?}");
    assert_eq!(counts.iter().sum::<u32>(), n as u32);
    assert_eq!(counts[3] + counts[5], 0);
}

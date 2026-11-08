//! WP-04 definition of done: all 12 testdata meshes give 100% quads.
//!
//! Each mesh goes through ingest (with `--features auto` style Hard edges at 30 degrees, and
//! once without features), is split into components scaled to quad units for 5000 quads in
//! total, with the DESIGN 7.1 budget floors (24 quads closed, 8 open). Debris (closed with
//! area < 6 h^2, or a budget below the floor) goes through `sf_batch`, the rest through
//! `sf_quadrangulate`, as the engine's M0 path does. Every output must be a valid, manifold,
//! all-quad mesh with the component's Euler characteristic and every vertex on the surface.
//! Flips are reported, not asserted: on these meshes at 5000 quads some parts (fingers, toes,
//! folds of the source) are thinner than the SF edge length, and the aq_prep proxy itself
//! already has triangles reversed against the source there (CHANGES.md, WP-04).

mod common;

use aq_core::{CtlState, Projector};
use aq_fallback::{SfParams, sf_batch, sf_quadrangulate};
use aq_prep::{ComponentMesh, ingest, split_components};
use aq_testkit::{load_testdata, testdata_names};
use common::*;

const N: f64 = 5000.0;

fn run_mesh(name: &str, auto: bool) -> (u32, u32) {
    let m = load_testdata(name).unwrap();
    let (pairs, kinds) = if auto { auto_features(&m, 30.0) } else { (Vec::new(), Vec::new()) };
    let st = CtlState::new();
    let ing = ingest(m, &pairs, &kinds, &st.unbounded()).unwrap();
    let comps = split_components(&ing);
    let a: f64 = comps.iter().map(|c| c.area).sum();
    let h0 = (a / N).sqrt();
    let scaled: Vec<ComponentMesh> = comps.iter().map(|c| c.scaled(1.0 / h0)).collect();
    let mut debris: Vec<(usize, u32)> = Vec::new();
    let mut total = 0u32;
    let mut target_total = 0u32;
    for (i, c) in scaled.iter().enumerate() {
        let floor = if c.closed { 24.0 } else { 8.0 };
        // N_c = N * A_c / A (DESIGN 7.1 pass 1), in quad units.
        let nc = N * c.area / (a / (h0 * h0));
        let t = nc.round().max(floor) as u32;
        target_total += t;
        if (c.closed && c.area < 6.0) || nc < floor {
            debris.push((i, t));
            continue;
        }
        let (fr, bvh) = c.source_geometry().unwrap();
        let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
        let q = sf_quadrangulate(c, &proj, &SfParams { target: t, relax_iters: 5 }, &st.unbounded())
            .unwrap_or_else(|e| panic!("{name} comp {i}: {e}"));
        check_with(&format!("{name}{} comp {i}", if auto { " (auto)" } else { "" }), &q, c, t, 1.0, f64::INFINITY);
        eprintln!("    angle deviation {:.1} deg", angle_dev_deg(&q));
        total += q.n_faces() as u32;
    }
    if !debris.is_empty() {
        let srcs: Vec<&ComponentMesh> = debris.iter().map(|&(i, _)| &scaled[i]).collect();
        let ps: Vec<SfParams> = debris.iter().map(|&(_, t)| SfParams { target: t, relax_iters: 5 }).collect();
        let qs = sf_batch(&srcs, &ps, &st.unbounded()).unwrap_or_else(|e| panic!("{name} debris: {e}"));
        for ((q, c), &(i, t)) in qs.iter().zip(&srcs).zip(&debris) {
            q.validate().unwrap();
            assert!(q.is_all_quads(), "{name} debris {i}: not all quads");
            assert_eq!(nonmanifold_vertices(q), 0, "{name} debris {i}");
            assert_eq!(euler(q), c.chi, "{name} debris {i}: chi");
            let flips = flipped_quads(q, &c.mesh);
            eprintln!("{name} debris {i}: target {t}, quads {}, flips {flips}", q.n_faces());
            total += q.n_faces() as u32;
        }
    }
    (total, target_total)
}

#[test]
fn all_testdata_meshes() {
    for name in testdata_names() {
        for auto in [false, true] {
            let t0 = std::time::Instant::now();
            let (n, t) = run_mesh(name, auto);
            eprintln!("== {name} auto={auto}: {n} quads for {t} ({:.2} s)", t0.elapsed().as_secs_f64());
            // With auto features the chain network can be denser than the target resolution
            // (basemesh_female: 1409 frozen chain corners for about 1250 SF vertices), so the
            // count is only checked without features.
            if !auto {
                assert!((n as f64 - t as f64).abs() <= 0.25 * t as f64, "{name}: {n} quads for {t}");
            }
        }
    }
}

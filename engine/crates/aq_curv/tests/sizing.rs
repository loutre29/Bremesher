//! DESIGN 6.5 sizing tests (WP-07), M1 part, plus the S3 driver, performance, determinism
//! and the testdata smoke run.

mod common;

use std::f64::consts::TAU;

use common::*;

use aq_core::{ErrKind, SizeField, TriMesh, math};
use aq_curv::{CurvParams, SizingParams, curvature, curvature_and_sizing, sizing, sizing_with};
use aq_testkit::Surface;

const EDGE: f64 = 1.0 / 3.0;

fn run(s: &Surface, sp: &SizingParams) -> (aq_core::CurvField, SizeField, aq_curv::SizingReport) {
    let st = ctl_state();
    let ctl = st.unbounded();
    let px = view(s);
    let c = curvature(&px, &vec![1.0; s.n_verts()], &CurvParams::default(), &ctl).unwrap();
    let (sf, rep) = sizing_with(&px, &c, sp, None, &ctl).unwrap();
    sf.validate(&s.frames, s.n_verts()).unwrap();
    (c, sf, rep)
}

/// `log h(y) <= log h(x) + grad |xy| / h(x)` on every edge, both directions; returns the
/// largest excess.
fn gradation_excess(s: &Surface, h: &[f64], grad: f64) -> f64 {
    let mut worst = f64::NEG_INFINITY;
    for e in 0..3 * s.n_faces() as u32 {
        if !s.topo.is_canonical(e) {
            continue;
        }
        let a = s.topo.tail(&s.mesh, e) as usize;
        let b = s.topo.head(&s.mesh, e) as usize;
        let l = math::dist(s.mesh.pos[a], s.mesh.pos[b]);
        for (x, y) in [(a, b), (b, a)] {
            worst = worst.max(h[y].ln() - h[x].ln() - grad * l / h[x]);
        }
    }
    worst
}

#[test]
fn uniform_without_adaptive_or_density() {
    let cases: Vec<(&str, TriMesh, f64)> = vec![
        ("icosphere R 4", aq_testkit::icosphere(4, 4.0).unwrap(), 201.0),
        ("jittered plane", aq_testkit::plane_grid(30, 24, EDGE, 0.3, 0.2, 1).unwrap(), 80.0),
        ("noisy sphere", {
            let mut m = aq_testkit::icosphere(4, 6.0).unwrap();
            aq_testkit::add_noise(&mut m, 0.05, 1.5, 2).unwrap();
            m
        }, 452.4),
    ];
    for (name, m, n) in cases {
        let s = surf(m);
        let sp = SizingParams::new(n);
        let (_, sf, rep) = run(&s, &sp);
        let (lo, hi) = minmax(&sf.h_vert);
        let (flo, fhi) = minmax(&sf.h_face);
        let count = sf.predicted_count(&s.frames);
        println!("{name}: h in [{lo}, {hi}], faces [{flo}, {fhi}], count {count} for N = {n}, report {rep:?}");
        assert!(hi - lo <= 1e-12 * hi, "{name}: h not uniform");
        assert!(fhi - flo <= 1e-12 * fhi && (fhi - hi).abs() <= 1e-12 * hi, "{name}: h_face not uniform");
        assert!((count - n).abs() <= 1e-9, "{name}: count {count} for {n}");
        assert!((sf.n_pred - n).abs() <= 1e-9, "{name}: n_pred {}", sf.n_pred);
        assert_eq!(rep.capped_verts, 0);
        assert_eq!(rep.graded_verts, 0);
        // h = sqrt(A / N) exactly in the uniform case.
        let a: f64 = s.frames.area.iter().sum();
        assert!((hi - (a / n).sqrt()).abs() <= 1e-12 * hi);
    }
}

/// A thin tube R = 0.5 next to a large flat sheet: the tube cap holds and the count is exact.
#[test]
fn thin_cylinder_tube_cap() {
    let r = 0.5;
    let len = 8.0;
    let tube = cylinder(r, len, EDGE, 0.2, 3);
    let n_tube_verts = tube.pos.len();
    let mut sheet = aq_testkit::plane_grid(60, 60, EDGE, 0.0, 0.2, 4).unwrap();
    translate(&mut sheet, [5.0, -10.0, 0.0]);
    let s = surf(aq_testkit::merge(&[tube, sheet]));
    let area: f64 = s.frames.area.iter().sum();
    let n = area; // h0 = 1 (quad units)
    let (_, sf, rep) = run(&s, &SizingParams::new(n));
    println!("tube + sheet: report {rep:?}");
    assert!(!rep.cap_infeasible);
    let count = sf.predicted_count(&s.frames);
    assert!((count - n).abs() <= 1e-9 * n.max(1.0), "count {count} for {n}");
    let mut worst = f64::INFINITY;
    for v in 0..n_tube_verts {
        worst = worst.min(TAU * r / sf.h_vert[v]);
    }
    for f in 0..s.n_faces() {
        if (s.mesh.tris[f][0] as usize) < n_tube_verts {
            worst = worst.min(TAU * r / sf.h_face[f]);
        }
    }
    println!("tube: min 2 pi R / h = {worst}");
    assert!(worst >= 8.0, "2 pi R / h = {worst}");
    // The sheet gets the rest of the budget: coarser than h0 = 1.
    assert!(sf.h_vert[s.n_verts() - 1] > 1.0);
    assert!(gradation_excess(&s, &sf.h_vert, 0.25) <= 1e-9);
}

/// The whole component is a thin tube: the caps alone exceed N, the count wins (CHANGES.md).
#[test]
fn thin_cylinder_alone_count_wins() {
    let s = surf(cylinder(0.5, 8.0, EDGE, 0.2, 5));
    let n: f64 = s.frames.area.iter().sum();
    let (_, sf, rep) = run(&s, &SizingParams::new(n));
    println!("tube alone: report {rep:?}");
    assert!(rep.cap_infeasible);
    assert!(rep.cap_relax > 1.0);
    let count = sf.predicted_count(&s.frames);
    assert!((count - n).abs() <= 1e-9 * n.max(1.0), "count {count} for {n}");
    assert!(gradation_excess(&s, &sf.h_vert, 0.25) <= 1e-9);
    // Same request with enough quads for the cap: feasible, cap holds.
    let (_, sf2, rep2) = run(&s, &SizingParams::new(10.0 * n));
    assert!(!rep2.cap_infeasible);
    let worst = sf2.h_vert.iter().map(|h| TAU * 0.5 / h).fold(f64::INFINITY, f64::min);
    assert!(worst >= 8.0, "{worst}");
}

/// A trumpet: a thin tube (radius 0.4) that flares to radius 3 over z in [5, 6.5]. The caps
/// jump along the axis, so the gradation is active. No triangulation jitter: on the 0.05 x 0.33
/// cells of the thin part, z jitter makes slivers whose clamped cotan weights over-diffuse
/// (an isotropic proxy has none).
fn trumpet(noise: bool) -> Surface {
    let len = 12.0;
    let mut m = aq_testkit::cylinder_open(1.0, len, 48, 36, 0.0, 8).unwrap();
    for p in m.pos.iter_mut() {
        let s = ((p[2] - 5.0) / 1.5).clamp(0.0, 1.0);
        let r = 0.4 + 2.6 * s * s * (3.0 - 2.0 * s);
        p[0] *= r;
        p[1] *= r;
    }
    if noise {
        aq_testkit::add_noise(&mut m, 0.01, 0.8, 9).unwrap();
    }
    surf(m)
}

#[test]
fn gradation_holds_on_every_edge() {
    let s = trumpet(false);
    let n: f64 = s.frames.area.iter().sum::<f64>() * 1.5;
    for grad in [0.25, 0.1, 1.0] {
        let sp = SizingParams { grad, ..SizingParams::new(n) };
        let (c, sf, rep) = run(&s, &sp);
        let ex = gradation_excess(&s, &sf.h_vert, grad);
        let count = sf.predicted_count(&s.frames);
        let (lo, hi) = minmax(&sf.h_vert);
        println!("trumpet grad {grad}: excess {ex:e}, h in [{lo:.4}, {hi:.4}], count {count} / {n}, {rep:?}");
        assert!(ex <= 1e-9, "gradation excess {ex}");
        assert!((count - n).abs() <= 1e-9 * n, "count {count}");
        if grad == 0.25 {
            assert!(!rep.cap_infeasible);
            assert!(rep.capped_verts > 0 && rep.graded_verts > 0, "cap and gradation must both be active");
        }
        if rep.cap_infeasible {
            // grad 0.1: the caps grade the whole flare down, so they alone exceed N.
            continue;
        }
        // The cap holds as defined: h_face <= 2 pi / (ring_min |k1|) on every tube face.
        let mut n_tube = 0;
        for f in 0..s.n_faces() {
            if c.conf[f] > aq_curv::TUBE_CONF && aq_curv::anisotropy(c.k1[f], c.k2[f]) > aq_curv::TUBE_ANISO {
                n_tube += 1;
                let cap = TAU / (8.0 * c.k1[f].abs());
                assert!(sf.h_face[f] <= cap * (1.0 + 1e-12), "face {f}: h {} > cap {cap}", sf.h_face[f]);
            }
        }
        assert!(n_tube > 100);
    }
}

#[test]
fn errors_and_cancel() {
    let s = surf(aq_testkit::icosphere(2, 2.0).unwrap());
    let st = ctl_state();
    let ctl = st.unbounded();
    let px = view(&s);
    let c = curvature(&px, &vec![1.0; s.n_verts()], &CurvParams::default(), &ctl).unwrap();
    for bad in [
        SizingParams::new(0.0),
        SizingParams::new(f64::NAN),
        SizingParams { adaptive: 1.5, ..SizingParams::new(10.0) },
        SizingParams { grad: 0.0, ..SizingParams::new(10.0) },
    ] {
        let e = sizing(&px, &c, &bad, &ctl).unwrap_err();
        assert_eq!(e.kind, ErrKind::InvalidInput);
        assert_eq!(e.stage, aq_core::Stage::Curv);
    }
    let sp = SizingParams::new(10.0);
    assert_eq!(sizing_with(&px, &c, &sp, Some(&[1.0; 2]), &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    let mut d = vec![1.0; s.n_verts()];
    d[0] = -1.0;
    assert_eq!(sizing_with(&px, &c, &sp, Some(&d), &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    let mut c2 = c.clone();
    c2.conf.pop();
    assert_eq!(sizing(&px, &c2, &sp, &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    // thickness = true runs the M2 cap (WP-20; it used to be reported as skipped), and its
    // parameters are checked.
    let (_, rep) = sizing_with(&px, &c, &SizingParams { thickness: true, ..sp }, None, &ctl).unwrap();
    assert!(!rep.thickness_skipped);
    for bad in [
        SizingParams { thickness: true, thick_budget: -1.0, ..sp },
        SizingParams { thickness: true, thick_floor_min: 0.0, ..sp },
        SizingParams { thickness: true, thick_floor: Some(0.0), ..sp },
    ] {
        assert_eq!(sizing_with(&px, &c, &bad, None, &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    }
    assert_eq!(
        aq_curv::sizing_diam(&px, &c, &SizingParams { thickness: true, ..sp }, None, Some(&[1.0; 2]), &ctl).unwrap_err().kind,
        ErrKind::InvalidInput
    );
    st.cancel();
    assert_eq!(sizing(&px, &c, &sp, &ctl).unwrap_err().kind, ErrKind::Cancelled);
}

#[test]
fn driver_passes_and_density() {
    let s = surf(aq_testkit::plane_grid(36, 36, EDGE, 0.2, 0.2, 6).unwrap());
    let st = ctl_state();
    let ctl = st.unbounded();
    let px = view(&s);
    let n: f64 = s.frames.area.iter().sum();
    let cp = CurvParams::default();
    let (c, sf, rep) = curvature_and_sizing(&px, None, &cp, &SizingParams::new(n), &ctl).unwrap();
    assert_eq!(rep.passes, 1);
    c.validate(s.n_faces(), s.n_verts()).unwrap();
    sf.validate(&s.frames, s.n_verts()).unwrap();
    // A density doubling on one half: two passes, denser there, count exact.
    let x_mid = s.mesh.pos.iter().map(|p| p[0]).sum::<f64>() / s.n_verts() as f64;
    let dens: Vec<f64> = s.mesh.pos.iter().map(|p| if p[0] > x_mid { 4.0 } else { 1.0 }).collect();
    let (_, sf2, rep2) = curvature_and_sizing(&px, Some(&dens), &cp, &SizingParams::new(n), &ctl).unwrap();
    assert_eq!(rep2.passes, 2);
    assert!(rep2.sizing.density_used);
    assert!((sf2.predicted_count(&s.frames) - n).abs() <= 1e-9 * n);
    let mean_h = |pred: &dyn Fn(f64) -> bool| {
        let v: Vec<f64> = s.mesh.pos.iter().zip(&sf2.h_vert).filter(|(p, _)| pred(p[0])).map(|(_, h)| *h).collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    // Away from the graded transition (about 3.2 units wide at grad 0.25).
    let dense = mean_h(&|x| x > x_mid + 2.0);
    let sparse = mean_h(&|x| x < x_mid - 4.5);
    println!("density 4x: mean h dense {dense:.4}, sparse {sparse:.4}");
    // rho^(-1/2) halves h where the density is 4x (away from the graded transition).
    assert!((sparse / dense - 2.0).abs() < 0.02, "ratio {}", sparse / dense);
}

/// M2 sanity only (the M2 gate is WP-20's): adaptive = 1 makes h smaller where kappa_hat is
/// larger.
#[test]
fn adaptive_sanity_bump() {
    let m = aq_testkit::heightfield(60, 10.0, |x, y| 1.5 * (-(x * x + y * y) / 2.0).exp()).unwrap();
    let s = surf(m);
    let n: f64 = s.frames.area.iter().sum();
    let (_, sf, _) = run(&s, &SizingParams { adaptive: 1.0, ..SizingParams::new(n) });
    let dens = |pred: &dyn Fn(f64) -> bool| {
        let v: Vec<f64> = s
            .mesh
            .pos
            .iter()
            .zip(&sf.h_vert)
            .filter(|(p, _)| pred(p[0].hypot(p[1])))
            .map(|(_, h)| 1.0 / (h * h))
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let bump = dens(&|r| r < 0.8);
    let flat = dens(&|r| r > 7.0 && r < 9.0);
    println!("adaptive = 1: quad density bump {bump:.4}, flat {flat:.4}, ratio {:.3}", bump / flat);
    assert!(bump > flat);
    assert!((sf.predicted_count(&s.frames) - n).abs() <= 1e-9 * n);
}

#[test]
fn deterministic_across_thread_counts() {
    let s = trumpet(true);
    let n: f64 = s.frames.area.iter().sum::<f64>() * 0.7;
    let dens: Vec<f64> = s.mesh.pos.iter().map(|p| 1.0 + 0.1 * p[2]).collect();
    let go = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let st = ctl_state();
            let sp = SizingParams { adaptive: 0.5, ..SizingParams::new(n) };
            curvature_and_sizing(&view(&s), Some(&dens), &CurvParams::default(), &sp, &st.unbounded()).unwrap()
        })
    };
    let (c1, s1, _) = go(1);
    for t in [4, 16] {
        let (c2, s2, _) = go(t);
        let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&c1.k1), bits(&c2.k1));
        assert_eq!(bits(&c1.phi), bits(&c2.phi));
        assert_eq!(bits(&c1.conf), bits(&c2.conf));
        assert_eq!(bits(&s1.h_vert), bits(&s2.h_vert));
        assert_eq!(bits(&s1.h_face), bits(&s2.h_face));
        assert_eq!(s1.n_pred.to_bits(), s2.n_pred.to_bits());
    }
}

/// DESIGN 6.5 performance: 100k triangles, both passes, < 1.5 s (release builds only).
#[test]
fn performance_100k_two_passes() {
    let mut m = aq_testkit::torus(13.0, 5.0, 250, 200, 0.2, 21).unwrap();
    aq_testkit::add_noise(&mut m, 0.03, 1.0, 22).unwrap();
    let s = surf(m);
    assert_eq!(s.n_faces(), 100_000);
    let n: f64 = s.frames.area.iter().sum::<f64>() / 9.0 * 0.95; // proxy edges of about h/3
    let st = ctl_state();
    let sp = SizingParams { adaptive: 1.0, ..SizingParams::new(n) };
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let (_, sf, rep) = curvature_and_sizing(&view(&s), None, &CurvParams::default(), &sp, &st.unbounded()).unwrap();
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
        assert_eq!(rep.passes, 2);
        assert!((sf.predicted_count(&s.frames) - n).abs() <= 1e-9 * n);
        println!(
            "100k triangles, 2 passes: {dt:.3} s (curvature {:.0} ms, sizing {:.0} ms, {} evaluations)",
            rep.curv_ms, rep.sizing_ms, rep.sizing.evaluations
        );
    }
    if cfg!(debug_assertions) {
        println!("debug build: timing not asserted");
    } else {
        assert!(best < 1.5, "best {best} s");
    }
}

/// Smoke run on the testdata meshes, scaled to quad units for N = 3000 (h0 = 1). Meshes whose
/// raw topology aq_core rejects (aq_prep repairs them in the pipeline) are skipped and listed.
#[test]
fn testdata_smoke() {
    let mut ran = 0;
    for name in aq_testkit::testdata_names() {
        let Ok(mut m) = aq_testkit::load_testdata(name) else {
            println!("{name}: cannot load");
            continue;
        };
        let area = m.total_area();
        let h0 = (area / 3000.0).sqrt();
        for p in m.pos.iter_mut() {
            *p = math::scale(*p, 1.0 / h0);
        }
        let s = match Surface::new(m) {
            Ok(s) => s,
            Err(e) => {
                println!("{name}: skipped ({e})");
                continue;
            }
        };
        let st = ctl_state();
        let px = view(&s);
        let t = std::time::Instant::now();
        let (c, sf, rep) =
            curvature_and_sizing(&px, None, &CurvParams::default(), &SizingParams::new(3000.0), &st.unbounded())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        c.validate(s.n_faces(), s.n_verts()).unwrap();
        sf.validate(&s.frames, s.n_verts()).unwrap();
        assert!((sf.predicted_count(&s.frames) - 3000.0).abs() <= 1e-9 * 3000.0);
        let (lo, hi) = minmax(&sf.h_vert);
        let hi_conf = frac(c.conf.iter().map(|x| *x > 0.5));
        println!(
            "{name}: {} faces, {:.0} ms, h in [{lo:.3}, {hi:.3}], conf > 0.5 on {hi_conf:.3}, tube verts {}, capped {}, infeasible {}",
            s.n_faces(),
            t.elapsed().as_secs_f64() * 1e3,
            rep.sizing.tube_verts,
            rep.sizing.capped_verts,
            rep.sizing.cap_infeasible
        );
        ran += 1;
    }
    assert!(ran >= 8, "only {ran} testdata meshes ran");
}

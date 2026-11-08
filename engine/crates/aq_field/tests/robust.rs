//! DESIGN 6.6: Poincare-Hopf on every testkit generator and on the 12 testdata meshes, noise
//! robustness, determinism and the time budget.
#![allow(clippy::needless_range_loop)]

mod common;

use std::time::Instant;

use aq_core::math;
use aq_core::CrossField;
use aq_field::{FieldParams, FieldReport, solve_field};
use aq_testkit::Surface;
use common::*;

fn solve(s: &Surface, h: f64) -> (CrossField, FieldReport) {
    let c = estimate_curv(s, h);
    let st = ctl_state();
    let (f, rep) = solve_field(&px(s), &c, &size(s, h), &FieldParams::default(), &st.unbounded()).unwrap();
    // Poincare-Hopf per component (also checked inside solve_field).
    aq_core::cross::poincare_hopf(&s.mesh, &s.topo, &f.index4).unwrap();
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
    (f, rep)
}

fn generators() -> Vec<(&'static str, Surface, f64)> {
    use aq_testkit::*;
    vec![
        ("plane", Surface::new(plane_grid(30, 24, 1.0 / 3.0, 0.4, 0.2, 1).unwrap()).unwrap(), 1.0),
        ("square_disk", Surface::new(square_disk(30, 10.0, 0.25, 2).unwrap()).unwrap(), 1.0),
        ("uv_sphere", Surface::new(uv_sphere(24, 48, 4.0).unwrap()).unwrap(), 1.0),
        ("icosphere", Surface::new(icosphere(4, 5.0).unwrap()).unwrap(), 1.0),
        ("octasphere", Surface::new(octasphere(16, 4.0).unwrap()).unwrap(), 1.0),
        ("cylinder_open", Surface::new(cylinder_open(2.0, 8.0, 40, 24, 0.2, 3).unwrap()).unwrap(), 1.0),
        ("cylinder_closed", Surface::new(cylinder_closed(2.0, 8.0, 40, 24, 0.2, 4).unwrap()).unwrap(), 1.0),
        ("torus", Surface::new(torus(3.0, 1.0, 60, 20, 0.2, 5).unwrap()).unwrap(), 1.0),
        ("double_torus", Surface::new(double_torus().unwrap()).unwrap(), 0.15),
        ("cube_sharp", Surface::from_synth(&cube_sharp(12, 4.0, 0.1, 6).unwrap()).unwrap(), 1.0),
        ("hemisphere", Surface::from_synth(&hemisphere(16, 4.0).unwrap()).unwrap(), 1.0),
        (
            "saddle",
            Surface::new(heightfield(30, 10.0, |x, y| (x * x - y * y) / 20.0).unwrap()).unwrap(),
            1.0,
        ),
        (
            "two_components",
            Surface::new(merge(&[icosphere(3, 3.0).unwrap(), torus(3.0, 1.0, 50, 16, 0.1, 7).unwrap()])).unwrap(),
            1.0,
        ),
        ("noisy_sphere", {
            let mut m = icosphere(4, 5.0).unwrap();
            add_noise(&mut m, 0.05, 0.6, 8).unwrap();
            Surface::new(m).unwrap()
        }, 1.0),
    ]
}

#[test]
fn poincare_hopf_on_every_generator() {
    for (name, s, h) in generators() {
        let (f, rep) = solve(&s, h);
        let sum: i64 = f.index4.iter().map(|&i| i as i64).sum();
        println!(
            "{name:16} faces {:6} chi {:3} sum index4 {:4} cones {:3} (initial {:3}, split {}, pushed {}, cancelled {}, reverted {}) eigen {}",
            s.n_faces(),
            s.chi_total(),
            sum,
            rep.cones_final,
            rep.cones_initial,
            rep.split,
            rep.pushed,
            rep.cancelled,
            rep.reverted,
            rep.eigen_path
        );
        assert_eq!(sum, 4 * s.chi_total(), "{name}");
    }
}

#[test]
fn poincare_hopf_on_testdata() {
    for name in aq_testkit::testdata_names() {
        let s = testdata_surface(name);
        let t0 = Instant::now();
        let (f, rep) = solve(&s, 1.0);
        let sum: i64 = f.index4.iter().map(|&i| i as i64).sum();
        let interior = interior_cones(&s, &f).len();
        println!(
            "{name:22} faces {:7} comps {:2} chi {:4} sum {:4} cones {:4} interior {:4} (initial {:4}, split {}, pushed {}, cancelled {}, reverted {}) {:.2} s",
            s.n_faces(),
            s.topo.n_comp,
            s.chi_total(),
            sum,
            rep.cones_final,
            interior,
            rep.cones_initial,
            rep.split,
            rep.pushed,
            rep.cancelled,
            rep.reverted,
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(sum, 4 * s.chi_total(), "{name}");
    }
}

/// A smooth blob of about `n` quads (h = 1), proxy edge about h/3, optionally with noise.
fn blob(n: f64, noise: Option<(f64, f64, u64)>) -> Surface {
    let r = (n / (4.0 * std::f64::consts::PI)).sqrt() * 0.93;
    let mut m = aq_testkit::icosphere(6, r).unwrap();
    for p in m.pos.iter_mut() {
        let u = math::scale(*p, 1.0 / r);
        let k = 1.0 + 0.15 * (2.1 * u[0] + 0.3).sin() * (1.7 * u[1]).sin() + 0.1 * (2.9 * u[2]).cos();
        *p = math::scale(*p, k);
    }
    if let Some((amp, wl, seed)) = noise {
        aq_testkit::add_noise(&mut m, amp, wl, seed).unwrap();
    }
    Surface::new(m).unwrap()
}

/// Radial lumps on a sphere mesh of radius `r`: an ellipsoid (1.5 : 1 : 0.75) displaced by 6
/// random waves of 6 to 13 h wavelength and relative amplitude 0.15, so that the curvature is
/// anisotropic at the quad scale; then optional noise of amplitude `amp` h, wavelength `wl` h.
fn lumpy(mut m: aq_core::TriMesh, r: f64, noise: Option<(f64, f64, u64)>) -> Surface {
    let mut rng = aq_core::SplitMix64::new(11);
    let mut waves = Vec::new();
    for _ in 0..6 {
        let z = rng.range_f64(-1.0, 1.0);
        let a = rng.range_f64(0.0, std::f64::consts::TAU);
        let s = (1.0 - z * z).sqrt();
        waves.push(([s * a.cos(), s * a.sin(), z], rng.range_f64(0.0, std::f64::consts::TAU), rng.range_f64(4.0, 8.0)));
    }
    for p in m.pos.iter_mut() {
        let u = math::scale(*p, 1.0 / r);
        let d: f64 = waves.iter().map(|(dir, ph, k)| (k * math::dot(*dir, u) + ph).sin()).sum::<f64>() / 6f64.sqrt();
        let k = 1.0 + 0.15 * d;
        *p = [p[0] * k * 1.5, p[1] * k, p[2] * k * 0.75];
    }
    if let Some((amp, wl, seed)) = noise {
        aq_testkit::add_noise(&mut m, amp, wl, seed).unwrap();
    }
    Surface::new(m).unwrap()
}

/// The lumpy blob at N = 5000 (h = 1, 81920 faces, proxy edge about h/3, area-weighted mean
/// conf about 0.11).
fn lumpy_blob(noise: Option<(f64, f64, u64)>) -> Surface {
    let r = (5000.0 / (4.0 * std::f64::consts::PI)).sqrt() * 0.8;
    lumpy(aq_testkit::icosphere(6, r).unwrap(), r, noise)
}

fn mean_conf(s: &Surface, c: &aq_core::CurvField) -> f64 {
    let a: f64 = s.frames.area.iter().sum();
    s.frames.area.iter().zip(&c.conf).map(|(x, y)| x * y).sum::<f64>() / a
}

/// Cone count, report and area-weighted mean conf of S4 with curvature_align `align`.
fn solve_with(s: &Surface, align: f64) -> (usize, FieldReport, f64) {
    let c = estimate_curv(s, 1.0);
    let st = ctl_state();
    let p = FieldParams { align, ..FieldParams::default() };
    let (f, rep) = solve_field(&px(s), &c, &size(s, 1.0), &p, &st.unbounded()).unwrap();
    aq_core::cross::poincare_hopf(&s.mesh, &s.topo, &f.index4).unwrap();
    (f.cones().len(), rep, mean_conf(s, &c))
}

/// DESIGN 6.6 "Noise robustness" on the linear (Knoppel) path: the clean lumpy blob and every
/// noisy copy have a mean conf well above 0.01, so curvature alignment is on.
#[test]
fn noise_robustness_lumpy_blob_5000() {
    let clean = lumpy_blob(None);
    let (nc, rc, mc) = solve_with(&clean, 1.0);
    let (nc0, _, _) = solve_with(&clean, 0.0);
    println!(
        "lumpy blob: {} faces, mean conf {mc:.3}, cones {nc} (initial {}), smooth field (align 0) {nc0}",
        clean.n_faces(),
        rc.cones_initial
    );
    assert!(!rc.eigen_path);
    assert!(mc > 0.05, "clean mean conf {mc}");
    // Noise of 0.05h to 0.1h at wavelength 0.4h to 1h.
    for (amp, wl, seed) in [(0.05, 0.4, 1u64), (0.05, 0.7, 2), (0.05, 1.0, 3), (0.1, 1.0, 5)] {
        let noisy = lumpy_blob(Some((amp, wl, seed)));
        let (nn, rn, mn) = solve_with(&noisy, 1.0);
        println!(
            "noise amp {amp} wl {wl}: mean conf {mn:.3}, cones {nn} (initial {}, cancelled {}), clean {nc}, ratio {:.2}",
            rn.cones_initial,
            rn.cancelled,
            nn as f64 / nc as f64
        );
        assert!(!rn.eigen_path, "noise {amp} {wl} took the eigen path");
        assert!(mn > 0.05, "noisy mean conf {mn}");
        assert!(nn as f64 <= 1.5 * nc as f64, "noise {amp} {wl}: {nn} cones, clean {nc}");
    }
}

/// Heavier noise (0.1h at 0.7h, 0.2h at 1h): here even the smoothest field (align 0, eigen
/// path) has about 2.8x and 4.6x the cones of the clean smooth field, because the noise itself
/// concentrates Gaussian curvature, and the DESIGN 1.5x criterion is not met (CHANGES.md, WP-08
/// review). What S4 owns is checked instead: alignment adds at most 1.5x over the smooth field
/// of the same noisy mesh.
#[test]
fn heavy_noise_alignment_adds_few_cones() {
    for (amp, wl, seed) in [(0.1, 0.7, 4u64), (0.2, 1.0, 6)] {
        let noisy = lumpy_blob(Some((amp, wl, seed)));
        let (n1, r1, mn) = solve_with(&noisy, 1.0);
        let (n0, _, _) = solve_with(&noisy, 0.0);
        println!(
            "heavy noise amp {amp} wl {wl}: mean conf {mn:.3}, cones {n1} (initial {}), smooth field {n0}",
            r1.cones_initial
        );
        assert!(!r1.eigen_path);
        assert!(n1 as f64 <= 1.5 * n0 as f64, "noise {amp} {wl}: {n1} cones, smooth field {n0}");
    }
}

/// testdata noisy_blob (suite target 5000, M1 gate 14 poles), locally repaired: the aligned
/// field has at most 1.5x the cones of the smooth field of the same mesh.
#[test]
fn noisy_blob_testdata_alignment() {
    let s = testdata_surface("noisy_blob");
    let (n1, r1, mn) = solve_with(&s, 1.0);
    let (n0, r0, _) = solve_with(&s, 0.0);
    println!("noisy_blob: mean conf {mn:.3}, cones {n1} (initial {}), smooth field {n0}", r1.cones_initial);
    assert!(!r1.eigen_path && r0.eigen_path);
    assert!(mn > 0.05);
    assert!(n1 as f64 <= 1.5 * n0 as f64, "noisy_blob: {n1} cones, smooth field {n0}");
}

/// The near-isotropic blob (mean conf below 0.01) takes the eigen path, with or without noise.
#[test]
fn near_isotropic_blob_eigen_path() {
    let (nc, rc, _) = solve_with(&blob(5000.0, None), 1.0);
    let (nn, rn, _) = solve_with(&blob(5000.0, Some((0.03, 0.5, 1))), 1.0);
    println!("isotropic blob: cones clean {nc}, noisy {nn}");
    assert!(rc.eigen_path && rn.eigen_path);
    assert!(nn as f64 <= 1.5 * nc as f64);
}

#[test]
fn deterministic_across_runs_and_threads() {
    let s = blob(800.0, Some((0.03, 0.5, 4)));
    let c = estimate_curv(&s, 1.0);
    let sz = size(&s, 1.0);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let st = ctl_state();
            solve_field(&px(&s), &c, &sz, &FieldParams::default(), &st.unbounded()).unwrap()
        })
    };
    let (a, ra) = run(1);
    let (b, rb) = run(4);
    let (c2, _) = run(1);
    assert_eq!(ra, rb);
    for (x, y) in [(&a, &b), (&a, &c2)] {
        assert_eq!(x.p, y.p);
        assert_eq!(x.index4, y.index4);
        assert!(x.theta.iter().zip(&y.theta).all(|(p, q)| p.to_bits() == q.to_bits()));
        assert!(x.delta.iter().zip(&y.delta).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

#[test]
fn time_budget_100k_faces() {
    if cfg!(debug_assertions) {
        return;
    }
    // About 100k faces: a torus at h = 1 (R = 16, r = 5, edge about h/3).
    let s = Surface::new(aq_testkit::torus(16.0, 5.0, 330, 152, 0.2, 9).unwrap()).unwrap();
    let c = estimate_curv(&s, 1.0);
    let sz = size(&s, 1.0);
    let mut best = f64::INFINITY;
    for _ in 0..2 {
        let st = ctl_state();
        let t0 = Instant::now();
        let (f, _) = solve_field(&px(&s), &c, &sz, &FieldParams::default(), &st.unbounded()).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        assert_eq!(f.index4.iter().map(|&i| i as i64).sum::<i64>(), 0);
    }
    // Eigen path at the same size (an octahedral sphere of 100352 faces without curvature
    // confidence).
    let sp = Surface::new(aq_testkit::octasphere(112, 19.5).unwrap()).unwrap();
    let (cz, szp) = (zero_curv(&sp), size(&sp, 1.0));
    let mut te = f64::INFINITY;
    for _ in 0..2 {
        let st = ctl_state();
        let t0 = Instant::now();
        let (_, rep) = solve_field(&px(&sp), &cz, &szp, &FieldParams::default(), &st.unbounded()).unwrap();
        te = te.min(t0.elapsed().as_secs_f64());
        assert!(rep.eigen_path);
    }
    // Linear path with curvature alignment on (coherence filter and normalized iterations): the
    // lumpy displacement on an octahedral sphere of 100352 faces.
    let r = 17.5;
    let lb = lumpy(aq_testkit::octasphere(112, r).unwrap(), r, None);
    let (cl, szl) = (estimate_curv(&lb, 1.0), size(&lb, 1.0));
    let mut tl = f64::INFINITY;
    for _ in 0..2 {
        let st = ctl_state();
        let t0 = Instant::now();
        let (_, rep) = solve_field(&px(&lb), &cl, &szl, &FieldParams::default(), &st.unbounded()).unwrap();
        tl = tl.min(t0.elapsed().as_secs_f64());
        assert!(!rep.eigen_path);
    }
    println!(
        "torus {} faces: {best:.3} s; eigen path sphere {} faces: {te:.3} s; aligned lumpy blob {} faces (mean conf {:.3}): {tl:.3} s",
        s.n_faces(),
        sp.n_faces(),
        lb.n_faces(),
        mean_conf(&lb, &cl)
    );
    assert!(tl < 3.0, "S4 aligned linear path took {tl} s at {} faces", lb.n_faces());
    assert!(best < 3.0, "S4 took {best} s at {} faces", s.n_faces());
    assert!(te < 3.0, "S4 eigen path took {te} s at {} faces", sp.n_faces());
}

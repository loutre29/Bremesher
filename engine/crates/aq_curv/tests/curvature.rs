//! DESIGN 6.5 curvature tests (WP-07). h = 1 (quad units), proxy-like edges of h/3.

mod common;

use common::*;

use aq_core::ErrKind;
use aq_curv::{CurvParams, curvature, curvature_detail, dihedral};

const EDGE: f64 = 1.0 / 3.0;

fn curv_of(s: &aq_testkit::Surface) -> aq_core::CurvField {
    let st = ctl_state();
    let ctl = st.unbounded();
    let h = vec![1.0; s.n_verts()];
    curvature(&view(s), &h, &CurvParams::default(), &ctl).unwrap()
}

#[test]
fn sphere_radius_2() {
    for (name, m) in [
        ("icosphere(3)", aq_testkit::icosphere(3, 2.0).unwrap()),
        ("uv_sphere(20, 38)", aq_testkit::uv_sphere(20, 38, 2.0).unwrap()),
    ] {
        let s = surf(m);
        let c = curv_of(&s);
        let mut worst = 0.0f64;
        let mut worst_conf = 0.0f64;
        for f in 0..s.n_faces() {
            worst = worst.max((c.k1[f] - 0.5).abs() / 0.5).max((c.k2[f] - 0.5).abs() / 0.5);
            worst_conf = worst_conf.max(c.conf[f]);
        }
        println!("{name}: {} faces, max rel error {worst:.4}, max conf {worst_conf:.4}", s.n_faces());
        assert!(worst < 0.03, "{name}: k1/k2 error {worst}");
        assert!(worst_conf < 0.05, "{name}: conf {worst_conf}");
        for &k in &c.kappa_hat {
            assert!((k - 0.5f64.hypot(0.5)).abs() < 0.03 * 0.5f64.hypot(0.5), "{name}: kappa_hat {k}");
        }
    }
}

#[test]
fn cylinders_r_2_5_10() {
    for (r, conf_min) in [(2.0, 0.95), (5.0, 0.85), (10.0, 0.25)] {
        let len = 2.0 * r + 8.0;
        let s = surf(cylinder(r, len, EDGE, 0.2, 7));
        let c = curv_of(&s);
        let faces: Vec<usize> = (0..s.n_faces())
            .filter(|&f| {
                let z = centroid(&s.mesh, f)[2];
                z > r && z < len - r
            })
            .collect();
        assert!(!faces.is_empty());
        let mut k1_err = 0.0f64;
        let mut k2_max = 0.0f64;
        let mut conf_min_seen = 1.0f64;
        let mut dir_ok = 0usize;
        for &f in &faces {
            k1_err = k1_err.max((c.k1[f].abs() - 1.0 / r).abs() * r);
            k2_max = k2_max.max(c.k2[f].abs() * r);
            conf_min_seen = conf_min_seen.min(c.conf[f]);
            let want = angle_in_face(&s, f, circ_dir(centroid(&s.mesh, f)));
            if line_angle_diff(c.phi[f], want) < 3f64.to_radians() {
                dir_ok += 1;
            }
            assert!(c.k1[f] > 0.0, "convex cylinder must have k1 > 0");
        }
        let dir_frac = dir_ok as f64 / faces.len() as f64;
        println!(
            "R = {r}: {} faces ({} measured), |k1| rel err {k1_err:.4}, |k2| R {k2_max:.4}, min conf {conf_min_seen:.4}, dir ok {dir_frac:.4}",
            s.n_faces(),
            faces.len()
        );
        assert!(k1_err < 0.03, "R = {r}: |k1| error {k1_err}");
        assert!(k2_max < 0.03, "R = {r}: |k2| R = {k2_max}");
        assert!(dir_frac >= 0.95, "R = {r}: direction ok on {dir_frac}");
        assert!(conf_min_seen >= conf_min, "R = {r}: conf {conf_min_seen} < {conf_min}");
    }
}

#[test]
fn noisy_cylinder_direction() {
    // DESIGN 6.5 does not fix R; the test runs at R = 2 (the noise amplitude scales with R and
    // the curvature signal with 1/R). R = 5 and 10 are measured for information (CHANGES.md).
    for r in [2.0, 5.0, 10.0] {
        let len = 2.0 * r + 8.0;
        let mut m = cylinder(r, len, EDGE, 0.2, 11);
        aq_testkit::add_noise(&mut m, 0.02 * r, 0.2, 99).unwrap();
        let s = surf(m);
        let c = curv_of(&s);
        let measured = |f: usize| {
            let z = centroid(&s.mesh, f)[2];
            z > r && z < len - r
        };
        let ok = frac((0..s.n_faces()).filter(|&f| measured(f)).map(|f| {
            let want = angle_in_face(&s, f, circ_dir(centroid(&s.mesh, f)));
            line_angle_diff(c.phi[f], want) < 5f64.to_radians()
        }));
        let ok_all = frac((0..s.n_faces()).map(|f| {
            let want = angle_in_face(&s, f, circ_dir(centroid(&s.mesh, f)));
            line_angle_diff(c.phi[f], want) < 5f64.to_radians()
        }));
        println!("noisy cylinder R = {r}: direction < 5 deg on {ok:.4} (middle), {ok_all:.4} (all faces)");
        if r == 2.0 {
            assert!(ok >= 0.90, "R = {r}: {ok}");
        }
    }
}

#[test]
fn saddle_conf_near_origin() {
    let m = aq_testkit::heightfield(36, 6.0, |x, y| (x * x - y * y) / 4.0).unwrap();
    let s = surf(m);
    let c = curv_of(&s);
    let mut n = 0;
    for f in 0..s.n_faces() {
        let p = centroid(&s.mesh, f);
        if p[0].hypot(p[1]) < 1.0 {
            n += 1;
            assert!(c.conf[f] > 0.5, "conf {} at {p:?}", c.conf[f]);
            assert!(c.k1[f] * c.k2[f] < 0.0, "saddle must have k1 k2 < 0");
            // Ambient diffusion over about h shrinks a saddle's curvature (the tensor rotates
            // across the kernel); only the sign structure and conf are DESIGN requirements.
            assert!(c.k1[f].abs() > 0.2 && c.k1[f].abs() < 0.55, "|k1| = {}", c.k1[f]);
        }
    }
    assert!(n > 20);
}

#[test]
fn dihedral_sign_convex_positive() {
    let cube = aq_testkit::cube_sharp(3, 2.0, 0.0, 0).unwrap();
    let s = aq_testkit::Surface::from_synth(&cube).unwrap();
    let px = view(&s);
    let mut n_sharp = 0;
    for h in 0..3 * s.n_faces() as u32 {
        if !s.topo.is_canonical(h) || s.topo.twin[h as usize] == aq_core::NONE {
            continue;
        }
        let b = dihedral(&px, h);
        assert!(b > -1e-12, "convex cube edge with negative dihedral {b}");
        if (b - std::f64::consts::FRAC_PI_2).abs() < 1e-9 {
            n_sharp += 1;
        }
    }
    assert_eq!(n_sharp, 12 * 3);
}

#[test]
fn errors_and_cancel() {
    let s = surf(aq_testkit::icosphere(2, 2.0).unwrap());
    let st = ctl_state();
    let ctl = st.unbounded();
    let p = CurvParams::default();
    let e = curvature(&view(&s), &[1.0; 3], &p, &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(e.stage, aq_core::Stage::Curv);
    let mut h = vec![1.0; s.n_verts()];
    h[5] = f64::NAN;
    assert_eq!(curvature(&view(&s), &h, &p, &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    let h = vec![1.0; s.n_verts()];
    let bad = CurvParams { scale: 0.0, ..p };
    assert_eq!(curvature(&view(&s), &h, &bad, &ctl).unwrap_err().kind, ErrKind::InvalidInput);
    st.cancel();
    assert_eq!(curvature(&view(&s), &h, &p, &ctl).unwrap_err().kind, ErrKind::Cancelled);
    let st2 = ctl_state();
    let late = st2.ctl(std::time::Instant::now() - std::time::Duration::from_millis(1));
    assert_eq!(curvature(&view(&s), &h, &p, &late).unwrap_err().kind, ErrKind::Timeout);
}

#[test]
fn detail_is_consistent_and_threads_deterministic() {
    let mut m = aq_testkit::torus(4.0, 1.5, 120, 48, 0.2, 3).unwrap();
    aq_testkit::add_noise(&mut m, 0.03, 0.7, 5).unwrap();
    let s = surf(m);
    let h = vec![1.0; s.n_verts()];
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let st = ctl_state();
            curvature_detail(&view(&s), &h, &CurvParams::default(), &st.unbounded()).unwrap()
        })
    };
    let (a, da) = run(1);
    let (b, db) = run(4);
    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.k1), bits(&b.k1));
    assert_eq!(bits(&a.k2), bits(&b.k2));
    assert_eq!(bits(&a.phi), bits(&b.phi));
    assert_eq!(bits(&a.conf), bits(&b.conf));
    assert_eq!(bits(&a.kappa_hat), bits(&b.kappa_hat));
    assert_eq!(da, db);
    assert_eq!(da.vert_tensor.len(), s.n_verts());
    a.validate(s.n_faces(), s.n_verts()).unwrap();
}

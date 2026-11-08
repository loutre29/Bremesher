//! DESIGN 6.9 tests of aq_param on testkit inputs.

mod common;

use std::f64::consts::TAU;

use aq_core::{CtlState, ErrKind, NONE, Topology};
use aq_param::{LoopSet, ParamParams, seamless};
use aq_testkit::*;
use common::*;

/// Plane with a constant field: the affine map, exactly (the expected uv is dyadic, so the snap
/// lands on it), det > 0 everywhere.
#[test]
fn plane_constant_field_is_affine() {
    for &(alpha, s) in &[(0.3, 1.0), (-1.1, 2.0), (0.0, 1.0)] {
        let c = plane_param(alpha, s).unwrap();
        let p = ParamParams {
            n_target: 64.0,
            ..ParamParams::default()
        };
        let (sp, diag) = run_case(&c, &p);
        check_exact(&c.surface, &c.cut, &sp, &c.loops);
        let mut err: f64 = 0.0;
        let mut err_f: f64 = 0.0;
        for f in 0..sp.uv.len() {
            for k in 0..3 {
                for q in 0..2 {
                    err = err.max((sp.uv[f][k][q] - c.param.uv[f][k][q]).abs());
                    err_f = err_f.max((diag.uv_float[f][k][q] - c.param.uv[f][k][q]).abs());
                }
            }
        }
        assert!(
            err < 1e-9 && err_f < 1e-9,
            "alpha {alpha}: affine error {err:e} (float {err_f:e})"
        );
        let dets = aq_param::face_dets(&px(&c.surface), &sp.uv);
        assert!(dets.iter().all(|&d| d > 0.0));
        assert_eq!(n_folded(&sp), 0);
        assert!(sp.report.e0 < 1e-12, "e0 {}", sp.report.e0);
        assert!((sp.report.n_pred - 64.0).abs() < 1e-9);
    }
}

/// Exactness after the snap on every analytic param of the testkit (own field, cut and loops).
#[test]
fn exact_on_analytic_params() {
    let mut cases = vec![
        ("plane", plane_param(0.7, 1.5).unwrap()),
        (
            "plane aligned",
            plane_param_with(0.2, 1.0, 6.0, ParamOpts::aligned(2)).unwrap(),
        ),
        ("cylinder", cylinder_param(12).unwrap()),
        (
            "cylinder aligned",
            cylinder_param_with(8, 4.0, ParamOpts::aligned(2)).unwrap(),
        ),
        ("torus", torus_param(16, 6).unwrap()),
        (
            "torus aligned",
            torus_param_with(12, 6, 3.0, 0.9, ParamOpts::aligned(2)).unwrap(),
        ),
        ("cube sphere", cube_sphere_param(4).unwrap()),
        (
            "perturbed cube sphere",
            perturbed_cube_sphere_param(4, 0.3, 9).unwrap(),
        ),
        (
            "jittered cube sphere",
            cube_sphere_param_with(
                3,
                CubeSphereOpts {
                    res: 3,
                    jitter: 0.2,
                    seed: 4,
                    radius: None,
                },
            )
            .unwrap(),
        ),
    ];
    for (name, c) in cases.iter_mut() {
        let p = ParamParams::default();
        let (sp, diag) = run_case(c, &p);
        check_exact(&c.surface, &c.cut, &sp, &c.loops);
        assert_eq!(n_folded(&sp), 0, "{name}: folds {:?}", diag.fold_history);
        assert!(
            (sp.report.n_pred - diag.n_c).abs() <= 1e-9 * diag.n_c,
            "{name}: n_pred"
        );
        assert!(
            (diag.n_pred_snapped - diag.n_c).abs() <= 1e-6 * diag.n_c,
            "{name}: snapped n_pred"
        );
        assert_eq!(sp.report.inconsistent, 0, "{name}");
        assert_eq!(sp.report.penalty_rows, 0, "{name}");
        assert_eq!(
            sp.loops_exact.len(),
            c.loops.len(),
            "{name}: loops {:?}",
            diag.loop_set
        );
    }
}

/// Exactness with cut graphs built by the testkit's spanning-tree cut (not the seams).
#[test]
fn exact_with_spanning_tree_cuts() {
    let cases = vec![
        ("torus", torus_param(16, 6).unwrap()),
        (
            "cube sphere",
            cube_sphere_param_with(
                4,
                CubeSphereOpts {
                    res: 2,
                    jitter: 0.2,
                    seed: 2,
                    radius: None,
                },
            )
            .unwrap(),
        ),
        (
            "perturbed cube sphere",
            perturbed_cube_sphere_param(4, 0.4, 3).unwrap(),
        ),
    ];
    for (name, c) in &cases {
        let cut = analytic_cut(&c.surface, &c.field).unwrap();
        let (sp, _) = run(
            &c.surface,
            &c.field,
            &cut,
            &c.size,
            &c.loops,
            &ParamParams::default(),
        );
        check_exact(&c.surface, &cut, &sp, &c.loops);
        assert_eq!(n_folded(&sp), 0, "{name}");
    }
}

/// Open cylinder: boundary loops iso exact, t_v = 0 exactly, t_u = 2 pi R / h within 1e-3.
#[test]
fn open_cylinder_translation() {
    let c = cylinder_param_with(
        16,
        8.0,
        ParamOpts {
            res: 4,
            jitter: 0.25,
            seed: 3,
        },
    )
    .unwrap();
    let (sp, _) = run_case(&c, &ParamParams::default());
    check_exact(&c.surface, &c.cut, &sp, &c.loops);
    let s = &c.surface;
    // Rims: every boundary halfedge carries a family and is iso exact (checked above).
    for lp in &s.topo.boundary_loops {
        for &h in lp {
            assert_eq!(sp.he_family[h as usize], 0, "rim he {h} is not U");
        }
    }
    assert!(!c.cut.arcs.is_empty());
    let radius = 16.0 / TAU;
    for a in &c.cut.arcs {
        let tr = sp.trans[a.hes[0] as usize];
        assert_eq!(tr.r, 0);
        assert!(tr.t[1] == 0.0, "t_v = {}", tr.t[1]);
        let rel = (tr.t[0].abs() - TAU * radius).abs() / (TAU * radius);
        assert!(rel < 1e-3, "t_u = {} vs {}", tr.t[0], TAU * radius);
    }
}

/// Torus with loops: transversal translations exactly 0 for both families, |grad u| h within
/// 5% of 1 on 95% of the faces, no fold. A slender torus (R/r = 25): an isotropic seamless map
/// of a torus has |grad u| proportional to 1/(R + r cos psi) (see CHANGES.md).
#[test]
fn torus_with_loops() {
    let (n, m) = (100u32, 4u32);
    let c = torus_param_with(
        n,
        m,
        n as f64 / TAU,
        m as f64 / TAU,
        ParamOpts {
            res: 3,
            jitter: 0.25,
            seed: 11,
        },
    )
    .unwrap();
    let (sp, diag) = run_case(&c, &ParamParams::default());
    check_exact(&c.surface, &c.cut, &sp, &c.loops);
    assert_eq!(diag.loop_set, LoopSet::All);
    assert_eq!(sp.loops_exact.len(), c.loops.len());
    for l in &c.loops {
        assert!(loop_drift(&c.cut, &sp.trans, l) == 0.0);
    }
    // Each seam translation is purely along its own family.
    for a in &c.cut.arcs {
        let tr = sp.trans[a.hes[0] as usize];
        assert!(
            tr.t[0] == 0.0 || tr.t[1] == 0.0,
            "arc translation {:?}",
            tr.t
        );
    }
    let g = grad_norms(&c.surface, &sp.uv);
    let good = g.iter().filter(|(gu, _)| (gu - 1.0).abs() < 0.05).count();
    assert!(
        good as f64 >= 0.95 * g.len() as f64,
        "{good} of {}",
        g.len()
    );
    assert_eq!(n_folded(&sp), 0);
}

/// Twisted torus field (0.3 rad): with loops the transversal translation is exactly 0; without
/// loops (control) it is above 0.05. The default energy guard drops these loops (they cost far
/// more than 1.3 E0), so the loop test disables it.
#[test]
fn twisted_torus_loops() {
    let c = torus_param(24, 6).unwrap();
    let s = &c.surface;
    let f = analytic_field(s, &FieldKind::TorusTwisted { twist: 0.3 }).unwrap();
    let cut = analytic_cut(s, &f).unwrap();
    let size = uniform_size(s, 1.0).unwrap();
    let with = ParamParams {
        loop_guard: f64::INFINITY,
        ..ParamParams::default()
    };
    let (sp, diag) = run(s, &f, &cut, &size, &c.loops, &with);
    check_exact(s, &cut, &sp, &c.loops);
    assert_eq!(diag.loop_set, LoopSet::All);
    assert_eq!(sp.loops_exact.len(), c.loops.len());
    for l in &c.loops {
        assert!(loop_drift(&cut, &sp.trans, l) == 0.0);
    }
    let without = ParamParams {
        loops: false,
        ..ParamParams::default()
    };
    let (sp0, _) = run(s, &f, &cut, &size, &c.loops, &without);
    check_exact(s, &cut, &sp0, &c.loops);
    assert!(sp0.loops_exact.is_empty());
    for (i, l) in c.loops.iter().enumerate() {
        let d = loop_drift(&cut, &sp0.trans, l);
        assert!(d.abs() > 0.05, "loop {i} (branch {}): drift {d}", l.branch);
    }
    // Default guard: E1 > 1.3 E0, so the loops are dropped (rings, then none).
    let (sp1, d1) = run(s, &f, &cut, &size, &c.loops, &ParamParams::default());
    check_exact(s, &cut, &sp1, &c.loops);
    assert!(d1.e1.is_some_and(|e1| e1 > 1.3 * sp1.report.e0));
    assert_ne!(d1.loop_set, LoopSet::All);
    assert!(sp1.report.loops_dropped > 0);
}

/// Cube-sphere with the 8-cone field: at most 0.1% folds before repair, 0 after, N_pred = N
/// within 1e-9 after scaling.
#[test]
fn cube_sphere_eight_cones() {
    for (res, jitter, own) in [(4u32, 0.2, false), (3, 0.0, false), (4, 0.2, true)] {
        let c = cube_sphere_param_with(
            4,
            CubeSphereOpts {
                res,
                jitter,
                seed: 5,
                radius: None,
            },
        )
        .unwrap();
        let s = &c.surface;
        let (f, cut) = if own {
            (c.field.clone(), c.cut.clone())
        } else {
            let f = analytic_field(s, &FieldKind::CubeAligned).unwrap();
            let cut = analytic_cut(s, &f).unwrap();
            (f, cut)
        };
        assert_eq!(f.cones().len(), 8);
        let n = 96.0;
        let p = ParamParams {
            n_target: n,
            ..ParamParams::default()
        };
        let (sp, diag) = run(s, &f, &cut, &c.size, &[], &p);
        check_exact(s, &cut, &sp, &[]);
        let nf = s.n_faces() as f64;
        assert!(
            sp.report.flips_initial as f64 <= 1e-3 * nf,
            "folds before repair {}",
            sp.report.flips_initial
        );
        assert_eq!(sp.report.folds_after_repair, 0, "{:?}", diag.repair);
        assert!(
            (sp.report.n_pred - n).abs() <= 1e-9 * n,
            "n_pred {}",
            sp.report.n_pred
        );
        assert_eq!(sp.cones.len(), 8);
    }
}

/// cube_sharp: chains iso exact, corners on both iso lines, no fold.
#[test]
fn cube_sharp_chains() {
    let sm = cube_sharp(6, 4.0, 0.2, 7).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    assert_eq!(s.feats.chains.len(), 12);
    let f = analytic_field(&s, &FieldKind::CubeAligned).unwrap();
    let cut = analytic_cut(&s, &f).unwrap();
    let size = uniform_size(&s, 1.0).unwrap();
    let (sp, diag) = run(&s, &f, &cut, &size, &[], &ParamParams::default());
    check_exact(&s, &cut, &sp, &[]);
    assert_eq!(n_folded(&sp), 0, "{:?}", diag.fold_history);
    assert_eq!(sp.report.penalty_rows, 0);
    // Every chain halfedge has a family.
    for ch in &s.feats.chains {
        for &h in &ch.hes {
            assert_ne!(sp.he_family[h as usize], aq_core::param::FAMILY_NONE);
        }
    }
    // Corners: seen from any face of the fan, the chain edges at the corner constrain both
    // coordinates (U and V families meet).
    let (m, t) = (&s.mesh, &s.topo);
    let mut n_corners = 0;
    for v in 0..s.n_verts() as u32 {
        if !s.feats.vert_corner[v as usize] {
            continue;
        }
        n_corners += 1;
        let fan: Vec<u32> = t.fan_out(v).collect();
        // Composite rotation from the chart of fan[0] to each face of the fan.
        let mut rot = vec![0u8; fan.len()];
        for i in 1..fan.len() {
            let h = Topology::prev(fan[i - 1]);
            rot[i] = (rot[i - 1] + sp.trans[h as usize].r) % 4;
        }
        let mut fams = [false; 2];
        for (i, &o) in fan.iter().enumerate() {
            let f = o / 3;
            for k in 0..3 {
                let h = 3 * f + k;
                let fam = sp.he_family[h as usize];
                if fam == aq_core::param::FAMILY_NONE {
                    continue;
                }
                if t.tail(m, h) == v || t.head(m, h) == v {
                    fams[((fam + rot[i]) % 2) as usize] = true;
                }
            }
        }
        assert!(
            fams[0] && fams[1],
            "corner {v} is constrained in one coordinate only"
        );
    }
    assert_eq!(n_corners, 8);
}

/// Hemisphere with a Symmetry boundary chain: iso exact on the equator.
#[test]
fn hemisphere_symmetry_chain() {
    let sm = hemisphere(6, 3.0).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    let f = analytic_field(&s, &FieldKind::CubeAligned).unwrap();
    let cut = analytic_cut(&s, &f).unwrap();
    let size = uniform_size(&s, 1.0).unwrap();
    let (sp, _) = run(&s, &f, &cut, &size, &[], &ParamParams::default());
    check_exact(&s, &cut, &sp, &[]);
    for lp in &s.topo.boundary_loops {
        for &h in lp {
            assert_ne!(sp.he_family[h as usize], aq_core::param::FAMILY_NONE);
        }
    }
}

/// Same inputs, same bits (also on another thread).
#[test]
fn deterministic() {
    let c = perturbed_cube_sphere_param(4, 0.3, 5).unwrap();
    let cut = analytic_cut(&c.surface, &c.field).unwrap();
    let a = run(
        &c.surface,
        &c.field,
        &cut,
        &c.size,
        &[],
        &ParamParams::default(),
    )
    .0;
    let b = run(
        &c.surface,
        &c.field,
        &cut,
        &c.size,
        &[],
        &ParamParams::default(),
    )
    .0;
    assert_eq!(a, b);
    let c2 = c.clone();
    let cut2 = cut.clone();
    let d = std::thread::spawn(move || {
        run(
            &c2.surface,
            &c2.field,
            &cut2,
            &c2.size,
            &[],
            &ParamParams::default(),
        )
        .0
    })
    .join()
    .unwrap();
    assert_eq!(a, d);
}

/// Input errors and cancellation.
#[test]
fn errors() {
    let c = plane_param(0.1, 1.0).unwrap();
    let st = CtlState::new();
    let ctl = st.unbounded();
    let p = ParamParams::default();
    let mut bad = c.size.clone();
    bad.h_face.pop();
    let e = seamless(&px(&c.surface), &c.field, &c.cut, &bad, &[], &p, &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(e.stage, aq_core::Stage::Param);
    let mut badcut = c.cut.clone();
    badcut.is_cut[0] = !badcut.is_cut[0];
    assert!(seamless(&px(&c.surface), &c.field, &badcut, &c.size, &[], &p, &ctl).is_err());
    st.cancel();
    let e = seamless(&px(&c.surface), &c.field, &c.cut, &c.size, &[], &p, &ctl).unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
    let _ = NONE;
}

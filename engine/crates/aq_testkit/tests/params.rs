//! Analytic params: exact seamless (SeamlessParam::validate), consistent field and cut, no
//! folds, expected cones, exact loops, determinism.

use std::f64::consts::TAU;

use aq_core::cross::{det2, jacobian};
use aq_core::tol::is_dyadic;
use aq_core::{NONE, Transition};
use aq_testkit::*;

fn common_checks(c: &ParamCase) {
    c.validate().unwrap();
    assert_eq!(c.n_folded(), 0, "folded faces");
    let s = &c.surface;
    for f in 0..s.n_faces() as u32 {
        assert!(det2(&jacobian(&s.frames, &s.mesh, f, &c.param.uv[f as usize])) > 0.0);
    }
    // Cut flags = non-identity transitions; transition rotation = combed matching = p.
    for h in 0..c.param.trans.len() {
        assert_eq!(c.param.is_cut[h], !c.param.trans[h].is_identity());
        assert_eq!(c.cut.is_cut[h], c.param.is_cut[h]);
        if s.topo.twin[h] != NONE {
            assert_eq!(c.cut.r[h], c.param.trans[h].r);
            assert_eq!(c.field.p[h], c.param.trans[h].r);
        }
    }
    assert!(c.cut.comb_offset.iter().all(|&o| o == 0));
    assert_eq!(c.param.cones, c.field.cones());
    // Rep wedge values reproduce the corners of the lowest face.
    for v in 0..s.n_verts() as u32 {
        let h = s.topo.fan_out(v).min_by_key(|&h| h / 3).unwrap();
        assert_eq!(c.rep_uv[v as usize], c.param.uv[(h / 3) as usize][(h % 3) as usize]);
    }
    // N_pred is the uv area.
    let area: f64 = c.param.uv.iter().map(|q| 0.5 * ((q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]))).sum();
    assert!((c.param.report.n_pred - area).abs() < 1e-9 * area);
    // Exact loops close bitwise: composing the transitions crossed gives the identity rotation
    // and zero transversal translation.
    for (i, l) in c.loops.iter().enumerate() {
        assert!(c.param.loops_exact.contains(&(i as u32)));
        let mut tot = Transition::IDENTITY;
        for &h in &l.hes {
            tot = tot.compose(&c.param.trans[h as usize]);
        }
        assert_eq!(tot.r, 0);
        let transversal = if l.branch % 2 == 0 { tot.t[1] } else { tot.t[0] };
        assert_eq!(transversal, 0.0);
    }
}

#[test]
fn plane_param_is_affine() {
    for (alpha, s) in [(0.0, 1.0), (0.37, 2.0), (-1.1, 0.5)] {
        let c = plane_param(alpha, s).unwrap();
        common_checks(&c);
        assert!(c.param.is_cut.iter().all(|&x| !x));
        // The only "cones" are the 4 boundary corners (index4 +1).
        assert_eq!(c.param.cones.len(), 4);
        assert!(c.param.cones.iter().all(|&(v, i)| i == 1 && c.surface.feats.vert_corner[v as usize]));
        let (sn, cs) = alpha.sin_cos();
        for f in 0..c.surface.n_faces() {
            for k in 0..3 {
                let p = c.surface.mesh.pos[c.surface.mesh.tris[f][k] as usize];
                let u = s * (cs * p[0] + sn * p[1]);
                let v = s * (-sn * p[0] + cs * p[1]);
                let q = c.param.uv[f][k];
                assert!((q[0] - u).abs() < 1e-9 && (q[1] - v).abs() < 1e-9);
            }
        }
        // 4 boundary segments, iso exact, U on the bottom/top and V on the sides.
        let ch = &c.surface.feats.chains[0];
        assert_eq!(ch.corners.len(), 4);
        for &h in &ch.hes {
            let (a, b) = c.param.he_uv(h);
            let fam = c.param.he_family[h as usize];
            if a[1] == b[1] && (a[1] == 0.0 || a[1] == 8.0) {
                assert_eq!(fam, 0);
            } else {
                assert_eq!(fam, 1);
                assert!(a[0] == b[0] && (a[0] == 0.0 || a[0] == 8.0));
            }
        }
        // Grad u has length s on every face (affine map).
        for f in 0..c.surface.n_faces() as u32 {
            let j = jacobian(&c.surface.frames, &c.surface.mesh, f, &c.param.uv[f as usize]);
            assert!((j[0][0].hypot(j[0][1]) - s).abs() < 1e-9);
        }
    }
    let a = plane_param_with(0.2, 1.0, 5.5, ParamOpts::aligned(2)).unwrap();
    common_checks(&a);
}

#[test]
fn cylinder_param_seam_and_rings() {
    for opts in [ParamOpts::default(), ParamOpts::aligned(3), ParamOpts { res: 2, jitter: 0.25, seed: 9 }] {
        let c = cylinder_param_with(12, 6.0, opts).unwrap();
        common_checks(&c);
        assert!(c.param.cones.is_empty());
        assert_eq!(c.surface.chi(0), 0);
        assert_eq!(c.cut.arcs.len(), 1);
        let arc = &c.cut.arcs[0];
        assert!(c.surface.topo.is_bnd_vert[arc.nodes[0] as usize] && c.surface.topo.is_bnd_vert[arc.nodes[1] as usize]);
        for &h in &arc.hes {
            let t = c.param.trans[h as usize];
            assert_eq!(t.r, 0);
            assert!(t.t == [12.0, 0.0] || t.t == [-12.0, 0.0]);
        }
        // Rims: U family, v = 0 and v = 6 exactly.
        assert_eq!(c.surface.feats.chains.len(), 2);
        for ch in &c.surface.feats.chains {
            for &h in &ch.hes {
                assert_eq!(c.param.he_family[h as usize], 0);
                let (a, _) = c.param.he_uv(h);
                assert!(a[1] == 0.0 || a[1] == 6.0);
            }
        }
        assert_eq!(c.loops.len(), 6);
        for l in &c.loops {
            assert!(l.ring && l.branch == 0);
            assert!((l.length - 12.0).abs() < 0.1, "ring length {}", l.length);
        }
        assert_eq!(c, cylinder_param_with(12, 6.0, opts).unwrap());
    }
    assert!(cylinder_param_with(12, 6.0, ParamOpts { res: 2, jitter: 0.3, seed: 9 }).is_err());
    let d = cylinder_param(8).unwrap();
    common_checks(&d);
    assert!(d.param.uv.iter().flatten().flatten().all(|&x| is_dyadic(x)));
}

#[test]
fn torus_param_two_seams_and_loops() {
    for opts in [ParamOpts::default(), ParamOpts::aligned(2)] {
        let (n, m) = (16u32, 6u32);
        let r = m as f64 / TAU;
        let c = torus_param_with(n, m, 2.5 * r, r, opts).unwrap();
        common_checks(&c);
        assert!(c.param.cones.is_empty());
        assert_eq!(c.cut.arcs.len(), 2);
        assert_eq!(c.cut.node_verts.len(), 1);
        assert_eq!(c.loops.len(), (n + m) as usize);
        let rings = c.loops.iter().filter(|l| l.ring).count();
        assert_eq!(rings, n as usize, "meridians are rings");
        assert!(c.loops.iter().filter(|l| l.ring).all(|l| l.branch == 1));
        for h in 0..c.param.trans.len() {
            if c.param.is_cut[h] {
                let t = c.param.trans[h].t;
                assert!(t[0].abs() == n as f64 && t[1] == 0.0 || t[0] == 0.0 && t[1].abs() == m as f64);
            }
        }
    }
    common_checks(&torus_param(12, 4).unwrap());
}

#[test]
fn cube_sphere_param_8_cones() {
    for (n, opts) in [
        (1u32, CubeSphereOpts::default()),
        (3, CubeSphereOpts::default()),
        (5, CubeSphereOpts::default()),
        (4, CubeSphereOpts { res: 3, jitter: 0.25, seed: 2, radius: None }),
        (3, CubeSphereOpts { res: 2, jitter: 0.0, seed: 1, radius: Some(1.0) }),
    ] {
        let c = cube_sphere_param_with(n, opts).unwrap();
        common_checks(&c);
        assert_eq!(c.param.cones.len(), 8, "n = {n}");
        assert!(c.param.cones.iter().all(|&(_, i)| i == 1));
        assert_eq!(c.cut.arcs.len(), 7);
        let mut rots = [0usize; 4];
        for a in &c.cut.arcs {
            rots[a.r as usize] += 1;
            assert_eq!(a.hes.len(), (n * opts.res) as usize);
        }
        assert!(rots[0] < 7, "the net has rotated transitions");
        // Cones at integer uv in every chart.
        for &(v, _) in &c.param.cones {
            for h in c.surface.topo.fan_out(v) {
                let q = c.param.uv[(h / 3) as usize][(h % 3) as usize];
                assert!(q[0].fract() == 0.0 && q[1].fract() == 0.0);
            }
        }
        for h in 0..c.param.trans.len() {
            let t = c.param.trans[h].t;
            assert!(t[0].fract() == 0.0 && t[1].fract() == 0.0);
        }
        assert!((c.param.report.n_pred - 6.0 * (n * n) as f64).abs() < 1e-9);
    }
    assert_eq!(cube_sphere_param(3).unwrap(), cube_sphere_param(3).unwrap());
}

#[test]
fn perturbed_cube_sphere_param_20_seeds() {
    let base = cube_sphere_param(6).unwrap();
    for seed in 0..20u64 {
        let c = perturbed_cube_sphere_param(6, 0.4, seed).unwrap();
        common_checks(&c);
        assert_eq!(c.param.cones, base.param.cones);
        assert_eq!(c.param.is_cut, base.param.is_cut);
        assert!(c.param.cones.iter().all(|&(_, i)| i == 1));
        // Cones moved off the integers (in at least one chart coordinate).
        let off = c
            .param
            .cones
            .iter()
            .filter(|&&(v, _)| {
                let q = c.rep_uv[v as usize];
                q[0].fract() != 0.0 || q[1].fract() != 0.0
            })
            .count();
        assert!(off >= 6, "seed {seed}: only {off} cones off the integers");
        // Translations are no longer integers on most arcs.
        let frac = c.cut.arcs.iter().filter(|a| {
            let t = c.param.trans[a.hes[0] as usize].t;
            t[0].fract() != 0.0 || t[1].fract() != 0.0
        });
        assert!(frac.count() >= 6);
    }
    let a = perturbed_cube_sphere_param(4, 0.3, 7).unwrap();
    assert_eq!(a, perturbed_cube_sphere_param(4, 0.3, 7).unwrap());
    let j = perturbed_cube_sphere_param_with(5, 0.3, 3, CubeSphereOpts { res: 2, jitter: 0.2, seed: 4, radius: None }).unwrap();
    common_checks(&j);
    // amp = 0 reproduces the base param.
    let z = perturbed_cube_sphere_param(6, 0.0, 1).unwrap();
    assert_eq!(z.param.uv, base.param.uv);
}

#[test]
fn param_from_corners_rejects_non_seamless() {
    let c = plane_param(0.0, 1.0).unwrap();
    let mut uv = c.param.uv.clone();
    uv[10][1][0] += 1.0 / 1024.0;
    assert!(param_from_corners(c.surface.mesh.clone(), &[], uv).is_err());
}

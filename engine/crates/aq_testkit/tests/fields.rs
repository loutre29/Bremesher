//! Analytic fields (Poincare-Hopf on every generator), curvature, size and cut graphs.

use std::f64::consts::{FRAC_PI_2, PI};

use aq_core::cross::wrap_pi;
use aq_core::{CrossField, NONE, math};
use aq_testkit::*;

fn sum_index(cf: &CrossField) -> i64 {
    cf.index4.iter().map(|&i| i as i64).sum()
}

fn cones(cf: &CrossField) -> Vec<(u32, i8)> {
    cf.cones()
}

/// Checks the cut contracts of DESIGN 6.8 on a field: valid, every cone on the cut, disk.
fn check_cut(s: &Surface, cf: &CrossField) -> aq_core::CutGraph {
    let cg = analytic_cut(s, cf).unwrap();
    cg.validate(&s.mesh, &s.topo, Some(cf)).unwrap();
    for (v, i) in cf.cones() {
        let on = s.topo.fan_out(v).any(|h| cg.is_cut[h as usize]);
        assert!(on || s.topo.is_bnd_vert[v as usize], "cone {v} ({i}) not on the cut");
    }
    for c in 0..s.topo.n_comp {
        assert_eq!(aq_testkit::cut::cut_mesh_euler(s, &cg.is_cut, c), 1);
    }
    for h in 0..cg.r.len() {
        if !cg.is_cut[h] {
            assert_eq!(cg.r[h], 0);
        }
    }
    assert_eq!(cg, analytic_cut(s, cf).unwrap(), "determinism");
    cg
}

#[test]
fn constant_field_on_rotated_plane() {
    let s = Surface::new(plane_grid(12, 9, 0.5, 0.3, 0.25, 2).unwrap()).unwrap();
    let cf = analytic_field(&s, &FieldKind::Constant { angle: 0.3 }).unwrap();
    assert_eq!(sum_index(&cf), 4);
    let c = cones(&cf);
    assert_eq!(c.len(), 4);
    assert!(c.iter().all(|&(v, i)| i == 1 && s.feats.vert_corner[v as usize]));
    // Interior faces keep the analytic direction exactly; boundary faces are hard.
    for f in 0..s.n_faces() {
        let d = s.frames.dir_at(f as u32, cf.theta[f]);
        let a = d[1].atan2(d[0]);
        assert!(wrap_pi(4.0 * (a - 0.3)).abs() < 1e-9);
        assert_eq!(cf.hard[f], (0..3).any(|k| s.topo.twin[3 * f + k] == NONE));
    }
    let cg = check_cut(&s, &cf);
    assert!(cg.arcs.iter().all(|a| a.r == 0));
}

#[test]
fn cylinder_fields() {
    let s = Surface::new(cylinder_open(2.0, 5.0, 40, 12, 0.25, 3).unwrap()).unwrap();
    let cf = analytic_field(&s, &FieldKind::CylinderCircumferential).unwrap();
    assert_eq!(sum_index(&cf), 0);
    assert!(cones(&cf).is_empty());
    // The circles projected into the (faceted, jittered) face planes: within 5 degrees of the
    // 3D circle direction off the rims (the rim faces are snapped to their chord).
    let mut good = 0;
    let soft = (0..s.n_faces()).filter(|&f| !cf.hard[f]).count();
    for f in (0..s.n_faces()).filter(|&f| !cf.hard[f]) {
        let c = aq_core::SurfacePoint { face: f as u32, bary: [1.0 / 3.0; 3] }.position(&s.mesh);
        let circ = math::normalize([-c[1], c[0], 0.0]).unwrap();
        let d = s.frames.dir_at(f as u32, cf.theta[f]);
        let a = math::angle_between(d, circ);
        let a = a.min(PI - a).min((FRAC_PI_2 - a).abs());
        if a < 5f64.to_radians() {
            good += 1;
        }
    }
    assert!(good as f64 >= 0.99 * soft as f64, "{good} of {soft}");
    check_cut(&s, &cf);
    let tw = analytic_field(&s, &FieldKind::CylinderTwisted { pitch: 2.0 }).unwrap();
    assert_eq!(sum_index(&tw), 0);
    check_cut(&s, &tw);

    let s = Surface::new(cylinder_closed(1.0, 3.0, 32, 8, 0.2, 1).unwrap()).unwrap();
    let cf = analytic_field(&s, &FieldKind::CylinderCircumferential).unwrap();
    assert_eq!(sum_index(&cf), 8);
    check_cut(&s, &cf);
}

#[test]
fn torus_fields() {
    let s = Surface::new(torus(3.0, 1.0, 48, 16, 0.25, 4).unwrap()).unwrap();
    for kind in [FieldKind::TorusCircumferential, FieldKind::TorusTwisted { twist: 0.3 }] {
        let cf = analytic_field(&s, &kind).unwrap();
        assert_eq!(sum_index(&cf), 0);
        assert!(cones(&cf).is_empty());
        let cg = check_cut(&s, &cf);
        // Two loops through one node of cut degree 4.
        assert_eq!(cg.arcs.len(), 2);
        assert!(cg.arcs.iter().all(|a| a.nodes[0] == a.nodes[1]));
    }
}

#[test]
fn cube_aligned_fields_have_8_corner_cones() {
    let sm = cube_sharp(5, 2.0, 0.2, 3).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    let cf = analytic_field(&s, &FieldKind::CubeAligned).unwrap();
    let c = cones(&cf);
    assert_eq!(c.len(), 8);
    assert!(c.iter().all(|&(v, i)| i == 1 && s.feats.vert_corner[v as usize]));
    // Every chain face is hard and aligned.
    for f in 0..s.n_faces() {
        let chain = (0..3).any(|k| s.feats.he_chain[3 * f + k] != NONE);
        assert_eq!(cf.hard[f], chain);
    }
    check_cut(&s, &cf);
    let cs = cube_sphere_param(4).unwrap();
    let cf = analytic_field(&cs.surface, &FieldKind::CubeAligned).unwrap();
    assert_eq!(cones(&cf).len(), 8);
    assert!(cones(&cf).iter().all(|&(_, i)| i == 1));
}

#[test]
fn poincare_hopf_on_every_generator() {
    let mut cases: Vec<(&str, Surface, i64)> = vec![
        ("uv_sphere", Surface::new(uv_sphere(10, 16, 1.0).unwrap()).unwrap(), 2),
        ("icosphere", Surface::new(icosphere(3, 1.0).unwrap()).unwrap(), 2),
        ("octasphere", Surface::new(octasphere(6, 1.0).unwrap()).unwrap(), 2),
        ("double_torus", Surface::new(double_torus().unwrap()).unwrap(), -2),
        ("hemisphere", Surface::from_synth(&hemisphere(6, 1.0).unwrap()).unwrap(), 1),
        ("square_disk", Surface::new(square_disk(8, 1.0, 0.2, 1).unwrap()).unwrap(), 1),
    ];
    let mut noisy = icosphere(3, 1.0).unwrap();
    add_noise(&mut noisy, 0.02, 0.4, 3).unwrap();
    cases.push(("noisy icosphere", Surface::new(noisy).unwrap(), 2));
    for (name, s, chi) in &cases {
        assert_eq!(s.chi_total(), *chi, "{name}");
        for kind in [FieldKind::Constant { angle: 0.7 }, FieldKind::CubeAligned, FieldKind::TorusCircumferential] {
            let cf = analytic_field(s, &kind).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(sum_index(&cf), 4 * chi, "{name} {kind:?}");
            check_cut(s, &cf);
        }
    }
}

#[test]
fn planted_singularities() {
    let n = 24u32;
    let s = Surface::new(plane_grid(n, n, 1.0, 0.0, 0.2, 8).unwrap()).unwrap();
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let (a, b) = (id(9, 12), id(15, 12));
    let kind = FieldKind::planted_pair(FieldKind::Constant { angle: 0.0 }, a, b, PlantChart::Plane);
    let cf = analytic_field(&s, &kind).unwrap();
    assert_eq!(cf.index4[a as usize], 1);
    assert_eq!(cf.index4[b as usize], -1);
    let interior: Vec<_> = cones(&cf).into_iter().filter(|&(v, _)| !s.topo.is_bnd_vert[v as usize]).collect();
    assert_eq!(interior, vec![(a, 1), (b, -1)]);
    check_cut(&s, &cf);
    // +2 vertex.
    let kind = FieldKind::Planted {
        base: Box::new(FieldKind::Constant { angle: 0.0 }),
        sings: vec![(id(12, 12), 2)],
        chart: PlantChart::Plane,
    };
    let cf = analytic_field(&s, &kind).unwrap();
    assert_eq!(cf.index4[id(12, 12) as usize], 2);
    // On a sphere: a pair planted on the cube-sphere 8-cone field.
    let cs = cube_sphere_param_with(6, CubeSphereOpts { res: 2, ..Default::default() }).unwrap();
    let ss = &cs.surface;
    let near = |p: [f64; 3]| {
        (0..ss.n_verts() as u32)
            .min_by(|&x, &y| math::dist(ss.mesh.pos[x as usize], p).total_cmp(&math::dist(ss.mesh.pos[y as usize], p)))
            .unwrap()
    };
    let r = math::norm(ss.mesh.pos[0]);
    let pa = near(math::scale(math::normalize([1.0, 0.1, 0.05]).unwrap(), r));
    let pb = near(math::scale(math::normalize([1.0, -0.15, 0.05]).unwrap(), r));
    assert_ne!(pa, pb);
    let kind = FieldKind::planted_pair(FieldKind::CubeAligned, pa, pb, PlantChart::Sphere { center: [0.0; 3] });
    let cf = analytic_field(ss, &kind).unwrap();
    assert_eq!(cf.index4[pa as usize], 1);
    assert_eq!(cf.index4[pb as usize], -1);
    assert_eq!(cones(&cf).len(), 10);
    assert_eq!(sum_index(&cf), 8);
    check_cut(ss, &cf);
}

#[test]
fn analytic_curvature_and_size() {
    let s = Surface::new(cylinder_open(5.0, 10.0, 64, 16, 0.2, 1).unwrap()).unwrap();
    let c = analytic_curv(&s, &CurvKind::Cylinder { radius: 5.0 }).unwrap();
    assert!(c.k1.iter().all(|&k| (k - 0.2).abs() < 1e-12));
    assert!(c.k2.iter().all(|&k| k == 0.0));
    assert!(c.conf.iter().all(|&x| x > 0.25 && x <= 1.0));
    let cf = analytic_field(&s, &FieldKind::CylinderCircumferential).unwrap();
    // The k1 direction is circumferential: phi matches theta mod pi/2 on interior faces.
    for f in 0..s.n_faces() {
        if !cf.hard[f] {
            assert!(wrap_pi(4.0 * (c.phi[f] - cf.theta[f])).abs() < 1e-9);
        }
    }
    let s = Surface::new(icosphere(3, 2.0).unwrap()).unwrap();
    let c = analytic_curv(&s, &CurvKind::Sphere { center: [0.0; 3], radius: 2.0 }).unwrap();
    assert!(c.k1.iter().zip(&c.k2).all(|(a, b)| *a == 0.5 && *b == 0.5));
    assert!(c.conf.iter().all(|&x| x < 0.05));
    let s = Surface::new(torus(3.0, 1.0, 32, 12, 0.0, 0).unwrap()).unwrap();
    let c = analytic_curv(&s, &CurvKind::Torus { big_r: 3.0, r: 1.0 }).unwrap();
    assert!(c.k1.iter().all(|&k| (k - 1.0).abs() < 1e-12));
    let p = analytic_curv(&s, &CurvKind::Plane).unwrap();
    assert!(p.conf.iter().all(|&x| x == 0.0));
    let sz = uniform_size(&s, 0.5).unwrap();
    assert!((sz.n_pred - s.frames.area.iter().sum::<f64>() * 4.0).abs() < 1e-9);
}

//! Generators: topology (chi, boundary loops), geometry invariants, features, determinism.

use aq_core::{FeatureKind, NONE, Topology, TriMesh, math};
use aq_testkit::*;

fn min_angle_deg(m: &TriMesh) -> f64 {
    let s = Surface::new(m.clone()).unwrap();
    s.frames.corner_angle.iter().copied().fold(f64::INFINITY, f64::min).to_degrees()
}

fn surf(m: TriMesh) -> Surface {
    Surface::new(m).unwrap()
}

#[test]
fn plane_grid_rotated_jittered() {
    let m = plane_grid(10, 7, 0.5, 0.4, 0.3, 3).unwrap();
    let s = surf(m.clone());
    assert_eq!(s.topo.n_comp, 1);
    assert_eq!(s.chi(0), 1);
    assert_eq!(s.topo.boundary_loops.len(), 1);
    // Boundary vertices stay on the 4 rotated sides.
    let (sn, cs) = 0.4f64.sin_cos();
    for v in 0..m.pos.len() {
        if !s.topo.is_bnd_vert[v] {
            continue;
        }
        let p = m.pos[v];
        let (x, y) = (cs * p[0] + sn * p[1], -sn * p[0] + cs * p[1]);
        let on = x.abs() < 1e-12 || (x - 5.0).abs() < 1e-12 || y.abs() < 1e-12 || (y - 3.5).abs() < 1e-12;
        assert!(on, "boundary vertex {v} off the sides: {x} {y}");
    }
    assert!(m.pos.iter().all(|p| p[2] == 0.0));
    assert!(min_angle_deg(&m) > 5.0);
    assert_eq!(m, plane_grid(10, 7, 0.5, 0.4, 0.3, 3).unwrap());
    assert_ne!(m, plane_grid(10, 7, 0.5, 0.4, 0.3, 4).unwrap());
}

#[test]
fn square_disk_has_one_boundary_chain_with_4_corners() {
    let s = surf(square_disk(12, 3.0, 0.3, 9).unwrap());
    assert_eq!(s.chi(0), 1);
    assert_eq!(s.feats.chains.len(), 1);
    let c = &s.feats.chains[0];
    assert!(c.closed);
    assert_eq!(c.kind, FeatureKind::Boundary);
    assert_eq!(c.corners.len(), 4);
    assert_eq!(c.corners[0], 0);
    assert_eq!(s.feats.vert_corner.iter().filter(|&&b| b).count(), 4);
    for (v, &cor) in s.feats.vert_corner.iter().enumerate() {
        if cor {
            let p = s.mesh.pos[v];
            assert!((p[0] == 0.0 || p[0] == 3.0) && (p[1] == 0.0 || p[1] == 3.0));
        }
    }
}

#[test]
fn spheres_have_chi_2() {
    for m in [uv_sphere(12, 20, 2.0).unwrap(), icosphere(4, 1.5).unwrap(), octasphere(6, 1.0).unwrap()] {
        let s = surf(m.clone());
        assert_eq!(s.topo.n_comp, 1);
        assert_eq!(s.chi(0), 2);
        assert!(s.topo.boundary_loops.is_empty());
        let r0 = math::norm(m.pos[0]);
        assert!(m.pos.iter().all(|p| (math::norm(*p) - r0).abs() < 1e-12));
    }
    assert_eq!(icosphere(4, 1.0).unwrap().tris.len(), 5120);
    assert!(min_angle_deg(&icosphere(3, 1.0).unwrap()) > 50.0);
    assert!(min_angle_deg(&octasphere(8, 1.0).unwrap()) > 30.0);
}

#[test]
fn cylinders() {
    let m = cylinder_open(1.5, 4.0, 32, 10, 0.2, 5).unwrap();
    let s = surf(m.clone());
    assert_eq!(s.chi(0), 0);
    assert_eq!(s.topo.boundary_loops.len(), 2);
    assert_eq!(s.feats.chains.len(), 2);
    assert!(s.feats.chains.iter().all(|c| c.closed && c.corners.is_empty() && c.kind == FeatureKind::Boundary));
    for (v, p) in m.pos.iter().enumerate() {
        assert!((p[0].hypot(p[1]) - 1.5).abs() < 1e-12);
        if s.topo.is_bnd_vert[v] {
            assert!(p[2] == 0.0 || p[2] == 4.0);
        }
    }
    assert!(min_angle_deg(&m) > 10.0);
    for (n, j) in [(24u32, 0.0), (40, 0.25), (7, 0.2)] {
        let m = cylinder_closed(1.0, 3.0, n, 8, j, 2).unwrap();
        let s = surf(m.clone());
        assert_eq!(s.chi(0), 2, "closed cylinder n = {n}");
        assert!(s.topo.boundary_loops.is_empty());
        assert!(min_angle_deg(&m) > if j == 0.0 { 15.0 } else { 5.0 }, "closed cylinder n = {n}: {}", min_angle_deg(&m));
    }
}

#[test]
fn torus_chi_0() {
    let m = torus(3.0, 1.0, 48, 16, 0.3, 1).unwrap();
    let s = surf(m.clone());
    assert_eq!(s.chi(0), 0);
    assert!(s.topo.boundary_loops.is_empty());
    for p in &m.pos {
        let rho = p[0].hypot(p[1]);
        assert!(((rho - 3.0).hypot(p[2]) - 1.0).abs() < 1e-12);
    }
}

#[test]
fn double_torus_chi_minus_2() {
    let m = double_torus().unwrap();
    let s = surf(m.clone());
    assert_eq!(s.topo.n_comp, 1);
    assert_eq!(s.chi(0), -2);
    assert!(s.topo.boundary_loops.is_empty());
    let ma = min_angle_deg(&m);
    let n_small = s.frames.corner_angle.iter().filter(|a| a.to_degrees() < 20.0).count();
    assert!(ma > 5.0, "double torus min angle {ma}");
    assert!((n_small as f64) < 0.03 * s.frames.corner_angle.len() as f64, "{n_small} angles < 20 deg");
    for p in &m.pos {
        assert!(aq_testkit::implicit::double_torus_sdf(*p).abs() < 1e-6);
    }
    assert!(m.tris.len() > 4000, "{} faces", m.tris.len());
    assert_eq!(m, double_torus().unwrap());
    let coarse = double_torus_with(0.15, 3).unwrap();
    assert_eq!(surf(coarse).chi(0), -2);
}

#[test]
fn cube_sharp_has_12_chains_and_8_corners() {
    let half = 1.0;
    let sm = cube_sharp(6, 2.0 * half, 0.3, 4).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    assert_eq!(s.chi(0), 2);
    assert_eq!(s.feats.chains.len(), 12);
    assert!(s.feats.chains.iter().all(|c| !c.closed && c.kind == FeatureKind::Hard && c.corners == vec![0]));
    assert!(s.feats.chains.iter().all(|c| c.hes.len() == 6));
    let corners: Vec<usize> = (0..s.n_verts()).filter(|&v| s.feats.vert_corner[v]).collect();
    assert_eq!(corners.len(), 8);
    for &v in &corners {
        assert!(sm.mesh.pos[v].iter().all(|x| x.abs() == half), "corner {v} not exact");
    }
    // Every chain vertex lies on a cube edge (two coordinates at +-1, exactly).
    for c in &s.feats.chains {
        for &h in &c.hes {
            let v = s.topo.tail(&s.mesh, h) as usize;
            let p = sm.mesh.pos[v];
            assert!(p.iter().filter(|x| x.abs() == half).count() >= 2);
        }
    }
    // Faces stay planar.
    for f in 0..s.n_faces() {
        let n = s.frames.n[f];
        assert!(n.iter().filter(|x| (x.abs() - 1.0).abs() < 1e-12).count() == 1);
    }
}

#[test]
fn hemisphere_symmetry_loop_has_no_corner() {
    let sm = hemisphere(8, 2.0).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    assert_eq!(s.chi(0), 1);
    assert_eq!(s.topo.boundary_loops.len(), 1);
    assert_eq!(s.feats.chains.len(), 1);
    let c = &s.feats.chains[0];
    assert!(c.closed && c.corners.is_empty());
    assert_eq!(c.kind, FeatureKind::Symmetry);
    assert_eq!(c.hes.len(), 32);
    assert!(c.hes.iter().all(|&h| s.topo.twin[h as usize] == NONE));
    assert!(s.feats.vert_corner.iter().all(|&b| !b));
    assert!(sm.mesh.pos.iter().all(|p| p[2] >= 0.0 && (math::norm(*p) - 2.0).abs() < 1e-12));
}

#[test]
fn feature_chain_classification() {
    // A polyline on a plane with a 45 degree kink and a 90 degree kink (Hard).
    let n = 12u32;
    let m = plane_grid(n, n, 1.0, 0.0, 0.0, 0).unwrap();
    let t = Topology::build(&m).unwrap();
    let id = |i: u32, j: u32| j * (n + 1) + i;
    // Straight run along j = 6 from i = 1 to 4, then 90 degrees up to (4, 9).
    let mut pairs = Vec::new();
    for i in 1..4 {
        pairs.push(([id(i, 6), id(i + 1, 6)], FeatureKind::Hard));
    }
    for j in 6..9 {
        pairs.push(([id(4, j), id(4, j + 1)], FeatureKind::Hard));
    }
    assert!(t.find_he(&m, id(4, 6), id(4, 7)).is_some());
    let s = Surface::with_features(m.clone(), &pairs, 50.0, 70.0).unwrap();
    let hard: Vec<_> = s.feats.chains.iter().filter(|c| c.kind == FeatureKind::Hard).collect();
    assert_eq!(hard.len(), 1);
    assert_eq!(hard[0].corners.len(), 2, "start corner + the 90 degree kink");
    assert!(s.feats.vert_corner[id(4, 6) as usize]);
    // 45 degrees: a diagonal step through a grid diagonal (diagonals alternate; find one).
    let mut pairs = Vec::new();
    for i in 1..6 {
        pairs.push(([id(i, 6), id(i + 1, 6)], FeatureKind::Hard));
    }
    // Cell (6, 6) has its diagonal from (6, 6) to (7, 7).
    let diag = [id(6, 6), id(7, 7)];
    assert!(t.find_he(&m, diag[0], diag[1]).is_some() || t.find_he(&m, diag[1], diag[0]).is_some());
    pairs.push((diag, FeatureKind::Hard));
    let s = Surface::with_features(m, &pairs, 50.0, 70.0).unwrap();
    let hard: Vec<_> = s.feats.chains.iter().filter(|c| c.kind == FeatureKind::Hard).collect();
    assert_eq!(hard.len(), 1);
    assert_eq!(hard[0].corners, vec![0], "45 degrees is not a corner");
}

#[test]
fn noise_heightfield_merge() {
    let mut m = icosphere(3, 1.0).unwrap();
    let base = m.clone();
    add_noise(&mut m, 0.02, 0.3, 7).unwrap();
    let d: Vec<f64> = m.pos.iter().zip(&base.pos).map(|(a, b)| math::dist(*a, *b)).collect();
    let rms = (d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt();
    assert!(rms > 0.005 && rms < 0.03, "rms {rms}");
    assert!(d.iter().all(|&x| x <= 0.02 * 8f64.sqrt() + 1e-12));
    let mut m2 = base.clone();
    add_noise(&mut m2, 0.02, 0.3, 7).unwrap();
    assert_eq!(m, m2);
    assert_eq!(surf(m).chi(0), 2);

    let h = heightfield(10, 1.0, |x, y| (x * x - y * y) / 4.0).unwrap();
    assert!(h.pos.iter().all(|p| (p[2] - (p[0] * p[0] - p[1] * p[1]) / 4.0).abs() < 1e-15));
    assert_eq!(surf(h).chi(0), 1);

    let a = plane_grid(4, 4, 0.1, 0.0, 0.0, 0).unwrap();
    let mut b = a.clone();
    for p in b.pos.iter_mut() {
        p[2] = 0.05;
    }
    let s = surf(merge(&[a, b]));
    assert_eq!(s.topo.n_comp, 2);
}

#[test]
fn testdata_loads() {
    let names = testdata_names();
    assert_eq!(names.len(), 12);
    for n in names {
        let m = load_testdata(n).unwrap();
        assert!(m.tris.len() > 1000, "{n}");
        m.validate().unwrap();
    }
    let m = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3 4/4/4\nf -4 -2 -1\n").unwrap();
    assert_eq!(m.tris, vec![[0, 1, 2], [0, 2, 3], [0, 2, 3]]);
    assert!(parse_obj("v 0 0 0\nf 1 2 3\n").is_err());
}

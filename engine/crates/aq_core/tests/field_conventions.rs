//! DESIGN 6.1: frames, transport, matching, index, Poincare-Hopf, planted singularities,
//! cube corners.

mod common;
use common::*;

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use aq_core::cross::{
    self, comb_matching, comb_theta, combed_to_raw_branch, index4_vertex, map_branch,
    poincare_hopf, raw_to_combed_branch,
};
use aq_core::math;
use aq_core::{CrossField, NONE, delta_from_p, edge_family, index4_all, matching_from_theta, wrap_pi};

fn noisy_sphere() -> Ctx {
    let mut m = icosphere(4, 1.0);
    add_radial_noise(&mut m, 0.02, 7);
    ctx(m)
}

#[test]
fn wrap_pi_range() {
    assert_eq!(wrap_pi(0.3), 0.3);
    assert_eq!(wrap_pi(PI), PI);
    assert!((wrap_pi(-PI) - PI).abs() < 1e-15);
    assert!((wrap_pi(3.0 * PI + 0.25) - (-PI + 0.25)).abs() < 1e-12);
    for i in -1000..1000 {
        let x = i as f64 * 0.0371;
        let y = wrap_pi(x);
        assert!(y > -PI && y <= PI);
        let k = ((x - y) / TAU).round();
        assert!((x - y - k * TAU).abs() < 1e-12);
    }
}

#[test]
fn frames_orthonormal() {
    let c = noisy_sphere();
    for f in 0..c.m.tris.len() {
        let (e1, e2, n) = (c.fr.e1[f], c.fr.e2[f], c.fr.n[f]);
        assert!((math::norm(e1) - 1.0).abs() < 1e-12);
        assert!(math::dot(e1, e2).abs() < 1e-12);
        assert!(math::dot(e1, n).abs() < 1e-12);
        // Right-handed: e2 = n x e1, normal from C1.
        let nn = math::normalize(c.m.face_cross(f as u32)).unwrap();
        assert!(math::dist(nn, n) < 1e-12);
        assert!(math::dist(math::cross(n, e1), e2) < 1e-12);
        // to/from face 2d round trip.
        let d = [0.3, -0.2, 0.7];
        let a = c.fr.to_face_2d(f as u32, d);
        let back = c.fr.from_face_2d(f as u32, a);
        let tang = math::sub(d, math::scale(n, math::dot(d, n)));
        assert!(math::dist(back, tang) < 1e-12);
    }
    // Corner angles of a face sum to pi.
    for f in 0..c.m.tris.len() {
        let s: f64 = (0..3).map(|k| c.fr.corner_angle[3 * f + k]).sum();
        assert!((s - PI).abs() < 1e-12);
    }
}

#[test]
fn transport() {
    // kappa antisymmetric bitwise.
    let c = noisy_sphere();
    for h in 0..c.t.twin.len() {
        let tw = c.t.twin[h];
        assert_ne!(tw, NONE);
        assert_eq!(c.fr.kappa[tw as usize].to_bits(), (-c.fr.kappa[h]).to_bits());
    }
    // Gauss-Bonnet per vertex: sum_{H_in} kappa == K mod 2pi within 1e-9.
    let mut total_k = 0.0;
    for v in 0..c.m.pos.len() as u32 {
        let s: f64 = c.t.incoming(&c.m, v).map(|h| c.fr.kappa[h as usize]).sum();
        let k = c.fr.angle_defect(&c.t, v);
        total_k += k;
        let r = wrap_pi(s - k);
        assert!(r.abs() < 1e-9, "vertex {v}: sum kappa {s} vs K {k}");
    }
    assert!((total_k - 2.0 * TAU).abs() < 1e-9, "discrete Gauss-Bonnet");
    // Transport meaning: the edge direction itself is transported to the same edge direction.
    for h in 0..c.t.twin.len() as u32 {
        let tw = c.t.twin[h as usize];
        let a = c.fr.edge_angle(&c.m, h);
        let b = c.fr.edge_angle(&c.m, tw);
        assert!(wrap_pi(a + c.fr.kappa[h as usize] - (b + PI)).abs() < 1e-12);
    }
    // Planar jittered grid. DESIGN 6.1 says |kappa| < 1e-12, which cannot hold with C4
    // per-face frames (each face has its own e1). The intended property is that flat
    // transport is trivial: a global direction keeps its global angle, i.e.
    // kappa[he] = beta_f - beta_g with beta the global angle of e1 (CHANGES.md). Also the
    // holonomy around every interior vertex is 0.
    let c = ctx(plane_grid(-1.0, 1.0, -1.0, 1.0, 20, 20, 0.4, 3));
    let beta = |f: usize| c.fr.e1[f][1].atan2(c.fr.e1[f][0]);
    for h in 0..c.t.twin.len() {
        let tw = c.t.twin[h];
        if tw == NONE {
            assert_eq!(c.fr.kappa[h], 0.0);
            continue;
        }
        let want = beta(h / 3) - beta(tw as usize / 3);
        assert!(wrap_pi(c.fr.kappa[h] - want).abs() < 1e-12, "planar transport {}", c.fr.kappa[h]);
        // A global direction (angle 0.7) expressed in f, transported, is the same direction in g.
        let a_f = 0.7 - beta(h / 3);
        let a_g = 0.7 - beta(tw as usize / 3);
        assert!(wrap_pi(a_f + c.fr.kappa[h] - a_g).abs() < 1e-12);
    }
    for v in 0..c.m.pos.len() as u32 {
        if !c.t.is_bnd_vert[v as usize] {
            let s: f64 = c.t.incoming(&c.m, v).map(|h| c.fr.kappa[h as usize]).sum();
            assert!(wrap_pi(s).abs() < 1e-12);
        }
    }
}

#[test]
fn matching() {
    let c = noisy_sphere();
    let theta = random_theta(c.m.tris.len(), 20.0, 11);
    let (p, delta) = matching_from_theta(&c.m, &c.t, &c.fr, &theta);
    for h in 0..p.len() {
        let tw = c.t.twin[h] as usize;
        assert!(p[h] < 4);
        assert_eq!(p[tw], (4 - p[h]) % 4);
        assert_eq!(delta[tw].to_bits(), (-delta[h]).to_bits());
        assert!(delta[h] >= -FRAC_PI_4 - 1e-12 && delta[h] <= FRAC_PI_4 + 1e-12, "delta {}", delta[h]);
        // C7 meaning: branch k of f maps to branch (k - p) of g, which is within pi/4 of the
        // transported direction.
        let (f, g) = (h / 3, tw / 3);
        for k in 0..4u8 {
            let a = theta[f] + k as f64 * FRAC_PI_2 + c.fr.kappa[h];
            let j = map_branch(k, p[h]);
            let b = theta[g] + j as f64 * FRAC_PI_2;
            assert!(wrap_pi(b - a).abs() <= FRAC_PI_4 + 1e-12);
        }
    }
    // delta_from_p agrees bitwise with the fresh residual.
    let d2 = delta_from_p(&c.m, &c.t, &c.fr, &theta, &p);
    for h in 0..p.len() {
        assert_eq!(d2[h].to_bits(), delta[h].to_bits());
    }
    // Rounding is half away from zero.
    assert_eq!(cross::matching_edge(0.0, FRAC_PI_4, 0.0), 1);
    assert_eq!(cross::matching_edge(0.0, -FRAC_PI_4, 0.0), 3);
    assert_eq!(cross::matching_edge(0.0, -1.5 * FRAC_PI_2, 0.0), 2);
}

fn assert_ph(c: &Ctx, theta: Vec<f64>, want: i64) -> CrossField {
    let nf = c.m.tris.len();
    let cf = CrossField::from_theta(&c.m, &c.t, &c.fr, theta, vec![false; nf]).unwrap();
    let s: i64 = cf.index4.iter().map(|&i| i as i64).sum();
    assert_eq!(s, want);
    assert_eq!(s, 4 * c.t.euler_char(&c.m, 0));
    cf.validate(&c.m, &c.t, &c.fr).unwrap();
    cf
}

#[test]
fn index_poincare_hopf_closed() {
    let c = ctx(icosphere(4, 1.0));
    for seed in 0..5 {
        assert_ph(&c, random_theta(c.m.tris.len(), 10.0, seed), 8);
    }
    let c = noisy_sphere();
    assert_ph(&c, random_theta(c.m.tris.len(), 10.0, 99), 8);
    let c = ctx(torus(3.0, 1.0, 48, 16));
    for seed in 0..3 {
        assert_ph(&c, random_theta(c.m.tris.len(), 10.0, seed), 0);
    }
    let c = ctx(double_torus());
    for seed in 0..3 {
        assert_ph(&c, random_theta(c.m.tris.len(), 10.0, seed), -8);
    }
}

#[test]
fn index_open_cylinder_boundary_aligned() {
    let c = ctx(open_cylinder(1.0, 3.0, 24, 8));
    // Circumferential field: z x n_f lies in the face plane and along the rims.
    let theta: Vec<f64> = (0..c.m.tris.len())
        .map(|f| angle_in_face(&c.fr, f, math::cross([0.0, 0.0, 1.0], c.fr.n[f])))
        .collect();
    let cf = assert_ph(&c, theta, 0);
    assert!(cf.index4.iter().all(|&i| i == 0));
}

#[test]
fn index_planar_square_boundary_aligned() {
    let c = ctx(plane_grid(-1.0, 1.0, -1.0, 1.0, 10, 10, 0.3, 5));
    let theta: Vec<f64> = (0..c.m.tris.len()).map(|f| angle_in_face(&c.fr, f, [1.0, 0.0, 0.0])).collect();
    let cf = assert_ph(&c, theta, 4);
    for v in 0..c.m.pos.len() {
        let p = c.m.pos[v];
        let corner = p[0].abs() == 1.0 && p[1].abs() == 1.0;
        assert_eq!(cf.index4[v], if corner { 1 } else { 0 }, "vertex {v} at {p:?}");
    }
}

#[test]
fn boundary_misaligned_field_is_an_invariant_error() {
    let c = ctx(plane_grid(-1.0, 1.0, -1.0, 1.0, 6, 6, 0.0, 5));
    // A rotating field: not aligned with the boundary.
    let theta: Vec<f64> = (0..c.m.tris.len())
        .map(|f| {
            let p = centroid(&c.m, f);
            let a = 0.5 * p[0] + 0.3 * p[1];
            angle_in_face(&c.fr, f, [a.cos(), a.sin(), 0.0])
        })
        .collect();
    let (_, delta) = matching_from_theta(&c.m, &c.t, &c.fr, &theta);
    let e = index4_all(&c.m, &c.t, &c.fr, &delta).unwrap_err();
    assert_eq!(e.kind, aq_core::ErrKind::Invariant);
}

#[test]
fn planted_planar_singularities() {
    // 21 x 21 vertices on [-1,1]^2: the origin is a vertex.
    let mut m = plane_grid(-1.0, 1.0, -1.0, 1.0, 20, 20, 0.3, 17);
    let center = (0..m.pos.len()).min_by(|&a, &b| {
        let d = |i: usize| m.pos[i][0].hypot(m.pos[i][1]);
        d(a).total_cmp(&d(b))
    }).unwrap();
    m.pos[center] = [0.0, 0.0, 0.0];
    let center = center as u32;
    let c = ctx(m);
    for (sign, want) in [(1.0, 1i8), (-1.0, -1i8)] {
        let theta: Vec<f64> = (0..c.m.tris.len())
            .map(|f| {
                let p = centroid(&c.m, f);
                let a = sign * p[1].atan2(p[0]) / 4.0;
                angle_in_face(&c.fr, f, [a.cos(), a.sin(), 0.0])
            })
            .collect();
        let (_, delta) = matching_from_theta(&c.m, &c.t, &c.fr, &theta);
        assert_eq!(index4_vertex(&c.m, &c.t, &c.fr, &delta, center).unwrap(), want);
        for v in 0..c.m.pos.len() as u32 {
            if v != center && !c.t.is_bnd_vert[v as usize] {
                assert_eq!(index4_vertex(&c.m, &c.t, &c.fr, &delta, v).unwrap(), 0, "vertex {v}");
            }
        }
    }
}

#[test]
fn cube_corners() {
    let (m, axis) = cube(4);
    let c = ctx(m);
    // Theta aligned with the cube edges: one in-plane cube axis per face.
    let theta: Vec<f64> = (0..c.m.tris.len())
        .map(|f| {
            let d = match axis[f] / 2 {
                0 => [0.0, 1.0, 0.0],
                1 => [0.0, 0.0, 1.0],
                _ => [1.0, 0.0, 0.0],
            };
            angle_in_face(&c.fr, f, d)
        })
        .collect();
    let cf = assert_ph(&c, theta, 8);
    let mut n = 0;
    for v in 0..c.m.pos.len() {
        let p = c.m.pos[v];
        let corner = p.iter().all(|x| x.abs() == 1.0);
        assert_eq!(cf.index4[v], if corner { 1 } else { 0 }, "vertex {v} at {p:?}");
        n += corner as u32;
    }
    assert_eq!(n, 8);
    poincare_hopf(&c.m, &c.t, &cf.index4).unwrap();
}

#[test]
fn index_independent_of_theta_under_fixed_matching() {
    // C9: the index depends only on p and the geometry: perturb theta, keep p, recompute delta.
    let c = noisy_sphere();
    let theta = random_theta(c.m.tris.len(), 3.0, 5);
    let (p, delta) = matching_from_theta(&c.m, &c.t, &c.fr, &theta);
    let i0 = index4_all(&c.m, &c.t, &c.fr, &delta).unwrap();
    let mut rng = aq_core::SplitMix64::new(1);
    let theta2: Vec<f64> = theta.iter().map(|t| t + rng.range_f64(-0.2, 0.2)).collect();
    let d2 = delta_from_p(&c.m, &c.t, &c.fr, &theta2, &p);
    let i1 = index4_all(&c.m, &c.t, &c.fr, &d2).unwrap();
    assert_eq!(i0, i1);
}

#[test]
fn combing_and_families() {
    // r = (p + c_g - c_f) mod 4 is the matching of the combed theta.
    let c = noisy_sphere();
    let theta = random_theta(c.m.tris.len(), 3.0, 8);
    let (p, _) = matching_from_theta(&c.m, &c.t, &c.fr, &theta);
    let mut rng = aq_core::SplitMix64::new(4);
    let comb: Vec<u8> = (0..c.m.tris.len()).map(|_| rng.below(4) as u8).collect();
    let tc: Vec<f64> = (0..c.m.tris.len()).map(|f| comb_theta(theta[f], comb[f])).collect();
    let (pc, _) = matching_from_theta(&c.m, &c.t, &c.fr, &tc);
    for h in 0..p.len() {
        let tw = c.t.twin[h] as usize;
        assert_eq!(pc[h], comb_matching(p[h], comb[h / 3], comb[tw / 3]));
    }
    // Raw branch k is combed branch (k - c): same direction.
    for k in 0..4u8 {
        for cc in 0..4u8 {
            let j = raw_to_combed_branch(k, cc);
            assert_eq!(combed_to_raw_branch(j, cc), k);
            let raw = 0.4 + k as f64 * FRAC_PI_2;
            let combed = comb_theta(0.4, cc) + j as f64 * FRAC_PI_2;
            assert!(wrap_pi(raw - combed).abs() < 1e-12);
        }
    }
    // Edge family on a plane with theta = 0: x-edges are U, y-edges are V.
    let c = ctx(plane_grid(0.0, 1.0, 0.0, 1.0, 4, 4, 0.0, 0));
    for h in 0..c.t.twin.len() as u32 {
        let f = (h / 3) as usize;
        let th = angle_in_face(&c.fr, f, [1.0, 0.0, 0.0]);
        let d = math::sub(c.m.pos[c.t.head(&c.m, h) as usize], c.m.pos[c.t.tail(&c.m, h) as usize]);
        let fam = edge_family(&c.fr, &c.m, h, th);
        if d[1].abs() < 1e-12 {
            assert_eq!(fam, 0);
        } else if d[0].abs() < 1e-12 {
            assert_eq!(fam, 1);
        }
    }
}

#[test]
fn jacobian_and_folds() {
    let c = ctx(plane_grid(0.0, 1.0, 0.0, 1.0, 4, 4, 0.2, 1));
    // uv = (x, y) gives J = rotation of the frame, det 1.
    let uv: Vec<[aq_core::V2; 3]> = c
        .m
        .tris
        .iter()
        .map(|t| {
            let q = |i: u32| [c.m.pos[i as usize][0], c.m.pos[i as usize][1]];
            [q(t[0]), q(t[1]), q(t[2])]
        })
        .collect();
    for f in 0..c.m.tris.len() {
        let j = cross::jacobian(&c.fr, &c.m, f as u32, &uv[f]);
        assert!((cross::det2(&j) - 1.0).abs() < 1e-12);
        // Row 0 is grad u expressed in (e1, e2): grad u = x axis.
        let gx = c.fr.from_face_2d(f as u32, j[0]);
        assert!(math::dist(gx, [1.0, 0.0, 0.0]) < 1e-12);
    }
    let mut bad = uv.clone();
    bad[3].swap(1, 2);
    let folded = cross::folded_faces(&c.fr, &c.m, &bad);
    assert_eq!(folded.iter().filter(|&&b| b).count(), 1);
    assert!(folded[3]);
}

//! Cotan Laplacian and lumped mass tests (DESIGN 6.3).

mod common;

use aq_core::{ErrKind, NONE};
use aq_linalg::{
    Chol, Csc, cotan_laplacian, cotan_weights, lumped_mass, try_cotan_laplacian, try_lumped_mass,
};
use common::*;

fn meshes() -> Vec<(&'static str, Ctx)> {
    let mut ico = icosphere(4, 1.0);
    add_radial_noise(&mut ico, 0.02, 3);
    vec![
        (
            "jittered plane",
            ctx(plane_grid(0.0, 3.0, 0.0, 2.0, 30, 20, 0.6, 1)),
        ),
        ("noisy icosphere", ctx(ico)),
        ("open cylinder", ctx(open_cylinder(1.0, 3.0, 24, 10))),
        ("torus", ctx(torus(3.0, 1.0, 40, 16))),
    ]
}

#[test]
fn laplacian_kills_constants() {
    for (name, c) in meshes() {
        let l = cotan_laplacian(&c.m, &c.t, &c.fr, None);
        l.validate().unwrap();
        assert_eq!(l.n(), c.m.pos.len());
        let one = vec![1.0; l.n()];
        let mut y = vec![0.0; l.n()];
        l.mul_vec(&one, &mut y);
        let max = y.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        println!("{name}: max |L 1| = {max:e}");
        assert!(max < 1e-12, "{name}: {max}");
    }
}

#[test]
fn laplacian_is_psd_with_positive_weights() {
    for (name, c) in meshes() {
        let l = cotan_laplacian(&c.m, &c.t, &c.fr, None);
        let n = l.n();
        // Off-diagonals are -w with w > 0 (clamped), diagonals positive.
        for j in 0..n {
            assert!(l.get(j, j) > 0.0, "{name}: diagonal {j}");
            for k in l.col_ptr()[j] + 1..l.col_ptr()[j + 1] {
                assert!(l.values()[k] < 0.0, "{name}: off-diagonal not negative");
            }
        }
        // x^T L x >= 0, and L + M factors.
        let x: Vec<f64> = (0..n).map(|i| ((i * 13) as f64).sin()).collect();
        assert!(l.quad_form(&x).unwrap() >= 0.0);
        let mut a = l.clone();
        a.add_diag_vec(&lumped_mass(&c.m, &c.t, &c.fr)).unwrap();
        Chol::new(&a).unwrap();
    }
}

#[test]
fn laplacian_values_on_regular_grid() {
    // Unit right triangles: the hypotenuse is opposite two right angles (cot = 0, so the
    // clamped minimum weight), each leg has weight 1/2 per adjacent triangle (cot 45 = 1).
    let c = ctx(plane_grid(0.0, 4.0, 0.0, 4.0, 4, 4, 0.0, 0));
    let w = cotan_weights(&c.m, &c.t, None).unwrap();
    let mut legs = 0;
    #[allow(clippy::needless_range_loop)]
    for h in 0..w.len() {
        let tw = c.t.twin[h];
        if tw != NONE && (tw as usize) < h {
            assert_eq!(w[h], 0.0);
            continue;
        }
        let f = h / 3;
        let k = h % 3;
        let a = c.m.pos[c.m.tris[f][k] as usize];
        let b = c.m.pos[c.m.tris[f][(k + 1) % 3] as usize];
        let len2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
        if (len2 - 2.0).abs() < 1e-12 {
            assert!(w[h] > 0.0 && w[h] < 1e-5, "hypotenuse weight {}", w[h]);
        } else if tw == NONE {
            assert!((w[h] - 0.5).abs() < 1e-12);
            legs += 1;
        } else {
            assert!((w[h] - 1.0).abs() < 1e-12);
            legs += 1;
        }
    }
    assert!(legs > 0);
    // Linear precision at interior vertices (up to the clamped hypotenuse weights).
    let l = cotan_laplacian(&c.m, &c.t, &c.fr, None);
    let xs: Vec<f64> = c.m.pos.iter().map(|p| 2.0 * p[0] - p[1]).collect();
    let mut y = vec![0.0; l.n()];
    l.mul_vec(&xs, &mut y);
    for (v, p) in c.m.pos.iter().enumerate() {
        if p[0] > 0.0 && p[0] < 4.0 && p[1] > 0.0 && p[1] < 4.0 {
            assert!(y[v].abs() < 1e-4, "vertex {v}: {}", y[v]);
        }
    }
}

#[test]
fn edge_scale_and_errors() {
    let c = ctx(plane_grid(0.0, 2.0, 0.0, 2.0, 6, 6, 0.3, 2));
    let nh = 3 * c.m.tris.len();
    let l1 = cotan_laplacian(&c.m, &c.t, &c.fr, None);
    let l2 = cotan_laplacian(&c.m, &c.t, &c.fr, Some(&vec![2.0; nh]));
    // A uniform scale scales every weight (the clamp bounds scale with the mean).
    let d = Csc::lin_comb(&l2, 1.0, &l1, -2.0).unwrap();
    assert!(d.values().iter().all(|v| v.abs() < 1e-12));
    assert!(l1.same_pattern(&l2));
    let e = try_cotan_laplacian(&c.m, &c.t, &c.fr, Some(&[1.0; 3])).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(cotan_laplacian(&c.m, &c.t, &c.fr, Some(&[1.0; 3])).n(), 0);
    // A malformed topology (twin out of range) is an error, not a panic.
    let mut bad = c.t.clone();
    let nh = bad.twin.len() as u32;
    bad.twin[0] = nh + 5;
    let e = try_cotan_laplacian(&c.m, &bad, &c.fr, None).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
}

#[test]
fn lumped_mass_sums_to_area() {
    for (name, c) in meshes() {
        let m = lumped_mass(&c.m, &c.t, &c.fr);
        assert_eq!(m, try_lumped_mass(&c.m, &c.t, &c.fr).unwrap());
        let s: f64 = m.iter().sum();
        let a: f64 = c.fr.area.iter().sum();
        assert!((s - a).abs() < 1e-12 * a, "{name}");
        assert!(m.iter().all(|&x| x > 0.0));
    }
}

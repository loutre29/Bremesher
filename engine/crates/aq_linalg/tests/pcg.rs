//! PCG tests (DESIGN 6.3).

use std::time::Duration;

use aq_core::{CtlState, ErrKind};
use aq_linalg::{Csc, Jacobi, Triplets, pcg, pcg_ctl};

/// 5-point 2D Poisson on an m x m grid (Dirichlet), scaled rows to make Jacobi matter.
fn poisson(m: usize, scaled: bool) -> Csc {
    let n = m * m;
    let id = |i: usize, j: usize| j * m + i;
    let s = |i: usize, j: usize| {
        if scaled {
            1.0 + ((i * 7 + j * 3) % 5) as f64
        } else {
            1.0
        }
    };
    let mut t = Triplets::new(n);
    for j in 0..m {
        for i in 0..m {
            let a = id(i, j);
            // Edge weights between neighbours; the diagonal gets their sum plus the Dirichlet
            // boundary weights (as if the missing neighbours were fixed to 0).
            let mut d = 0.0;
            for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (ii, jj) = (i as i64 + di, j as i64 + dj);
                let w = 0.5
                    * (s(i, j)
                        + if ii >= 0 && jj >= 0 && (ii as usize) < m && (jj as usize) < m {
                            s(ii as usize, jj as usize)
                        } else {
                            s(i, j)
                        });
                d += w;
                if ii >= 0 && jj >= 0 && (ii as usize) < m && (jj as usize) < m {
                    let b = id(ii as usize, jj as usize);
                    if b < a {
                        t.push(a, b, -w);
                    }
                }
            }
            t.push(a, a, d);
        }
    }
    t.to_csc().unwrap()
}

fn rel_res(a: &Csc, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.mul_vec(x, &mut ax);
    let r: f64 = ax
        .iter()
        .zip(b)
        .map(|(p, q)| (p - q).powi(2))
        .sum::<f64>()
        .sqrt();
    r / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn poisson_jacobi_converges() {
    for scaled in [false, true] {
        let m = 64;
        let a = poisson(m, scaled);
        let n = a.n();
        let b: Vec<f64> = (0..n).map(|i| ((i * 31 % 17) as f64 - 8.0) / 8.0).collect();
        let jac = Jacobi::new(&a);
        let pre = |r: &[f64], z: &mut [f64]| jac.apply(r, z);
        let mut x = vec![0.0; n];
        let it = pcg(&a, &b, &mut x, &pre, 1e-10, 500);
        let r = rel_res(&a, &x, &b);
        println!("2D Poisson {m}x{m} (scaled {scaled}): {it} iterations, relative residual {r:e}");
        assert!(it < 500, "{it} iterations");
        assert!(r <= 1.0001e-10, "residual {r}");
        // The report agrees.
        let mut x2 = vec![0.0; n];
        let info = pcg_ctl(&a, &b, &mut x2, &pre, 1e-10, 500, None).unwrap();
        assert!(info.converged);
        assert_eq!(info.iters, it);
        assert_eq!(x, x2);
    }
}

#[test]
fn pcg_edge_cases() {
    let a = poisson(8, false);
    let n = a.n();
    let id = |r: &[f64], z: &mut [f64]| z.copy_from_slice(r);
    // b = 0 gives x = 0 immediately.
    let mut x = vec![1.0; n];
    assert_eq!(pcg(&a, &vec![0.0; n], &mut x, &id, 1e-10, 100), 0);
    assert!(x.iter().all(|&v| v == 0.0));
    // The exact solution as initial guess needs no iteration.
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut b = vec![0.0; n];
    a.mul_vec(&xs, &mut b);
    let mut x = xs.clone();
    assert_eq!(pcg(&a, &b, &mut x, &id, 1e-10, 100), 0);
    // Wrong lengths do not panic.
    let mut short = vec![0.0; 3];
    assert_eq!(pcg(&a, &b, &mut short, &id, 1e-10, 100), 0);
    assert_eq!(
        pcg_ctl(&a, &b, &mut short, &id, 1e-10, 100, None)
            .unwrap_err()
            .kind,
        ErrKind::InvalidInput
    );
    // Iteration cap is reported as not converged.
    let mut x = vec![0.0; n];
    let info = pcg_ctl(&a, &b, &mut x, &id, 1e-14, 2, None).unwrap();
    assert_eq!(info.iters, 2);
    assert!(!info.converged);
}

#[test]
fn pcg_cancel_and_timeout() {
    let a = poisson(32, false);
    let n = a.n();
    let b = vec![1.0; n];
    let id = |r: &[f64], z: &mut [f64]| z.copy_from_slice(r);
    let st = CtlState::new();
    st.cancel();
    let ctl = st.unbounded();
    let mut x = vec![0.0; n];
    assert_eq!(
        pcg_ctl(&a, &b, &mut x, &id, 1e-12, 100, Some(&ctl))
            .unwrap_err()
            .kind,
        ErrKind::Cancelled
    );
    let st = CtlState::new();
    let ctl = st.ctl_for(Duration::ZERO);
    let mut x = vec![0.0; n];
    assert_eq!(
        pcg_ctl(&a, &b, &mut x, &id, 1e-12, 100, Some(&ctl))
            .unwrap_err()
            .kind,
        ErrKind::Timeout
    );
}

//! Small dense helpers.

use aq_core::ErrKind;
use aq_linalg::dense::{lu_solve, sym_eigen};

#[test]
fn lu_solve_small() {
    let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
    let x = [1.0, -2.0, 3.0];
    let mut b: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum())
        .collect();
    lu_solve(3, &a, &mut b).unwrap();
    for i in 0..3 {
        assert!((b[i] - x[i]).abs() < 1e-14);
    }
    // Needs pivoting.
    let a = [0.0, 1.0, 1.0, 0.0];
    let mut b = vec![3.0, 5.0];
    lu_solve(2, &a, &mut b).unwrap();
    assert_eq!(b, vec![5.0, 3.0]);
    // Singular and malformed inputs are errors.
    let mut b = vec![1.0, 1.0];
    assert_eq!(
        lu_solve(2, &[1.0, 2.0, 2.0, 4.0], &mut b).unwrap_err().kind,
        ErrKind::Numerical
    );
    assert_eq!(
        lu_solve(2, &[1.0; 3], &mut b).unwrap_err().kind,
        ErrKind::InvalidInput
    );
}

#[test]
fn sym_eigen_small() {
    let a = [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 5.0];
    let (vals, vecs) = sym_eigen(3, &a).unwrap();
    let want = [1.0, 3.0, 5.0];
    for k in 0..3 {
        assert!((vals[k] - want[k]).abs() < 1e-12);
        let v = &vecs[k * 3..k * 3 + 3];
        // A v = lambda v, |v| = 1.
        for i in 0..3 {
            let av: f64 = (0..3).map(|j| a[i * 3 + j] * v[j]).sum();
            assert!((av - vals[k] * v[i]).abs() < 1e-12);
        }
        assert!(((v.iter().map(|x| x * x).sum::<f64>()) - 1.0).abs() < 1e-12);
    }
}

#[test]
fn dense_block_limit() {
    use aq_linalg::dense::MAX_DENSE_N;
    let n = MAX_DENSE_N + 1;
    let a = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    assert_eq!(lu_solve(n, &a, &mut b).unwrap_err().kind, ErrKind::Budget);
    assert_eq!(sym_eigen(n, &a).unwrap_err().kind, ErrKind::Budget);
}

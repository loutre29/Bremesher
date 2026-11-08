//! Small dense blocks through faer (DESIGN 3.3: "dense LU/eigen for small blocks"), so that no
//! other crate needs to import faer. Matrices are row-major `n * n` slices, n at most
//! [`MAX_DENSE_N`] (`ErrKind::Budget` above).

use std::panic::{AssertUnwindSafe, catch_unwind};

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

use aq_core::{AqError, ErrKind};

use crate::sparse::err;

/// Largest block side: faer's dense matrices (n * n values, plus the eigenvector output) are
/// allocated infallibly, so blocks are limited to 16 MB of values (DESIGN 3.4).
pub const MAX_DENSE_N: usize = 1448;

fn check(n: usize, a: &[f64], what: &str) -> Result<(), AqError> {
    if n.checked_mul(n) != Some(a.len()) {
        return Err(err(
            ErrKind::InvalidInput,
            format!("{what}: {} values for n = {n}", a.len()),
        ));
    }
    if n > MAX_DENSE_N {
        return Err(err(
            ErrKind::Budget,
            format!("{what}: n = {n} is above the dense block limit {MAX_DENSE_N}"),
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(err(
            ErrKind::Numerical,
            format!("{what}: non-finite matrix entry"),
        ));
    }
    Ok(())
}

/// Solves `A x = b` in place by LU with partial pivoting. A singular or ill-conditioned
/// matrix whose solution is not finite gives `Numerical`.
pub fn lu_solve(n: usize, a: &[f64], b: &mut [f64]) -> Result<(), AqError> {
    check(n, a, "lu_solve")?;
    if b.len() != n {
        return Err(err(
            ErrKind::InvalidInput,
            format!("lu_solve: rhs length {} for n = {n}", b.len()),
        ));
    }
    if n == 0 {
        return Ok(());
    }
    let x = catch_unwind(AssertUnwindSafe(|| {
        let m = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let rhs = Mat::from_fn(n, 1, |i, _| b[i]);
        let lu = m.partial_piv_lu();
        let x = lu.solve(&rhs);
        (0..n).map(|i| x[(i, 0)]).collect::<Vec<f64>>()
    }))
    .map_err(|_| err(ErrKind::Internal, "lu_solve: faer panicked"))?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(err(ErrKind::Numerical, "lu_solve: matrix is singular"));
    }
    b.copy_from_slice(&x);
    Ok(())
}

/// Eigen-decomposition of a symmetric matrix (only the lower triangle is read). Returns the
/// eigenvalues in nondecreasing order and the eigenvectors, row-major by vector:
/// `vecs[k * n + i]` is component i of the eigenvector of `vals[k]` (unit length).
pub fn sym_eigen(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), AqError> {
    check(n, a, "sym_eigen")?;
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let m = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        m.self_adjoint_eigen(Side::Lower).ok().map(|e| {
            let u = e.U();
            let s = e.S();
            let vals: Vec<f64> = (0..n).map(|k| s[k]).collect();
            let mut vecs = vec![0.0; n * n];
            for k in 0..n {
                for i in 0..n {
                    vecs[k * n + i] = u[(i, k)];
                }
            }
            (vals, vecs)
        })
    }))
    .map_err(|_| err(ErrKind::Internal, "sym_eigen: faer panicked"))?;
    r.ok_or_else(|| err(ErrKind::Numerical, "sym_eigen: did not converge"))
}

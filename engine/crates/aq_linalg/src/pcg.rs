//! Preconditioned conjugate gradients (DESIGN 6.3).
//!
//! Every operation is sequential with a fixed order, so the iterates are bit-identical for
//! any thread count.

use aq_core::alloc::try_vec;
use aq_core::{AqError, Ctl, ErrKind};

use crate::sparse::{Csc, err};

/// Outcome of [`pcg_ctl`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgInfo {
    /// Iterations performed.
    pub iters: usize,
    /// Final `||b - A x|| / ||b||` (0 when b = 0).
    pub rel_res: f64,
    /// `rel_res <= tol`.
    pub converged: bool,
}

/// Solves `A x = b` by PCG from the initial guess in `x`. `pre(r, z)` writes `z = M^-1 r`.
/// Stops when `||r|| <= tol ||b||` or after `max_it` iterations, and returns the number of
/// iterations. On a length mismatch x is filled with NaN and 0 is returned. Use
/// [`pcg_ctl`] for the residual, the convergence flag, errors and cancellation.
pub fn pcg(
    a: &Csc,
    b: &[f64],
    x: &mut [f64],
    pre: &dyn Fn(&[f64], &mut [f64]),
    tol: f64,
    max_it: usize,
) -> usize {
    match pcg_ctl(a, b, x, pre, tol, max_it, None) {
        Ok(info) => info.iters,
        Err(_) => {
            x.fill(f64::NAN);
            0
        }
    }
}

/// [`pcg`] with a convergence report and an optional [`Ctl`], checked every iteration
/// (DESIGN 3.4). A breakdown (`p^T A p <= 0` or a non-finite value) stops the iteration with
/// `converged = false`, keeping the last finite iterate.
pub fn pcg_ctl(
    a: &Csc,
    b: &[f64],
    x: &mut [f64],
    pre: &dyn Fn(&[f64], &mut [f64]),
    tol: f64,
    max_it: usize,
    ctl: Option<&Ctl>,
) -> Result<PcgInfo, AqError> {
    let n = a.n();
    if b.len() != n || x.len() != n {
        return Err(err(
            ErrKind::InvalidInput,
            format!("pcg: lengths b {} x {} for n = {n}", b.len(), x.len()),
        ));
    }
    let bnorm = norm(b);
    if !bnorm.is_finite() || x.iter().any(|v| !v.is_finite()) {
        return Err(err(
            ErrKind::Numerical,
            "pcg: non-finite right-hand side or initial guess",
        ));
    }
    if bnorm == 0.0 {
        x.fill(0.0);
        return Ok(PcgInfo {
            iters: 0,
            rel_res: 0.0,
            converged: true,
        });
    }
    let mut r = try_vec(n, 0.0)?;
    let mut z = try_vec(n, 0.0)?;
    let mut ap = try_vec(n, 0.0)?;
    a.try_mul_vec(x, &mut r)?;
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let mut rel = norm(&r) / bnorm;
    if rel <= tol {
        return Ok(PcgInfo {
            iters: 0,
            rel_res: rel,
            converged: true,
        });
    }
    pre(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut it = 0;
    while it < max_it {
        if let Some(c) = ctl {
            c.check()?;
        }
        a.try_mul_vec(&p, &mut ap)?;
        let pap = dot(&p, &ap);
        if pap.is_nan() || pap <= 0.0 || !pap.is_finite() || !rz.is_finite() {
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        it += 1;
        rel = norm(&r) / bnorm;
        if !rel.is_finite() {
            return Err(err(ErrKind::Numerical, "pcg: residual is not finite"));
        }
        if rel <= tol {
            break;
        }
        pre(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    Ok(PcgInfo {
        iters: it,
        rel_res: rel,
        converged: rel <= tol,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Jacobi (diagonal) preconditioner. A non-positive or non-finite diagonal entry acts as 1.
#[derive(Clone, Debug)]
pub struct Jacobi {
    pub inv_diag: Vec<f64>,
}

impl Jacobi {
    pub fn new(a: &Csc) -> Self {
        let inv_diag = a
            .diag()
            .into_iter()
            .map(|d| {
                if d > 0.0 && d.is_finite() {
                    1.0 / d
                } else {
                    1.0
                }
            })
            .collect();
        Jacobi { inv_diag }
    }

    /// `z = D^-1 r`.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        for ((zi, ri), di) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *zi = ri * di;
        }
    }
}

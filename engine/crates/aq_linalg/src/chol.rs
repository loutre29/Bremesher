//! Sparse Cholesky LL^T through faer, with the symbolic analysis (AMD ordering) separated
//! from the numeric factorization so the analysis is reused across refactorizations of the
//! same pattern (DESIGN 6.3, 6.9 stiffening, 6.6 re-solves).
//!
//! Ordering ([`Ordering::Auto`]): AMD, and above [`ND_MIN_N`] unknowns also a nested
//! dissection; the one with the smaller operation count is kept (as CHOLMOD does with AMD and
//! METIS). On irregular closed meshes the nested dissection typically halves the work.
//!
//! Parallelism: faer's numeric factorization and solves are called with `Par::Seq`. faer
//! 0.24 parallelizes only inside the dense supernode kernels, and its results then depend on
//! the thread count (measured: about half of the solution entries differ in their last bits
//! between 1 and 8 threads), which DESIGN 3.4 forbids. Sequential is also as fast or faster
//! on surface meshes, whose supernodes are small. Callers parallelize across independent
//! systems instead.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::linalg::solvers::LltError;
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::{
    LltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering, factorize_symbolic_cholesky,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};

use aq_core::alloc::try_vec;
use aq_core::{AqError, Ctl, ErrKind};

use crate::order;
use crate::sparse::{Csc, err};

/// Parallelism handed to faer: always sequential (determinism, see the module documentation).
fn par() -> Par {
    Par::Seq
}

/// Runs a faer call, turning a panic into `ErrKind::Internal` (a function never panics on
/// user data, DESIGN 3.4).
fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T, AqError>) -> Result<T, AqError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(err(ErrKind::Internal, format!("{what}: faer panicked"))),
    }
}

/// Fill-reducing ordering of [`Chol::analyze_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// AMD below [`ND_MIN_N`] unknowns; above, both AMD and nested dissection are computed
    /// (concurrently on the rayon pool) and the one with the smaller operation count is kept
    /// (AMD on a tie).
    Auto,
    /// Approximate minimum degree (faer).
    Amd,
    /// Nested dissection with BFS level-set separators, AMD on the leaves.
    NestedDissection,
}

/// Size from which [`Ordering::Auto`] also tries nested dissection.
pub const ND_MIN_N: usize = 20_000;

/// Leaf size of the nested dissection.
const ND_LEAF: usize = 512;

fn choose_ordering(
    a: &Csc,
    ord: Ordering,
    ctl: Option<&Ctl>,
) -> Result<(Ordering, Vec<usize>, Vec<usize>, f64), AqError> {
    type Cand = Option<(Ordering, Vec<usize>, Vec<usize>, f64)>;
    let want_amd = ord != Ordering::NestedDissection;
    let want_nd = ord == Ordering::NestedDissection || (ord == Ordering::Auto && a.n() >= ND_MIN_N);
    let amd = || -> Result<Cand, AqError> {
        if !want_amd {
            return Ok(None);
        }
        let (f, i) = order::amd_order(a)?;
        let (fl, _) = order::chol_flops(a, &i, ctl)?;
        Ok(Some((Ordering::Amd, f, i, fl)))
    };
    let nd = || -> Result<Cand, AqError> {
        if !want_nd {
            return Ok(None);
        }
        let g = order::Graph::from_lower(a)?;
        let (f, i) = order::nested_dissection(&g, ND_LEAF, ctl)?;
        let (fl, _) = order::chol_flops(a, &i, ctl)?;
        Ok(Some((Ordering::NestedDissection, f, i, fl)))
    };
    // Two independent computations: running them concurrently cannot change the result.
    let (ra, rn) = rayon::join(amd, nd);
    let (ra, rn) = (ra?, rn?);
    let best = match (ra, rn) {
        (Some(x), Some(y)) => Some(if y.3 < x.3 { y } else { x }),
        (x, y) => x.or(y),
    };
    best.ok_or_else(|| err(ErrKind::Internal, "choose_ordering: no ordering"))
}

/// Symbolic analysis of a pattern: fill-reducing ordering plus the elimination structure. Cheap to
/// clone (reference counted), `Send + Sync`.
#[derive(Clone, Debug)]
pub struct Symbolic {
    n: usize,
    inner: Option<Arc<SymbolicCholesky<usize>>>, // None for n = 0
    col_ptr: Arc<[usize]>,
    row_idx: Arc<[usize]>,
    ordering: Ordering,
    flops: f64,
}

impl Symbolic {
    pub fn n(&self) -> usize {
        self.n
    }

    /// True when `a` has the pattern this analysis was computed for.
    pub fn matches(&self, a: &Csc) -> bool {
        a.n() == self.n && a.col_ptr() == &self.col_ptr[..] && a.row_idx() == &self.row_idx[..]
    }

    /// Number of stored values of the factor L.
    pub fn factor_len(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.len_val())
    }

    /// The ordering that was used ([`Ordering::Amd`] or [`Ordering::NestedDissection`]).
    pub fn ordering(&self) -> Ordering {
        self.ordering
    }

    /// Operation count proxy `sum_j c_j^2` of the factor (`c_j` = column counts of L).
    pub fn flops(&self) -> f64 {
        self.flops
    }

    /// True when faer chose the supernodal factorization (false: simplicial).
    pub fn is_supernodal(&self) -> bool {
        matches!(
            self.inner.as_deref().map(|s| s.raw()),
            Some(SymbolicCholeskyRaw::Supernodal(_))
        )
    }

    /// Fill-reducing permutation: `perm[k]` is the original index eliminated k-th.
    pub fn perm(&self) -> Vec<usize> {
        match self.inner.as_ref().and_then(|s| s.perm()) {
            Some(p) => p.arrays().0.to_vec(),
            None => (0..self.n).collect(),
        }
    }
}

/// Numeric Cholesky factor `P A P^T = L L^T` (P from the symbolic ordering).
#[derive(Clone, Debug)]
pub struct Chol {
    sym: Symbolic,
    l: Vec<f64>,
    ok: bool,
}

/// Numeric factorization of `a` into `l` (length `sym.factor_len()`).
fn numeric(sym: &Symbolic, a: &Csc, l: &mut [f64]) -> Result<(), AqError> {
    if !sym.matches(a) {
        return Err(err(
            ErrKind::InvalidInput,
            "Chol::factor: matrix pattern differs from the symbolic analysis",
        ));
    }
    if let Some(k) = a.values().iter().position(|v| !v.is_finite()) {
        return Err(err(
            ErrKind::Numerical,
            format!("Chol::factor: value {k} is not finite"),
        ));
    }
    let Some(inner) = sym.inner.as_ref() else {
        return Ok(());
    };
    if l.len() != inner.len_val() {
        return Err(err(
            ErrKind::Internal,
            "Chol::factor: factor storage has the wrong size",
        ));
    }
    let n = sym.n;
    let p = par();
    guarded("Chol::factor", || {
        let s = SymbolicSparseColMatRef::new_checked(n, n, a.col_ptr(), None, a.row_idx());
        let am = SparseColMatRef::new(s, a.values());
        let req = inner.factorize_numeric_llt_scratch::<f64>(p, Default::default());
        let mut mem = MemBuffer::try_new(req).map_err(|_| {
            err(
                ErrKind::Budget,
                "Chol::factor: out of memory for the workspace",
            )
        })?;
        let stack = MemStack::new(&mut mem);
        match inner.factorize_numeric_llt::<f64>(
            l,
            am,
            Side::Lower,
            LltRegularization::default(),
            p,
            stack,
            Default::default(),
        ) {
            Ok(_) => Ok(()),
            Err(LltError::NonPositivePivot { index }) => {
                // faer 0.24 reports the permuted pivot 1-based from the simplicial path and
                // 0-based from the supernodal one.
                let index = if sym.is_supernodal() {
                    index
                } else {
                    index.saturating_sub(1)
                };
                let col = sym.perm().get(index).copied().unwrap_or(index);
                Err(err(
                    ErrKind::Numerical,
                    format!("Chol::factor: matrix is not SPD (non-positive pivot at column {col})"),
                ))
            }
        }
    })?;
    if let Some(k) = l.iter().position(|v| !v.is_finite()) {
        return Err(err(
            ErrKind::Numerical,
            format!("Chol::factor: factor value {k} is not finite"),
        ));
    }
    Ok(())
}

impl Chol {
    /// Symbolic analysis with [`Ordering::Auto`]. Only the structure of `a` is read.
    /// Errors: `Invariant` for a malformed matrix, `Budget` when out of memory.
    pub fn analyze(a: &Csc) -> Result<Symbolic, AqError> {
        Chol::analyze_with(a, Ordering::Auto)
    }

    /// Symbolic analysis with the given fill-reducing ordering.
    pub fn analyze_with(a: &Csc, ord: Ordering) -> Result<Symbolic, AqError> {
        Chol::analyze_ctl(a, ord, None)
    }

    /// [`Chol::analyze_with`] that calls `ctl.check()` (DESIGN 3.4) inside the nested
    /// dissection (once per task) and the operation counts (every 4096 columns), and before
    /// and after faer's AMD and symbolic factorization, which cannot be interrupted (about
    /// 0.1 s at 200k unknowns). `Cancelled` / `Timeout` are returned as `ctl` reports them.
    pub fn analyze_ctl(a: &Csc, ord: Ordering, ctl: Option<&Ctl>) -> Result<Symbolic, AqError> {
        let check = || ctl.map_or(Ok(()), |c| c.check());
        check()?;
        a.validate()?;
        let n = a.n();
        let col_ptr: Arc<[usize]> = Arc::from(a.col_ptr());
        let row_idx: Arc<[usize]> = Arc::from(a.row_idx());
        if n == 0 {
            return Ok(Symbolic {
                n,
                inner: None,
                col_ptr,
                row_idx,
                ordering: Ordering::Amd,
                flops: 0.0,
            });
        }
        let (used, fwd, inv, flops) = guarded("Chol::analyze", || choose_ordering(a, ord, ctl))?;
        check()?;
        let inner = guarded("Chol::analyze", || {
            let s = SymbolicSparseColMatRef::new_checked(n, n, a.col_ptr(), None, a.row_idx());
            let p = PermRef::new_checked(&fwd, &inv, n);
            factorize_symbolic_cholesky(
                s,
                Side::Lower,
                SymmetricOrdering::Custom(p),
                Default::default(),
            )
            .map_err(|e| err(ErrKind::Budget, format!("Chol::analyze: {e:?}")))
        })?;
        check()?;
        Ok(Symbolic {
            n,
            inner: Some(Arc::new(inner)),
            col_ptr,
            row_idx,
            ordering: used,
            flops,
        })
    }

    /// Numeric factorization reusing `sym`. `a` must have the analyzed pattern
    /// (`InvalidInput` otherwise) and finite values (`Numerical` otherwise). A matrix that is
    /// not numerically SPD gives `ErrKind::Numerical` naming the original column index.
    pub fn factor(sym: &Symbolic, a: &Csc) -> Result<Self, AqError> {
        let mut l = try_vec(sym.factor_len(), 0.0f64)?;
        numeric(sym, a, &mut l)?;
        Ok(Chol {
            sym: sym.clone(),
            l,
            ok: true,
        })
    }

    /// Numeric refactorization in place: same symbolic, new values of the same pattern, and
    /// the factor storage is reused (no allocation). This is the cheap step of a stiffening or
    /// re-solve loop. On an error the factor is left invalid: [`Chol::try_solve`] then returns
    /// `Numerical` until a refactor succeeds.
    pub fn refactor(&mut self, a: &Csc) -> Result<(), AqError> {
        self.ok = false;
        numeric(&self.sym, a, &mut self.l)?;
        self.ok = true;
        Ok(())
    }

    /// `analyze` then `factor`.
    pub fn new(a: &Csc) -> Result<Self, AqError> {
        let s = Chol::analyze(a)?;
        Chol::factor(&s, a)
    }

    pub fn n(&self) -> usize {
        self.sym.n
    }

    pub fn symbolic(&self) -> &Symbolic {
        &self.sym
    }

    /// Solves `A x = b` in place. On a length mismatch or a failed workspace allocation, b
    /// is filled with NaN (use [`Chol::try_solve`] to get the error).
    pub fn solve(&self, b: &mut [f64]) {
        if self.try_solve(b).is_err() {
            b.fill(f64::NAN);
        }
    }

    /// Solves `A X = B` in place for several right-hand sides at once. On an error every
    /// right-hand side is filled with NaN (use [`Chol::try_solve_multi`]).
    pub fn solve_multi(&self, b: &mut [Vec<f64>]) {
        if self.try_solve_multi(b).is_err() {
            for v in b.iter_mut() {
                v.fill(f64::NAN);
            }
        }
    }

    /// [`Chol::solve`] returning `InvalidInput` (length), `Budget` (workspace) or
    /// `Numerical` (after a failed [`Chol::refactor`]).
    pub fn try_solve(&self, b: &mut [f64]) -> Result<(), AqError> {
        let n = self.sym.n;
        if b.len() != n {
            return Err(err(
                ErrKind::InvalidInput,
                format!("Chol::solve: rhs length {} for n = {n}", b.len()),
            ));
        }
        self.solve_cols(b, 1)
    }

    /// [`Chol::solve_multi`] returning its error.
    pub fn try_solve_multi(&self, b: &mut [Vec<f64>]) -> Result<(), AqError> {
        let n = self.sym.n;
        if let Some(v) = b.iter().find(|v| v.len() != n) {
            return Err(err(
                ErrKind::InvalidInput,
                format!("Chol::solve_multi: rhs length {} for n = {n}", v.len()),
            ));
        }
        let k = b.len();
        if k == 0 || n == 0 {
            return Ok(());
        }
        if k == 1 {
            return self.solve_cols(&mut b[0], 1);
        }
        let mut buf = try_vec(n * k, 0.0f64)?;
        for (j, v) in b.iter().enumerate() {
            buf[j * n..(j + 1) * n].copy_from_slice(v);
        }
        self.solve_cols(&mut buf, k)?;
        for (j, v) in b.iter_mut().enumerate() {
            v.copy_from_slice(&buf[j * n..(j + 1) * n]);
        }
        Ok(())
    }

    /// Solves in place for `k` column-major right-hand sides stored in `buf` (n * k).
    fn solve_cols(&self, buf: &mut [f64], k: usize) -> Result<(), AqError> {
        if !self.ok {
            return Err(err(
                ErrKind::Numerical,
                "Chol::solve: the last refactorization failed",
            ));
        }
        let Some(inner) = self.sym.inner.as_ref() else {
            return Ok(());
        };
        let n = self.sym.n;
        let p = par();
        guarded("Chol::solve", || {
            let req = inner.solve_in_place_scratch::<f64>(k, p);
            let mut mem = MemBuffer::try_new(req).map_err(|_| {
                err(
                    ErrKind::Budget,
                    "Chol::solve: out of memory for the workspace",
                )
            })?;
            let stack = MemStack::new(&mut mem);
            let rhs = MatMut::from_column_major_slice_mut(buf, n, k);
            LltRef::new(inner, &self.l).solve_in_place_with_conj(Conj::No, rhs, p, stack);
            Ok(())
        })
    }
}

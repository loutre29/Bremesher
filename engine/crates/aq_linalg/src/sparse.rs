//! Sparse symmetric matrices in compressed sparse column form, lower storage (DESIGN 6.3).
//!
//! * [`Triplets`] collects `(i, j, v)` entries of a **symmetric** matrix. An off-diagonal pair
//!   is pushed once, in either orientation: `push(i, j, v)` adds `v` to both `A[i][j]` and
//!   `A[j][i]`. Duplicates are summed in insertion order (deterministic).
//! * [`Csc`] stores the lower triangle (row >= column) with sorted row indices. Every
//!   diagonal entry is present in the pattern (an explicit 0 when nothing was pushed), and
//!   explicit zeros are never dropped, so assembling the same entries with other values always
//!   gives the same pattern: a [`crate::Symbolic`] analysis can be reused (stiffening,
//!   refactoring after a regularization).
//! * [`CTriplets`] and [`hermitian_to_real`]: a complex Hermitian n x n matrix as a real
//!   symmetric 2n x 2n one. Unknown `z_i = x_i + i y_i` is stored interleaved: `x_i` at `2i`,
//!   `y_i` at `2i + 1` ([`complex_to_real`], [`real_to_complex`]).

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::{AqError, ErrKind, Stage};

/// Errors of this crate carry `Stage::Engine` (like aq_core); callers restamp them with
/// `AqError::at(stage)`.
pub(crate) fn err(kind: ErrKind, msg: impl Into<String>) -> AqError {
    AqError::new(Stage::Engine, kind, msg)
}

/// Entries of a symmetric n x n matrix. See the module documentation for the convention.
#[derive(Clone, Debug, Default)]
pub struct Triplets {
    n: usize,
    ent: Vec<(u32, u32, f64)>, // (row, col) with row >= col
    bad: Option<(usize, usize)>,
}

impl Triplets {
    /// An empty n x n symmetric matrix.
    pub fn new(n: usize) -> Self {
        Triplets {
            n,
            ent: Vec::new(),
            bad: None,
        }
    }

    /// Like [`Triplets::new`] with room for `cap` entries (`Err(Budget)` when it cannot be
    /// allocated).
    pub fn with_capacity(n: usize, cap: usize) -> Result<Self, AqError> {
        Ok(Triplets {
            n,
            ent: try_with_capacity(cap)?,
            bad: None,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of pushed entries (before merging duplicates).
    pub fn len(&self) -> usize {
        self.ent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ent.is_empty()
    }

    /// Adds `v` to `A[i][j]` and, when `i != j`, to `A[j][i]`. An index out of range is
    /// remembered and reported by [`Triplets::to_csc`] as `InvalidInput`.
    pub fn push(&mut self, i: usize, j: usize, v: f64) {
        if i >= self.n || j >= self.n || self.n > u32::MAX as usize {
            if self.bad.is_none() {
                self.bad = Some((i, j));
            }
            return;
        }
        let (r, c) = if i >= j { (i, j) } else { (j, i) };
        if self.ent.len() == self.ent.capacity()
            && self.ent.try_reserve(self.ent.len().max(64)).is_err()
        {
            // Remember the failure; to_csc reports it as Budget.
            self.bad = Some((usize::MAX, usize::MAX));
            return;
        }
        self.ent.push((r as u32, c as u32, v));
    }

    /// Removes every entry, keeping n and the allocation.
    pub fn clear(&mut self) {
        self.ent.clear();
        self.bad = None;
    }

    /// Compresses into lower CSC storage. Duplicates are summed in insertion order; every
    /// diagonal entry exists. Errors: an index out of range is `InvalidInput`, a non-finite
    /// value `Numerical`, a failed allocation `Budget`.
    pub fn to_csc(&self) -> Result<Csc, AqError> {
        if let Some((i, j)) = self.bad {
            if i == usize::MAX {
                return Err(err(
                    ErrKind::Budget,
                    "Triplets: allocation failed while pushing",
                ));
            }
            return Err(err(
                ErrKind::InvalidInput,
                format!("Triplets: entry ({i}, {j}) out of range for n = {}", self.n),
            ));
        }
        if let Some(k) = self.ent.iter().position(|e| !e.2.is_finite()) {
            let (r, c, v) = self.ent[k];
            return Err(err(
                ErrKind::Numerical,
                format!("Triplets: entry ({r}, {c}) is {v}"),
            ));
        }
        let n = self.n;
        // Counting sort by column (stable), with one diagonal placeholder first per column.
        let mut cnt = try_vec(n + 1, 0usize)?;
        for &(_, c, _) in &self.ent {
            cnt[c as usize + 1] += 1;
        }
        for c in cnt.iter_mut().skip(1) {
            *c += 1; // the diagonal placeholder of the column
        }
        for j in 0..n {
            cnt[j + 1] += cnt[j];
        }
        let total = cnt[n];
        let mut buf: Vec<(u32, f64)> = try_vec(total, (0u32, 0.0f64))?;
        let mut fill = cnt.clone();
        for (j, f) in fill.iter_mut().take(n).enumerate() {
            buf[*f] = (j as u32, 0.0);
            *f += 1;
        }
        for &(r, c, v) in &self.ent {
            let f = &mut fill[c as usize];
            buf[*f] = (r, v);
            *f += 1;
        }
        // Per column: stable sort by row, then merge duplicates in order.
        let mut col_ptr = try_vec(n + 1, 0usize)?;
        let mut row_idx: Vec<usize> = try_with_capacity(total)?;
        let mut val: Vec<f64> = try_with_capacity(total)?;
        for j in 0..n {
            let s = &mut buf[cnt[j]..cnt[j + 1]];
            s.sort_by_key(|e| e.0);
            let mut k = 0;
            while k < s.len() {
                let r = s[k].0;
                let mut acc = s[k].1;
                k += 1;
                while k < s.len() && s[k].0 == r {
                    acc += s[k].1;
                    k += 1;
                }
                row_idx.push(r as usize);
                val.push(acc);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        Ok(Csc {
            n,
            col_ptr,
            row_idx,
            val,
        })
    }
}

/// Symmetric n x n matrix, lower triangle in compressed sparse column form.
///
/// Invariants (checked by [`Csc::validate`]): `col_ptr.len() == n + 1`, row indices of a
/// column are strictly increasing and `>= column`, and the first entry of every column is its
/// diagonal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csc {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    val: Vec<f64>,
}

impl Csc {
    /// Builds from raw parts, then validates them.
    pub fn from_parts(
        n: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        val: Vec<f64>,
    ) -> Result<Self, AqError> {
        let a = Csc {
            n,
            col_ptr,
            row_idx,
            val,
        };
        a.validate()?;
        Ok(a)
    }

    /// Shorthand for `t.to_csc()`.
    pub fn from_triplets(t: &Triplets) -> Result<Self, AqError> {
        t.to_csc()
    }

    /// The diagonal matrix `diag(d)`.
    pub fn from_diag(d: &[f64]) -> Result<Self, AqError> {
        let n = d.len();
        let mut col_ptr = try_with_capacity(n + 1)?;
        col_ptr.extend(0..=n);
        let mut row_idx = try_with_capacity(n)?;
        row_idx.extend(0..n);
        let mut val = try_with_capacity(n)?;
        val.extend_from_slice(d);
        Ok(Csc {
            n,
            col_ptr,
            row_idx,
            val,
        })
    }

    /// The n x n identity.
    pub fn identity(n: usize) -> Result<Self, AqError> {
        Csc::from_diag(&try_vec(n, 1.0)?)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Stored entries (lower triangle including the diagonal).
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    pub fn values(&self) -> &[f64] {
        &self.val
    }

    /// Mutable values with the pattern fixed (numeric refactorization with the same symbolic).
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.val
    }

    /// Checks the structural invariants (`Invariant` on failure).
    pub fn validate(&self) -> Result<(), AqError> {
        let n = self.n;
        let bad = |m: String| Err(err(ErrKind::Invariant, format!("Csc: {m}")));
        if self.col_ptr.len() != n + 1 {
            return bad(format!(
                "col_ptr has {} entries for n = {n}",
                self.col_ptr.len()
            ));
        }
        if self.col_ptr[0] != 0
            || self.col_ptr[n] != self.row_idx.len()
            || self.row_idx.len() != self.val.len()
        {
            return bad("col_ptr ends or value count inconsistent".into());
        }
        for j in 0..n {
            let (s, e) = (self.col_ptr[j], self.col_ptr[j + 1]);
            if e < s || e > self.row_idx.len() {
                return bad(format!("col_ptr not monotone at column {j}"));
            }
            if s == e || self.row_idx[s] != j {
                return bad(format!("column {j} does not start with its diagonal"));
            }
            for k in s + 1..e {
                if self.row_idx[k] <= self.row_idx[k - 1] || self.row_idx[k] >= n {
                    return bad(format!(
                        "column {j}: row indices not strictly increasing or out of range"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Same n and the same pattern as `o`.
    pub fn same_pattern(&self, o: &Csc) -> bool {
        self.n == o.n && self.col_ptr == o.col_ptr && self.row_idx == o.row_idx
    }

    /// `A[i][j]` (0 outside the pattern or out of range).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (r, c) = if i >= j { (i, j) } else { (j, i) };
        if c >= self.n {
            return 0.0;
        }
        let s = &self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]];
        match s.binary_search(&r) {
            Ok(k) => self.val[self.col_ptr[c] + k],
            Err(_) => 0.0,
        }
    }

    /// The diagonal.
    pub fn diag(&self) -> Vec<f64> {
        (0..self.n).map(|j| self.val[self.col_ptr[j]]).collect()
    }

    /// Mean of the diagonal (0 for n = 0).
    pub fn mean_diag(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (0..self.n).map(|j| self.val[self.col_ptr[j]]).sum::<f64>() / self.n as f64
    }

    /// `A += s I` (for example the `1e-10 * mean diagonal` regularization of 6.9).
    pub fn add_diag(&mut self, s: f64) {
        for j in 0..self.n {
            self.val[self.col_ptr[j]] += s;
        }
    }

    /// `A += diag(d)` (for example Laplacian plus lumped mass). `InvalidInput` when
    /// `d.len() != n`.
    pub fn add_diag_vec(&mut self, d: &[f64]) -> Result<(), AqError> {
        if d.len() != self.n {
            return Err(err(
                ErrKind::InvalidInput,
                format!("add_diag_vec: {} values for n = {}", d.len(), self.n),
            ));
        }
        for (j, &x) in d.iter().enumerate() {
            self.val[self.col_ptr[j]] += x;
        }
        Ok(())
    }

    /// `A *= s`.
    pub fn scale(&mut self, s: f64) {
        for v in &mut self.val {
            *v *= s;
        }
    }

    /// `alpha A + beta B` on the union of the patterns (explicit zeros kept).
    pub fn lin_comb(a: &Csc, alpha: f64, b: &Csc, beta: f64) -> Result<Csc, AqError> {
        if a.n != b.n {
            return Err(err(
                ErrKind::InvalidInput,
                format!("lin_comb: sizes {} and {}", a.n, b.n),
            ));
        }
        let n = a.n;
        let mut col_ptr = try_vec(n + 1, 0usize)?;
        let mut row_idx = try_with_capacity(a.nnz() + b.nnz())?;
        let mut val = try_with_capacity(a.nnz() + b.nnz())?;
        for j in 0..n {
            let (mut p, pe) = (a.col_ptr[j], a.col_ptr[j + 1]);
            let (mut q, qe) = (b.col_ptr[j], b.col_ptr[j + 1]);
            while p < pe || q < qe {
                let ra = if p < pe { a.row_idx[p] } else { usize::MAX };
                let rb = if q < qe { b.row_idx[q] } else { usize::MAX };
                if ra < rb {
                    row_idx.push(ra);
                    val.push(alpha * a.val[p]);
                    p += 1;
                } else if rb < ra {
                    row_idx.push(rb);
                    val.push(beta * b.val[q]);
                    q += 1;
                } else {
                    row_idx.push(ra);
                    val.push(alpha * a.val[p] + beta * b.val[q]);
                    p += 1;
                    q += 1;
                }
            }
            col_ptr[j + 1] = row_idx.len();
        }
        Ok(Csc {
            n,
            col_ptr,
            row_idx,
            val,
        })
    }

    /// `y = A x` with the full symmetric A. Sequential, so the result is bit-identical for
    /// any thread count. When a length differs from n, y is filled with NaN (use
    /// [`Csc::try_mul_vec`] to get an error instead).
    pub fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        if self.try_mul_vec(x, y).is_err() {
            y.fill(f64::NAN);
        }
    }

    /// `y = A x`, `InvalidInput` on a length mismatch.
    pub fn try_mul_vec(&self, x: &[f64], y: &mut [f64]) -> Result<(), AqError> {
        let n = self.n;
        if x.len() != n || y.len() != n {
            return Err(err(
                ErrKind::InvalidInput,
                format!("mul_vec: lengths {} and {} for n = {n}", x.len(), y.len()),
            ));
        }
        y.fill(0.0);
        for j in 0..n {
            let (s, e) = (self.col_ptr[j], self.col_ptr[j + 1]);
            let xj = x[j];
            // Diagonal first.
            let mut acc = self.val[s] * xj;
            for k in s + 1..e {
                let i = self.row_idx[k];
                let a = self.val[k];
                y[i] += a * xj;
                acc += a * x[i];
            }
            y[j] += acc;
        }
        Ok(())
    }

    /// `x^T A x`.
    pub fn quad_form(&self, x: &[f64]) -> Result<f64, AqError> {
        let mut y = try_vec(self.n, 0.0)?;
        self.try_mul_vec(x, &mut y)?;
        Ok(x.iter().zip(&y).map(|(a, b)| a * b).sum())
    }

    /// Dense full matrix, row-major `n * n` (for tests and small blocks; `Budget` above
    /// 16M entries).
    pub fn to_dense(&self) -> Result<Vec<f64>, AqError> {
        let n = self.n;
        let nn = n
            .checked_mul(n)
            .filter(|&x| x <= 1 << 24)
            .ok_or_else(|| err(ErrKind::Budget, format!("to_dense: n = {n} is too large")))?;
        let mut d = try_vec(nn, 0.0)?;
        for j in 0..n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                d[i * n + j] = self.val[k];
                d[j * n + i] = self.val[k];
            }
        }
        Ok(d)
    }
}

/// Entries of a complex Hermitian n x n matrix H, as `[re, im]` pairs.
///
/// `push(i, j, [re, im])` adds `re + i im` to `H[i][j]` and its conjugate to `H[j][i]`.
/// On the diagonal (i == j) only `re` is used: a Hermitian diagonal is real.
#[derive(Clone, Debug, Default)]
pub struct CTriplets {
    n: usize,
    ent: Vec<(u32, u32, [f64; 2])>,
    bad: Option<(usize, usize)>,
}

impl CTriplets {
    pub fn new(n: usize) -> Self {
        CTriplets {
            n,
            ent: Vec::new(),
            bad: None,
        }
    }

    pub fn with_capacity(n: usize, cap: usize) -> Result<Self, AqError> {
        Ok(CTriplets {
            n,
            ent: try_with_capacity(cap)?,
            bad: None,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn len(&self) -> usize {
        self.ent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ent.is_empty()
    }

    /// Adds `z` to `H[i][j]` and `conj(z)` to `H[j][i]` (see the type documentation).
    pub fn push(&mut self, i: usize, j: usize, z: [f64; 2]) {
        if i >= self.n || j >= self.n || 2 * self.n > u32::MAX as usize {
            if self.bad.is_none() {
                self.bad = Some((i, j));
            }
            return;
        }
        if self.ent.len() == self.ent.capacity()
            && self.ent.try_reserve(self.ent.len().max(64)).is_err()
        {
            self.bad = Some((usize::MAX, usize::MAX));
            return;
        }
        self.ent.push((i as u32, j as u32, z));
    }
}

/// Hermitian n x n to real symmetric 2n x 2n (DESIGN 6.3). With `H = A + iB` and the
/// interleaved unknowns `(x_0, y_0, x_1, y_1, ...)`, block (i, j) is
/// `[[A_ij, -B_ij], [B_ij, A_ij]]`, so `H z = c` becomes `R [x; y] = [Re c; Im c]`
/// (interleaved the same way). R is SPD exactly when H is Hermitian positive definite.
///
/// On an error (index out of range, non-finite value, allocation) this returns an empty
/// 0 x 0 matrix; use [`try_hermitian_to_real`] to get the error.
pub fn hermitian_to_real(h: &CTriplets) -> Csc {
    try_hermitian_to_real(h).unwrap_or_default()
}

/// [`hermitian_to_real`] returning its error.
pub fn try_hermitian_to_real(h: &CTriplets) -> Result<Csc, AqError> {
    if let Some((i, j)) = h.bad {
        if i == usize::MAX {
            return Err(err(
                ErrKind::Budget,
                "CTriplets: allocation failed while pushing",
            ));
        }
        return Err(err(
            ErrKind::InvalidInput,
            format!("CTriplets: entry ({i}, {j}) out of range for n = {}", h.n),
        ));
    }
    let mut t = Triplets::with_capacity(2 * h.n, 4 * h.ent.len())?;
    for &(i, j, [re, im]) in &h.ent {
        let (i, j) = (i as usize, j as usize);
        if i == j {
            t.push(2 * i, 2 * i, re);
            t.push(2 * i + 1, 2 * i + 1, re);
            continue;
        }
        // Block (i, j) = [[re, -im], [im, re]]; block (j, i) is its transpose (implied).
        t.push(2 * i, 2 * j, re);
        t.push(2 * i, 2 * j + 1, -im);
        t.push(2 * i + 1, 2 * j, im);
        t.push(2 * i + 1, 2 * j + 1, re);
    }
    t.to_csc()
}

/// Complex vector to the interleaved real layout of [`hermitian_to_real`].
pub fn complex_to_real(z: &[[f64; 2]]) -> Vec<f64> {
    z.iter().flat_map(|c| [c[0], c[1]]).collect()
}

/// Inverse of [`complex_to_real`] (a trailing odd entry is ignored).
pub fn real_to_complex(x: &[f64]) -> Vec<[f64; 2]> {
    x.as_chunks::<2>().0.to_vec()
}

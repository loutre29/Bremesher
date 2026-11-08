//! # aq_linalg: sparse linear algebra of the AutoQuad engine (DESIGN 6.3, WP-02)
//!
//! * [`Triplets`] and [`Csc`]: symmetric sparse matrices in lower CSC storage. The pattern
//!   always holds the diagonal and never drops explicit zeros, so the same assembly with other
//!   values reuses a [`Symbolic`] analysis.
//! * [`Chol`]: faer sparse Cholesky LL^T with a symbolic/numeric split ([`Chol::analyze`],
//!   [`Chol::factor`], in-place [`Chol::refactor`]). The ordering is AMD, or a nested
//!   dissection when it is cheaper ([`Ordering`]); [`Chol::analyze_ctl`] checks a `Ctl`
//!   during the ordering. A non-SPD matrix gives `ErrKind::Numerical`
//!   naming the column.
//! * [`pcg()`] / [`pcg_ctl`] with a [`Jacobi`] preconditioner.
//! * [`cotan_laplacian`] (PSD, clamped weights) and [`lumped_mass`] ([`try_lumped_mass`]).
//! * [`hermitian_to_real`]: complex Hermitian n x n as real symmetric 2n x 2n, interleaved
//!   unknowns (`x_i` at 2i, `y_i` at 2i + 1).
//! * [`Elim`]: exact integer elimination of homogeneous constraints with +-1 pivots (aq_param);
//!   [`Elim::try_new`] for large variable counts.
//! * [`dense`]: small dense LU solve and symmetric eigen-decomposition.
//!
//! Functions with the exact DESIGN 6.3 signatures that cannot return a `Result`
//! (`cotan_laplacian`, `hermitian_to_real`, `Chol::solve`, `pcg`) have a `try_` (or `_ctl`)
//! twin returning the error. Errors carry `Stage::Engine`; callers restamp them with
//! `AqError::at(stage)`.
//!
//! Determinism: everything runs sequentially in a fixed order (faer included, see [`chol`]), so
//! results are bit-identical for any rayon thread count (tested). Callers get parallelism by
//! running independent systems concurrently.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod chol;
pub mod dense;
pub mod elim;
pub mod lap;
mod order;
pub mod pcg;
pub mod sparse;

pub use chol::{Chol, ND_MIN_N, Ordering, Symbolic};
pub use elim::{Elim, RowOutcome};
pub use lap::{cotan_laplacian, cotan_weights, lumped_mass, try_cotan_laplacian, try_lumped_mass};
pub use pcg::{Jacobi, PcgInfo, pcg, pcg_ctl};
pub use sparse::{
    CTriplets, Csc, Triplets, complex_to_real, hermitian_to_real, real_to_complex,
    try_hermitian_to_real,
};

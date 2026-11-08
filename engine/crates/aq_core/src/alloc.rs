//! Fallible allocation helpers (DESIGN 3.4: `try_reserve` for every buffer larger than 16 MB;
//! a failure returns `ErrKind::Budget`). They are cheap, so they may be used for any size.

use crate::error::AqError;

/// Buffers above this many bytes must be allocated fallibly.
pub const LARGE_BUFFER_BYTES: usize = 16 << 20;

/// `Vec::with_capacity(n)` that returns `Err(Budget)` instead of aborting.
pub fn try_with_capacity<T>(n: usize) -> Result<Vec<T>, AqError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|e| AqError::budget(format!("allocation of {n} elements failed: {e}")))?;
    Ok(v)
}

/// `vec![value; n]` that returns `Err(Budget)` instead of aborting.
pub fn try_vec<T: Clone>(n: usize, value: T) -> Result<Vec<T>, AqError> {
    let mut v = try_with_capacity(n)?;
    v.resize(n, value);
    Ok(v)
}

/// Reserves `extra` more elements in `v`, or `Err(Budget)`.
pub fn try_reserve<T>(v: &mut Vec<T>, extra: usize) -> Result<(), AqError> {
    v.try_reserve(extra)
        .map_err(|e| AqError::budget(format!("reserve of {extra} elements failed: {e}")))
}

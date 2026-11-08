//! C14: units and tolerances. Absolute tolerances are in quad units (h0 = 1).
//!
//! The Python side sends meshes with a bounding box diagonal of 10. The engine computes
//! `h0 = sqrt(A_total / N_target)`, scales all positions by `1/h0` and works in quad units,
//! where the target edge length is 1 and the local size is `h(x) = g(x)` (about 1).

/// Geometric epsilon (in h).
pub const GEOM_EPS: f64 = 1e-9;
/// A face is zero-area when `area < ZERO_AREA_REL * diag_q^2` ([`is_zero_area`]).
pub const ZERO_AREA_REL: f64 = 1e-14;
/// Dyadic uv quantum, 2^-24 (exact power of two).
pub const DYADIC_QUANTUM: f64 = 1.0 / 16_777_216.0;
/// Inverse of the dyadic quantum, 2^24.
pub const DYADIC_SCALE: f64 = 16_777_216.0;
/// Bound on |uv|: 2^27, otherwise `ErrKind::Numerical`.
pub const UV_BOUND: f64 = 134_217_728.0;
/// Angle tests (radians).
pub const ANGLE_EPS: f64 = 1e-12;
/// Index residual bound of C9 (radians).
pub const INDEX_RESIDUAL: f64 = 1e-6;
/// Weights are clamped to [WEIGHT_CLAMP_LO, WEIGHT_CLAMP_HI] times their mean ([`clamp_weight`]).
pub const WEIGHT_CLAMP_LO: f64 = 1e-6;
pub const WEIGHT_CLAMP_HI: f64 = 1e6;
/// C11: `eps_det = FOLD_EPS_REL * median(|det J|)`.
pub const FOLD_EPS_REL: f64 = 1e-10;

/// Zero-area test of C14, `diag_q` being the bounding box diagonal in quad units.
/// NaN areas count as zero-area.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn is_zero_area(area: f64, diag_q: f64) -> bool {
    !(area >= ZERO_AREA_REL * diag_q * diag_q)
}

/// Clamps a weight to [1e-6, 1e6] times `mean` (C14). NaN gives the lower bound.
pub fn clamp_weight(w: f64, mean: f64) -> f64 {
    let lo = WEIGHT_CLAMP_LO * mean;
    let hi = WEIGHT_CLAMP_HI * mean;
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    if w.is_nan() { lo } else { w.clamp(lo, hi) }
}

/// Rounds `x` to the nearest multiple of 2^-24 (half away from zero). Exact for |x| < 2^28.
pub fn snap_dyadic(x: f64) -> f64 {
    (x * DYADIC_SCALE).round() * DYADIC_QUANTUM + 0.0
}

/// True when `x` is an exact multiple of 2^-24.
pub fn is_dyadic(x: f64) -> bool {
    let y = x * DYADIC_SCALE;
    y.is_finite() && y == y.trunc()
}

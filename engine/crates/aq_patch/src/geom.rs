//! Small geometry helpers: per-face 2D frames (C4), gradient bases, 2x2 matrices.

use aq_core::math;
use aq_core::{AqError, ErrKind, Stage, V2, V3};

/// 2x2 matrix, row-major. A Jacobian has row 0 = grad U, row 1 = grad V (C11).
pub type M2 = [[f64; 2]; 2];

/// Errors of this crate carry `Stage::Patch`.
pub(crate) fn err(kind: ErrKind, msg: impl Into<String>) -> AqError {
    AqError::new(Stage::Patch, kind, msg)
}

pub(crate) fn inv_err(msg: impl Into<String>) -> AqError {
    err(ErrKind::Invariant, msg)
}

pub(crate) fn input_err(msg: impl Into<String>) -> AqError {
    err(ErrKind::InvalidInput, msg)
}

/// Relative threshold below which a face counts as degenerate for the energies:
/// `2 A <= DEGEN_REL * (longest edge)^2`.
pub(crate) const DEGEN_REL: f64 = 1e-14;

/// Geometry of one triangle in its C4 frame (`e1 = normalize(p1 - p0)`, `e2 = n x e1`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceGeo {
    /// Gradients of the 3 hat functions in the face frame.
    pub g: [V2; 3],
    pub area: f64,
}

/// Corner coordinates of a triangle in its C4 frame and its area, `None` when degenerate.
pub(crate) fn face_coords(p: [V3; 3]) -> Option<([V2; 3], f64)> {
    let d1 = math::sub(p[1], p[0]);
    let d2 = math::sub(p[2], p[0]);
    let n = math::cross(d1, d2);
    let twice = math::norm(n);
    let l1 = math::norm(d1);
    let lmax2 = math::norm2(d1).max(math::norm2(d2)).max(math::dist2(p[1], p[2]));
    if !(twice.is_finite() && l1 > 0.0 && twice > DEGEN_REL * lmax2) {
        return None;
    }
    let e1 = math::scale(d1, 1.0 / l1);
    let nn = math::scale(n, 1.0 / twice);
    let e2 = math::cross(nn, e1);
    let q2 = [math::dot(d2, e1), math::dot(d2, e2)];
    Some(([[0.0, 0.0], [l1, 0.0], q2], 0.5 * twice))
}

/// Hat function gradients of a CCW 2D triangle: `g_k = R90(q_{k+2} - q_{k+1}) / (2A)`.
pub(crate) fn hat_grads(q: [V2; 3], area: f64) -> [V2; 3] {
    let s = 1.0 / (2.0 * area);
    let mut g = [[0.0; 2]; 3];
    for (k, gk) in g.iter_mut().enumerate() {
        let a = q[(k + 1) % 3];
        let b = q[(k + 2) % 3];
        let d = [b[0] - a[0], b[1] - a[1]];
        *gk = [-d[1] * s, d[0] * s];
    }
    g
}

pub(crate) fn face_geo(p: [V3; 3]) -> Option<FaceGeo> {
    let (q, area) = face_coords(p)?;
    let g = hat_grads(q, area);
    if g.iter().all(|x| x[0].is_finite() && x[1].is_finite()) { Some(FaceGeo { g, area }) } else { None }
}

/// Jacobian of the per-corner values `x` (rows: coordinate, columns: frame axis).
#[inline]
pub(crate) fn jac(g: &[V2; 3], x: [V2; 3]) -> M2 {
    let mut j = [[0.0; 2]; 2];
    for c in 0..2 {
        for a in 0..2 {
            j[c][a] = x[0][c] * g[0][a] + x[1][c] * g[1][a] + x[2][c] * g[2][a];
        }
    }
    j
}

#[inline]
pub(crate) fn det(m: &M2) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

#[inline]
pub(crate) fn mul(a: &M2, b: &M2) -> M2 {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

/// Inverse, `None` when singular or not finite.
#[inline]
pub(crate) fn inv(m: &M2) -> Option<M2> {
    let d = det(m);
    if !(d.is_finite() && d != 0.0) {
        return None;
    }
    let s = 1.0 / d;
    Some([[m[1][1] * s, -m[0][1] * s], [-m[1][0] * s, m[0][0] * s]])
}

#[inline]
pub(crate) fn frob2(m: &M2) -> f64 {
    m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[1][0] * m[1][0] + m[1][1] * m[1][1]
}

/// Closest rotation to `m` (2D polar decomposition); identity when undefined.
#[inline]
pub(crate) fn closest_rotation(m: &M2) -> M2 {
    let a = m[0][0] + m[1][1];
    let b = m[1][0] - m[0][1];
    let r = (a * a + b * b).sqrt();
    if !(r.is_finite() && r > 0.0) {
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let (c, s) = (a / r, b / r);
    [[c, -s], [s, c]]
}

/// Twice the signed area of a 2D triangle (floating point).
#[inline]
pub(crate) fn cross_tri(a: V2, b: V2, c: V2) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

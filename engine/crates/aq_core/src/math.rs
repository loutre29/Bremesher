//! Small vector helpers on `[f64; 3]` and `[f64; 2]`.

use crate::mesh::{V2, V3};

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
pub fn norm2(a: V3) -> f64 {
    dot(a, a)
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub fn dist(a: V3, b: V3) -> f64 {
    norm(sub(a, b))
}
#[inline]
pub fn dist2(a: V3, b: V3) -> f64 {
    norm2(sub(a, b))
}
/// `a / |a|`, or `None` when |a| is zero or not finite.
#[inline]
pub fn normalize(a: V3) -> Option<V3> {
    let l = norm(a);
    if l > 0.0 && l.is_finite() { Some(scale(a, 1.0 / l)) } else { None }
}
/// `a + t (b - a)`.
#[inline]
pub fn lerp(a: V3, b: V3, t: f64) -> V3 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}
/// Barycentric combination `w0 a + w1 b + w2 c`.
#[inline]
pub fn bary_point(a: V3, b: V3, c: V3, w: V3) -> V3 {
    [
        w[0] * a[0] + w[1] * b[0] + w[2] * c[0],
        w[0] * a[1] + w[1] * b[1] + w[2] * c[1],
        w[0] * a[2] + w[1] * b[2] + w[2] * c[2],
    ]
}
#[inline]
pub fn is_finite3(a: V3) -> bool {
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite()
}

#[inline]
pub fn add2(a: V2, b: V2) -> V2 {
    [a[0] + b[0], a[1] + b[1]]
}
#[inline]
pub fn sub2(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}
#[inline]
pub fn dot2(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
/// z component of the 2D cross product.
#[inline]
pub fn cross2(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
#[inline]
pub fn is_finite2(a: V2) -> bool {
    a[0].is_finite() && a[1].is_finite()
}

/// Angle between two vectors in [0, pi], robust for tiny and nearly parallel vectors.
#[inline]
pub fn angle_between(a: V3, b: V3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

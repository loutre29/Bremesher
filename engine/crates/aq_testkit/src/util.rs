//! Small private helpers.

use aq_core::math;
use aq_core::{AqError, TriMesh, V2, V3};

pub(crate) fn bad(msg: impl Into<String>) -> AqError {
    AqError::invalid_input(msg)
}

pub(crate) fn inv(msg: impl Into<String>) -> AqError {
    AqError::invariant(msg)
}

pub(crate) fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), AqError> {
    if cond { Ok(()) } else { Err(inv(msg())) }
}

/// Rounds x to a multiple of `q` (a power of two), exact.
pub(crate) fn snap_to(x: f64, q: f64) -> f64 {
    (x / q).round() * q + 0.0
}

pub(crate) fn centroid(m: &TriMesh, f: u32) -> V3 {
    let [a, b, c] = m.tri_pos(f);
    math::scale(math::add(math::add(a, b), c), 1.0 / 3.0)
}

/// Area-weighted vertex normals (zero vector for an isolated vertex).
pub(crate) fn vertex_normals(m: &TriMesh) -> Vec<V3> {
    let mut n = vec![[0.0; 3]; m.pos.len()];
    for f in 0..m.tris.len() as u32 {
        let c = m.face_cross(f);
        for &v in &m.tris[f as usize] {
            n[v as usize] = math::add(n[v as usize], c);
        }
    }
    n.iter().map(|x| math::normalize(*x).unwrap_or([0.0; 3])).collect()
}

/// Two unit vectors (t1, t2) with t1 x t2 = n (n unit).
pub(crate) fn tangent_basis(n: V3) -> (V3, V3) {
    let a = if n[0].abs() < 0.6 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let t1 = math::normalize(math::sub(a, math::scale(n, math::dot(a, n)))).unwrap_or([1.0, 0.0, 0.0]);
    let t2 = math::cross(n, t1);
    (t1, t2)
}

#[inline]
pub(crate) fn rot2(a: f64, p: V2) -> V2 {
    let (s, c) = a.sin_cos();
    [c * p[0] - s * p[1], s * p[0] + c * p[1]]
}

//! Cotan Laplacian and lumped mass (DESIGN 6.3).
//!
//! Sign convention: L is positive semi-definite, `L[i][j] = -w_ij` for an edge (i, j) and
//! `L[i][i] = sum_j w_ij`, so `L * 1 = 0` and `x^T L x = sum_edges w_ij (x_i - x_j)^2`.
//!
//! `w_ij = (cot alpha + cot beta) / 2` over the angles opposite the edge (one angle on the
//! boundary), times `edge_scale[he]` (read on the canonical halfedge he of the edge) when
//! given, then clamped to [1e-6, 1e6] times the mean |w| (C14, [`aq_core::tol::clamp_weight`]).
//! The clamp makes every weight positive, so `L + M` is SPD on each component.

use aq_core::alloc::try_vec;
use aq_core::tol::clamp_weight;
use aq_core::{AqError, ErrKind, Frames, NONE, Topology, TriMesh, math};

use crate::sparse::{Csc, Triplets, err};

/// Cotangent of the angle at `o` in the triangle (o, a, b), `dot / |cross|`.
fn cot_at(o: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let u = math::sub(a, o);
    let v = math::sub(b, o);
    math::dot(u, v) / math::norm(math::cross(u, v))
}

/// Clamped edge weights, one per canonical halfedge (0 elsewhere).
pub fn cotan_weights(
    m: &TriMesh,
    t: &Topology,
    edge_scale: Option<&[f64]>,
) -> Result<Vec<f64>, AqError> {
    let nh = 3 * m.tris.len();
    if t.twin.len() != nh {
        return Err(err(
            ErrKind::InvalidInput,
            "cotan_weights: topology does not match the mesh",
        ));
    }
    if let Some(s) = edge_scale
        && s.len() != nh
    {
        return Err(err(
            ErrKind::InvalidInput,
            format!(
                "cotan_weights: edge_scale has {} values for {nh} halfedges",
                s.len()
            ),
        ));
    }
    if m.tris.iter().flatten().any(|&v| v as usize >= m.pos.len()) {
        return Err(err(
            ErrKind::InvalidInput,
            "cotan_weights: vertex index out of range",
        ));
    }
    if nh > NONE as usize || t.twin.iter().any(|&tw| tw != NONE && tw as usize >= nh) {
        return Err(err(
            ErrKind::InvalidInput,
            "cotan_weights: twin halfedge out of range",
        ));
    }
    // Half cotangent of the angle opposite each halfedge (C2: the corner of prev(h)).
    let half_cot = |h: u32| -> f64 {
        let a = m.pos[t.tail(m, h) as usize];
        let b = m.pos[t.head(m, h) as usize];
        let o = m.pos[t.tail(m, Topology::prev(h)) as usize];
        0.5 * cot_at(o, a, b)
    };
    let mut w = try_vec(nh, 0.0f64)?;
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for h in 0..nh as u32 {
        if !t.is_canonical(h) {
            continue;
        }
        let tw = t.twin_of(h);
        let mut x = half_cot(h);
        if tw != NONE {
            x += half_cot(tw);
        }
        if let Some(s) = edge_scale {
            x *= s[h as usize];
        }
        w[h as usize] = x;
        if x.is_finite() {
            sum += x.abs();
            cnt += 1;
        }
    }
    let mean = if cnt > 0 && sum > 0.0 && sum.is_finite() {
        sum / cnt as f64
    } else {
        1.0
    };
    for (h, wh) in w.iter_mut().enumerate() {
        if !t.is_canonical(h as u32) {
            continue;
        }
        // Inf clamps to the upper bound, NaN to the lower one.
        *wh = clamp_weight(*wh, mean);
    }
    Ok(w)
}

/// [`cotan_laplacian`] returning its error (`InvalidInput` for mismatched inputs, `Budget`).
pub fn try_cotan_laplacian(
    m: &TriMesh,
    t: &Topology,
    _fr: &Frames,
    edge_scale: Option<&[f64]>,
) -> Result<Csc, AqError> {
    let w = cotan_weights(m, t, edge_scale)?;
    let nv = m.pos.len();
    let mut tr = Triplets::with_capacity(nv, 2 * w.len() / 3 + nv + 8)?;
    for (h, &wh) in w.iter().enumerate() {
        let h = h as u32;
        if !t.is_canonical(h) {
            continue;
        }
        tr.push(t.tail(m, h) as usize, t.head(m, h) as usize, -wh);
    }
    let mut l = tr.to_csc()?;
    // The diagonal is the sum of the stored off-diagonal weights of its row, so L * 1 = 0
    // up to the rounding of that sum.
    let mut diag = try_vec(nv, 0.0f64)?;
    for j in 0..nv {
        for k in l.col_ptr()[j] + 1..l.col_ptr()[j + 1] {
            let (i, v) = (l.row_idx()[k], l.values()[k]);
            diag[i] -= v;
            diag[j] -= v;
        }
    }
    // The diagonal is stored first in each column: values[col_ptr[j]].
    let mut cp = try_vec(nv, 0usize)?;
    cp.copy_from_slice(&l.col_ptr()[..nv]);
    let vm = l.values_mut();
    for (j, &c) in cp.iter().enumerate() {
        vm[c] = diag[j];
    }
    Ok(l)
}

/// Cotan Laplacian (PSD, clamped weights; see the module documentation), n_verts x n_verts.
/// On an error this returns an empty 0 x 0 matrix; use [`try_cotan_laplacian`] to get it.
pub fn cotan_laplacian(m: &TriMesh, t: &Topology, fr: &Frames, edge_scale: Option<&[f64]>) -> Csc {
    try_cotan_laplacian(m, t, fr, edge_scale).unwrap_or_default()
}

/// Barycentric lumped mass: each vertex gets a third of the area of its faces.
/// A face whose area or indices are invalid contributes nothing. When the n_verts buffer
/// cannot be allocated this returns an empty vector; use [`try_lumped_mass`] to get the error.
pub fn lumped_mass(m: &TriMesh, t: &Topology, fr: &Frames) -> Vec<f64> {
    try_lumped_mass(m, t, fr).unwrap_or_default()
}

/// [`lumped_mass`] returning `Budget` when the buffer cannot be allocated.
pub fn try_lumped_mass(m: &TriMesh, _t: &Topology, fr: &Frames) -> Result<Vec<f64>, AqError> {
    let mut mass = try_vec(m.pos.len(), 0.0f64)?;
    for (f, tri) in m.tris.iter().enumerate() {
        let a = fr.area.get(f).copied().unwrap_or(0.0);
        if !a.is_finite() {
            continue;
        }
        for &v in tri {
            if let Some(x) = mass.get_mut(v as usize) {
                *x += a / 3.0;
            }
        }
    }
    Ok(mass)
}

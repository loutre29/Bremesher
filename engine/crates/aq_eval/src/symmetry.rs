//! Symmetry utilities (DESIGN 6.15, 8.3, 9): `bisect` (keep one side of a plane, report the
//! cut edges for FeatureKind::Symmetry) and `mirror_weld` (mirror a half quad mesh and weld
//! it along the plane, reporting open edges and seam poles).
//!
//! Planes are `[a, b, c, d]` for `a x + b y + c z + d = 0`; the normal is normalized
//! internally, the positive side is `a x + b y + c z + d > 0`.

use std::collections::BTreeMap;

use aq_core::math;
use aq_core::{QuadMesh, TriMesh, V3};
use serde::{Deserialize, Serialize};

use crate::poly::{edge_census, is_irregular, valid_faces};

/// Unit normal and offset of a plane, None when the normal is zero or not finite.
fn unit_plane(plane: [f64; 4]) -> Option<(V3, f64)> {
    let n = [plane[0], plane[1], plane[2]];
    let len = math::norm(n);
    if len.is_nan() || len <= 0.0 || !len.is_finite() || !plane[3].is_finite() {
        return None;
    }
    Some((math::scale(n, 1.0 / len), plane[3] / len))
}

fn signed(n: V3, d: f64, p: V3) -> f64 {
    math::dot(n, p) + d
}

/// Orthogonal projection onto the plane.
fn project(n: V3, d: f64, p: V3) -> V3 {
    math::sub(p, math::scale(n, signed(n, d, p)))
}

/// Keeps the part of m on one side of the plane (the positive side when `keep_positive`).
///
/// * Vertices within `1e-9 * bbox_diag` of the plane count as on it and are projected onto
///   it; triangles crossing the plane are clipped, new vertices are created once per cut
///   edge (interpolated from the lower vertex id, then projected), a clipped quadrilateral
///   is split along its shorter diagonal.
/// * Triangles lying in the plane (all three vertices on it) are dropped; triangles with an
///   out-of-range index are skipped. Unused vertices are removed.
/// * The second result is the list of cut edges for FeatureKind::Symmetry: every boundary
///   edge of the result whose two vertices lie on the plane, as `[tail, head]` of its
///   triangle side, sorted.
///
/// A zero or non-finite plane normal returns m unchanged and no cut edges.
pub fn bisect(m: &TriMesh, plane: [f64; 4], keep_positive: bool) -> (TriMesh, Vec<[u32; 2]>) {
    let Some((n, d)) = unit_plane(plane) else {
        return (m.clone(), Vec::new());
    };
    let nv = m.pos.len();
    let diag = m.bbox_diag();
    let eps = if diag.is_finite() { 1e-9 * diag } else { 0.0 };
    let sgn = if keep_positive { 1.0 } else { -1.0 };
    let s: Vec<f64> = m
        .pos
        .iter()
        .map(|&p| {
            let x = sgn * signed(n, d, p);
            if x.abs() <= eps { 0.0 } else { x }
        })
        .collect();

    let mut out = TriMesh::default();
    let mut on_plane: Vec<bool> = Vec::new();
    let mut new_id = vec![u32::MAX; nv];
    let mut cut_vert: BTreeMap<(u32, u32), u32> = BTreeMap::new();

    for t in &m.tris {
        if !t.iter().all(|&v| (v as usize) < nv) {
            continue;
        }
        let sv = [s[t[0] as usize], s[t[1] as usize], s[t[2] as usize]];
        if sv.iter().all(|&x| x == 0.0) {
            continue;
        }
        let mut poly: Vec<u32> = Vec::with_capacity(4);
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (sa, sb) = (sv[k], sv[(k + 1) % 3]);
            if sa >= 0.0 {
                let id = &mut new_id[a as usize];
                if *id == u32::MAX {
                    *id = out.pos.len() as u32;
                    let p = m.pos[a as usize];
                    out.pos.push(if sa == 0.0 { project(n, d, p) } else { p });
                    on_plane.push(sa == 0.0);
                }
                poly.push(*id);
            }
            if (sa > 0.0 && sb < 0.0) || (sa < 0.0 && sb > 0.0) {
                let key = (a.min(b), a.max(b));
                let id = *cut_vert.entry(key).or_insert_with(|| {
                    let (lo, hi) = (key.0 as usize, key.1 as usize);
                    let w = s[lo] / (s[lo] - s[hi]);
                    let p = project(n, d, math::lerp(m.pos[lo], m.pos[hi], w));
                    out.pos.push(p);
                    on_plane.push(true);
                    (out.pos.len() - 1) as u32
                });
                poly.push(id);
            }
        }
        match poly.len() {
            3 => out.tris.push([poly[0], poly[1], poly[2]]),
            4 => {
                let p: Vec<V3> = poly.iter().map(|&v| out.pos[v as usize]).collect();
                if math::dist2(p[0], p[2]) <= math::dist2(p[1], p[3]) {
                    out.tris.push([poly[0], poly[1], poly[2]]);
                    out.tris.push([poly[0], poly[2], poly[3]]);
                } else {
                    out.tris.push([poly[1], poly[2], poly[3]]);
                    out.tris.push([poly[1], poly[3], poly[0]]);
                }
            }
            _ => {}
        }
    }

    let mut directed: Vec<(u32, u32)> = Vec::with_capacity(3 * out.tris.len());
    for t in &out.tris {
        for k in 0..3 {
            directed.push((t[k], t[(k + 1) % 3]));
        }
    }
    directed.sort_unstable();
    let mut cut: Vec<[u32; 2]> = directed
        .iter()
        .filter(|&&(a, b)| on_plane[a as usize] && on_plane[b as usize] && directed.binary_search(&(b, a)).is_err())
        .map(|&(a, b)| [a, b])
        .collect();
    cut.dedup();
    (out, cut)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WeldReport {
    /// Vertices within tol of the plane, snapped onto it and shared by both halves.
    pub seam_verts: u32,
    /// Boundary edges of the welded mesh (edges used by one face side).
    pub open_edges: u32,
    /// Open edges with both vertices on the seam (a weld that failed).
    pub seam_open_edges: u32,
    /// Seam vertices that are irregular in the welded mesh (quad_stats.py rules).
    pub seam_poles: u32,
    /// Edges used by more than two face sides in the welded mesh.
    pub nonmanifold_edges: u32,
}

/// Mirrors q across the plane and welds the copy to the original.
///
/// Vertices with `|distance| <= tol` are the seam: they are snapped onto the plane and shared.
/// Every other used vertex gets a reflected copy (same vflag). Each valid face gets a
/// mirrored copy with reversed vertex order (same face_comp), except faces lying entirely on
/// the seam. A zero or non-finite plane normal returns q unchanged with its open edges.
pub fn mirror_weld(q: &QuadMesh, plane: [f64; 4], tol: f64) -> (QuadMesh, WeldReport) {
    let faces = valid_faces(q);
    let nv = q.pos.len();
    let Some((n, d)) = unit_plane(plane) else {
        let c = edge_census(nv, &faces);
        let rep = WeldReport {
            open_edges: c.edges.iter().filter(|e| e.1 == 1).count() as u32,
            nonmanifold_edges: c.edges.iter().filter(|e| e.1 > 2).count() as u32,
            ..Default::default()
        };
        return (q.clone(), rep);
    };
    let mut used = vec![false; nv];
    for fv in &faces {
        for &v in *fv {
            used[v as usize] = true;
        }
    }
    let seam: Vec<bool> = q.pos.iter().map(|&p| signed(n, d, p).abs() <= tol).collect();
    let mut out = QuadMesh::new();
    out.pos = q.pos.iter().enumerate().map(|(v, &p)| if seam[v] { project(n, d, p) } else { p }).collect();
    out.vflag = (0..nv).map(|v| q.vflag.get(v).copied().unwrap_or(0)).collect();
    let mut mirror = vec![0u32; nv];
    for v in 0..nv {
        if seam[v] || !used[v] {
            mirror[v] = v as u32;
            continue;
        }
        let p = q.pos[v];
        mirror[v] = out.pos.len() as u32;
        out.pos.push(math::sub(p, math::scale(n, 2.0 * signed(n, d, p))));
        out.vflag.push(out.vflag[v]);
    }
    let mut comps: Vec<u32> = Vec::with_capacity(faces.len());
    let mut fi = 0usize;
    for f in 0..q.n_faces() {
        let (a, b) = (q.face_off[f] as usize, q.face_off[f + 1] as usize);
        let Some(fv) = q.face_verts.get(a..b) else { continue };
        if fv.is_empty() || !fv.iter().all(|&v| (v as usize) < nv) {
            continue;
        }
        let comp = q.face_comp.get(f).copied().unwrap_or(0);
        out.push_face(fv, comp);
        comps.push(comp);
        fi += 1;
    }
    debug_assert_eq!(fi, faces.len());
    for (fv, &comp) in faces.iter().zip(&comps) {
        if fv.iter().all(|&v| seam[v as usize]) {
            continue;
        }
        let mf: Vec<u32> = fv.iter().rev().map(|&v| mirror[v as usize]).collect();
        out.push_face(&mf, comp);
    }

    let ofaces = valid_faces(&out);
    let c = edge_census(out.pos.len(), &ofaces);
    let mut rep = WeldReport {
        seam_verts: (0..nv).filter(|&v| seam[v] && used[v]).count() as u32,
        open_edges: c.edges.iter().filter(|e| e.1 == 1).count() as u32,
        nonmanifold_edges: c.edges.iter().filter(|e| e.1 > 2).count() as u32,
        ..Default::default()
    };
    rep.seam_open_edges =
        c.edges.iter().filter(|&&([a, b], k)| k == 1 && seam.get(a as usize) == Some(&true) && seam.get(b as usize) == Some(&true)).count() as u32;
    rep.seam_poles = (0..nv)
        .filter(|&v| seam[v] && used[v] && c.valence[v] > 0 && is_irregular(c.valence[v], c.boundary[v]))
        .count() as u32;
    (out, rep)
}

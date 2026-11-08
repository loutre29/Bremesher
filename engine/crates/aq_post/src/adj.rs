//! Vertex-face incidence of a polygon mesh (CSR) and boundary vertices.

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::math;
use aq_core::{AqError, QuadMesh, V3};

/// Faces around each vertex: `inc[off[v]..off[v + 1]]` holds `(face, corner index in face)`,
/// sorted by face id.
pub(crate) struct VertFaces {
    pub off: Vec<u32>,
    pub inc: Vec<(u32, u8)>,
    /// True when an edge at the vertex is used by one face only.
    pub is_bnd: Vec<bool>,
}

impl VertFaces {
    /// Requires a mesh that passed `QuadMesh::validate`.
    pub fn build(q: &QuadMesh) -> Result<Self, AqError> {
        let nv = q.n_verts();
        let nf = q.n_faces();
        let mut cnt: Vec<u32> = try_vec(nv + 1, 0)?;
        for f in 0..nf {
            for &v in q.face(f) {
                cnt[v as usize + 1] += 1;
            }
        }
        for v in 0..nv {
            cnt[v + 1] += cnt[v];
        }
        let off = cnt.clone();
        let mut fill = cnt;
        let mut inc: Vec<(u32, u8)> = try_vec(q.face_verts.len(), (0, 0))?;
        for f in 0..nf {
            for (k, &v) in q.face(f).iter().enumerate() {
                let s = &mut fill[v as usize];
                inc[*s as usize] = (f as u32, k as u8);
                *s += 1;
            }
        }
        // Directed edges, sorted, for twin lookups.
        let mut edges: Vec<(u32, u32)> = try_with_capacity(q.face_verts.len())?;
        for f in 0..nf {
            let fv = q.face(f);
            for (k, &v) in fv.iter().enumerate() {
                edges.push((v, fv[(k + 1) % fv.len()]));
            }
        }
        edges.sort_unstable();
        let mut is_bnd: Vec<bool> = try_vec(nv, false)?;
        for &(a, b) in &edges {
            if edges.binary_search(&(b, a)).is_err() {
                is_bnd[a as usize] = true;
                is_bnd[b as usize] = true;
            }
        }
        Ok(VertFaces { off, inc, is_bnd })
    }

    #[inline]
    pub fn of(&self, v: usize) -> &[(u32, u8)] {
        &self.inc[self.off[v] as usize..self.off[v + 1] as usize]
    }
}

/// Polygon area vector (Newell): its norm is twice the area for a planar polygon, its
/// direction the normal. For a quad this is `(p2 - p0) x (p3 - p1)`.
pub(crate) fn newell(pos: &[V3], fv: &[u32]) -> V3 {
    if fv.len() == 4 {
        let [a, b, c, d] = [fv[0], fv[1], fv[2], fv[3]].map(|i| pos[i as usize]);
        return math::cross(math::sub(c, a), math::sub(d, b));
    }
    let mut n = [0.0; 3];
    let o = pos[fv[0] as usize];
    for k in 1..fv.len().saturating_sub(1) {
        let a = math::sub(pos[fv[k] as usize], o);
        let b = math::sub(pos[fv[k + 1] as usize], o);
        n = math::add(n, math::cross(a, b));
    }
    n
}

/// Area and centroid (vertex mean) of a face.
pub(crate) fn area_centroid(pos: &[V3], fv: &[u32]) -> (f64, V3) {
    let mut c = [0.0; 3];
    for &v in fv {
        c = math::add(c, pos[v as usize]);
    }
    let c = math::scale(c, 1.0 / fv.len().max(1) as f64);
    (0.5 * math::norm(newell(pos, fv)), c)
}

/// Corner normal of face `fv` at corner k: `(p_next - p_k) x (p_prev - p_k)`, outward for a
/// CCW face.
#[inline]
pub(crate) fn corner_normal(pos: &[V3], fv: &[u32], k: usize) -> V3 {
    let n = fv.len();
    let p = pos[fv[k] as usize];
    let a = pos[fv[(k + 1) % n] as usize];
    let b = pos[fv[(k + n - 1) % n] as usize];
    math::cross(math::sub(a, p), math::sub(b, p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incidence_and_boundary() {
        // Two quads side by side: 6 vertices, all on the boundary.
        let mut q = QuadMesh::new();
        q.pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        q.vflag = vec![0; 6];
        q.push_face(&[0, 1, 4, 3], 0);
        q.push_face(&[1, 2, 5, 4], 0);
        let vf = VertFaces::build(&q).unwrap_or_else(|_| unreachable!());
        assert_eq!(vf.of(1), &[(0, 1), (1, 0)]);
        assert_eq!(vf.of(4), &[(0, 2), (1, 3)]);
        assert!(vf.is_bnd.iter().all(|&b| b));
        let (a, c) = area_centroid(&q.pos, q.face(0));
        assert!((a - 1.0).abs() < 1e-15);
        assert_eq!(c, [0.5, 0.5, 0.0]);
        assert!(corner_normal(&q.pos, q.face(0), 0)[2] > 0.0);
    }
}

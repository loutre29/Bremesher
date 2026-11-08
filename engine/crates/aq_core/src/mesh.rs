//! C1: meshes. `TriMesh { pos, tris }`, triangles CCW seen from outside,
//! `n_f = normalize((p1-p0) x (p2-p0))`, `NONE = u32::MAX`.

use crate::error::{AqError, ensure};
use crate::math;

pub type V2 = [f64; 2];
pub type V3 = [f64; 3];
/// The "no index" sentinel.
pub const NONE: u32 = u32::MAX;

/// Triangle mesh (C1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriMesh {
    pub pos: Vec<V3>,
    pub tris: Vec<[u32; 3]>,
}

impl TriMesh {
    pub fn n_verts(&self) -> usize {
        self.pos.len()
    }
    pub fn n_faces(&self) -> usize {
        self.tris.len()
    }

    /// The 3 corner positions of face f.
    pub fn tri_pos(&self, f: u32) -> [V3; 3] {
        let t = self.tris[f as usize];
        [self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]]
    }

    /// Unnormalized face normal `(p1-p0) x (p2-p0)` (C1); its length is twice the area.
    pub fn face_cross(&self, f: u32) -> V3 {
        let [p0, p1, p2] = self.tri_pos(f);
        math::cross(math::sub(p1, p0), math::sub(p2, p0))
    }

    /// Unit face normal (C1), `None` for a zero-area face.
    pub fn face_normal(&self, f: u32) -> Option<V3> {
        math::normalize(self.face_cross(f))
    }

    pub fn face_area(&self, f: u32) -> f64 {
        0.5 * math::norm(self.face_cross(f))
    }

    pub fn total_area(&self) -> f64 {
        (0..self.tris.len() as u32).map(|f| self.face_area(f)).sum()
    }

    /// Axis-aligned bounding box (lo, hi); `None` for an empty mesh.
    pub fn bbox(&self) -> Option<(V3, V3)> {
        let first = *self.pos.first()?;
        let mut lo = first;
        let mut hi = first;
        for p in &self.pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Some((lo, hi))
    }

    /// Bounding box diagonal (0 for an empty mesh).
    pub fn bbox_diag(&self) -> f64 {
        self.bbox().map(|(lo, hi)| math::dist(lo, hi)).unwrap_or(0.0)
    }

    /// Structural validation: finite positions, indices in range, no repeated index in a
    /// triangle, sizes addressable by u32 halfedge ids.
    pub fn validate(&self) -> Result<(), AqError> {
        let nv = self.pos.len();
        if nv >= NONE as usize || self.tris.len().saturating_mul(3) >= NONE as usize {
            return Err(AqError::invalid_input("mesh too large for u32 ids"));
        }
        for (i, p) in self.pos.iter().enumerate() {
            if !math::is_finite3(*p) {
                return Err(AqError::invalid_input(format!("vertex {i} is not finite")));
            }
        }
        for (f, t) in self.tris.iter().enumerate() {
            for &v in t {
                if v as usize >= nv {
                    return Err(AqError::invalid_input(format!(
                        "face {f} index {v} out of range ({nv} vertices)"
                    )));
                }
            }
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                return Err(AqError::invalid_input(format!("face {f} has a repeated index")));
            }
        }
        Ok(())
    }
}

/// Polygon mesh in CSR form (quads have 4 vertices).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuadMesh {
    pub pos: Vec<V3>,
    /// n_faces + 1 offsets into `face_verts` (CSR).
    pub face_off: Vec<u32>,
    pub face_verts: Vec<u32>,
    /// Per vertex, DESIGN 8.4.
    pub vflag: Vec<u32>,
    /// Per face.
    pub face_comp: Vec<u32>,
}

impl QuadMesh {
    /// An empty mesh with a valid CSR header.
    pub fn new() -> Self {
        QuadMesh { face_off: vec![0], ..Default::default() }
    }

    pub fn n_faces(&self) -> usize {
        self.face_off.len().saturating_sub(1)
    }

    pub fn n_verts(&self) -> usize {
        self.pos.len()
    }

    /// Vertex ids of face f.
    pub fn face(&self, f: usize) -> &[u32] {
        let a = self.face_off[f] as usize;
        let b = self.face_off[f + 1] as usize;
        &self.face_verts[a..b]
    }

    /// Appends a face; `comp` goes to `face_comp`.
    pub fn push_face(&mut self, verts: &[u32], comp: u32) {
        if self.face_off.is_empty() {
            self.face_off.push(0);
        }
        self.face_verts.extend_from_slice(verts);
        self.face_off.push(self.face_verts.len() as u32);
        self.face_comp.push(comp);
    }

    /// True when every face has 4 vertices.
    pub fn is_all_quads(&self) -> bool {
        (0..self.n_faces()).all(|f| self.face_off[f + 1] - self.face_off[f] == 4)
    }

    /// Validation: CSR header, sizes, finite positions, indices in range, faces with at least
    /// 3 distinct vertices, every directed edge used once and every undirected edge by at most
    /// two faces (edge-manifold and consistently oriented).
    pub fn validate(&self) -> Result<(), AqError> {
        let nv = self.pos.len();
        ensure(!self.face_off.is_empty() && self.face_off[0] == 0, || {
            "QuadMesh.face_off must start with 0".into()
        })?;
        let nf = self.n_faces();
        ensure(*self.face_off.last().unwrap_or(&0) as usize == self.face_verts.len(), || {
            "QuadMesh.face_off last entry must equal face_verts.len()".into()
        })?;
        ensure(self.vflag.len() == nv, || format!("vflag len {} != {nv}", self.vflag.len()))?;
        ensure(self.face_comp.len() == nf, || {
            format!("face_comp len {} != {nf}", self.face_comp.len())
        })?;
        for (i, p) in self.pos.iter().enumerate() {
            ensure(math::is_finite3(*p), || format!("vertex {i} is not finite"))?;
        }
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(self.face_verts.len());
        for f in 0..nf {
            let a = self.face_off[f];
            let b = self.face_off[f + 1];
            ensure(b >= a + 3, || format!("face {f} has fewer than 3 vertices"))?;
            let fv = &self.face_verts[a as usize..b as usize];
            for (i, &v) in fv.iter().enumerate() {
                ensure((v as usize) < nv, || format!("face {f} index {v} out of range"))?;
                ensure(!fv[..i].contains(&v), || format!("face {f} repeats vertex {v}"))?;
                edges.push((v, fv[(i + 1) % fv.len()]));
            }
        }
        edges.sort_unstable();
        for w in edges.windows(2) {
            ensure(w[0] != w[1], || {
                format!("directed edge {:?} used twice (non-manifold or flipped)", w[0])
            })?;
        }
        Ok(())
    }
}

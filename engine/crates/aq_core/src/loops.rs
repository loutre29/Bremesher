//! Surface points and closed streamline loop candidates (DESIGN 5 `aq_core::loops`, 6.7).

use crate::error::{AqError, ensure};
use crate::math;
use crate::mesh::{NONE, TriMesh, V3};
use crate::topo::Topology;

/// A point on a face, by barycentric coordinates relative to `tris[face]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfacePoint {
    pub face: u32,
    pub bary: V3,
}

impl SurfacePoint {
    /// 3D position on mesh m.
    pub fn position(&self, m: &TriMesh) -> V3 {
        let [a, b, c] = m.tri_pos(self.face);
        math::bary_point(a, b, c, self.bary)
    }

    /// face < n_faces, bary finite, each >= -1e-9, sum 1 within 1e-9.
    pub fn validate(&self, n_faces: usize) -> Result<(), AqError> {
        ensure((self.face as usize) < n_faces, || format!("SurfacePoint face {} out of range", self.face))?;
        let b = self.bary;
        ensure(
            math::is_finite3(b)
                && b.iter().all(|&x| x >= -1e-9)
                && (b[0] + b[1] + b[2] - 1.0).abs() <= 1e-9,
            || format!("SurfacePoint bary invalid: {b:?}"),
        )
    }
}

/// A closed streamline found by S5.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopCand {
    /// Dual cycle: halfedges crossed in order, ends back in `start_face`.
    pub hes: Vec<u32>,
    pub start_face: u32,
    pub seed: SurfacePoint,
    /// Raw-theta branch (0..3) tangent to the loop at the seed.
    pub branch: u8,
    pub length: f64,
    /// Wraps a tube (length < 1.5 * 2pi / mean|k1|).
    pub ring: bool,
}

impl LoopCand {
    /// Checks: non-empty, every crossed halfedge interior, consecutive in the dual
    /// (`face(twin(hes[i])) == face(hes[i+1])`), starting in and returning to `start_face`,
    /// seed valid and in `start_face`, branch < 4, length finite and positive.
    pub fn validate(&self, m: &TriMesh, t: &Topology) -> Result<(), AqError> {
        let nh = 3 * m.tris.len();
        ensure(!self.hes.is_empty(), || "LoopCand is empty".into())?;
        ensure(self.branch < 4, || "LoopCand branch >= 4".into())?;
        ensure(self.length.is_finite() && self.length > 0.0, || "LoopCand length".into())?;
        self.seed.validate(m.tris.len())?;
        ensure(self.seed.face == self.start_face, || "LoopCand seed not in start_face".into())?;
        let mut face = self.start_face;
        for (i, &h) in self.hes.iter().enumerate() {
            ensure((h as usize) < nh, || format!("LoopCand he {h} out of range"))?;
            ensure(h / 3 == face, || format!("LoopCand dual path broken at position {i}"))?;
            let tw = t.twin[h as usize];
            ensure(tw != NONE, || format!("LoopCand crosses boundary he {h}"))?;
            face = tw / 3;
        }
        ensure(face == self.start_face, || "LoopCand does not return to start_face".into())
    }
}

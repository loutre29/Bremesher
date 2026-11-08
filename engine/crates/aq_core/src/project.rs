//! Local projection with a normal check (DESIGN 5 `aq_core::project`).
//!
//! `project` walks from the hint face over a growing neighbourhood (at most 3 vertex rings).
//! The walk converges when the closest face of the ring-r neighbourhood already lies in the
//! ring-(r-1) neighbourhood (a local minimum surrounded by searched faces). The result is
//! accepted only if `n_src . n_ref > 0.5` (n_src = unit normal of the closest face, n_ref
//! normalized). Otherwise, or when the walk does not converge within 3 rings, the BVH gives
//! the global closest point, accepted with the same test; otherwise `None`.

use crate::bvh::{Bvh, closest_on_triangle};
use crate::frames::Frames;
use crate::loops::SurfacePoint;
use crate::math;
use crate::mesh::{NONE, TriMesh, V3};
use crate::topo::Topology;

/// Number of vertex rings the walk may grow.
pub const MAX_RINGS: usize = 3;
/// Normal agreement threshold.
pub const NORMAL_DOT_MIN: f64 = 0.5;

/// Projection onto a source mesh.
#[derive(Clone, Copy, Debug)]
pub struct Projector<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub frames: &'a Frames,
    pub bvh: &'a Bvh,
}

#[derive(Clone, Copy)]
struct Best {
    face: u32,
    bary: V3,
    pos: V3,
    d2: f64,
}

impl<'a> Projector<'a> {
    pub fn new(mesh: &'a TriMesh, topo: &'a Topology, frames: &'a Frames, bvh: &'a Bvh) -> Self {
        Projector { mesh, topo, frames, bvh }
    }

    fn eval(&self, p: V3, faces: &[u32], best: &mut Option<Best>) {
        for &f in faces {
            let [a, b, c] = self.mesh.tri_pos(f);
            let (bary, q) = closest_on_triangle(p, a, b, c);
            let d2 = math::dist2(p, q);
            let better = match best {
                None => true,
                Some(bb) => d2 < bb.d2 || (d2 == bb.d2 && f < bb.face),
            };
            if better {
                *best = Some(Best { face: f, bary, pos: q, d2 });
            }
        }
    }

    fn normal_ok(&self, f: u32, n_ref: Option<V3>) -> bool {
        match (n_ref, self.frames.n.get(f as usize)) {
            (Some(nr), Some(&nf)) => math::dot(nf, nr) > NORMAL_DOT_MIN,
            _ => false,
        }
    }

    /// The walk only: closest point in the converged neighbourhood of `hint_face`, or `None`
    /// when the walk does not converge within [`MAX_RINGS`] rings. No normal test.
    pub fn walk(&self, p: V3, hint_face: u32) -> Option<(SurfacePoint, V3)> {
        let nf = self.mesh.tris.len();
        if hint_face as usize >= nf {
            return None;
        }
        let mut region: Vec<u32> = vec![hint_face]; // sorted
        let mut frontier: Vec<u32> = vec![hint_face];
        let mut best: Option<Best> = None;
        self.eval(p, &frontier, &mut best);
        for _ in 0..MAX_RINGS {
            let mut cand: Vec<u32> = Vec::new();
            for &f in &frontier {
                for &v in &self.mesh.tris[f as usize] {
                    for h in self.topo.fan_out(v) {
                        cand.push(h / 3);
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();
            cand.retain(|f| region.binary_search(f).is_err());
            if cand.is_empty() {
                // The whole connected neighbourhood has been searched.
                return best.map(|b| (SurfacePoint { face: b.face, bary: b.bary }, b.pos));
            }
            let prev = best;
            let mut nb: Option<Best> = None;
            self.eval(p, &cand, &mut nb);
            let improved = match (prev, nb) {
                (Some(a), Some(b)) => b.d2 < a.d2,
                (None, Some(_)) => true,
                _ => false,
            };
            if !improved {
                return prev.map(|b| (SurfacePoint { face: b.face, bary: b.bary }, b.pos));
            }
            best = nb;
            region.extend_from_slice(&cand);
            region.sort_unstable();
            frontier = cand;
        }
        None
    }

    /// Projects p (see the module doc). `n_ref` need not be normalized; a zero or
    /// non-finite `n_ref` fails the normal test.
    pub fn project(&self, p: V3, hint_face: u32, n_ref: V3) -> Option<(SurfacePoint, V3)> {
        if !math::is_finite3(p) {
            return None;
        }
        let nr = math::normalize(n_ref);
        if let Some((sp, q)) = self.walk(p, hint_face)
            && self.normal_ok(sp.face, nr)
        {
            return Some((sp, q));
        }
        let (sp, q, _) = self.bvh.closest(p);
        if sp.face != NONE && self.normal_ok(sp.face, nr) {
            return Some((sp, q));
        }
        None
    }
}

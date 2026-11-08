//! Layout contract (DESIGN 5 `aq_core::layout`, 6.10).

use crate::cross::Transition;
use crate::error::{AqError, ensure};
use crate::mesh::{NONE, TriMesh, V2};
use crate::tmesh::TMesh;
use crate::topo::Topology;

/// Rectangle mismatch statistics (DESIGN 6.10 "Health"): per patch
/// `m_P = max(|l0 - l2| / max(l0, l2, 1), |l1 - l3| / max(l1, l3, 1))`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutHealth {
    /// m_P per patch.
    pub per_patch: Vec<f64>,
    pub mean: f64,
    pub max: f64,
    /// Fraction of patches with m_P > 0.1.
    pub frac_above: f64,
}

impl LayoutHealth {
    /// Threshold of `frac_above`.
    pub const BAD: f64 = 0.1;

    /// Builds the statistics from per-patch side lengths (l0, l1, l2, l3).
    pub fn from_sides(sides: &[[f64; 4]]) -> Self {
        let per_patch: Vec<f64> = sides
            .iter()
            .map(|l| {
                let a = (l[0] - l[2]).abs() / l[0].max(l[2]).max(1.0);
                let b = (l[1] - l[3]).abs() / l[1].max(l[3]).max(1.0);
                a.max(b)
            })
            .collect();
        let n = per_patch.len();
        let mean = if n == 0 { 0.0 } else { per_patch.iter().sum::<f64>() / n as f64 };
        let max = per_patch.iter().copied().fold(0.0, f64::max);
        let bad = per_patch.iter().filter(|&&m| m > Self::BAD).count();
        let frac_above = if n == 0 { 0.0 } else { bad as f64 / n as f64 };
        LayoutHealth { per_patch, mean, max, frac_above }
    }

    pub fn validate(&self) -> Result<(), AqError> {
        ensure(
            self.mean.is_finite()
                && self.max.is_finite()
                && (0.0..=1.0).contains(&self.frac_above)
                && self.per_patch.iter().all(|x| x.is_finite() && *x >= 0.0),
            || "LayoutHealth invalid".into(),
        )
    }
}

/// S8 output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout {
    /// W' = proxy split along all arcs.
    pub wmesh: TriMesh,
    pub wtopo: Topology,
    /// Param restricted to W' (exact).
    pub wuv: Vec<[V2; 3]>,
    pub wtrans: Vec<Transition>,
    /// W' face -> proxy face.
    pub parent_face: Vec<u32>,
    pub tmesh: TMesh,
    /// TNode -> W' vertex.
    pub node_vert: Vec<u32>,
    /// W' vertex chain `ends[0] -> ends[1]`, per arc.
    pub arc_verts: Vec<Vec<u32>>,
    /// Cumulative parametric length along `arc_verts`.
    pub arc_s: Vec<Vec<f64>>,
    pub patch_faces: Vec<Vec<u32>>,
    pub patch_chart_face: Vec<u32>,
    /// W' vertex -> feature chain id or NONE.
    pub vert_chain: Vec<u32>,
    pub health: LayoutHealth,
}

impl Layout {
    /// Checks: W' mesh and topology, sizes, `parent_face < n_proxy_faces`, T-mesh, node and arc
    /// vertex maps (arc chains start and end at their node vertices and are W' edge chains),
    /// `arc_s` starting at 0, non-decreasing, ending at the arc length within 1e-9 relative,
    /// patch faces partitioning W' with the chart face inside, health.
    pub fn validate(&self, n_proxy_faces: usize) -> Result<(), AqError> {
        let m = &self.wmesh;
        m.validate()?;
        self.wtopo.validate(m)?;
        let nf = m.tris.len();
        let nv = m.pos.len();
        ensure(self.wuv.len() == nf && self.wtrans.len() == 3 * nf && self.parent_face.len() == nf, || {
            "Layout per-face sizes".into()
        })?;
        ensure(self.vert_chain.len() == nv, || "Layout.vert_chain size".into())?;
        for &p in &self.parent_face {
            ensure((p as usize) < n_proxy_faces, || format!("parent_face {p} out of range"))?;
        }
        for tr in &self.wtrans {
            tr.validate()?;
        }
        let t = &self.tmesh;
        t.validate()?;
        ensure(self.node_vert.len() == t.nodes.len(), || "Layout.node_vert size".into())?;
        ensure(self.arc_verts.len() == t.arcs.len() && self.arc_s.len() == t.arcs.len(), || {
            "Layout arc sizes".into()
        })?;
        for &v in &self.node_vert {
            ensure((v as usize) < nv, || format!("node vertex {v} out of range"))?;
        }
        for (a, (vs, ss)) in self.arc_verts.iter().zip(&self.arc_s).enumerate() {
            ensure(vs.len() >= 2 && vs.len() == ss.len(), || format!("arc {a} vertex chain size"))?;
            let e = t.arcs[a].ends;
            let (Some(&f), Some(&l)) = (vs.first(), vs.last()) else {
                return Err(AqError::invariant("empty arc chain"));
            };
            ensure(f == self.node_vert[e[0] as usize] && l == self.node_vert[e[1] as usize], || {
                format!("arc {a} chain does not join its node vertices")
            })?;
            for w in vs.windows(2) {
                ensure((w[0] as usize) < nv && (w[1] as usize) < nv, || format!("arc {a} vertex out of range"))?;
                ensure(self.wtopo.find_he(m, w[0], w[1]).is_some() || self.wtopo.find_he(m, w[1], w[0]).is_some(), || {
                    format!("arc {a} step {}-{} is not a W' edge", w[0], w[1])
                })?;
            }
            ensure(ss[0] == 0.0, || format!("arc {a} arc_s does not start at 0"))?;
            for w in ss.windows(2) {
                ensure(w[1] >= w[0] && w[1].is_finite(), || format!("arc {a} arc_s decreasing"))?;
            }
            let len = t.arcs[a].len;
            let last = ss[ss.len() - 1];
            ensure((last - len).abs() <= 1e-9 * len.max(1.0), || {
                format!("arc {a} arc_s ends at {last}, len {len}")
            })?;
        }
        ensure(self.patch_faces.len() == t.patches.len() && self.patch_chart_face.len() == t.patches.len(), || {
            "Layout patch sizes".into()
        })?;
        let mut owner = vec![NONE; nf];
        for (p, fs) in self.patch_faces.iter().enumerate() {
            for &f in fs {
                ensure((f as usize) < nf, || format!("patch {p} face {f} out of range"))?;
                ensure(owner[f as usize] == NONE, || format!("face {f} in two patches"))?;
                owner[f as usize] = p as u32;
            }
            let cf = self.patch_chart_face[p];
            ensure((cf as usize) < nf && owner[cf as usize] == p as u32, || {
                format!("patch {p} chart face not in the patch")
            })?;
        }
        ensure(owner.iter().all(|&o| o != NONE), || "a W' face belongs to no patch".into())?;
        self.health.validate()
    }
}

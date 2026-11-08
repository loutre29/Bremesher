//! Cut graph and combing (DESIGN 5 `aq_core::cut`, 6.8, C12).

use crate::cross::{CrossField, comb_matching, comb_theta};
use crate::error::{AqError, ensure};
use crate::mesh::{NONE, TriMesh};
use crate::topo::Topology;

/// A maximal path of cut edges between two nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CutArc {
    /// Cut halfedges of side A in path order, from `nodes[0]` to `nodes[1]`. Side A is the
    /// side whose faces lie on the left of the walk (each halfedge's own face).
    pub hes: Vec<u32>,
    pub nodes: [u32; 2],
    /// Combed rotation of side A (constant along the arc).
    pub r: u8,
}

/// S6 output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CutGraph {
    /// Per halfedge (both halves); boundary halfedges false.
    pub is_cut: Vec<bool>,
    /// Per halfedge: (arc id, +1 side A / -1 side B), or (NONE, 0).
    pub he_arc: Vec<(u32, i8)>,
    pub arcs: Vec<CutArc>,
    /// Cut-graph vertices of degree != 2, cones, cut/boundary junctions.
    pub node_verts: Vec<u32>,
    /// Per face, C12 `theta + c pi/2`.
    pub theta_combed: Vec<f64>,
    /// Per halfedge, combed matching (0 off the cut).
    pub r: Vec<u8>,
    /// Per face, the comb offset c_f in 0..4.
    pub comb_offset: Vec<u8>,
}

impl CutGraph {
    /// Checks sizes; cut flags symmetric and false on the boundary; `r[twin] = (4-r)%4`, r < 4,
    /// r = 0 off the cut; arcs consecutive from `nodes[0]` to `nodes[1]`, made of cut
    /// halfedges with `he_arc` (a, +1) and twins (a, -1), r constant equal to `arc.r`; every
    /// cut halfedge on exactly one arc side. With `cf`: `r = (p + c_g - c_f) mod 4` and
    /// `theta_combed = theta + c pi/2` bitwise.
    pub fn validate(&self, m: &TriMesh, t: &Topology, cf: Option<&CrossField>) -> Result<(), AqError> {
        let nf = m.tris.len();
        let nh = 3 * nf;
        ensure(
            self.is_cut.len() == nh
                && self.he_arc.len() == nh
                && self.r.len() == nh
                && self.theta_combed.len() == nf
                && self.comb_offset.len() == nf,
            || "CutGraph sizes".into(),
        )?;
        for (f, &c) in self.comb_offset.iter().enumerate() {
            ensure(c < 4, || format!("comb_offset[{f}] >= 4"))?;
        }
        for h in 0..nh {
            let tw = t.twin[h];
            ensure(self.r[h] < 4, || format!("r[{h}] >= 4"))?;
            if tw == NONE {
                ensure(!self.is_cut[h] && self.r[h] == 0, || format!("boundary he {h} is cut"))?;
                continue;
            }
            let tw = tw as usize;
            ensure(self.is_cut[tw] == self.is_cut[h], || format!("cut flag asymmetric at {h}"))?;
            ensure(self.r[tw] == (4 - self.r[h]) % 4, || format!("r[twin] != (4-r)%4 at {h}"))?;
            if !self.is_cut[h] {
                ensure(self.r[h] == 0, || format!("r != 0 off the cut at {h}"))?;
            }
            if let Some(cf) = cf {
                let (f, g) = (h / 3, tw / 3);
                let want = comb_matching(cf.p[h], self.comb_offset[f], self.comb_offset[g]);
                ensure(self.r[h] == want, || format!("r[{h}] != (p + c_g - c_f) mod 4"))?;
            }
        }
        if let Some(cf) = cf {
            ensure(cf.theta.len() == nf, || "CrossField size".into())?;
            for f in 0..nf {
                let want = comb_theta(cf.theta[f], self.comb_offset[f]);
                ensure(self.theta_combed[f].to_bits() == want.to_bits(), || {
                    format!("theta_combed[{f}] != theta + c pi/2")
                })?;
            }
        }
        let mut seen = vec![false; nh];
        for (ai, a) in self.arcs.iter().enumerate() {
            ensure(!a.hes.is_empty(), || format!("cut arc {ai} is empty"))?;
            ensure(a.r < 4, || format!("cut arc {ai} r >= 4"))?;
            for (i, &h) in a.hes.iter().enumerate() {
                ensure((h as usize) < nh && self.is_cut[h as usize], || {
                    format!("cut arc {ai}: he {h} is not cut")
                })?;
                let tw = t.twin[h as usize];
                ensure(tw != NONE, || format!("cut arc {ai}: he {h} on the boundary"))?;
                ensure(self.he_arc[h as usize] == (ai as u32, 1), || {
                    format!("he_arc of side A he {h} != ({ai}, +1)")
                })?;
                ensure(self.he_arc[tw as usize] == (ai as u32, -1), || {
                    format!("he_arc of side B he {tw} != ({ai}, -1)")
                })?;
                ensure(self.r[h as usize] == a.r, || format!("r not constant along arc {ai}"))?;
                ensure(!seen[h as usize] && !seen[tw as usize], || format!("he {h} in two arcs"))?;
                seen[h as usize] = true;
                seen[tw as usize] = true;
                if i + 1 < a.hes.len() {
                    ensure(t.head(m, h) == t.tail(m, a.hes[i + 1]), || {
                        format!("cut arc {ai} broken after position {i}")
                    })?;
                }
            }
            let (Some(&first), Some(&last)) = (a.hes.first(), a.hes.last()) else {
                return Err(AqError::invariant("empty cut arc"));
            };
            ensure(t.tail(m, first) == a.nodes[0] && t.head(m, last) == a.nodes[1], || {
                format!("cut arc {ai} endpoints differ from its nodes")
            })?;
        }
        for (h, &sn) in seen.iter().enumerate() {
            if self.is_cut[h] {
                ensure(sn, || format!("cut he {h} belongs to no arc"))?;
            } else {
                ensure(self.he_arc[h] == (NONE, 0), || format!("he_arc set on non-cut he {h}"))?;
            }
        }
        for &v in &self.node_verts {
            ensure((v as usize) < m.pos.len(), || format!("cut node {v} out of range"))?;
        }
        Ok(())
    }
}

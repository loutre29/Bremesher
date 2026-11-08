//! Mesh-free T-mesh and combinatorial quantized quad mesh (DESIGN 5 `aq_core::tmesh`,
//! 6.10, 6.11). Used by aq_layout, aq_quantize and aq_patch.

use crate::error::{AqError, ensure};
use crate::mesh::NONE;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum NodeKind {
    #[default]
    Regular = 0,
    Cone = 1,
    FeatureCorner = 2,
    BoundaryCorner = 3,
    ChainEnd = 4,
    Seed = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ArcKind {
    #[default]
    Separatrix = 0,
    SeedTrace = 1,
    Loop = 2,
    Feature = 3,
    Boundary = 4,
    Symmetry = 5,
    Guide = 6,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TNode {
    pub kind: NodeKind,
    pub index4: i8,
    /// Feature chain id or NONE.
    pub chain: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TArc {
    pub ends: [u32; 2],
    /// Parametric length (sum of |delta varying coordinate|).
    pub len: f64,
    pub kind: ArcKind,
    /// Lower bound on the quantized length.
    pub lb: u32,
    pub weight: f64,
    pub wrap: bool,
    /// Feature chain id or NONE.
    pub chain: u32,
}

/// A 4-sided patch. Each side is a list of (arc, reversed); sides are CCW: side0 runs +u in
/// the patch chart, side1 +v, side2 -u, side3 -v.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TPatch {
    pub sides: [Vec<(u32, bool)>; 4],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TMesh {
    pub nodes: Vec<TNode>,
    pub arcs: Vec<TArc>,
    pub patches: Vec<TPatch>,
    /// Closed arc cycles that must keep >= 3 segments.
    pub rings: Vec<Vec<(u32, bool)>>,
    pub chi: i32,
}

impl TMesh {
    /// Start and end node of arc `a` traversed with the given reversal flag.
    pub fn arc_walk(&self, a: u32, reversed: bool) -> (u32, u32) {
        let e = self.arcs[a as usize].ends;
        if reversed { (e[1], e[0]) } else { (e[0], e[1]) }
    }

    /// Number of occurrences of each arc on patch sides.
    pub fn arc_occurrences(&self) -> Vec<u32> {
        let mut occ = vec![0u32; self.arcs.len()];
        for p in &self.patches {
            for s in &p.sides {
                for &(a, _) in s {
                    if let Some(o) = occ.get_mut(a as usize) {
                        *o += 1;
                    }
                }
            }
        }
        occ
    }

    /// The 4 corner nodes of patch `p` (start node of each side).
    pub fn patch_corners(&self, p: u32) -> [u32; 4] {
        let pa = &self.patches[p as usize];
        let mut c = [NONE; 4];
        for (ck, side) in c.iter_mut().zip(&pa.sides) {
            if let Some(&(a, r)) = side.first() {
                *ck = self.arc_walk(a, r).0;
            }
        }
        c
    }

    /// Checks: indices in range; arc lengths and weights finite and >= 0; every side
    /// non-empty and connected (consecutive arcs share nodes, side k ends where side k+1
    /// starts); every arc occurs 1 or 2 times on patch sides, twice with opposite traversal
    /// directions; rings closed; Euler `nodes - arcs + patches == chi`.
    pub fn validate(&self) -> Result<(), AqError> {
        let nn = self.nodes.len() as u32;
        let na = self.arcs.len() as u32;
        for (i, n) in self.nodes.iter().enumerate() {
            ensure((-4..=4).contains(&n.index4), || format!("TNode {i} index4 {}", n.index4))?;
        }
        for (i, a) in self.arcs.iter().enumerate() {
            ensure(a.ends[0] < nn && a.ends[1] < nn, || format!("TArc {i} ends out of range"))?;
            ensure(a.len.is_finite() && a.len >= 0.0, || format!("TArc {i} len {}", a.len))?;
            ensure(a.weight.is_finite() && a.weight >= 0.0, || format!("TArc {i} weight"))?;
        }
        let mut dir_count = vec![[0u32; 2]; na as usize];
        for (pi, p) in self.patches.iter().enumerate() {
            let mut first_start = NONE;
            let mut prev_end = NONE;
            for (k, s) in p.sides.iter().enumerate() {
                ensure(!s.is_empty(), || format!("patch {pi} side {k} is empty"))?;
                for &(a, r) in s {
                    ensure(a < na, || format!("patch {pi} arc {a} out of range"))?;
                    dir_count[a as usize][r as usize] += 1;
                    let (st, en) = self.arc_walk(a, r);
                    if first_start == NONE {
                        first_start = st;
                    } else {
                        ensure(st == prev_end, || format!("patch {pi} side {k} disconnected"))?;
                    }
                    prev_end = en;
                }
            }
            ensure(prev_end == first_start, || format!("patch {pi} boundary does not close"))?;
        }
        for (a, c) in dir_count.iter().enumerate() {
            let tot = c[0] + c[1];
            ensure(tot == 1 || (c[0] == 1 && c[1] == 1), || {
                format!("arc {a} occurs {:?} times (forward, reversed) on patch sides", c)
            })?;
        }
        for (ri, ring) in self.rings.iter().enumerate() {
            ensure(!ring.is_empty(), || format!("ring {ri} empty"))?;
            for (i, &(a, r)) in ring.iter().enumerate() {
                ensure(a < na, || format!("ring {ri} arc out of range"))?;
                let (_, en) = self.arc_walk(a, r);
                let (b, rb) = ring[(i + 1) % ring.len()];
                ensure(b < na && self.arc_walk(b, rb).0 == en, || format!("ring {ri} not closed"))?;
            }
        }
        let euler = nn as i64 - na as i64 + self.patches.len() as i64;
        ensure(euler == self.chi as i64, || format!("T-mesh Euler {euler} != chi {}", self.chi))
    }
}

/// Combinatorial quantized quad mesh (S9).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QComb {
    pub n_verts: u32,
    /// Per TNode.
    pub node_id: Vec<u32>,
    /// Per arc, n_a + 1 ids from `ends[0]` to `ends[1]`.
    pub arc_ids: Vec<Vec<u32>>,
    /// W, H per patch (0 for a glued zero patch).
    pub patch_dims: Vec<[u32; 2]>,
    /// (W+1)*(H+1) ids per patch, row-major, `j*(W+1)+i`.
    pub patch_ids: Vec<Vec<u32>>,
    pub quads: Vec<[u32; 4]>,
    pub quad_patch: Vec<u32>,
}

impl QComb {
    /// Checks against its T-mesh: sizes, ids in range, arc id chains start and end at their
    /// node ids, patch grids sized (W+1)(H+1) (or empty for a zero patch), quads with 4
    /// distinct vertices, each directed quad edge used once (manifold, oriented), and Euler
    /// `n_verts - edges + quads == chi`.
    pub fn validate(&self, t: &TMesh) -> Result<(), AqError> {
        let nv = self.n_verts;
        ensure(self.node_id.len() == t.nodes.len(), || "QComb.node_id size".into())?;
        ensure(self.arc_ids.len() == t.arcs.len(), || "QComb.arc_ids size".into())?;
        ensure(self.patch_dims.len() == t.patches.len() && self.patch_ids.len() == t.patches.len(), || {
            "QComb patch sizes".into()
        })?;
        ensure(self.quad_patch.len() == self.quads.len(), || "QComb.quad_patch size".into())?;
        for &i in &self.node_id {
            ensure(i < nv, || format!("QComb node id {i} out of range"))?;
        }
        for (a, ids) in self.arc_ids.iter().enumerate() {
            ensure(!ids.is_empty(), || format!("QComb arc {a} has no ids"))?;
            ensure(ids.iter().all(|&i| i < nv), || format!("QComb arc {a} id out of range"))?;
            let e = t.arcs[a].ends;
            let (Some(&f), Some(&l)) = (ids.first(), ids.last()) else {
                return Err(AqError::invariant("empty arc ids"));
            };
            ensure(f == self.node_id[e[0] as usize] && l == self.node_id[e[1] as usize], || {
                format!("QComb arc {a} ids do not start/end at its node ids")
            })?;
        }
        for (p, (d, ids)) in self.patch_dims.iter().zip(&self.patch_ids).enumerate() {
            let want = (d[0] as usize + 1) * (d[1] as usize + 1);
            ensure(ids.len() == want || (ids.is_empty() && (d[0] == 0 || d[1] == 0)), || {
                format!("QComb patch {p} has {} ids for dims {d:?}", ids.len())
            })?;
            ensure(ids.iter().all(|&i| i < nv), || format!("QComb patch {p} id out of range"))?;
        }
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(4 * self.quads.len());
        for (qi, q) in self.quads.iter().enumerate() {
            ensure((self.quad_patch[qi] as usize) < t.patches.len(), || format!("quad {qi} patch"))?;
            for k in 0..4 {
                ensure(q[k] < nv, || format!("quad {qi} id out of range"))?;
                ensure(!q[..k].contains(&q[k]), || format!("quad {qi} repeats a vertex"))?;
                edges.push((q[k], q[(k + 1) % 4]));
            }
        }
        edges.sort_unstable();
        for w in edges.windows(2) {
            ensure(w[0] != w[1], || format!("QComb directed edge {:?} used twice", w[0]))?;
        }
        let mut und: Vec<(u32, u32)> = edges.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        und.sort_unstable();
        und.dedup();
        let euler = nv as i64 - und.len() as i64 + self.quads.len() as i64;
        ensure(euler == t.chi as i64, || format!("QComb Euler {euler} != chi {}", t.chi))
    }
}

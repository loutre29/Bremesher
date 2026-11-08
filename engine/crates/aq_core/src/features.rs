//! Feature chains (DESIGN 5 `aq_core::features`, 6.4).

use crate::error::{AqError, ensure};
use crate::mesh::{NONE, TriMesh};
use crate::topo::Topology;

/// Feature kind. Values match `AqMeshIn.feat_kind` (DESIGN 8.1).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum FeatureKind {
    #[default]
    None = 0,
    Hard = 1,
    Guide = 2,
    GuideHard = 3,
    Boundary = 4,
    Symmetry = 5,
}

impl FeatureKind {
    /// Inverse of `kind as u8`.
    pub fn from_u8(x: u8) -> Option<Self> {
        match x {
            0 => Some(FeatureKind::None),
            1 => Some(FeatureKind::Hard),
            2 => Some(FeatureKind::Guide),
            3 => Some(FeatureKind::GuideHard),
            4 => Some(FeatureKind::Boundary),
            5 => Some(FeatureKind::Symmetry),
            _ => None,
        }
    }
}

/// A chain of consecutive feature halfedges.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Chain {
    /// Consecutive halfedges, `head(hes[i]) == tail(hes[i+1])`; boundary chains use the
    /// boundary halfedges.
    pub hes: Vec<u32>,
    pub kind: FeatureKind,
    pub closed: bool,
    /// Indices i into `hes`: a segment starts at `tail(hes[i])`; sorted; empty for a closed
    /// chain with no corner.
    pub corners: Vec<u32>,
}

/// Per-halfedge feature labels and the chains.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureSet {
    /// Per halfedge; both halves of an interior feature edge carry the kind.
    pub he_kind: Vec<FeatureKind>,
    /// Per halfedge: chain id or NONE.
    pub he_chain: Vec<u32>,
    pub chains: Vec<Chain>,
    /// Per vertex: feature or boundary corner vertex.
    pub vert_corner: Vec<bool>,
}

impl FeatureSet {
    /// A feature set with no chains: boundary halfedges get `Boundary`, all others `None`.
    pub fn new(m: &TriMesh, t: &Topology) -> Self {
        let nh = 3 * m.tris.len();
        let he_kind = (0..nh)
            .map(|h| {
                if t.twin.get(h).copied().unwrap_or(NONE) == NONE {
                    FeatureKind::Boundary
                } else {
                    FeatureKind::None
                }
            })
            .collect();
        FeatureSet {
            he_kind,
            he_chain: vec![NONE; nh],
            chains: Vec::new(),
            vert_corner: vec![false; m.pos.len()],
        }
    }

    /// True when halfedge h lies on a chain.
    pub fn is_chain_he(&self, h: u32) -> bool {
        self.he_chain.get(h as usize).is_some_and(|&c| c != NONE)
    }

    /// Checks sizes; both halves of an interior feature edge carry the same kind; boundary
    /// halfedges carry Boundary or Symmetry; chains are consecutive (closed chains close),
    /// non-empty, of a kind other than None, with sorted in-range corners; every chain
    /// halfedge has `he_kind == chain.kind` and `he_chain == chain id`, and every halfedge
    /// with a chain id is in that chain (itself or its twin).
    pub fn validate(&self, m: &TriMesh, t: &Topology) -> Result<(), AqError> {
        let nh = 3 * m.tris.len();
        ensure(self.he_kind.len() == nh && self.he_chain.len() == nh, || {
            "FeatureSet per-halfedge sizes".into()
        })?;
        ensure(self.vert_corner.len() == m.pos.len(), || "FeatureSet.vert_corner size".into())?;
        for h in 0..nh {
            let tw = t.twin[h];
            if tw == NONE {
                ensure(
                    matches!(self.he_kind[h], FeatureKind::Boundary | FeatureKind::Symmetry),
                    || format!("boundary he {h} has kind {:?}", self.he_kind[h]),
                )?;
            } else {
                ensure(self.he_kind[tw as usize] == self.he_kind[h], || {
                    format!("he {h} and its twin carry different kinds")
                })?;
            }
        }
        let mut owned = vec![false; nh];
        for (ci, c) in self.chains.iter().enumerate() {
            ensure(!c.hes.is_empty(), || format!("chain {ci} is empty"))?;
            ensure(c.kind != FeatureKind::None, || format!("chain {ci} has kind None"))?;
            for (i, &h) in c.hes.iter().enumerate() {
                ensure((h as usize) < nh, || format!("chain {ci} he {h} out of range"))?;
                ensure(self.he_kind[h as usize] == c.kind, || {
                    format!("chain {ci} he {h}: he_kind differs from the chain kind")
                })?;
                ensure(self.he_chain[h as usize] == ci as u32, || {
                    format!("chain {ci} he {h}: he_chain is {}", self.he_chain[h as usize])
                })?;
                ensure(!owned[h as usize], || format!("he {h} in two chains"))?;
                owned[h as usize] = true;
                if i + 1 < c.hes.len() {
                    ensure(t.head(m, h) == t.tail(m, c.hes[i + 1]), || {
                        format!("chain {ci} broken after position {i}")
                    })?;
                }
            }
            if c.closed {
                let (Some(&a), Some(&b)) = (c.hes.first(), c.hes.last()) else {
                    return Err(AqError::invariant("empty chain"));
                };
                ensure(t.head(m, b) == t.tail(m, a), || format!("closed chain {ci} is open"))?;
            }
            for (i, &k) in c.corners.iter().enumerate() {
                ensure((k as usize) < c.hes.len(), || format!("chain {ci} corner out of range"))?;
                if i > 0 {
                    ensure(c.corners[i - 1] < k, || format!("chain {ci} corners not sorted"))?;
                }
            }
        }
        for h in 0..nh {
            let c = self.he_chain[h];
            if c == NONE {
                continue;
            }
            ensure((c as usize) < self.chains.len(), || format!("he_chain[{h}] out of range"))?;
            ensure(self.he_kind[h] != FeatureKind::None, || format!("he {h} on a chain has kind None"))?;
            let tw = t.twin[h];
            ensure(owned[h] || (tw != NONE && owned[tw as usize] && self.he_chain[tw as usize] == c), || {
                format!("he {h} has chain id {c} but is not in that chain")
            })?;
        }
        Ok(())
    }
}

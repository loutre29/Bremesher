//! Shared read-only context of one layout run: proxy, param and derived per-element flags.

use std::collections::BTreeMap;

use aq_core::exact::orient2d;
use aq_core::{AqError, FeatureKind, FeatureSet, NONE, SeamlessParam, Topology, TriMesh, V2, math};

use crate::ProxyView;
use crate::geom::{Line, Pt};

/// True for the chain kinds that are pre-inserted constraint curves in M1.
pub fn is_constraint_kind(k: FeatureKind) -> bool {
    matches!(k, FeatureKind::Hard | FeatureKind::GuideHard | FeatureKind::Boundary | FeatureKind::Symmetry)
}

pub struct Ctx<'a> {
    pub m: &'a TriMesh,
    pub t: &'a Topology,
    pub feats: &'a FeatureSet,
    pub prm: &'a SeamlessParam,
    /// Per halfedge: on a constraint chain, or on the boundary.
    pub constraint_he: Vec<bool>,
    /// Chain id -> curve id (NONE for chains that are not constraints).
    pub chain_curve: Vec<u32>,
    pub n_chain_curves: u32,
    /// Vertex -> index4 for every listed cone.
    pub cone_index: BTreeMap<u32, i8>,
    /// Per face: sign of the uv orientation (+1, 0 degenerate, -1 folded).
    pub orient: Vec<i8>,
    /// Per component: uv area.
    pub comp_area: Vec<f64>,
}

impl<'a> Ctx<'a> {
    pub fn new(px: &ProxyView<'a>, prm: &'a SeamlessParam) -> Result<Self, AqError> {
        let m = px.mesh;
        let t = px.topo;
        let feats = px.feats;
        let nf = m.tris.len();
        let nh = 3 * nf;
        if prm.uv.len() != nf || prm.trans.len() != nh || t.twin.len() != nh {
            return Err(AqError::invalid_input("layout: param or topology sizes do not match the proxy"));
        }
        if feats.he_chain.len() != nh || feats.he_kind.len() != nh {
            return Err(AqError::invalid_input("layout: feature set sizes do not match the proxy"));
        }
        if t.face_comp.len() != nf || t.vert_he.len() != m.pos.len() || t.is_bnd_vert.len() != m.pos.len() {
            return Err(AqError::invalid_input("layout: topology sizes do not match the proxy"));
        }
        for (f, tri) in prm.uv.iter().enumerate() {
            if !tri.iter().all(|p| math::is_finite2(*p)) {
                return Err(AqError::invalid_input(format!("layout: uv of face {f} not finite")));
            }
        }
        for tr in &prm.trans {
            tr.validate()?;
        }
        let mut chain_curve = vec![NONE; feats.chains.len()];
        let mut n_chain_curves = 0u32;
        for (i, ch) in feats.chains.iter().enumerate() {
            if is_constraint_kind(ch.kind) && !ch.hes.is_empty() {
                chain_curve[i] = n_chain_curves;
                n_chain_curves += 1;
            }
        }
        let mut constraint_he = vec![false; nh];
        for (h, c) in constraint_he.iter_mut().enumerate() {
            let ch = feats.he_chain[h];
            let on_chain = ch != NONE
                && (ch as usize) < feats.chains.len()
                && chain_curve[ch as usize] != NONE
                && is_constraint_kind(feats.he_kind[h]);
            *c = on_chain || t.twin[h] == NONE;
        }
        let mut cone_index = BTreeMap::new();
        for &(v, i) in &prm.cones {
            if (v as usize) < m.pos.len() && i != 0 {
                cone_index.insert(v, i);
            }
        }
        let orient: Vec<i8> = prm
            .uv
            .iter()
            .map(|u| {
                let o = orient2d(u[0], u[1], u[2]);
                if o > 0.0 {
                    1
                } else if o < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        let mut comp_area = vec![0.0; t.n_comp as usize];
        for (f, u) in prm.uv.iter().enumerate() {
            let c = t.face_comp[f] as usize;
            if let Some(a) = comp_area.get_mut(c) {
                *a += 0.5 * math::cross2(math::sub2(u[1], u[0]), math::sub2(u[2], u[0])).abs();
            }
        }
        Ok(Ctx { m, t, feats, prm, constraint_he, chain_curve, n_chain_curves, cone_index, orient, comp_area })
    }

    #[inline]
    pub fn n_faces(&self) -> usize {
        self.m.tris.len()
    }

    #[inline]
    pub fn tri(&self, f: u32) -> &[V2; 3] {
        &self.prm.uv[f as usize]
    }

    #[inline]
    pub fn vid(&self, f: u32, k: u8) -> u32 {
        self.m.tris[f as usize][k as usize % 3]
    }

    pub fn corner_of(&self, f: u32, v: u32) -> Option<u8> {
        let tr = self.m.tris[f as usize];
        (0..3u8).find(|&k| tr[k as usize] == v)
    }

    /// Canonical halfedge of the edge of h.
    #[inline]
    pub fn canon(&self, h: u32) -> u32 {
        let tw = self.t.twin[h as usize];
        if tw == NONE || h < tw { h } else { tw }
    }

    /// Maximal trace length for a curve living in face f's component.
    pub fn max_len(&self, f: u32, user: f64) -> f64 {
        if user > 0.0 {
            return user;
        }
        let c = self.t.face_comp.get(f as usize).copied().unwrap_or(0) as usize;
        4.0 * self.comp_area.get(c).copied().unwrap_or(0.0).max(1.0).sqrt()
    }

    /// Crosses local edge k of face f: the twin face, its local edge, the line and the point
    /// in the twin chart. None on the boundary.
    pub fn cross(&self, f: u32, k: u8, l: Line, p: Pt) -> Option<(u32, u8, Line, Pt)> {
        let h = 3 * f + k as u32;
        let tw = self.t.twin[h as usize];
        if tw == NONE {
            return None;
        }
        let g = tw / 3;
        let kg = (tw % 3) as u8;
        let tr = &self.prm.trans[h as usize];
        let l2 = l.map(tr);
        let p2 = match p {
            Pt::V(c) => Pt::V(self.corner_of(g, self.vid(f, c))?),
            Pt::E(_) => Pt::E(kg),
            Pt::X(x) => {
                let q = tr.apply(l.point(x));
                Pt::X(q[l2.a()] + 0.0)
            }
        };
        Some((g, kg, l2, p2))
    }

    /// Maps a direction code across the transition of halfedge h.
    #[inline]
    pub fn map_dir(&self, h: u32, d: u8) -> u8 {
        (d + 4 - self.prm.trans[h as usize].r % 4) % 4
    }
}

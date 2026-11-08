//! [`Surface`]: mesh + topology + frames + feature chains, the analytic stand-in for an
//! aq_prep `Proxy` (which adds only source tracking and density).

use aq_core::math;
use aq_core::{AqError, Chain, FeatureKind, FeatureSet, Frames, NONE, Topology, TriMesh};

use crate::meshes::SynthMesh;
use crate::util::bad;

/// Default corner angle for Hard, Guide, GuideHard and Boundary chains (DESIGN 6.4).
pub const CORNER_DEG: f64 = 50.0;
/// Default corner angle for Symmetry chains (DESIGN 6.4).
pub const SYM_CORNER_DEG: f64 = 70.0;

/// A validated mesh with its topology, frames and feature chains.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    pub mesh: TriMesh,
    pub topo: Topology,
    pub frames: Frames,
    pub feats: FeatureSet,
}

impl Surface {
    /// Mesh with boundary chains only (one chain per boundary loop, corners where the boundary
    /// turns by at least 50 degrees).
    pub fn new(mesh: TriMesh) -> Result<Self, AqError> {
        Self::with_features(mesh, &[], CORNER_DEG, SYM_CORNER_DEG)
    }

    /// From a [`SynthMesh`] with its feature edges and the default corner angles.
    pub fn from_synth(s: &SynthMesh) -> Result<Self, AqError> {
        Self::with_features(s.mesh.clone(), &s.feature_edges(), CORNER_DEG, SYM_CORNER_DEG)
    }

    /// Mesh with the given feature edges (vertex pairs that must be mesh edges). Boundary edges
    /// always carry Boundary unless flagged Symmetry. Chains are maximal runs between vertices
    /// of chain degree other than 2 (or where the kind changes); corners are those vertices plus
    /// the chain vertices whose turning angle reaches `corner_deg` (`sym_corner_deg` for
    /// Symmetry). This is DESIGN 6.4 with a per-vertex (not windowed) turning angle.
    pub fn with_features(
        mesh: TriMesh,
        edges: &[([u32; 2], FeatureKind)],
        corner_deg: f64,
        sym_corner_deg: f64,
    ) -> Result<Self, AqError> {
        mesh.validate()?;
        let topo = Topology::build(&mesh)?;
        topo.validate(&mesh)?;
        let frames = Frames::new(&mesh, &topo)?;
        frames.validate(&mesh, &topo)?;
        let feats = build_features(&mesh, &topo, edges, corner_deg, sym_corner_deg)?;
        feats.validate(&mesh, &topo)?;
        Ok(Surface { mesh, topo, frames, feats })
    }

    pub fn n_faces(&self) -> usize {
        self.mesh.tris.len()
    }

    pub fn n_verts(&self) -> usize {
        self.mesh.pos.len()
    }

    /// Euler characteristic of component `comp`.
    pub fn chi(&self, comp: u32) -> i64 {
        self.topo.euler_char(&self.mesh, comp)
    }

    /// Sum of chi over all components.
    pub fn chi_total(&self) -> i64 {
        (0..self.topo.n_comp).map(|c| self.chi(c)).sum()
    }
}

/// Builds a [`FeatureSet`] from feature edges (see [`Surface::with_features`]).
pub fn build_features(
    m: &TriMesh,
    t: &Topology,
    edges: &[([u32; 2], FeatureKind)],
    corner_deg: f64,
    sym_corner_deg: f64,
) -> Result<FeatureSet, AqError> {
    let mut fs = FeatureSet::new(m, t);
    let nh = 3 * m.tris.len();
    let nv = m.pos.len();
    for &([a, b], kind) in edges {
        if kind == FeatureKind::None {
            continue;
        }
        if a as usize >= nv || b as usize >= nv {
            return Err(bad(format!("feature pair ({a}, {b}) out of range")));
        }
        let h = t
            .find_he(m, a, b)
            .or_else(|| t.find_he(m, b, a))
            .ok_or_else(|| bad(format!("feature pair ({a}, {b}) is not a mesh edge")))?;
        let tw = t.twin[h as usize];
        if tw == NONE {
            if kind == FeatureKind::Symmetry {
                fs.he_kind[h as usize] = FeatureKind::Symmetry;
            }
        } else {
            fs.he_kind[h as usize] = kind;
            fs.he_kind[tw as usize] = kind;
        }
    }
    // Canonical feature halfedges per vertex.
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for h in 0..nh as u32 {
        if fs.he_kind[h as usize] != FeatureKind::None && t.is_canonical(h) {
            inc[t.tail(m, h) as usize].push(h);
            inc[t.head(m, h) as usize].push(h);
        }
    }
    let is_bnd = |h: u32| t.twin[h as usize] == NONE;
    let breaker: Vec<bool> = (0..nv)
        .map(|v| {
            let e = &inc[v];
            match e.len() {
                0 => false,
                2 => {
                    fs.he_kind[e[0] as usize] != fs.he_kind[e[1] as usize] || is_bnd(e[0]) != is_bnd(e[1])
                }
                _ => true,
            }
        })
        .collect();
    // Orients canonical feature edge e to start at vertex v, if allowed.
    let orient = |e: u32, v: u32| -> Option<u32> {
        if t.tail(m, e) == v {
            Some(e)
        } else if !is_bnd(e) && t.head(m, e) == v {
            Some(t.twin[e as usize])
        } else {
            None
        }
    };
    let mut visited = vec![false; nh];
    let mut chains: Vec<(Vec<u32>, bool)> = Vec::new();
    for v in 0..nv as u32 {
        if !breaker[v as usize] {
            continue;
        }
        for &e in &inc[v as usize] {
            if visited[e as usize] {
                continue;
            }
            let Some(h0) = orient(e, v) else { continue };
            visited[e as usize] = true;
            let mut hes = vec![h0];
            let mut cur = t.head(m, h0);
            let mut guard = 0usize;
            while !breaker[cur as usize] {
                guard += 1;
                if guard > nh {
                    return Err(AqError::internal("chain walk does not terminate"));
                }
                let Some(&next) = inc[cur as usize].iter().find(|&&x| !visited[x as usize]) else {
                    break;
                };
                let Some(h) = orient(next, cur) else {
                    return Err(bad(format!("boundary feature chain reversed at vertex {cur}")));
                };
                visited[next as usize] = true;
                hes.push(h);
                cur = t.head(m, h);
            }
            chains.push((hes, false));
        }
    }
    for e in 0..nh as u32 {
        if visited[e as usize] || fs.he_kind[e as usize] == FeatureKind::None || !t.is_canonical(e) {
            continue;
        }
        let start = t.tail(m, e);
        visited[e as usize] = true;
        let mut hes = vec![e];
        let mut cur = t.head(m, e);
        let mut guard = 0usize;
        while cur != start {
            guard += 1;
            if guard > nh {
                return Err(AqError::internal("closed chain walk does not terminate"));
            }
            let Some(&next) = inc[cur as usize].iter().find(|&&x| !visited[x as usize]) else {
                return Err(AqError::invariant(format!("closed chain broken at vertex {cur}")));
            };
            let Some(h) = orient(next, cur) else {
                return Err(bad(format!("boundary feature chain reversed at vertex {cur}")));
            };
            visited[next as usize] = true;
            hes.push(h);
            cur = t.head(m, h);
        }
        chains.push((hes, true));
    }
    let dir = |h: u32| -> [f64; 3] {
        math::sub(m.pos[t.head(m, h) as usize], m.pos[t.tail(m, h) as usize])
    };
    for (ci, (mut hes, closed)) in chains.into_iter().enumerate() {
        let kind = fs.he_kind[hes[0] as usize];
        let lim = if kind == FeatureKind::Symmetry { sym_corner_deg } else { corner_deg }.to_radians();
        let n = hes.len();
        let mut corners: Vec<u32> = Vec::new();
        for i in 0..n {
            let v = t.tail(m, hes[i]);
            let mut is_c = breaker[v as usize];
            if !is_c && (i > 0 || closed) {
                let prev = hes[(i + n - 1) % n];
                is_c = math::angle_between(dir(prev), dir(hes[i])) >= lim;
            }
            if is_c {
                corners.push(i as u32);
            }
        }
        if closed && !corners.is_empty() && corners[0] != 0 {
            let s = corners[0] as usize;
            hes.rotate_left(s);
            for c in corners.iter_mut() {
                *c = ((*c as usize + n - s) % n) as u32;
            }
            corners.sort_unstable();
        }
        for (i, &h) in hes.iter().enumerate() {
            fs.he_chain[h as usize] = ci as u32;
            let tw = t.twin[h as usize];
            if tw != NONE {
                fs.he_chain[tw as usize] = ci as u32;
            }
            if corners.binary_search(&(i as u32)).is_ok() {
                fs.vert_corner[t.tail(m, h) as usize] = true;
            }
        }
        fs.chains.push(Chain { hes, kind, closed, corners });
    }
    for v in 0..nv {
        if breaker[v] {
            fs.vert_corner[v] = true;
        }
    }
    Ok(fs)
}

//! # aq_prep: S0 ingest, S2 proxy remesh, chain classification, feature isolation
//!
//! DESIGN 6.4 (WP-03).
//!
//! * [`ingest`] (S0) validates and repairs the user mesh: degenerate faces are collapsed,
//!   non-manifold edges and vertices are split, every component is oriented (closed ones with a
//!   positive volume), and the feature pairs are mapped to halfedges (rerouted along a short
//!   edge path when they are not an edge). The result is a manifold, consistently oriented
//!   mesh with a [`FeatureSet`] and maps back to the input ids.
//! * [`split_components`] cuts an [`Ingested`] mesh into [`ComponentMesh`]es.
//! * [`build_proxy`] (S2) is a Botsch-Kobbelt isotropic remesh (split, collapse, flip,
//!   tangential smoothing, projection, then a sliver repair) to the target length `len(x)`. It
//!   keeps every feature chain exactly on its source polyline and every chain corner exactly in
//!   place.
//! * [`isolate_features`] splits bridge edges and faces with two or more chain edges.
//! * [`classify_chains`] rebuilds the chains with their corners (windowed turning angle over
//!   0.5 h) and applies `hard_mode`.
//! * [`transfer_density`] interpolates the user density map onto the proxy vertices.
//!
//! Every function is deterministic: same input, same bits, for any rayon thread count
//! (parallel loops compute independent per-vertex values; all topological edits are
//! sequential and ordered).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// `!(x > y)` comparisons are deliberate: they also reject NaN.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::neg_cmp_op_on_partial_ord
)]

mod chains;
mod components;
mod density;
mod dynmesh;
mod ingest;
mod isolate;
mod remesh;

use aq_core::{AqError, Bvh, FeatureSet, Frames, NONE, Stage, Topology, TriMesh, V3};

pub use chains::{
    CornerRule, apply_hard_mode, build_feature_set, chain_vertex_degree, count_bridge_edges,
    count_multi_chain_faces,
};
pub use components::split_components;
pub use density::transfer_density;
pub use ingest::ingest;
pub use isolate::{isolate_features, isolate_features_with};
pub use remesh::build_proxy;

/// What [`ingest`] repaired.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IngestReport {
    /// Faces deleted: exact duplicates of an earlier face, and degenerate faces that no
    /// collapse could remove.
    pub dropped_faces: u32,
    /// Faces removed by degenerate-face collapses (the degenerate face itself, faces with a
    /// repeated index, and the neighbours that vanish with a collapsed edge).
    pub collapsed_faces: u32,
    /// Edge occurrences detached: the extra faces of non-manifold edges (3 or more faces),
    /// edges cut to make a component orientable, and faces fully detached as a last resort.
    pub split_edges: u32,
    /// Vertex copies created so that every vertex has a single fan.
    pub split_verts: u32,
    /// Faces whose final orientation is reversed with respect to the input.
    pub flipped_faces: u32,
    /// Feature pairs dropped (not routable within 2 rings, or collapsed to one vertex).
    pub dropped_features: u32,
    /// Feature pairs that were not an edge and were routed along an edge path.
    pub rerouted_features: u32,
    /// Connected components of the output.
    pub components: u32,
}

/// Output of S0.
#[derive(Clone, Debug, Default)]
pub struct Ingested {
    pub mesh: TriMesh,
    pub topo: Topology,
    pub feats: FeatureSet,
    /// Output vertex -> input vertex.
    pub vert_src: Vec<u32>,
    /// Output face -> input face.
    pub face_src: Vec<u32>,
    pub report: IngestReport,
}

/// One connected component of an [`Ingested`] mesh.
#[derive(Clone, Debug, Default)]
pub struct ComponentMesh {
    pub mesh: TriMesh,
    pub topo: Topology,
    pub feats: FeatureSet,
    /// Component face -> **input** face (composed through the ingest map).
    pub face_src: Vec<u32>,
    /// Component vertex -> **input** vertex (composed through the ingest map).
    pub vert_src: Vec<u32>,
    pub area: f64,
    pub chi: i64,
    /// No boundary.
    pub closed: bool,
}

impl ComponentMesh {
    /// A copy with every position multiplied by `s` (the engine's `1/h0`, DESIGN 2). The area
    /// scales by `s^2`; topology and features are unchanged.
    pub fn scaled(&self, s: f64) -> ComponentMesh {
        let mut c = self.clone();
        for p in c.mesh.pos.iter_mut() {
            *p = [p[0] * s, p[1] * s, p[2] * s];
        }
        c.area *= s * s;
        c
    }

    /// Frames and BVH of the component mesh, the parts a `Projector` over it borrows:
    /// `let (fr, bvh) = c.source_geometry()?; let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);`
    pub fn source_geometry(&self) -> Result<(Frames, Bvh), AqError> {
        let fr = Frames::new(&self.mesh, &self.topo).map_err(|e| e.at(Stage::Proxy))?;
        let bvh = Bvh::try_build(&self.mesh).map_err(|e| e.at(Stage::Proxy))?;
        Ok((fr, bvh))
    }
}

/// Parameters of S2 (DESIGN 6.4, 13).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProxyParams {
    /// Remesh iterations (5).
    pub iters: u32,
    /// Triangle budget of this component (its share of `max_proxy_tris`, 150000 in total).
    /// 0 means no budget.
    pub max_tris: u32,
    /// Corner turning angle for Hard, Guide, GuideHard and Boundary chains (50 degrees).
    pub corner_deg: f64,
    /// Corner turning angle for Symmetry chains (70 degrees).
    pub sym_corner_deg: f64,
    /// `AqParams.hard_edges`: 0 drops Hard chains, 1 turns them into soft guides, 2 keeps
    /// them crisp.
    pub hard_mode: u8,
}

impl Default for ProxyParams {
    fn default() -> Self {
        ProxyParams { iters: 5, max_tris: 150_000, corner_deg: 50.0, sym_corner_deg: 70.0, hard_mode: 2 }
    }
}

/// Minimum proxy triangle count per component (DESIGN 6.4 budget).
pub const MIN_PROXY_TRIS: u32 = 2000;

/// Output of S2: the isotropic proxy mesh the solvers work on.
#[derive(Clone, Debug, Default)]
pub struct Proxy {
    pub mesh: TriMesh,
    pub topo: Topology,
    pub frames: Frames,
    pub feats: FeatureSet,
    /// Per proxy vertex: face of the projector's mesh (the source component) under it.
    pub src_face: Vec<u32>,
    /// Per proxy vertex: barycentric coordinates in `src_face`.
    pub src_bary: Vec<V3>,
    /// Per proxy vertex: density multiplier (1 without a density map), see
    /// [`transfer_density`].
    pub density: Vec<f64>,
}

impl Proxy {
    /// A proxy that is the mesh itself (no remesh): `src_face[v]` is a face of v and
    /// `src_bary` the matching corner, density 1. A feature set of the wrong size is replaced
    /// by `FeatureSet::new` (boundary chains only, without chain lists). Used by tests and by
    /// callers that already have a clean mesh.
    pub fn from_mesh(mesh: TriMesh, feats: FeatureSet) -> Result<Proxy, AqError> {
        let topo = Topology::build(&mesh).map_err(|e| e.at(Stage::Proxy))?;
        let frames = Frames::new(&mesh, &topo).map_err(|e| e.at(Stage::Proxy))?;
        let nv = mesh.pos.len();
        let mut src_face = vec![NONE; nv];
        let mut src_bary = vec![[0.0; 3]; nv];
        for v in 0..nv {
            let h = topo.vert_he[v];
            if h != NONE {
                src_face[v] = h / 3;
                src_bary[v][(h % 3) as usize] = 1.0;
            }
        }
        let feats = if feats.he_kind.len() == 3 * mesh.tris.len() && feats.vert_corner.len() == nv {
            feats
        } else {
            FeatureSet::new(&mesh, &topo)
        };
        Ok(Proxy { mesh, topo, frames, feats, src_face, src_bary, density: vec![1.0; nv] })
    }

    /// Checks the mesh, topology, frames and features, and the per-vertex array sizes.
    pub fn validate(&self) -> Result<(), AqError> {
        let st = |e: AqError| e.at(Stage::Proxy);
        self.mesh.validate().map_err(st)?;
        self.topo.validate(&self.mesh).map_err(st)?;
        self.frames.validate(&self.mesh, &self.topo).map_err(st)?;
        self.feats.validate(&self.mesh, &self.topo).map_err(st)?;
        let nv = self.mesh.pos.len();
        if self.src_face.len() != nv || self.src_bary.len() != nv || self.density.len() != nv {
            return Err(AqError::invariant("Proxy per-vertex array sizes").at(Stage::Proxy));
        }
        Ok(())
    }
}

/// Rebuilds the chains of a proxy and their corners, and applies `hard_mode`
/// (DESIGN 6.4 "Chain classification").
///
/// * Chains are maximal sequences of feature edges between vertices of chain degree other
///   than 2 (or where the kind, or boundary versus interior, changes). A closed loop forms one
///   chain.
/// * Corners: every chain vertex of chain degree other than 2, every point where a Symmetry
///   chain meets another kind, and every vertex whose windowed turning angle (the angle
///   between the mean tangents over `0.5 h_vert[v]` on each side) is at least `corner_deg`
///   (`sym_corner_deg` for Symmetry). Of several such vertices within one window only the one
///   with the largest angle is kept.
/// * `hard_mode` 0 drops Hard chains (GuideHard becomes Guide), 1 turns Hard and GuideHard into
///   Guide, 2 keeps them.
///
/// A missing or invalid `h_vert[v]` is read as h = 1. On an internal error the feature set is
/// left unchanged.
pub fn classify_chains(px: &mut Proxy, h_vert: &[f64], p: &ProxyParams) {
    let mut kinds = px.feats.he_kind.clone();
    if kinds.len() != 3 * px.mesh.tris.len() {
        return;
    }
    apply_hard_mode(&mut kinds, p.hard_mode);
    let win = |v: u32| -> f64 {
        let h = h_vert.get(v as usize).copied().unwrap_or(1.0);
        if h.is_finite() && h > 0.0 { 0.5 * h } else { 0.5 }
    };
    let rule = CornerRule { corner_deg: p.corner_deg, sym_corner_deg: p.sym_corner_deg, window: &win, fixed: None };
    if let Ok(fs) = build_feature_set(&px.mesh, &px.topo, kinds, &rule) {
        px.feats = fs;
    }
}

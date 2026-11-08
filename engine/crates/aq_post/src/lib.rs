//! # aq_post: S11 relaxation and cleanup (DESIGN 6.13, WP-14)
//!
//! Input: the assembled quad mesh of one component (S10, quad units, so h = 1) and the proxy
//! it was built on.
//!
//! * [`relax`]: `iters` sweeps of tangential smoothing. A free vertex moves to the
//!   area-weighted mean of its incident face centroids, projected to the tangent plane of the
//!   proxy and then onto the proxy with [`aq_core::Projector`] (hint = its current source
//!   face). Arc vertices are free, so loops straighten across patch borders. A chain vertex
//!   slides along its proxy chain polyline (it stays on it to rounding). Corner and cone
//!   vertices are frozen for the first `freeze_nodes_iters` sweeps; cones then move like the
//!   others, corners stay fixed. A move is rejected when an incident quad corner normal would
//!   turn more than 60 degrees away from the surface normal.
//! * [`cleanup`]: removes interior valence-2 vertices (doublets) by merging their two quads,
//!   then collapses 3-3 diagonal pairs on quads of area below 0.3 h^2, and reports the count
//!   change and the old-to-new vertex map.
//!
//! ## Deviations (see CHANGES.md, WP-14)
//!
//! * DESIGN 6.13 takes `px: &Proxy`, an aq_prep type that aq_post cannot name (DESIGN 3.2:
//!   aq_post depends on aq_core only). [`PostProxy`] holds borrowed references to the same
//!   proxy parts plus the BVH that the projector needs; the engine builds it from its
//!   `aq_prep::Proxy` with [`PostProxy::new`].
//! * The sweeps are Gauss-Seidel in vertex id order (single-threaded, so bit-identical for any
//!   thread count); the start-up projections run in parallel.
//!
//! Vertex roles come from `QuadMesh.vflag` (DESIGN 8.4): bit 24 corner, bit 25 cone, bit 26
//! boundary. A quad-mesh boundary vertex without a chain is kept fixed (it has no polyline to
//! slide on).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop)]

mod adj;
mod chain;
mod cleanup;
mod relax;

use aq_core::{Bvh, FeatureSet, Frames, Topology, TriMesh};

pub use chain::dist_to_polyline;
pub use cleanup::{CleanupParams, CleanupReport, cleanup, cleanup_with};
pub use relax::{CHAIN_SNAP_MAX, relax};

/// vflag bits 0..19: chain id + 1 (0 = none).
pub const VF_CHAIN_MASK: u32 = (1 << 20) - 1;
/// vflag bit 24: corner node.
pub const VF_CORNER: u32 = 1 << 24;
/// vflag bit 25: cone (singular vertex of the layout).
pub const VF_CONE: u32 = 1 << 25;
/// vflag bit 26: boundary vertex.
pub const VF_BOUNDARY: u32 = 1 << 26;

/// cos(60 degrees): the move rejection threshold of DESIGN 6.13.
pub const COS_MAX_TURN: f64 = 0.5;

/// Relaxation parameters (DESIGN 6.13, defaults from section 13).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostParams {
    /// Number of smoothing sweeps (8).
    pub iters: u32,
    /// Corner and cone vertices are fixed during the first sweeps (4); cones move afterwards.
    pub freeze_nodes_iters: u32,
}

impl Default for PostParams {
    fn default() -> Self {
        PostParams { iters: 8, freeze_nodes_iters: 4 }
    }
}

/// The parts of the proxy (aq_prep `Proxy`) that S11 reads, plus its BVH.
#[derive(Clone, Copy, Debug)]
pub struct PostProxy<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub frames: &'a Frames,
    /// Feature chains; `chain_of_vert` indexes `feats.chains`.
    pub feats: &'a FeatureSet,
    /// BVH of `mesh` (fallback of the projector and start-up hints).
    pub bvh: &'a Bvh,
}

impl<'a> PostProxy<'a> {
    pub fn new(
        mesh: &'a TriMesh,
        topo: &'a Topology,
        frames: &'a Frames,
        feats: &'a FeatureSet,
        bvh: &'a Bvh,
    ) -> Self {
        PostProxy { mesh, topo, frames, feats, bvh }
    }
}

/// What [`relax`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RelaxReport {
    /// Sweeps run.
    pub iters: u32,
    /// Vertices by role (a cone is counted as interior or chain, and also in `cone_verts`).
    pub interior_verts: u32,
    pub chain_verts: u32,
    /// Corners, boundary vertices without a chain, isolated vertices, chain vertices too far
    /// from their polyline.
    pub fixed_verts: u32,
    /// Cones (frozen during the first `freeze_nodes_iters` sweeps).
    pub cone_verts: u32,
    /// Accepted moves over all sweeps.
    pub moved: u64,
    /// Moves rejected by the 60 degree normal test.
    pub rejected_normal: u64,
    /// Moves dropped because the projector found no valid point.
    pub projection_failed: u64,
    /// Chain vertices snapped onto their polyline at start-up, and the largest snap distance.
    pub chain_snapped: u32,
    pub max_snap: f64,
    /// Chain vertices farther than [`CHAIN_SNAP_MAX`] from their polyline (kept fixed).
    pub chain_off: u32,
    /// Largest single move, and largest total displacement of a vertex.
    pub max_step: f64,
    pub max_disp: f64,
    /// Mean total displacement over the vertices that are not fixed.
    pub mean_disp: f64,
}

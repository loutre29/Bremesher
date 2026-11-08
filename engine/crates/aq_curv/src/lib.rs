//! # aq_curv: S3 curvature at the quad scale and sizing (DESIGN 6.5, WP-07)
//!
//! * [`curvature`]: normal-cycle tensor per vertex, implicit heat diffusion over about one
//!   quad (`(M + L_t) T = M T0`, one factorization, 6 right-hand sides), per-face principal
//!   curvatures k1, k2 (|k1| >= |k2|, signed, positive on convex parts), the k1 direction phi
//!   in the face frame, the confidence conf and kappa_hat per vertex ([`aq_core::CurvField`]).
//! * [`sizing`] / [`sizing_with`]: base size g (adaptive term and density, M2), tube cap,
//!   Persson gradation and exact normalization to the target count
//!   ([`aq_core::SizeField`], `n_pred = sum A/h^2 = N`).
//! * [`curvature_and_sizing`]: the two-pass S3 driver (pass 2 only when adaptive > 0 or a
//!   density is given).
//! * [`shape_diameter`]: the shape diameter d(v) of the M2 thickness cap (WP-20, [`thick`]).
//!
//! The stage takes a [`ProxyRef`] (borrowed mesh, topology, frames, features) instead of
//! aq_prep's `Proxy`, because aq_curv depends on aq_core and aq_linalg only (DESIGN 3.2);
//! it is field for field the view aq_cut and aq_loops use (CHANGES.md, WP-09 and WP-07).
//! Everything is in quad units (C14). Errors carry `Stage::Curv`.
//!
//! Deviations from DESIGN 6.5 are listed in CHANGES.md under WP-07 (normal leak return,
//! normalization order, infeasible caps) and WP-20 (thickness cap).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop)]

pub mod curv;
pub mod sizing;
pub mod thick;

use aq_core::{AqError, Ctl, CurvField, FeatureSet, Frames, SizeField, Stage, Topology, TriMesh};

pub use curv::{
    CurvDetail, CurvParams, Sym3, anisotropy, confidence, curvature, curvature_detail, dihedral,
    smoothstep,
};
pub use sizing::{
    G_MAX, G_MIN, KAPPA_REF_MIN, SizingParams, SizingReport, THICK_BUDGET, THICK_FLOOR_MIN, TUBE_ANISO,
    TUBE_CONF,
    sizing, sizing_diam, sizing_with, thickness_demand, thickness_floor,
};
pub use thick::{RayBvh, ThickPlan, shape_diameter, shape_diameter_mesh, subdivide_to, thickness_plan};

/// Borrowed proxy view (the parts of aq_prep's `Proxy` that S3 reads). The engine builds it
/// with `ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats)`.
#[derive(Clone, Copy, Debug)]
pub struct ProxyRef<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub frames: &'a Frames,
    pub feats: &'a FeatureSet,
}

impl<'a> ProxyRef<'a> {
    pub fn new(mesh: &'a TriMesh, topo: &'a Topology, frames: &'a Frames, feats: &'a FeatureSet) -> Self {
        ProxyRef { mesh, topo, frames, feats }
    }
}

pub(crate) fn bad(msg: impl Into<String>) -> AqError {
    AqError::invalid_input(msg).at(Stage::Curv)
}

pub(crate) fn restamp(e: AqError) -> AqError {
    e.at(Stage::Curv)
}

/// Size consistency of the view (`InvalidInput`).
pub(crate) fn check_proxy(px: &ProxyRef) -> Result<(), AqError> {
    let nf = px.mesh.tris.len();
    let nv = px.mesh.pos.len();
    let fr = px.frames;
    if nf == 0 {
        return Err(bad("proxy has no faces"));
    }
    if px.topo.twin.len() != 3 * nf || px.topo.vert_he.len() != nv {
        return Err(bad("proxy topology does not match the mesh"));
    }
    if fr.e1.len() != nf || fr.e2.len() != nf || fr.n.len() != nf || fr.area.len() != nf {
        return Err(bad("proxy frames do not match the mesh"));
    }
    if px.mesh.tris.iter().flatten().any(|&v| v as usize >= nv) {
        return Err(bad("proxy vertex index out of range"));
    }
    if px.topo.twin.iter().any(|&t| t != aq_core::NONE && t as usize >= 3 * nf) {
        return Err(bad("proxy twin out of range"));
    }
    Ok(())
}

/// Report of [`curvature_and_sizing`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct S3Report {
    /// 1, or 2 when adaptive > 0 or a density was given.
    pub passes: u32,
    /// The report of the last sizing pass.
    pub sizing: SizingReport,
    pub curv_ms: f64,
    pub sizing_ms: f64,
}

/// S3 driver (DESIGN 6.5 "Two passes"): pass 1 measures curvature with the uniform h0 = 1 and
/// sizes; pass 2 (only when `sp.adaptive > 0` or a density is given) measures again with the
/// diffusion time from pass 1's h and sizes again. (The M2 adaptive proxy re-remesh between
/// the passes belongs to aq_prep and the engine.)
pub fn curvature_and_sizing(
    px: &ProxyRef,
    density: Option<&[f64]>,
    cp: &CurvParams,
    sp: &SizingParams,
    ctl: &Ctl,
) -> Result<(CurvField, SizeField, S3Report), AqError> {
    let nv = px.mesh.pos.len();
    let mut rep = S3Report { passes: 1, ..Default::default() };
    let h1 = aq_core::alloc::try_vec(nv, 1.0f64).map_err(restamp)?;
    let t0 = std::time::Instant::now();
    let mut c = curvature(px, &h1, cp, ctl)?;
    let t1 = std::time::Instant::now();
    let td = std::time::Instant::now();
    let diam = if sp.thickness && sp.ring_min > 0 { Some(thick::shape_diameter(px, ctl)?) } else { None };
    let thick_ms = td.elapsed().as_secs_f64() * 1e3;
    let (mut s, mut r) = sizing_diam(px, &c, sp, density, diam.as_deref(), ctl)?;
    let t2 = std::time::Instant::now();
    rep.curv_ms = (t1 - t0).as_secs_f64() * 1e3;
    rep.sizing_ms = (t2 - t1).as_secs_f64() * 1e3;
    if sp.adaptive > 0.0 || density.is_some() {
        rep.passes = 2;
        c = curvature(px, &s.h_vert, cp, ctl)?;
        let t3 = std::time::Instant::now();
        (s, r) = sizing_diam(px, &c, sp, density, diam.as_deref(), ctl)?;
        let t4 = std::time::Instant::now();
        rep.curv_ms += (t3 - t2).as_secs_f64() * 1e3;
        rep.sizing_ms += (t4 - t3).as_secs_f64() * 1e3;
    }
    r.thick_ms += thick_ms;
    rep.sizing = r;
    Ok((c, s, rep))
}

//! # aq_field: S4 cross field (DESIGN 6.6, WP-08)
//!
//! Pipeline of [`solve_field`]:
//!
//! 1. **Constraints** (`setup`): hard faces (a chain edge of kind Boundary or Symmetry, plus
//!    Hard and GuideHard when `hard_mode == 2`), their segment-smoothed chain tangent psi (a
//!    window of 0.5h along the chain, never across a corner), soft guide targets (kind Guide,
//!    and Hard / GuideHard when `hard_mode == 1`, spread to the faces within 0.5h), the
//!    smoothness weights `w_e = |e|^2 / (A_f + A_g)` (clamped) and the curvature alignment
//!    weights `lambda_f A_f conf'_f` with `lambda_f = align / h_f^2`, where `conf'` and the
//!    target are those of the coherence filter (the curvature target smoothed over 1.5h with
//!    the 4-connection, so that incoherent, noisy directions lose their weight; CHANGES.md,
//!    WP-08 review).
//! 2. **Knoppel power field** ([`knoppel`]): `z_f = exp(4 i theta_f)`, hard faces eliminated,
//!    solved as a real 2n system with Cholesky, then at most 16 normalized iterations on the
//!    same factorization that keep |z| from collapsing (and creating cones) where alignment is
//!    weak; the eigen path (inverse power iteration) for components without hard faces and
//!    with a mean confidence below 0.01. Hard faces are then snapped to the **exact** direction
//!    of their constraint edge (see CHANGES.md, WP-08).
//! 3. **Split** interior non-corner vertices with |index4| >= 2 by one-edge path edits.
//! 4. **Path edits** ([`edit_path`]) move one quarter of index along a vertex path.
//! 5. **Corner merge** (WP-23, M2): a free cone next to an interior feature corner (0.5h) or
//!    next to a boundary vertex of opposite index sign (`boundary_merge`, 1h) moves into it.
//!    **Push-off** of cones near chains, **same-sign repulsion**, then **pair cancellation**
//!    of close +1/-1 pairs, with the `|delta| <= pi/3` check and revert; then (WP-23) the
//!    energy-guarded **far cancellation** of K-free pairs in low confidence (`far_cancel`)
//!    and the **merge of -1 pairs** into -2 cones, interior valence 6 poles (`merge_neg`).
//! 6. **Fixed-matching re-solve** ([`resolve_fixed_matching`]) after every edit batch and last.
//!
//! The functions take a [`ProxyRef`] (borrowed mesh, topology, frames and features), because
//! the aq_prep `Proxy` type is not visible from this crate (DESIGN 3.2: aq_field depends on
//! aq_core and aq_linalg only). The type is field for field the one of aq_cut and aq_loops
//! (CHANGES.md, WP-09 core change request).
//!
//! Conventions C4 to C9 come from aq_core (`matching_from_theta`, `delta_from_p`,
//! `index4_all`); this crate never re-derives them. Errors carry `Stage::Field`.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod edit;
mod graph;
mod knoppel;
mod resolve;
mod setup;

use aq_core::{AqError, CrossField, Ctl, CurvField, FeatureSet, Frames, SizeField, Stage, Topology, TriMesh};

pub use edit::{edit_path, try_edit_path};
pub use resolve::field_energy;

/// Borrowed view of a proxy: everything S4 reads from an aq_prep `Proxy`.
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

/// Parameters of S4 (DESIGN 6.6, defaults from DESIGN 13).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldParams {
    /// curvature_align, the numerator of `lambda_f = align / h_f^2` (default 1.0).
    pub align: f64,
    /// guide_strength: soft guide weight `20 * guide_w / h^2` (default 1.0).
    pub guide_w: f64,
    /// 0 ignore Hard chains, 1 Hard chains as soft guides, 2 Hard and GuideHard chains are hard
    /// constraints (default 2).
    pub hard_mode: u8,
    /// Pair cancellation distance in h (default 1.2).
    pub cancel_dist: f64,
    /// Pair cancellation distance in h in low-confidence regions (default 2.5).
    pub cancel_dist_lowconf: f64,
    /// Rounds of pair cancellation (default 3).
    pub max_cancel_rounds: u32,
    /// Clearance from Symmetry chains in h (default 0.5).
    pub sym_clearance: f64,
    /// M2 size-consistent connection beta_c. M1 ignores it (0 is the M1 default); see CHANGES.md.
    pub size_consistency: f64,
    /// Drives the deterministic start vector of the eigen path.
    pub seed: u64,
    /// WP-23 (M2): energy-guarded cancellation of +1/-1 pairs up to this distance in h, after
    /// the distance-only rounds of step 5 (0 switches it off; default 6). See
    /// `edit::cancel_far`.
    pub far_cancel: f64,
    /// WP-23 (M2): largest change of the step 7 energy one far pair may cause (dimensionless:
    /// a field off by 1 rad over one quad of conf 1 at align 1 costs 1). A negative value
    /// demands an energy drop. Default 1.0.
    pub far_cancel_energy: f64,
    /// WP-23 (M2): two free -1 cones within this distance in h merge into one -2 cone (an
    /// interior valence 6 pole) at the middle of their path, after the cancellation (0
    /// switches it off). See `edit::merge_negative`.
    pub merge_neg: f64,
    /// WP-23 (M2): a free cone within this distance in h of a boundary vertex whose index4 has
    /// the opposite sign merges into it, in the corner merge of step 6 (0 switches it off).
    pub boundary_merge: f64,
    /// M2 integration: points (proxy coordinates) near which the WP-23 edits (far
    /// cancellation, -1 pair merge, boundary merge) are not made: an edit whose cones lie
    /// within [`KEEP_OUT`] h of one of them is skipped. The engine fills it with the S7 folds
    /// and bad cones of a run whose S8 failed, before it drops the edits altogether. Default
    /// empty.
    pub keep_out: Vec<[f64; 3]>,
}

impl Default for FieldParams {
    fn default() -> Self {
        FieldParams {
            align: 1.0,
            guide_w: 1.0,
            hard_mode: 2,
            cancel_dist: 1.2,
            cancel_dist_lowconf: 2.5,
            max_cancel_rounds: 3,
            sym_clearance: 0.5,
            size_consistency: 0.0,
            seed: 0,
            far_cancel: FAR_CANCEL,
            far_cancel_energy: FAR_CANCEL_ENERGY,
            merge_neg: MERGE_NEG,
            boundary_merge: BOUNDARY_MERGE,
            keep_out: Vec::new(),
        }
    }
}

/// Default of [`FieldParams::far_cancel`] in h (WP-23).
pub const FAR_CANCEL: f64 = 6.0;
/// Default of [`FieldParams::far_cancel_energy`] (WP-23).
pub const FAR_CANCEL_ENERGY: f64 = 1.0;
/// Default of [`FieldParams::merge_neg`] in h (WP-23).
pub const MERGE_NEG: f64 = 1.5;
/// Radius in h (Euclidean, at the cone) of a [`FieldParams::keep_out`] point.
pub const KEEP_OUT: f64 = 1.5;
/// Default of [`FieldParams::boundary_merge`] in h (WP-23).
pub const BOUNDARY_MERGE: f64 = 1.0;
/// Clearance from non-Symmetry chains in h (DESIGN 13: chain clearance 0.35 h).
pub const CHAIN_CLEARANCE: f64 = 0.35;
/// Push-off target distance from all chains in h (DESIGN 13: push target 0.75 h).
pub const PUSH_TARGET: f64 = 0.75;
/// Maximum |delta| on an edited path after the re-solve (DESIGN 13: pi/3).
pub const MAX_EDIT_DELTA: f64 = std::f64::consts::FRAC_PI_3;
/// Low-confidence threshold of the cancellation disc (DESIGN 6.6 step 5).
pub const LOW_CONF: f64 = 0.2;
/// Eigen path when no hard faces and the area-weighted mean conf is below this (DESIGN 6.6).
pub const EIGEN_CONF: f64 = 0.01;

/// What S4 did (DESIGN 6.6).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldReport {
    /// Vertices with index4 != 0 right after the Knoppel solve (corners included).
    pub cones_initial: u32,
    /// Vertices with index4 != 0 at the end (corners included).
    pub cones_final: u32,
    /// Quarter moves made by the high-index split.
    pub split: u32,
    /// Cancelled +1/-1 pairs.
    pub cancelled: u32,
    /// Pairs reverted because an edited edge kept |delta| > pi/3.
    pub reverted: u32,
    /// Quarter moves made by the push-off.
    pub pushed: u32,
    /// max |delta| over all halfedges at the end.
    pub max_abs_delta: f64,
    /// True when at least one component used the eigen path.
    pub eigen_path: bool,
    /// M2 pole sanity cap (WP-23): true when the cones left exceed
    /// `max(1.5 S_ref, S_ref + 8)`, S_ref the cone count of the pure smooth field (alignment
    /// 0). Set by the orchestrator with [`excess_poles`]; `solve_field` leaves it false.
    pub excess_poles: bool,
    /// Pairs removed by the energy-guarded far cancellation (WP-23), included in `cancelled`.
    pub far_cancelled: u32,
    /// Cones merged into an interior feature corner (WP-23, DESIGN 6.6 step 6 M2).
    pub corner_merged: u32,
    /// Free cones merged into a boundary vertex of opposite index sign (WP-23).
    pub boundary_merged: u32,
    /// Pairs of -1 cones merged into one -2 cone (WP-23).
    pub merged_neg: u32,
}

/// WP-23 pole sanity cap (DESIGN 6.6 M2): true when a field with `cones` cones exceeds
/// `max(1.5 S_ref, S_ref + 8)`, with `s_ref` the cone count of the pure smooth run (the same
/// S4 with alignment 0). The orchestrator then re-runs S3 and S4 with curvature scale x1.5
/// and align x0.7.
pub fn excess_poles(cones: u32, s_ref: u32) -> bool {
    let r = s_ref as f64;
    cones as f64 > (1.5 * r).max(r + 8.0)
}

pub(crate) fn bad(msg: impl Into<String>) -> AqError {
    AqError::invalid_input(msg).at(Stage::Field)
}

pub(crate) fn inv(msg: impl Into<String>) -> AqError {
    AqError::invariant(msg).at(Stage::Field)
}

/// Checks the sizes and values of the inputs.
pub(crate) fn check_inputs(px: &ProxyRef, c: &CurvField, s: &SizeField, p: &FieldParams) -> Result<(), AqError> {
    let nf = px.mesh.tris.len();
    let nv = px.mesh.pos.len();
    if nf == 0 {
        return Err(bad("aq_field: empty mesh"));
    }
    if px.topo.twin.len() != 3 * nf
        || px.topo.face_comp.len() != nf
        || px.topo.vert_he.len() != nv
        || px.topo.is_bnd_vert.len() != nv
    {
        return Err(bad("aq_field: topology does not match the mesh"));
    }
    if px.frames.area.len() != nf || px.frames.kappa.len() != 3 * nf || px.frames.corner_angle.len() != 3 * nf {
        return Err(bad("aq_field: frames do not match the mesh"));
    }
    if px.feats.he_kind.len() != 3 * nf || px.feats.he_chain.len() != 3 * nf || px.feats.vert_corner.len() != nv {
        return Err(bad("aq_field: feature set does not match the mesh"));
    }
    if c.phi.len() != nf || c.conf.len() != nf {
        return Err(bad("aq_field: curvature field does not match the mesh"));
    }
    if s.h_face.len() != nf || s.h_vert.len() != nv {
        return Err(bad("aq_field: size field does not match the mesh"));
    }
    for (i, h) in s.h_face.iter().chain(&s.h_vert).enumerate() {
        if !(h.is_finite() && *h > 0.0) {
            return Err(bad(format!("aq_field: size entry {i} is {h}")));
        }
    }
    for f in 0..nf {
        if !c.phi[f].is_finite() || !c.conf[f].is_finite() {
            return Err(bad(format!("aq_field: curvature of face {f} not finite")));
        }
    }
    let finite = [p.align, p.guide_w, p.cancel_dist, p.cancel_dist_lowconf, p.sym_clearance, p.far_cancel, p.merge_neg, p.boundary_merge];
    if finite.iter().any(|x| !x.is_finite() || *x < 0.0)
        || !p.far_cancel_energy.is_finite()
        || p.keep_out.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(bad("aq_field: parameters must be finite and >= 0 (far_cancel_energy: finite)"));
    }
    for (ci, ch) in px.feats.chains.iter().enumerate() {
        if ch.hes.iter().any(|&h| h as usize >= 3 * nf) || ch.corners.iter().any(|&k| k as usize >= ch.hes.len()) {
            return Err(bad(format!("aq_field: chain {ci} out of range")));
        }
    }
    Ok(())
}

/// S4: Knoppel power field, singularity edits and the fixed-matching re-solve.
///
/// The result is validated (`CrossField::validate`, Poincare-Hopf included), interior
/// non-corner cones have |index4| <= 1 or index4 -2 ([`interior_index_ok`]), and every hard face is aligned with its constraint
/// edge to rounding (DESIGN 7.3, after S4).
pub fn solve_field(
    px: &ProxyRef,
    c: &CurvField,
    s: &SizeField,
    p: &FieldParams,
    ctl: &Ctl,
) -> Result<(CrossField, FieldReport), AqError> {
    check_inputs(px, c, s, p)?;
    ctl.progress(Stage::Field, 0.0);
    let st = setup::Setup::new(px, c, s, p, Some(ctl))?;
    ctl.check()?;
    let (theta, eigen) = knoppel::knoppel_theta(px, &st, p, ctl)?;
    ctl.progress(Stage::Field, 0.4);
    let mut f = CrossField::from_theta(px.mesh, px.topo, px.frames, theta, st.hard.clone())
        .map_err(|e| e.at(Stage::Field))?;
    let cones_initial = f.index4.iter().filter(|&&i| i != 0).count() as u32;
    let mut rep = edit::simplify(px, &st, s, &mut f, p, ctl)?;
    rep.cones_initial = cones_initial;
    rep.eigen_path = eigen;
    check_field(px, &st, &f)?;
    ctl.progress(Stage::Field, 1.0);
    Ok((f, rep))
}

/// Steps 3 to 7 on a given field (any source, for example a planted test field): split of high
/// indices, push-off, pair cancellation and the fixed-matching re-solves. Hard faces are snapped
/// to their constraint edges first. `cones_initial` is the cone count of the input field;
/// `eigen_path` is false.
pub fn edit_singularities(
    px: &ProxyRef,
    c: &CurvField,
    s: &SizeField,
    f: &mut CrossField,
    p: &FieldParams,
    ctl: &Ctl,
) -> Result<FieldReport, AqError> {
    check_inputs(px, c, s, p)?;
    check_field_sizes(px, f)?;
    let st = setup::Setup::new(px, c, s, p, Some(ctl))?;
    let cones_initial = f.index4.iter().filter(|&&i| i != 0).count() as u32;
    f.hard.clone_from(&st.hard);
    let mut rep = edit::simplify(px, &st, s, f, p, ctl)?;
    rep.cones_initial = cones_initial;
    check_field(px, &st, f)?;
    Ok(rep)
}

/// The Knoppel solve alone (DESIGN 6.6 step 2): theta per face, hard faces snapped to the exact
/// direction of their constraint edge.
pub fn knoppel(px: &ProxyRef, c: &CurvField, s: &SizeField, p: &FieldParams, ctl: &Ctl) -> Result<Vec<f64>, AqError> {
    check_inputs(px, c, s, p)?;
    let st = setup::Setup::new(px, c, s, p, Some(ctl))?;
    Ok(knoppel::knoppel_theta(px, &st, p, ctl)?.0)
}

/// Step 7: with p fixed, minimizes `sum_e w_e (theta_g - theta_f - kappa_e - p_e pi/2)^2 +
/// sum_f lambda_f A_f conf_f (theta_f - phihat_f)^2` over the free faces (hard faces fixed at
/// the branch of their constraint edge nearest their current theta), then recomputes delta.
/// p and index4 are unchanged (asserted: a change is `Invariant`). `f.hard` is set to the hard
/// faces of DESIGN 6.6 step 1.
pub fn resolve_fixed_matching(
    px: &ProxyRef,
    c: &CurvField,
    s: &SizeField,
    f: &mut CrossField,
    p: &FieldParams,
) -> Result<(), AqError> {
    check_inputs(px, c, s, p)?;
    check_field_sizes(px, f)?;
    let st = setup::Setup::new(px, c, s, p, None)?;
    f.hard.clone_from(&st.hard);
    let mut rs = resolve::Resolver::new(px, &st, f, None)?;
    rs.solve(px, &st, f)
}

pub(crate) fn check_field_sizes(px: &ProxyRef, f: &CrossField) -> Result<(), AqError> {
    let nf = px.mesh.tris.len();
    if f.theta.len() != nf
        || f.hard.len() != nf
        || f.p.len() != 3 * nf
        || f.delta.len() != 3 * nf
        || f.index4.len() != px.mesh.pos.len()
    {
        return Err(bad("aq_field: cross field does not match the mesh"));
    }
    if f.theta.iter().any(|t| !t.is_finite()) || f.p.iter().any(|&p| p >= 4) {
        return Err(bad("aq_field: cross field has a non-finite theta or p >= 4"));
    }
    Ok(())
}

/// Index4 an interior non-corner vertex may keep after S4: -2 (a merged pair of -1 cones, an
/// interior valence 6 pole, WP-23), -1, 0 or +1 (DESIGN 7.3: |index4| <= 1, plus the -2).
pub fn interior_index_ok(i: i8) -> bool {
    (-2..=1).contains(&i)
}

/// The S4 invariants of DESIGN 7.3: field valid (twin rules, delta, index, Poincare-Hopf),
/// interior non-corner cones with [`interior_index_ok`], hard faces aligned with their constraint edge.
pub(crate) fn check_field(px: &ProxyRef, st: &setup::Setup, f: &CrossField) -> Result<(), AqError> {
    f.validate(px.mesh, px.topo, px.frames).map_err(|e| e.at(Stage::Field))?;
    for (v, &i) in f.index4.iter().enumerate() {
        if !interior_index_ok(i) && !px.topo.is_bnd_vert[v] && !px.feats.vert_corner[v] {
            return Err(inv(format!("interior non-corner vertex {v} keeps index4 {i}")));
        }
    }
    for fc in 0..px.mesh.tris.len() {
        if st.hard[fc] {
            let d = setup::branch_offset(f.theta[fc], st.edge_angle[fc]);
            if d.abs() > 1e-9 {
                return Err(inv(format!("hard face {fc} is off its constraint edge by {d:e} rad")));
            }
        }
    }
    Ok(())
}

//! # aq_fallback: the SF quadder (DESIGN 6.14, WP-04)
//!
//! The fallback that always produces an all-quad mesh for a component:
//!
//! 1. **Isotropic remesh** at `l_sf = 2.15 h`, with `h = sqrt(area / target)`, so the triangle
//!    count is about `target / 2`. When that count reaches `aq_prep::MIN_PROXY_TRIS` the remesh
//!    is `aq_prep::build_proxy`; below it (where `build_proxy` would clamp the count up to its
//!    minimum) and for [`sf_batch`] the crate's own Botsch-Kobbelt remesher runs on the
//!    component (see `remesh`, CHANGES.md WP-04). A count feedback then rescales the length
//!    (at most [`COUNT_ROUNDS`] times) while the predicted quad count is off by more than
//!    [`COUNT_TOL`].
//! 2. **Greedy pairing** of adjacent triangles across interior non-chain edges whose dihedral
//!    is below 30 degrees, best quad first (smallest maximum |corner angle - 90 degrees|).
//! 3. **Topological split**: a paired quad gives 4 quads, an unpaired triangle 3 (edge
//!    midpoints and a face center), so the result is conforming and 100% quads. The new
//!    vertices are projected onto the source.
//! 4. **Relaxation** (`relax_iters`, 5) of the free vertices, tangential and projected onto
//!    the source after every iteration, with a guard that never lets a quad get worse (more
//!    folded corners, or a normal against its parent or the source).
//!
//! Feature chains (including boundaries) are never paired across and their vertices never
//! move. The output carries vflag bits (DESIGN 8.4) in component-local chain ids; the engine
//! renumbers them when it merges components.
//!
//! Everything is deterministic: all topological edits are sequential and ordered, and the
//! parallel loops compute independent per-vertex values.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// `!(x > y)` comparisons are deliberate: they also reject NaN.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]

mod quad;
mod remesh;

use remesh::Remesher;

use rayon::prelude::*;

use aq_core::math;
use aq_core::{AqError, Ctl, ErrKind, FeatureKind, NONE, Projector, QuadMesh, Stage, SurfacePoint, V3};
use aq_prep::{ComponentMesh, MIN_PROXY_TRIS, ProxyParams, build_proxy};

/// SF edge length in units of `h = sqrt(area / target)` (DESIGN 6.14, 13).
pub const L_SF: f64 = 2.15;
/// Pairing: largest dihedral angle across the diagonal (DESIGN 6.14 step 2).
pub const PAIR_MAX_DIHEDRAL_DEG: f64 = 30.0;
/// Pairing: largest quality (max |corner - 90 deg|) of a paired quad. Keeps every paired quad
/// strictly convex (corners in (10, 170) degrees) so its 4-way split cannot fold
/// (CHANGES.md, WP-04).
pub const PAIR_MAX_QUALITY_DEG: f64 = 80.0;
/// Relaxation step toward the area-weighted centroid of the incident quads.
pub const RELAX_LAMBDA: f64 = 0.5;

/// vflag bits (DESIGN 8.4).
pub const VFLAG_CHAIN_MASK: u32 = (1 << 20) - 1;
pub const VFLAG_KIND_SHIFT: u32 = 20;
pub const VFLAG_CORNER: u32 = 1 << 24;
pub const VFLAG_CONE: u32 = 1 << 25;
pub const VFLAG_BOUNDARY: u32 = 1 << 26;
pub const VFLAG_SF: u32 = 1 << 27;

/// Parameters of the SF quadder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SfParams {
    /// Target quad count of the component (at least 1).
    pub target: u32,
    /// Relaxation iterations after the split (5).
    pub relax_iters: u32,
}

impl Default for SfParams {
    fn default() -> Self {
        SfParams { target: 1000, relax_iters: 5 }
    }
}

/// The triangle mesh the pairing and split work on.
#[derive(Clone, Debug, Default)]
pub(crate) struct SfTri {
    pub pos: Vec<V3>,
    pub tris: Vec<[u32; 3]>,
    /// Per halfedge (C2), NONE on the boundary.
    pub twin: Vec<u32>,
    /// Per halfedge: chain id (both halves) or NONE.
    pub hchain: Vec<u32>,
    /// Per chain id: its kind.
    pub chain_kind: Vec<FeatureKind>,
    /// Per vertex: corner of the chain network.
    pub vcorner: Vec<bool>,
    /// Per vertex: a face of `proj.mesh` near it (projection hint).
    pub vhint: Vec<u32>,
}

pub(crate) fn ferr(kind: ErrKind, msg: impl Into<String>) -> AqError {
    AqError::new(Stage::Fallback, kind, msg)
}

fn check_component(src: &ComponentMesh, p: &SfParams) -> Result<(), AqError> {
    let nf = src.mesh.tris.len();
    if nf == 0 {
        return Err(ferr(ErrKind::InvalidInput, "SF: empty component"));
    }
    if p.target == 0 {
        return Err(ferr(ErrKind::InvalidInput, "SF: target must be at least 1"));
    }
    if src.topo.twin.len() != 3 * nf
        || src.feats.he_kind.len() != 3 * nf
        || src.feats.he_chain.len() != 3 * nf
        || src.feats.vert_corner.len() != src.mesh.pos.len()
        || src.topo.vert_he.len() != src.mesh.pos.len()
    {
        return Err(ferr(ErrKind::InvalidInput, "SF: component arrays do not match its mesh"));
    }
    src.mesh.validate().map_err(|e| e.at(Stage::Fallback))?;
    Ok(())
}

/// `(l_sf, expected triangle count)` of a component for a target.
fn scales(area: f64, target: u32) -> Result<(f64, f64), AqError> {
    if !(area.is_finite() && area > 0.0) {
        return Err(ferr(ErrKind::InvalidInput, format!("SF: component area {area}")));
    }
    let h = (area / target as f64).sqrt();
    let l = L_SF * h;
    let t = area / (0.25 * 3f64.sqrt() * l * l);
    Ok((l, t))
}

fn fatal(e: &AqError) -> bool {
    matches!(e.kind, ErrKind::Cancelled | ErrKind::Timeout | ErrKind::Budget)
}

/// Relative quad count error accepted without a length correction.
pub const COUNT_TOL: f64 = 0.05;
/// Length corrections at most (each runs [`FIX_ITERS`] local remesh iterations).
pub const COUNT_ROUNDS: u32 = 3;
/// Local remesh iterations per length correction.
pub const FIX_ITERS: usize = 3;

fn rel_err(n: f64, target: u32) -> f64 {
    (n / target as f64 - 1.0).abs()
}

/// Remesh (proxy or local), count feedback, then pairing, split and relaxation.
fn sf_core(src: &ComponentMesh, proj: &Projector, p: &SfParams, ctl: &Ctl, allow_proxy: bool) -> Result<QuadMesh, AqError> {
    check_component(src, p)?;
    let (l0, t_exp) = scales(src.mesh.total_area(), p.target)?;
    let mut first: Option<SfTri> = None;
    if allow_proxy && t_exp >= MIN_PROXY_TRIS as f64 {
        let pp = ProxyParams { iters: 5, max_tris: 0, corner_deg: 50.0, sym_corner_deg: 70.0, hard_mode: 2 };
        match build_proxy(src, proj, &|_| l0, &pp, ctl) {
            Ok(px) => {
                first = Some(SfTri {
                    pos: px.mesh.pos,
                    tris: px.mesh.tris,
                    twin: px.topo.twin,
                    hchain: px.feats.he_chain,
                    chain_kind: px.feats.chains.iter().map(|c| c.kind).collect(),
                    vcorner: px.feats.vert_corner,
                    vhint: px.src_face,
                })
            }
            Err(e) if fatal(&e) => return Err(e.at(Stage::Fallback)),
            // A proxy failure (Topology, Numerical, Invariant) falls back to the local remesher.
            Err(_) => {}
        }
    }
    let mut rem: Option<Remesher> = None;
    let mut best = match first {
        Some(t) => t,
        None => {
            let mut r = Remesher::from_component(src, proj)?;
            r.run(proj, l0, remesh::ITERS, ctl)?;
            let t = r.result()?;
            rem = Some(r);
            t
        }
    };
    ctl.progress(Stage::Fallback, 0.5);
    // Count feedback: N ~ 1/l^2, so l is scaled by sqrt(N_pred / target).
    let mut n = quad::predicted_count(&best, ctl)?;
    let mut best_err = rel_err(n, p.target);
    let mut l = l0;
    for _ in 0..COUNT_ROUNDS {
        if rel_err(n, p.target) <= COUNT_TOL {
            break;
        }
        l *= (n / p.target as f64).sqrt();
        if rem.is_none() {
            rem = Some(Remesher::from_tri(&best, src.chi)?);
        }
        let Some(r) = rem.as_mut() else { break };
        r.run(proj, l, FIX_ITERS, ctl)?;
        let t = match r.result() {
            Ok(t) => t,
            Err(e) if fatal(&e) => return Err(e),
            Err(_) => break,
        };
        n = quad::predicted_count(&t, ctl)?;
        let e = rel_err(n, p.target);
        if e < best_err {
            best_err = e;
            best = t;
        }
    }
    ctl.check()?;
    ctl.progress(Stage::Fallback, 0.6);
    quad::quadrangulate(&best, proj, p.relax_iters, src.chi, ctl)
}

/// The SF quadder on one component (DESIGN 6.14). `proj` projects onto the component's
/// surface (built over `src.mesh`, as for `build_proxy`). Returns an all-quad, manifold,
/// consistently oriented mesh with about `p.target` quads, `face_comp = 0`, and vflag bits in
/// component-local chain ids with bit 27 (SF) set on every vertex.
///
/// Errors: `InvalidInput` (empty or inconsistent component, target 0, zero area),
/// `Cancelled` / `Timeout` from `ctl`, `Budget` on a failed allocation, `Topology` when the
/// remeshed surface is not manifold or changed its Euler characteristic (the engine then goes
/// to R3).
pub fn sf_quadrangulate(src: &ComponentMesh, proj: &Projector, p: &SfParams, ctl: &Ctl) -> Result<QuadMesh, AqError> {
    ctl.progress(Stage::Fallback, 0.0);
    let q = sf_core(src, proj, p, ctl, true)?;
    ctl.progress(Stage::Fallback, 1.0);
    Ok(q)
}

/// The SF quadder on many tiny components (DESIGN 7.1 debris), with no `aq_prep` proxy each:
/// every component goes through the local remesher (which coarsens or refines it to `l_sf`)
/// and the quadder, with a projector over its own mesh. `params[i]` belongs to `srcs[i]`.
/// Components run in parallel; the output (order and bits) does not depend on the thread
/// count. The first error in input order is returned.
pub fn sf_batch(srcs: &[&ComponentMesh], params: &[SfParams], ctl: &Ctl) -> Result<Vec<QuadMesh>, AqError> {
    if srcs.len() != params.len() {
        return Err(ferr(ErrKind::InvalidInput, "sf_batch: one SfParams per component"));
    }
    ctl.progress(Stage::Fallback, 0.0);
    let res: Vec<Result<QuadMesh, AqError>> = srcs
        .par_iter()
        .zip(params.par_iter())
        .map(|(src, p)| -> Result<QuadMesh, AqError> {
            ctl.check()?;
            let (fr, bvh) = src.source_geometry().map_err(|e| e.at(Stage::Fallback))?;
            let proj = Projector::new(&src.mesh, &src.topo, &fr, &bvh);
            sf_core(src, &proj, p, ctl, false)
        })
        .collect();
    let mut out = Vec::with_capacity(res.len());
    for r in res {
        out.push(r?);
    }
    ctl.progress(Stage::Fallback, 1.0);
    Ok(out)
}

/// Projection of a new vertex onto the source, always onto the surface when the source is not
/// empty: `Projector::project` with the reference normal, then with the source normal at the
/// hint face, then the local walk without a normal test, then the BVH closest point. The last
/// two only happen where the surface is thinner than the SF resolution.
pub(crate) fn project_onto(proj: &Projector, p: V3, hint: u32, n_ref: V3) -> Option<(SurfacePoint, V3)> {
    if let Some(r) = proj.project(p, hint, n_ref) {
        return Some(r);
    }
    if let Some(&n2) = proj.frames.n.get(hint as usize)
        && let Some(r) = proj.project(p, hint, n2)
    {
        return Some(r);
    }
    if let Some(r) = proj.walk(p, hint) {
        return Some(r);
    }
    let (sp, q, _) = proj.bvh.closest(p);
    if sp.face != NONE && math::is_finite3(q) { Some((sp, q)) } else { None }
}

/// Chain id (component-local) stored in a vflag, or NONE.
pub fn vflag_chain(f: u32) -> u32 {
    let c = f & VFLAG_CHAIN_MASK;
    if c == 0 { NONE } else { c - 1 }
}

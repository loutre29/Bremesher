//! # aq_core: contracts and conventions of the AutoQuad engine
//!
//! Every contract type of DESIGN.md section 5 and every convention C1 to C15 of section 4
//! lives here. **No other crate re-derives these formulas**: they call the functions below.
//! The executable tests in `tests/` pin every sign (DESIGN 6.1).
//!
//! The conventions, with the function that implements each one:
//!
//! * **C1 Mesh** ([`mesh`]): `TriMesh { pos, tris }`, triangles CCW seen from outside,
//!   `n_f = normalize((p1-p0) x (p2-p0))`, `NONE = u32::MAX`.
//! * **C2 Halfedges** ([`topo`]): `he = 3f + k` runs from `tris[f][k]` to `tris[f][(k+1)%3]`;
//!   corner `c = 3f + k` sits at the tail of halfedge `c`; canonical halfedge:
//!   `twin == NONE || he < twin` ([`Topology::is_canonical`]).
//! * **C3 Fans** ([`Topology::fan_out`], [`Topology::incoming`], [`Topology::ccw_out`]):
//!   `vert_he[v]` is the CW-most outgoing halfedge (the boundary one on the boundary),
//!   `ccw_out(he) = twin(prev(he))`, `H_in(v)` = interior halfedges with head v in CCW order.
//! * **C4 Face frame** ([`Frames`]): `e1 = normalize(p1-p0)`, `e2 = n x e1`,
//!   `edge_angle(he) = atan2(d.e2, d.e1)` ([`Frames::edge_angle`]).
//! * **C5 Transport** ([`Frames::kappa`]): `kappa[he] = wrap_pi(edge_angle(twin) + pi - edge_angle(he))`
//!   on the canonical halfedge, `kappa[twin] = -kappa[he]` exactly.
//! * **C6 Cross field** ([`cross::CrossField`]): one real `theta[f]`, branch k = `theta + k pi/2`,
//!   u-axis = branch 0 ([`cross::branch_dir`]).
//! * **C7 Matching** ([`cross::matching_from_theta`]): `p = round((theta_g - theta_f - kappa)/(pi/2)) mod 4`,
//!   `p[twin] = (4 - p) mod 4`, branch k of f maps to branch `(k - p) mod 4` of g.
//! * **C8 Residual** ([`cross::delta_edge`], [`cross::delta_from_p`]):
//!   `delta = theta_g - theta_f - kappa - p pi/2`, reduced modulo 2pi into (-pi, pi]
//!   (see CHANGES.md: p is stored mod 4, so the representative is the one nearest to zero).
//! * **C9 Index** ([`cross::index4_all`], [`cross::index4_vertex`]) and Poincare-Hopf
//!   ([`cross::poincare_hopf`]).
//! * **C10 Rotations and transitions** ([`cross::rot`], [`cross::Transition`]):
//!   `phi_g = R(-r) phi_f + t`, twin = inverse.
//! * **C11 Orientation and folds** ([`cross::jacobian`], [`cross::fold_eps`]).
//! * **C12 Combing** ([`cross::comb_theta`], [`cross::comb_matching`], [`cross::raw_to_combed_branch`]).
//! * **C13 Edge family** ([`cross::edge_family`]).
//! * **C14 Units and tolerances** ([`tol`]).
//! * **C15 Determinism and seed** ([`rng::SplitMix64`]).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod alloc;
pub mod bvh;
pub mod ctl;
pub mod cross;
pub mod cut;
pub mod error;
pub mod exact;
pub mod features;
pub mod fields;
pub mod frames;
pub mod layout;
pub mod loops;
pub mod math;
pub mod mesh;
pub mod param;
pub mod project;
pub mod rng;
pub mod tmesh;
pub mod tol;
pub mod topo;

pub use bvh::Bvh;
pub use ctl::{Ctl, CtlState};
pub use cross::{
    CrossField, Transition, delta_from_p, edge_family, index4_all, matching_from_theta, rot,
    wrap_pi,
};
pub use cut::{CutArc, CutGraph};
pub use error::{AqError, ErrKind, Stage};
pub use features::{Chain, FeatureKind, FeatureSet};
pub use fields::{CurvField, SizeField};
pub use frames::Frames;
pub use layout::{Layout, LayoutHealth};
pub use loops::{LoopCand, SurfacePoint};
pub use mesh::{NONE, QuadMesh, TriMesh, V2, V3};
pub use param::{ParamReport, SeamlessParam};
pub use project::Projector;
pub use rng::SplitMix64;
pub use tmesh::{ArcKind, NodeKind, QComb, TArc, TMesh, TNode, TPatch};
pub use topo::{FanIter, InIter, Topology};

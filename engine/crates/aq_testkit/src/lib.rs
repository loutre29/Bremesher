//! # aq_testkit: synthetic meshes and analytic stage outputs (DESIGN 6.2, WP-01b)
//!
//! Dev-only. Every generator is parameterized and deterministic (same arguments, same bits).
//! It makes the stage crates testable without their upstream crates:
//!
//! * [`meshes`]: plane grids (rotation, jitter), a planar square disk, uv sphere, icosphere,
//!   octahedral sphere, open and closed cylinders, torus, cube with sharp edges, hemisphere
//!   with a Symmetry cut chain, height fields, the noise adder.
//! * [`implicit`]: the double torus (an implicit genus-2 surface, marching tetrahedra, then a
//!   small isotropic cleanup remesh).
//! * [`surface`]: [`Surface`] bundles mesh, topology, frames and a [`aq_core::FeatureSet`]
//!   built from feature edges (chains and corners as in DESIGN 6.4). It holds everything an
//!   aq_prep `Proxy` holds apart from the source tracking.
//! * [`obj`]: OBJ loader for `engine/testdata`.
//! * [`field`]: analytic cross fields ([`FieldKind`]: constant, circumferential and twisted on
//!   a cylinder or torus, cube-aligned, planted singularities), analytic curvature, uniform size.
//! * [`cut`]: [`analytic_cut`] (dual tree, prune, combing, arcs as in DESIGN 6.8) and the arc
//!   builder used by the params.
//! * [`param`]: exact seamless params ([`ParamCase`]): [`plane_param`], [`cylinder_param`],
//!   [`torus_param`], [`cube_sphere_param`], [`perturbed_cube_sphere_param`], with exact loops.
//! * [`tmesh_gen`]: T-mesh generators: [`tmesh_rect`], [`tmesh_grid`], [`tmesh_cube`],
//!   [`tmesh_random_bidirected`], [`tmesh_random_dissection`], [`tmesh_helix`].
//!
//! Every output is checked with the aq_core validators before it is returned.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

pub mod cut;
pub mod field;
pub mod implicit;
pub mod meshes;
pub mod obj;
pub mod param;
pub mod surface;
pub mod tmesh_gen;
mod util;

pub use cut::{analytic_cut, cut_graph_from_flags};
pub use field::{CurvKind, FieldKind, PlantChart, analytic_curv, analytic_field, uniform_size};
pub use implicit::{double_torus, double_torus_with};
pub use meshes::{
    SynthMesh, add_noise, cube_sharp, cylinder_closed, cylinder_open, heightfield, hemisphere,
    icosphere, merge, octasphere, plane_grid, square_disk, torus, uv_sphere,
};
pub use obj::{load_obj, load_testdata, parse_obj, testdata_dir, testdata_names};
pub use param::{
    CubeSphereOpts, ParamCase, ParamOpts, cube_sphere_param, cube_sphere_param_with,
    cylinder_param, cylinder_param_with, param_from_corners, perturbed_cube_sphere_param,
    perturbed_cube_sphere_param_with, plane_param, plane_param_with, torus_param,
    torus_param_with,
};
pub use surface::Surface;
pub use tmesh_gen::{
    arc_defaults, tmesh_box, tmesh_cube, tmesh_from_rects, tmesh_grid, tmesh_grid_sized,
    tmesh_helix, tmesh_random_bidirected, tmesh_random_dissection, tmesh_rect,
};

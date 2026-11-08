//! # aq_io: OBJ read/write and debug dumps (DESIGN 6.15, WP-01)
//!
//! * [`obj`]: OBJ parsing (`v`, `f` with any `/` suffixes, negative indices, polygons kept
//!   or fan-triangulated) and writing (triangles, quads as polygons), bit-exact round trip.
//! * [`dump`]: the stage dumps `proxy.obj`, `field.obj`, `cones.obj`, `cut.obj`,
//!   `param.obj`, `layout.obj`, `quant.json`, `patches.obj`, and [`dump::DumpDir`].
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod dump;
pub mod obj;

pub use dump::{
    DumpDir, dump_cones, dump_cut, dump_field, dump_layout, dump_param, dump_patches, dump_proxy, dump_quant,
};
pub use obj::{
    ObjData, parse_obj, read_obj, read_obj_poly, read_obj_tri, write_file, write_obj_quad, write_obj_tri,
    write_quad_obj, write_tri_obj,
};

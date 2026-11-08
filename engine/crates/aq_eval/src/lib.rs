//! # aq_eval: evaluation metrics (DESIGN 6.15, WP-01)
//!
//! * [`quad_stats`]: the statistics of `tools/quad_stats.py`, same rules (parity tested on
//!   fixtures produced by the script).
//! * [`hausdorff`] and [`vertex_distance`]: sampled distances between the output and the input.
//! * [`loop_closure`]: edge loop closure and helix suspects.
//! * [`bisect`] and [`mirror_weld`]: the symmetry pre and post steps of the CLI and add-on.
//! * [`fan_triangulate`] and [`poly_from_tris`]: conversions used by the metrics.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod hausdorff;
pub mod loops;
mod poly;
pub mod stats;
pub mod symmetry;

pub use hausdorff::{Hausdorff, hausdorff, vertex_distance};
pub use loops::{LoopReport, loop_closure};
pub use poly::{fan_triangulate, poly_from_tris};
pub use stats::{QuadStats, py_round, quad_stats};
pub use symmetry::{WeldReport, bisect, mirror_weld};

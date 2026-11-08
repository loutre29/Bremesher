//! # aq_engine: orchestration, components, fallback ladder, budgets, report (DESIGN 7)
//!
//! WP-05 (M0 path) with the hook for WP-16 in [`pipeline`].
//!
//! [`remesh`] is the whole engine call behind the C ABI (aq_ffi) and the CLI (aq_cli):
//!
//! 1. **Ingest** (S0, `aq_prep::ingest`) and the S0 invariants (7.3).
//! 2. **Split components**, area and chi per component, global scale `1/h0` with
//!    `h0 = sqrt(A_total / N_target)` (quad units, DESIGN 2).
//! 3. **Budgets** ([`budget`]): per-component targets with floors (24 closed, 8 open), debris,
//!    proxy triangle shares and time shares.
//! 4. **Per component, in parallel** (one rayon pool per call, `threads` workers, `install`),
//!    each under `catch_unwind`: the ladder R0, R1 ([`pipeline`], not wired at M0), R2 (the SF
//!    quadder, `aq_fallback`), R3 (the component's input triangles go to `failed_tris`). Debris
//!    goes to SF directly with no proxy (`sf_batch`). Every result is verified (valid, all
//!    quads, Euler characteristic) before it is accepted.
//! 5. **Count correction** (7.1 step 8), **merge** ([`merge`]) and the **report JSON** (DESIGN 9).
//!
//! The call never panics: a panic in a component sends only that component down the ladder,
//! and a panic anywhere else gives [`Status::Panic`]. There is no global state.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod budget;
pub mod host;
pub mod merge;
pub mod pipeline;
mod report;
mod run;

use std::path::PathBuf;

use aq_core::{Ctl, QuadMesh, TriMesh};

pub use merge::{MergeReport, euler_char};

/// The C ABI version this engine implements (DESIGN 8.1 `AQ_ABI_VERSION`).
pub const ABI_VERSION: u32 = 1;

/// Name of the engine path, written to the report (`"engine"` key).
pub const ENGINE_PATH: &str = if pipeline::AVAILABLE { "M1" } else { "M0" };

/// Largest accepted `target_quads` (a larger value is `Status::InvalidInput`).
pub const MAX_TARGET_QUADS: u32 = 4_000_000;

/// `debug_flags` bits (DESIGN 8.1, plus two test-only bits, see CHANGES.md WP-05).
pub mod debug {
    /// Write stage dumps to `dump_dir`.
    pub const DUMP: u32 = 1 << 0;
    /// Skip R0 and start the ladder at R1.
    pub const FORCE_R1: u32 = 1 << 1;
    /// Send every component to SF.
    pub const FORCE_SF: u32 = 1 << 2;
    /// Test only: panic inside the job of component 0 (exercises the per-component catch).
    pub const PANIC_COMPONENT: u32 = 1 << 30;
    /// Test only: panic in the engine outside any component (exercises the top-level catch).
    pub const PANIC_TOP: u32 = 1 << 31;
}

/// Engine parameters: `AqParams` (DESIGN 8.1) without `struct_size`, plus CLI-only fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub target_quads: u32,
    pub adaptive_size: f32,
    pub curvature_align: f32,
    pub flow_smoothness: f32,
    pub density_strength: f32,
    pub guide_strength: f32,
    /// 0 ignore, 1 soft (field only), 2 crisp (field + iso + arcs).
    pub hard_edges: u8,
    pub close_loops: u8,
    pub allow_collapse: u8,
    pub ring_min: u32,
    pub max_proxy_tris: u32,
    /// 0 = all logical cores.
    pub threads: u32,
    /// Seconds; +inf means no limit.
    pub time_limit_s: f32,
    pub debug_flags: u32,
    pub seed: u64,
    pub dump_dir: Option<PathBuf>,
    /// CLI `--stop-after S0..S11` (0..11). M0 honours 0 and 1 (the ingested mesh is returned);
    /// later stages are ignored with a warning.
    pub stop_after: Option<u8>,
}

impl Default for Params {
    /// The defaults of DESIGN 8.1.
    fn default() -> Self {
        Params {
            target_quads: 5000,
            adaptive_size: 0.5,
            curvature_align: 1.0,
            flow_smoothness: 1.0,
            density_strength: 1.0,
            guide_strength: 1.0,
            hard_edges: 2,
            close_loops: 1,
            allow_collapse: 1,
            ring_min: 8,
            max_proxy_tris: 150_000,
            threads: 0,
            time_limit_s: 60.0,
            debug_flags: 0,
            seed: 0,
            dump_dir: None,
            stop_after: None,
        }
    }
}

/// Overall status (`AqResult.status`, DESIGN 8.1).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Status {
    /// Every component succeeded.
    Ok = 0,
    /// Usable output, but at least one component failed (its triangles are in `failed_tris`)
    /// or went down the ladder after a failure.
    Degraded = 1,
    /// No component produced quads; `failed_tris` lists every input triangle that survived
    /// validation (Python runs QuadriFlow on them).
    Failed = 2,
    /// The caller cancelled; the output is empty.
    Cancelled = 3,
    /// Bad parameters or input data; the output is empty.
    InvalidInput = 4,
    /// A panic was caught outside the per-component catch; the output is empty.
    Panic = 5,
}

impl Status {
    pub fn code(self) -> i32 {
        self as i32
    }
    /// The status with this code, if any.
    pub fn from_code(code: i32) -> Option<Status> {
        Some(match code {
            0 => Status::Ok,
            1 => Status::Degraded,
            2 => Status::Failed,
            3 => Status::Cancelled,
            4 => Status::InvalidInput,
            5 => Status::Panic,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Degraded => "degraded",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
            Status::InvalidInput => "invalid_input",
            Status::Panic => "panic",
        }
    }
}

/// Per-component status (`AqResult.comp_status`, DESIGN 8.1).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompStatus {
    /// R0, the full pipeline.
    Full = 0,
    /// R1, the retried pipeline.
    Retried = 1,
    /// R2, the SF quadder.
    Sf = 2,
    /// R3, failed: the component's input triangles are in `failed_tris` (QuadriFlow).
    Failed = 3,
    /// Debris sent to SF directly.
    DebrisSf = 4,
}

impl CompStatus {
    pub fn name(self) -> &'static str {
        match self {
            CompStatus::Full => "full",
            CompStatus::Retried => "retried",
            CompStatus::Sf => "sf",
            CompStatus::Failed => "failed",
            CompStatus::DebrisSf => "debris_sf",
        }
    }
    /// The ladder rung (0 to 3) that produced the component.
    pub fn rung(self) -> u8 {
        match self {
            CompStatus::Full => 0,
            CompStatus::Retried => 1,
            CompStatus::Sf | CompStatus::DebrisSf => 2,
            CompStatus::Failed => 3,
        }
    }
}

/// Input of one engine call. Positions are expected with a bounding box diagonal of about 10
/// (DESIGN C14), but any finite scale works.
#[derive(Clone, Debug, Default)]
pub struct Input<'a> {
    pub mesh: TriMesh,
    /// Feature edges as vertex index pairs.
    pub feat_pairs: &'a [[u32; 2]],
    /// One kind per pair: 1 Hard, 2 Guide, 3 GuideHard, 5 Symmetry (0 ignored).
    pub feat_kind: &'a [u8],
    /// Per input vertex density multipliers (2 = twice the quads), or None.
    pub density: Option<&'a [f32]>,
}

/// Result of one engine call.
#[derive(Clone, Debug)]
pub struct Output {
    pub status: Status,
    /// The merged quad mesh, in input units (empty for statuses 3, 4 and 5).
    pub mesh: QuadMesh,
    /// One entry per component, in component order (`face_comp` indexes it).
    pub comp_status: Vec<CompStatus>,
    /// Input triangle indices of the failed components (sorted).
    pub failed_tris: Vec<u32>,
    /// The report (DESIGN 9 "Report JSON").
    pub report: serde_json::Value,
    /// "" when the status is Ok.
    pub error: String,
}

impl Output {
    /// The report as compact JSON text.
    pub fn report_json(&self) -> String {
        serde_json::to_string(&self.report).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Runs the engine on one mesh. Never panics, never retains its inputs.
///
/// The effective deadline is the earlier of `ctl`'s deadline and `time_limit_s` from now.
/// Progress and stage are published through `ctl`.
pub fn remesh(input: Input, params: &Params, ctl: &Ctl) -> Output {
    run::remesh(input, params, ctl)
}

/// The report JSON (DESIGN 9) of a call that produced nothing, for errors found outside the
/// engine (aq_ffi argument checks). Every DESIGN 9 key is present, with zero timings.
pub fn error_report(status: Status, msg: &str) -> serde_json::Value {
    run::error_report(status, msg)
}

//! Typed errors (DESIGN 5, `aq_core::error`).
//!
//! Every public function of the engine returns `Result<T, AqError>`. The error carries the
//! stage that produced it and a kind that drives the fallback ladder (DESIGN 7.2).
//!
//! Functions of `aq_core` itself do not know which stage calls them: their errors carry
//! [`Stage::Engine`]. A caller restamps them with [`AqError::at`] (or
//! `.map_err(|e| e.at(Stage::Field))`).

use std::fmt;

/// Pipeline stage. The discriminant is the value written to `AqControl.stage` (DESIGN 8.1).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Ingest = 0,
    Proxy = 1,
    Curv = 2,
    Field = 3,
    Loops = 4,
    Cut = 5,
    Param = 6,
    Layout = 7,
    Quantize = 8,
    Patch = 9,
    Post = 10,
    Fallback = 11,
    Engine = 12,
    Ffi = 13,
}

impl Stage {
    /// All stages in discriminant order.
    pub const ALL: [Stage; 14] = [
        Stage::Ingest,
        Stage::Proxy,
        Stage::Curv,
        Stage::Field,
        Stage::Loops,
        Stage::Cut,
        Stage::Param,
        Stage::Layout,
        Stage::Quantize,
        Stage::Patch,
        Stage::Post,
        Stage::Fallback,
        Stage::Engine,
        Stage::Ffi,
    ];

    /// Inverse of `stage as u32`.
    pub fn from_u32(x: u32) -> Option<Stage> {
        Stage::ALL.get(x as usize).copied()
    }

    /// Lower-case name, as used by the report JSON `timing_ms` keys (DESIGN 9).
    pub fn name(self) -> &'static str {
        match self {
            Stage::Ingest => "ingest",
            Stage::Proxy => "proxy",
            Stage::Curv => "curv",
            Stage::Field => "field",
            Stage::Loops => "loops",
            Stage::Cut => "cut",
            Stage::Param => "param",
            Stage::Layout => "layout",
            Stage::Quantize => "quantize",
            Stage::Patch => "patch",
            Stage::Post => "post",
            Stage::Fallback => "fallback",
            Stage::Engine => "engine",
            Stage::Ffi => "ffi",
        }
    }
}

/// Error kind (DESIGN 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrKind {
    /// Bad user data (NaN, out-of-range index, empty mesh...).
    InvalidInput,
    /// Non-manifold or inconsistently oriented mesh.
    Topology,
    /// NaN, non-SPD factorization, bound violations.
    Numerical,
    /// A convention or contract invariant does not hold (a bug or a degenerate case).
    Invariant,
    /// No solution exists (quantization, layout).
    Infeasible,
    /// The caller set the cancel flag.
    Cancelled,
    /// The deadline passed.
    Timeout,
    /// An allocation failed or a size budget is exceeded.
    Budget,
    /// Anything else (caught panics, internal bugs).
    Internal,
}

impl ErrKind {
    pub fn name(self) -> &'static str {
        match self {
            ErrKind::InvalidInput => "invalid_input",
            ErrKind::Topology => "topology",
            ErrKind::Numerical => "numerical",
            ErrKind::Invariant => "invariant",
            ErrKind::Infeasible => "infeasible",
            ErrKind::Cancelled => "cancelled",
            ErrKind::Timeout => "timeout",
            ErrKind::Budget => "budget",
            ErrKind::Internal => "internal",
        }
    }
}

/// The engine's error type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AqError {
    pub stage: Stage,
    pub kind: ErrKind,
    pub msg: String,
}

impl AqError {
    pub fn new(stage: Stage, kind: ErrKind, msg: impl Into<String>) -> Self {
        AqError { stage, kind, msg: msg.into() }
    }

    /// Error raised inside aq_core (stage [`Stage::Engine`] until restamped).
    pub fn core(kind: ErrKind, msg: impl Into<String>) -> Self {
        AqError::new(Stage::Engine, kind, msg)
    }

    /// Restamps the stage.
    pub fn at(mut self, stage: Stage) -> Self {
        self.stage = stage;
        self
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        AqError::core(ErrKind::InvalidInput, msg)
    }
    pub fn topology(msg: impl Into<String>) -> Self {
        AqError::core(ErrKind::Topology, msg)
    }
    pub fn numerical(msg: impl Into<String>) -> Self {
        AqError::core(ErrKind::Numerical, msg)
    }
    pub fn invariant(msg: impl Into<String>) -> Self {
        AqError::core(ErrKind::Invariant, msg)
    }
    pub fn budget(msg: impl Into<String>) -> Self {
        AqError::core(ErrKind::Budget, msg)
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        AqError::core(ErrKind::Internal, msg)
    }
}

impl fmt::Display for AqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}/{}] {}", self.stage.name(), self.kind.name(), self.msg)
    }
}

impl std::error::Error for AqError {}

/// Shorthand used by the validators.
pub(crate) fn inv(msg: impl Into<String>) -> AqError {
    AqError::invariant(msg)
}

/// Returns `Err(Invariant)` with `msg` when `cond` is false.
pub(crate) fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), AqError> {
    if cond { Ok(()) } else { Err(inv(msg())) }
}

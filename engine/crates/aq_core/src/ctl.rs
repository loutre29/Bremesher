//! Cancellation, progress and deadline control (DESIGN 5 `aq_core::ctl`, 8.2).
//!
//! `Ctl` borrows three atomics owned by the caller (the FFI points them at `AqControl`) and
//! carries a deadline. Every long loop calls [`Ctl::check`] at least every 10 ms of work.
//! There is no global state: concurrent calls each own their `Ctl`.

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::error::{AqError, ErrKind, Stage};

/// A far deadline (about 100 years) used when there is no time limit.
const FAR: Duration = Duration::from_secs(100 * 365 * 24 * 3600);

/// Borrowed control block. Cheap to copy; `Send + Sync`.
#[derive(Clone, Copy, Debug)]
pub struct Ctl<'a> {
    cancel: &'a AtomicI32,
    progress: &'a AtomicU32,
    stage: &'a AtomicU32,
    deadline: Instant,
}

impl<'a> Ctl<'a> {
    pub fn new(
        cancel: &'a AtomicI32,
        progress: &'a AtomicU32,
        stage: &'a AtomicU32,
        deadline: Instant,
    ) -> Self {
        Ctl { cancel, progress, stage, deadline }
    }

    /// `Err(Cancelled)` when the caller set the cancel flag, `Err(Timeout)` when the deadline
    /// passed. The error carries the stage last reported through [`Ctl::progress`].
    pub fn check(&self) -> Result<(), AqError> {
        if self.cancel.load(Ordering::Relaxed) != 0 {
            return Err(AqError::new(self.stage(), ErrKind::Cancelled, "cancelled by caller"));
        }
        if Instant::now() >= self.deadline {
            return Err(AqError::new(self.stage(), ErrKind::Timeout, "deadline reached"));
        }
        Ok(())
    }

    /// True when the cancel flag is set (no deadline test).
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) != 0
    }

    /// Publishes the current stage and its progress fraction (clamped to [0, 1], NaN gives 0).
    /// The progress is stored as f32 bits with a relaxed store (DESIGN 8.2).
    pub fn progress(&self, stage: Stage, frac: f32) {
        let f = if frac.is_nan() { 0.0 } else { frac.clamp(0.0, 1.0) };
        self.stage.store(stage as u32, Ordering::Relaxed);
        self.progress.store(f.to_bits(), Ordering::Relaxed);
    }

    /// The stage last published (Engine when the stored value is unknown).
    pub fn stage(&self) -> Stage {
        Stage::from_u32(self.stage.load(Ordering::Relaxed)).unwrap_or(Stage::Engine)
    }

    /// Time left before the deadline (zero when it passed).
    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// A control block sharing the same atomics with the earlier of the two deadlines.
    pub fn sub(&self, deadline: Instant) -> Ctl<'_> {
        Ctl {
            cancel: self.cancel,
            progress: self.progress,
            stage: self.stage,
            deadline: deadline.min(self.deadline),
        }
    }
}

/// Owned atomics for callers that do not have an `AqControl` (CLI, tests).
#[derive(Debug, Default)]
pub struct CtlState {
    pub cancel: AtomicI32,
    pub progress: AtomicU32,
    pub stage: AtomicU32,
}

impl CtlState {
    pub fn new() -> Self {
        CtlState {
            cancel: AtomicI32::new(0),
            progress: AtomicU32::new(0),
            stage: AtomicU32::new(Stage::Engine as u32),
        }
    }

    /// A `Ctl` with the given deadline.
    pub fn ctl(&self, deadline: Instant) -> Ctl<'_> {
        Ctl::new(&self.cancel, &self.progress, &self.stage, deadline)
    }

    /// A `Ctl` whose deadline is `limit` from now.
    pub fn ctl_for(&self, limit: Duration) -> Ctl<'_> {
        let now = Instant::now();
        let d = now.checked_add(limit).or_else(|| now.checked_add(FAR)).unwrap_or(now);
        self.ctl(d)
    }

    /// A `Ctl` with no practical deadline.
    pub fn unbounded(&self) -> Ctl<'_> {
        self.ctl_for(FAR)
    }

    /// Sets the cancel flag.
    pub fn cancel(&self) {
        self.cancel.store(1, Ordering::Relaxed);
    }

    /// The last published progress fraction.
    pub fn progress_frac(&self) -> f32 {
        f32::from_bits(self.progress.load(Ordering::Relaxed))
    }
}

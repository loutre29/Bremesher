//! S8 exact uv motorcycle layout and T-mesh extraction (DESIGN 6.10, WP-11).
//!
//! Every curve is an iso-line of the seamless param, traced face by face with exact
//! predicates ([`geom`]). The pipeline of one round is:
//!
//! 1. constraint curves: the constraint feature chains (Hard, GuideHard, Boundary, Symmetry)
//!    and the exact loops ([`curves::trace_loops`]);
//! 2. emitters at cones, chain corners, open chain ends and seeds ([`curves::emitters`]);
//! 3. simultaneous motorcycles ([`trace_all`]);
//! 4. embedding of every curve into the split proxy W' ([`embed`]);
//! 5. T-mesh extraction and patch validation ([`extract`]).
//!
//! [`build_layout`] re-runs 2 to 5 with accumulated seeds while a patch is not a 4-corner
//! disk, at most `max_rounds` extra times.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

pub mod ctx;
pub mod curves;
pub mod embed;
pub mod extract;
pub mod geom;
pub mod trace;

use aq_core::{AqError, Ctl, ErrKind, FeatureSet, Layout, LoopCand, SeamlessParam, Stage, Topology, TriMesh, V2};

pub use embed::{Embedded, embed};
pub use extract::{NodeHint, extract, node_hints};
pub use geom::{Line, Pt};
pub use trace::{TraceStats, trace_all};

/// The proxy surface as seen by S8. aq_prep's `Proxy` is not a dependency of aq_layout, so
/// the layout takes the three parts it needs (see CHANGES.md, WP-11).
#[derive(Clone, Copy, Debug)]
pub struct ProxyView<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub feats: &'a FeatureSet,
}

/// S8 parameters (DESIGN 6.10 API).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutParams {
    /// Length cap of a trace in quad units; `<= 0` selects `4 sqrt(A_c)` per component.
    pub max_trace_len: f64,
    /// Seeded re-runs after the first run.
    pub max_rounds: u32,
    /// M2: continuation traces at open guide ends (no effect in M1: guides are not traced).
    pub guide_continuation: bool,
    /// M2: separatrix tail simplification (no effect in M1).
    pub simplify: bool,
}

impl Default for LayoutParams {
    fn default() -> Self {
        LayoutParams { max_trace_len: 0.0, max_rounds: 3, guide_continuation: true, simplify: false }
    }
}

/// Source of a curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CurveKind {
    /// A constraint feature chain (its edges are proxy edges; no pieces).
    Chain,
    /// An exact loop (constraint).
    Loop,
    /// A trace from a cone, a chain corner or an open chain end.
    Separatrix,
    /// A trace from a validation seed.
    SeedTrace,
}

impl CurveKind {
    /// Constraint curves are never crossed and never truncated.
    pub fn is_constraint(self) -> bool {
        matches!(self, CurveKind::Chain | CurveKind::Loop)
    }
}

/// The part of a curve inside one face, in that face's chart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    pub face: u32,
    pub line: Line,
    /// First point in travel order.
    pub start: Pt,
    /// Last point in travel order.
    pub end: Pt,
    /// Local edge the curve entered through (255 at an emitter).
    pub entry: u8,
    /// Local edge the full chord leaves through.
    pub exit: u8,
    /// Local edge the piece runs along, or 255.
    pub on_edge: u8,
    /// Parametric time at `start`.
    pub t_in: f64,
    /// Approximate varying coordinate of `start`.
    pub x_in: f64,
}

impl Piece {
    /// Parametric time at a point of the piece.
    pub fn time_at(&self, tri: &[V2; 3], p: Pt) -> f64 {
        self.t_in + (geom::aval(tri, self.line, p) - self.x_in).abs()
    }
    /// True when start and end are the same point.
    pub fn is_zero(&self, tri: &[V2; 3]) -> bool {
        geom::cmp_a(tri, self.line, self.start, self.end) == std::cmp::Ordering::Equal
    }
}

/// A curve: a chain (no pieces), an exact loop or a trace.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub kind: CurveKind,
    /// Feature chain id or NONE.
    pub chain: u32,
    /// Source index: chain id, LoopCand index or emitter index.
    pub src: u32,
    pub pieces: Vec<Piece>,
    pub closed: bool,
}

/// A trace start: a ray leaving the corner at `vert` of face `face` along direction code
/// `dir` (0 +u, 1 +v, 2 -u, 3 -v) of that face's chart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Emitter {
    pub vert: u32,
    pub face: u32,
    pub dir: u8,
    pub kind: CurveKind,
}

/// A validation seed at proxy vertex `vert`: emits the ray `dir` given in the chart of
/// `face`, or every free axis when `dir == 255`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Seed {
    pub vert: u32,
    pub face: u32,
    pub dir: u8,
}

/// Outcome of a layout step that is not a plain error (DESIGN 6.10 API).
#[derive(Clone, Debug, PartialEq)]
pub enum LayoutIssue {
    /// Some patch is not a 4-corner disk: re-run with these extra seeds.
    Validation(Vec<Seed>),
    /// The trace with this index (into the emitter list) exceeded `max_trace_len`.
    Runaway(u32),
    Fatal(AqError),
}

impl From<AqError> for LayoutIssue {
    fn from(e: AqError) -> Self {
        LayoutIssue::Fatal(e)
    }
}

/// Statistics of a [`build_layout_with`] run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutStats {
    /// Runs performed (1 = no seed round).
    pub runs: u32,
    /// All seeds used by the final run.
    pub seeds: Vec<Seed>,
    pub n_loops: usize,
    pub n_traces: usize,
    pub n_events: u64,
    /// Stop events voided by the validation of the event engine.
    pub n_voided: u64,
}

/// S8 (DESIGN 6.10): builds the layout of `prm` over the proxy.
pub fn build_layout(
    px: &ProxyView,
    prm: &SeamlessParam,
    loops: &[LoopCand],
    p: &LayoutParams,
    ctl: &Ctl,
) -> Result<Layout, AqError> {
    build_layout_with(px, prm, loops, &[], p, ctl).map(|(l, _)| l)
}

/// [`build_layout`] with initial seeds, also returning statistics. Each run starts from
/// scratch (pure function of the inputs and the accumulated seeds).
pub fn build_layout_with(
    px: &ProxyView,
    prm: &SeamlessParam,
    loops: &[LoopCand],
    seeds0: &[Seed],
    p: &LayoutParams,
    ctl: &Ctl,
) -> Result<(Layout, LayoutStats), AqError> {
    let st = |e: AqError| e.at(Stage::Layout);
    ctl.check().map_err(st)?;
    let cx = ctx::Ctx::new(px, prm).map_err(st)?;
    let mut base = curves::chain_curves(&cx);
    let loop_curves = curves::trace_loops(&cx, loops, p, ctl).map_err(st)?;
    let n_loops = loop_curves.len();
    base.extend(loop_curves);
    let mut seeds: Vec<Seed> = Vec::new();
    for s in seeds0 {
        if !seeds.contains(s) {
            seeds.push(*s);
        }
    }
    let mut stats = LayoutStats { n_loops, ..Default::default() };
    for _round in 0..=p.max_rounds {
        ctl.check().map_err(st)?;
        stats.runs += 1;
        let ems = curves::emitters(&cx, &base, &seeds).map_err(st)?;
        let (traces, ts) = match trace::trace_curves(&cx, &base, &ems, p, ctl) {
            Ok(x) => x,
            Err(LayoutIssue::Runaway(i)) => {
                return Err(AqError::new(
                    Stage::Layout,
                    ErrKind::Infeasible,
                    format!("trace {i} exceeded max_trace_len (runaway separatrix)"),
                ));
            }
            Err(LayoutIssue::Fatal(e)) => return Err(st(e)),
            Err(LayoutIssue::Validation(_)) => return Err(st(AqError::internal("unexpected validation issue"))),
        };
        stats.n_traces = traces.len();
        stats.n_events += ts.events;
        stats.n_voided += ts.voided;
        let mut all = base.clone();
        all.extend(traces);
        let emb = embed::embed_ctx(&cx, &all).map_err(st)?;
        let hints = extract::node_hints_ctx(&cx, &ems);
        match extract::extract_ctx(&cx, &emb, &hints) {
            Ok(l) => {
                stats.seeds = seeds;
                return Ok((l, stats));
            }
            Err(LayoutIssue::Validation(new)) => {
                let before = seeds.len();
                for s in new {
                    if !seeds.contains(&s) {
                        seeds.push(s);
                    }
                }
                if seeds.len() == before {
                    return Err(AqError::new(
                        Stage::Layout,
                        ErrKind::Invariant,
                        "layout validation requested only seeds that are already present",
                    ));
                }
            }
            Err(LayoutIssue::Runaway(i)) => {
                return Err(AqError::new(Stage::Layout, ErrKind::Infeasible, format!("runaway trace {i}")));
            }
            Err(LayoutIssue::Fatal(e)) => return Err(st(e)),
        }
    }
    Err(AqError::new(
        Stage::Layout,
        ErrKind::Invariant,
        format!("layout still has invalid patches after {} seed rounds", p.max_rounds),
    ))
}

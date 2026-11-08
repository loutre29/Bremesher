//! # aq_quantize: S9 integer arc lengths (DESIGN 6.11, WP-12)
//!
//! Input: the mesh-free [`TMesh`] of a component (S8) and the target quad count N_c.
//! Output: an integer length `n_a` per arc and the combinatorial quad mesh [`QComb`].
//!
//! * **Constraints.** One node per (patch, axis) plus a ground G with a free balance. An arc
//!   occurrence on side k of patch P adds `sigma n_a` to node (P, k mod 2) with sigma = +1 on
//!   sides 0 and 1 and -1 on sides 2 and 3; a boundary arc's other end is G. Every node other
//!   than G must balance, and `n_a >= lb_a`.
//! * **Cost.** `f_a(n) = weight (n - sigma l_a)^2 / max(sigma l_a, 0.5)`, target 0 for wrap
//!   arcs; weights and lower bounds follow the DESIGN table ([`arc_weight`], [`table_lb`]).
//! * **Solver** ([`solve_bimdf`]): min-cost flow on the double cover of the bi-directed graph
//!   (successive shortest paths, Dijkstra with potentials, capacity scaling above a total
//!   excess of 2000), parity repair along closed walks, integer repair by cheapest double
//!   cover paths, and a polish by improving closed walks.
//! * **Validity** ([`build_comb`]): zero arcs merge nodes and zero patches are glued; the
//!   QComb is checked for manifoldness, valence and index, corner merges, pinches, short rings
//!   and its Euler characteristic. A violation sets lb = 1 on the zero arcs that caused it and
//!   the system is solved again (lazy cuts).
//! * **Count** ([`quantize`]): the scale sigma is found by a secant method on
//!   `log Q(sigma) - log N_c`, with the validity rounds inside each step and at most 24
//!   solves in total.
//!
//! Additions and deviations are listed in CHANGES.md (WP-12).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop)]

mod bimdf;
mod comb;
mod graph;

use aq_core::{AqError, Ctl, ErrKind, QComb, Stage, TMesh};

pub use comb::Violation;
pub use graph::{arc_target, arc_weight, objective, table_lb};

use bimdf::Dijkstra;
use graph::{CGraph, err};

/// Total number of solves allowed in one [`quantize`] call (DESIGN 6.11 step 5).
pub const MAX_SOLVES: u32 = 24;

/// Parameters of S9 (DESIGN 6.11, defaults from DESIGN 13).
#[derive(Clone, Debug, PartialEq)]
pub struct QuantParams {
    /// Separatrix and seed-trace arcs may collapse to 0 (lb 0).
    pub allow_collapse: bool,
    /// Relative count tolerance of the secant (0.04).
    pub count_tol: f64,
    /// Secant iterations after the first solve (5). 0 disables count targeting.
    pub max_secant: u32,
    /// Lazy-cut rounds per secant step before every arc gets lb = 1 (8).
    pub max_valid_rounds: u32,
    /// Minimum ring length in edges; a ring must keep `max(3, ring_min / 2)` (8).
    pub ring_min: u32,
}

impl Default for QuantParams {
    fn default() -> Self {
        QuantParams { allow_collapse: true, count_tol: 0.04, max_secant: 5, max_valid_rounds: 8, ring_min: 8 }
    }
}

/// Result of S9.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantResult {
    /// Integer length per arc.
    pub n: Vec<u32>,
    /// The scale applied to the arc lengths.
    pub sigma: f64,
    /// Quad count `sum_P W_P H_P`.
    pub count: u64,
    /// Number of Bi-MDF solves.
    pub solves: u32,
    /// Number of lower bounds raised by validity cuts.
    pub cuts: u32,
    pub comb: QComb,
    pub warnings: Vec<String>,
    /// Final lower bound per arc (table bounds at `sigma` raised by the cuts). Addition.
    pub lb: Vec<u32>,
}

/// Solves the Bi-MDF quantization at scale `sigma` with lower bounds `lb` (one per arc).
/// Returns the integer lengths; `Err(Infeasible)` when no integer solution was found.
pub fn solve_bimdf(t: &TMesh, sigma: f64, lb: &[u32]) -> Result<Vec<u32>, AqError> {
    let g = CGraph::new(t)?;
    let mut dij = Dijkstra::new(g.dn());
    bimdf::solve(t, &g, sigma, lb, None, &mut dij)
}

/// Builds the combinatorial quad mesh of `t` with lengths `n` and checks it (ring_min = 8).
pub fn build_comb(t: &TMesh, n: &[u32]) -> Result<QComb, Vec<Violation>> {
    build_comb_with(t, n, QuantParams::default().ring_min)
}

/// [`build_comb`] with an explicit `ring_min`. Addition.
pub fn build_comb_with(t: &TMesh, n: &[u32], ring_min: u32) -> Result<QComb, Vec<Violation>> {
    match comb::check_comb(t, n, ring_min) {
        Ok(c) if c.violations.is_empty() => Ok(c.comb),
        Ok(c) => Err(c.violations),
        Err(_) => Err(vec![Violation::Unbalanced(aq_core::NONE)]),
    }
}

/// Validity check returning every violation with the zero arcs that caused it. Addition,
/// used by the tests and the debug dumps.
pub fn comb_violations(t: &TMesh, n: &[u32], ring_min: u32) -> Result<Vec<(Violation, Vec<u32>)>, AqError> {
    let c = comb::check_comb(t, n, ring_min).map_err(|e| e.at(Stage::Quantize))?;
    Ok(c.violations.into_iter().zip(c.causes).collect())
}

/// One valid evaluation at a scale.
struct Eval {
    sigma: f64,
    n: Vec<u32>,
    lb: Vec<u32>,
    comb: QComb,
    count: u64,
}

struct Driver<'a> {
    t: &'a TMesh,
    g: CGraph,
    dij: Dijkstra,
    p: &'a QuantParams,
    cut_lb: Vec<u32>,
    all_one: bool,
    solves: u32,
    cuts: u32,
    warnings: Vec<String>,
}

impl Driver<'_> {
    fn lb(&self, sigma: f64) -> Vec<u32> {
        self.t
            .arcs
            .iter()
            .zip(&self.cut_lb)
            .map(|(a, &c)| {
                let l = table_lb(a, sigma, self.p.allow_collapse).max(c);
                if self.all_one { l.max(1) } else { l }
            })
            .collect()
    }

    /// Solves at `sigma` with the validity rounds. `Ok(None)` when the solve budget ran out.
    fn eval(&mut self, sigma: f64, ctl: &Ctl) -> Result<Option<Eval>, AqError> {
        let mut rounds = 0u32;
        loop {
            ctl.check()?;
            if self.solves >= MAX_SOLVES {
                return Ok(None);
            }
            let lb = self.lb(sigma);
            self.solves += 1;
            ctl.progress(Stage::Quantize, (self.solves as f32 / MAX_SOLVES as f32).min(1.0));
            let n = bimdf::solve(self.t, &self.g, sigma, &lb, Some(ctl), &mut self.dij)?;
            let chk = comb::check_comb(self.t, &n, self.p.ring_min).map_err(|e| e.at(Stage::Quantize))?;
            if chk.violations.is_empty() {
                let count = chk.comb.quads.len() as u64;
                return Ok(Some(Eval { sigma, n, lb, comb: chk.comb, count }));
            }
            if self.all_one {
                return Err(err(
                    ErrKind::Infeasible,
                    format!("no valid quantization with lb = 1 on every arc: {:?}", chk.violations.first()),
                ));
            }
            rounds += 1;
            let added = self.apply_cuts(&chk.violations, &chk.causes, &n, sigma);
            self.cuts += added;
            if added == 0 || rounds >= self.p.max_valid_rounds {
                self.all_one = true;
                self.warnings.push(format!(
                    "validity cuts did not converge after {rounds} rounds ({:?}); lb = 1 on every arc",
                    chk.violations.first()
                ));
            }
        }
    }

    /// Raises lb on the zero arcs that caused each violation. A short ring without a zero
    /// arc gets one more segment on its arc with the largest `sigma l / (n + 1)`.
    fn apply_cuts(&mut self, viol: &[Violation], causes: &[Vec<u32>], n: &[u32], sigma: f64) -> u32 {
        let mut added = 0;
        for (v, c) in viol.iter().zip(causes) {
            for &a in c {
                if self.cut_lb[a as usize] < 1 {
                    self.cut_lb[a as usize] = 1;
                    added += 1;
                }
            }
            if let (Violation::ShortRing(r), true) = (v, c.is_empty())
                && let Some(ring) = self.t.rings.get(*r as usize)
            {
                let mut best: Option<(f64, u32)> = None;
                for &(a, _) in ring {
                    let s = sigma * self.t.arcs[a as usize].len / (n[a as usize] as f64 + 1.0);
                    if best.is_none_or(|(bs, _)| s > bs) {
                        best = Some((s, a));
                    }
                }
                if let Some((_, a)) = best {
                    let want = n[a as usize] + 1;
                    if self.cut_lb[a as usize] < want {
                        self.cut_lb[a as usize] = want;
                        added += 1;
                    }
                }
            }
        }
        added
    }
}

/// S9: quantizes the arc lengths of `t` to integers with about `n_target` quads.
///
/// Errors: an invalid T-mesh or target (`InvalidInput`), no feasible or valid quantization
/// (`Infeasible`), cancellation and timeout from `ctl`.
pub fn quantize(t: &TMesh, n_target: f64, p: &QuantParams, ctl: &Ctl) -> Result<QuantResult, AqError> {
    ctl.progress(Stage::Quantize, 0.0);
    t.validate().map_err(|e| AqError::new(Stage::Quantize, ErrKind::InvalidInput, e.msg))?;
    if !(n_target.is_finite() && n_target > 0.0) {
        return Err(err(ErrKind::InvalidInput, format!("target count {n_target} must be positive")));
    }
    if !(p.count_tol.is_finite() && p.count_tol >= 0.0) {
        return Err(err(ErrKind::InvalidInput, "count_tol must be >= 0"));
    }
    let g = CGraph::new(t)?;
    let dn = g.dn();
    let mut d = Driver {
        t,
        g,
        dij: Dijkstra::new(dn),
        p,
        cut_lb: vec![0; t.arcs.len()],
        all_one: false,
        solves: 0,
        cuts: 0,
        warnings: Vec::new(),
    };
    let rel = |q: u64| (q as f64 - n_target).abs() / n_target;
    let logf = |q: u64| (q.max(1) as f64).ln() - n_target.ln();
    let Some(e0) = d.eval(1.0, ctl)? else {
        return Err(err(ErrKind::Infeasible, "solve budget exhausted before a valid quantization"));
    };
    let mut best = e0;
    if p.max_secant > 0 && rel(best.count) >= p.count_tol {
        // Secant on F(x) = log Q(e^x) - log N with x = log sigma; Q grows like sigma^2, which
        // gives the first step and the fallback slope.
        let (mut x0, mut f0) = (best.sigma.ln(), logf(best.count));
        let mut x1 = x0 - f0 / 2.0;
        for _ in 0..p.max_secant {
            let r = match d.eval(x1.exp(), ctl) {
                Ok(Some(r)) => r,
                Ok(None) => {
                    d.warnings.push(format!("solve budget of {MAX_SOLVES} reached during count targeting"));
                    break;
                }
                Err(e) if e.kind == ErrKind::Infeasible => {
                    d.warnings.push(format!("count targeting stopped at sigma {}: {}", x1.exp(), e.msg));
                    break;
                }
                Err(e) => return Err(e),
            };
            let f1 = logf(r.count);
            let done = rel(r.count) < p.count_tol;
            if rel(r.count) < rel(best.count) {
                best = r;
            }
            if done {
                break;
            }
            let mut slope = (f1 - f0) / (x1 - x0);
            if !(slope.is_finite() && slope > 0.25) {
                slope = 2.0;
            }
            let step = (-f1 / slope).clamp(-1.5, 1.5);
            if step == 0.0 {
                break;
            }
            x0 = x1;
            f0 = f1;
            x1 += step;
        }
        if rel(best.count) >= p.count_tol {
            d.warnings.push(format!(
                "count {} misses the target {n_target:.1} by {:.1}%",
                best.count,
                100.0 * rel(best.count)
            ));
        }
    }
    ctl.progress(Stage::Quantize, 1.0);
    Ok(QuantResult {
        n: best.n,
        sigma: best.sigma,
        count: best.count,
        solves: d.solves,
        cuts: d.cuts,
        comb: best.comb,
        warnings: d.warnings,
        lb: best.lb,
    })
}

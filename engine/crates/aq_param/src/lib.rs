//! # aq_param: S7 seamless parametrization (DESIGN 6.9, WP-10)
//!
//! [`seamless`] turns a combed cross field and its cut graph into a [`SeamlessParam`]: dyadic
//! per-corner uv (multiples of 2^-24, quad units) whose transitions across every cut halfedge,
//! chain iso-lines and exact loop closures hold **bitwise**.
//!
//! Pipeline (DESIGN 6.9):
//! 1. **Variables** ([`system`]): one rep wedge per vertex (the wedge of its lowest-index face)
//!    with two scalar variables, two translation variables per cut arc. Each corner is an
//!    integer expression `R(k) x_v + sum R(k_j)(+-t_a)` built by walking the fan CCW from the
//!    rep wedge (C3, C10).
//! 2. **Constraints**, eliminated exactly with +-1 pivots (`aq_linalg::Elim`), in this order:
//!    fan closures `(I - R_tot) x_v - tau_tot = 0` (pivots prefer translations), chain iso
//!    rows (U edge: equal v, V edge: equal u, C13), loop rows (transversal translation 0), the
//!    gauge (first corner of the lowest face of each component at (0, 0)). Rows without a unit pivot become
//!    penalties of weight `penalty_w * mean diag`.
//! 3. **Energy** ([`solve`]): `sum_f w_f A_f (|grad u - d'/h|^2 + |grad v - n x d'/h|^2)`
//!    over the free variables, sparse Cholesky. Loop energy guard (1.3), count scaling
//!    `nu = sqrt(N_c / N_pred)` (exact: every row is homogeneous), stiffening (at most
//!    `stiff_iters` numeric refactors with the same symbolic analysis).
//! 4. **Injectivity repair** ([`repair`]) of fold regions without cones.
//! 5. **Dyadic snap**: every free variable is rounded to 2^-24, then every corner and every
//!    translation is evaluated from its integer expression in exact integer arithmetic, so the
//!    transitions are exact at both endpoints of every cut edge.
//!
//! Errors carry `Stage::Param`: `InvalidInput` for inconsistent inputs, `Numerical` (NaN,
//! non-SPD after regularization, `N_pred <= 0`, |uv| >= 2^27), `Invariant` (transitions,
//! chain iso-lines or exact loops not exact after the snap, which happens when one of their
//! rows had no unit pivot), `Cancelled` / `Timeout` from `ctl`.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::needless_range_loop)]

mod expr;
pub mod repair;
pub mod solve;
pub mod system;

use aq_core::cross::{det2, folded_faces, jacobian};
use aq_core::tol::{DYADIC_QUANTUM, DYADIC_SCALE, UV_BOUND};
use aq_core::{
    AqError, CrossField, Ctl, CutGraph, FeatureSet, Frames, LoopCand, NONE, ParamReport,
    SeamlessParam, SizeField, Stage, Topology, Transition, TriMesh, V2,
};

pub use repair::{RepairInput, RepairStats};
pub use system::System;

use crate::solve::{Assembly, Solver, n_pred, singular_values};
use crate::system::Raw;

/// Borrowed view of the S2 proxy (`aq_prep::Proxy` lives in aq_prep, which this crate may not
/// depend on). Field for field the same as `aq_cut::ProxyRef`.
#[derive(Clone, Copy, Debug)]
pub struct ProxyRef<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub frames: &'a Frames,
    pub feats: &'a FeatureSet,
}

impl<'a> ProxyRef<'a> {
    pub fn new(
        mesh: &'a TriMesh,
        topo: &'a Topology,
        frames: &'a Frames,
        feats: &'a FeatureSet,
    ) -> Self {
        ProxyRef {
            mesh,
            topo,
            frames,
            feats,
        }
    }
}

/// Sweeps of the per-vertex untangling after the region repair (WP-16).
pub const UNTANGLE_SWEEPS: u32 = 40;

/// S7 parameters (DESIGN 6.9 and 13).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamParams {
    /// At most this many stiffening refactors (6).
    pub stiff_iters: u32,
    /// Run the injectivity repair.
    pub repair: bool,
    /// Extra 2-ring growths of a region that could not be repaired (3).
    pub repair_max_grow: u32,
    /// Use the loop rows (S5 streamline loops).
    pub loops: bool,
    /// Target quad count N_c; `<= 0` uses `SizeField::predicted_count`.
    pub n_target: f64,
    /// Penalty weight relative to the mean diagonal (1e4).
    pub penalty_w: f64,
    /// Loop energy guard (1.3): loops are kept when `E1 <= loop_guard * E0`.
    pub loop_guard: f64,
}

impl Default for ParamParams {
    fn default() -> Self {
        ParamParams {
            stiff_iters: 6,
            repair: true,
            repair_max_grow: 3,
            loops: true,
            n_target: 0.0,
            penalty_w: 1e4,
            loop_guard: 1.3,
        }
    }
}

/// Which loop rows the final system holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopSet {
    #[default]
    None,
    Rings,
    All,
}

/// Diagnostics of one [`seamless_detailed`] run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamDiag {
    pub loop_set: LoopSet,
    /// Energies of the guard (E0 is `ParamReport::e0`).
    pub e1: Option<f64>,
    pub e2: Option<f64>,
    /// Fold count of every solve (initial, then each stiffening refactor).
    pub fold_history: Vec<u32>,
    /// Fold count of the kept iterate after each solve.
    pub kept_folds: Vec<u32>,
    pub repair: RepairStats,
    /// Final uv before the snap (after scaling and repair).
    pub uv_float: Vec<[V2; 3]>,
    /// N_pred of the snapped uv.
    pub n_pred_snapped: f64,
    /// The target count used.
    pub n_c: f64,
}

/// A solved system before stiffening, repair and snap (for tests and tools).
pub struct Prepared {
    pub sys: System,
    /// Free values, scaled to the count.
    pub y: Vec<f64>,
    /// Per vertex, its lowest-index face.
    pub rep_face: Vec<u32>,
    pub nu: f64,
    pub energy: f64,
}

fn check_inputs(
    px: &ProxyRef,
    f: &CrossField,
    cut: &CutGraph,
    s: &SizeField,
) -> Result<(), AqError> {
    let (m, t) = (px.mesh, px.topo);
    let nf = m.tris.len();
    let nh = 3 * nf;
    let bad = |what: &str| Err(AqError::invalid_input(format!("aq_param: {what}")));
    if t.twin.len() != nh || px.frames.area.len() != nf || t.vert_he.len() != m.pos.len() {
        return bad("proxy topology or frames do not match the mesh");
    }
    if px.feats.he_kind.len() != nh || px.feats.he_chain.len() != nh {
        return bad("feature set does not match the mesh");
    }
    if f.theta.len() != nf || f.p.len() != nh || f.index4.len() != m.pos.len() {
        return bad("cross field does not match the mesh");
    }
    if s.h_face.len() != nf {
        return bad("size field does not match the mesh");
    }
    cut.validate(m, t, None)
        .map_err(|e| AqError::invalid_input(format!("aq_param: cut graph: {}", e.msg)))?;
    if cut.theta_combed.iter().any(|x| !x.is_finite()) {
        return bad("combed theta is not finite");
    }
    Ok(())
}

fn target_count(px: &ProxyRef, s: &SizeField, p: &ParamParams) -> Result<f64, AqError> {
    let nc = if p.n_target > 0.0 {
        p.n_target
    } else {
        s.predicted_count(px.frames)
    };
    if !(nc.is_finite() && nc > 0.0) {
        return Err(AqError::invalid_input(format!(
            "aq_param: target count {nc}"
        )));
    }
    Ok(nc)
}

/// Scales y so that N_pred = nc; returns nu.
fn scale_to_count(px: &ProxyRef, sys: &System, y: &mut [f64], nc: f64) -> Result<f64, AqError> {
    let uv = sys.corners(y);
    let np = n_pred(px.mesh, px.frames, &uv);
    if !(np.is_finite() && np > 0.0) {
        return Err(AqError::numerical(format!(
            "aq_param: N_pred = {np} (heavily folded)"
        )));
    }
    let nu = (nc / np).sqrt();
    for v in y.iter_mut() {
        *v *= nu;
    }
    Ok(nu)
}

fn count_folds(px: &ProxyRef, uv: &[[V2; 3]]) -> u32 {
    folded_faces(px.frames, px.mesh, uv)
        .iter()
        .filter(|&&b| b)
        .count() as u32
}

/// Builds one system, solves it with unit weights.
#[allow(clippy::too_many_arguments)]
fn make_solved(
    px: &ProxyRef,
    cut: &CutGraph,
    s: &SizeField,
    raw: &Raw,
    base: &aq_linalg::Elim,
    base_st: &system::RowStats,
    sel: &[u32],
    p: &ParamParams,
    ctl: &Ctl,
) -> Result<(System, Solver, Vec<f64>, f64), AqError> {
    let sys = System::build(px, raw, base, base_st, sel, ctl)?;
    let asm = Assembly::new(&sys, px.mesh, px.frames, &cut.theta_combed, &s.h_face, ctl)?;
    let mut sol = Solver::new(asm);
    let w = vec![1.0; px.mesh.tris.len()];
    let y = sol.solve(&w, p.penalty_w, ctl)?;
    let e = sol.asm.energy(&y, None);
    Ok((sys, sol, y, e))
}

/// Builds and solves the system for an explicit loop selection (indices into `loops`) with
/// unit weights, scaled to the target count. No guard, stiffening, repair or snap.
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    px: &ProxyRef,
    f: &CrossField,
    cut: &CutGraph,
    s: &SizeField,
    loops: &[LoopCand],
    sel: &[u32],
    p: &ParamParams,
    ctl: &Ctl,
) -> Result<Prepared, AqError> {
    (|| {
        check_inputs(px, f, cut, s)?;
        let nc = target_count(px, s, p)?;
        let raw = Raw::build(px, cut, loops, ctl)?;
        let (base, base_st) = raw.base_elim(ctl)?;
        let (sys, _sol, mut y, energy) =
            make_solved(px, cut, s, &raw, &base, &base_st, sel, p, ctl)?;
        let nu = scale_to_count(px, &sys, &mut y, nc)?;
        Ok(Prepared {
            sys,
            y,
            rep_face: raw.rep_face,
            nu,
            energy,
        })
    })()
    .map_err(|e: AqError| e.at(Stage::Param))
}

/// S7: the seamless parametrization (DESIGN 6.9).
pub fn seamless(
    px: &ProxyRef,
    f: &CrossField,
    cut: &CutGraph,
    s: &SizeField,
    loops: &[LoopCand],
    p: &ParamParams,
    ctl: &Ctl,
) -> Result<SeamlessParam, AqError> {
    seamless_detailed(px, f, cut, s, loops, p, ctl).map(|r| r.0)
}

/// [`seamless`] with its diagnostics.
pub fn seamless_detailed(
    px: &ProxyRef,
    f: &CrossField,
    cut: &CutGraph,
    s: &SizeField,
    loops: &[LoopCand],
    p: &ParamParams,
    ctl: &Ctl,
) -> Result<(SeamlessParam, ParamDiag), AqError> {
    run(px, f, cut, s, loops, p, ctl).map_err(|e| e.at(Stage::Param))
}

fn run(
    px: &ProxyRef,
    f: &CrossField,
    cut: &CutGraph,
    s: &SizeField,
    loops: &[LoopCand],
    p: &ParamParams,
    ctl: &Ctl,
) -> Result<(SeamlessParam, ParamDiag), AqError> {
    let (m, t, fr) = (px.mesh, px.topo, px.frames);
    let nf = m.tris.len();
    ctl.progress(Stage::Param, 0.0);
    check_inputs(px, f, cut, s)?;
    let nc = target_count(px, s, p)?;
    let mut diag = ParamDiag {
        n_c: nc,
        ..Default::default()
    };
    let raw = Raw::build(px, cut, loops, ctl)?;
    let (base, base_st) = raw.base_elim(ctl)?;
    ctl.progress(Stage::Param, 0.1);
    // Loop energy guard (at most 3 symbolic analyses).
    let all: Vec<u32> = if p.loops {
        (0..raw.loops.len() as u32).collect()
    } else {
        Vec::new()
    };
    let has_rows = |sel: &[u32]| {
        sel.iter().any(|&i| {
            raw.loops[i as usize]
                .row
                .as_ref()
                .is_some_and(|r| !r.is_empty())
        })
    };
    let slack = 1e-9 * fr.area.iter().sum::<f64>();
    let guard = |e: f64, e0: f64| e <= p.loop_guard * e0 + slack;
    let (chosen, e0) = if !has_rows(&all) {
        let c = make_solved(px, cut, s, &raw, &base, &base_st, &all, p, ctl)?;
        diag.loop_set = if all.is_empty() {
            LoopSet::None
        } else {
            LoopSet::All
        };
        let e = c.3;
        (c, e)
    } else {
        // WP-16 round 3: the two solves are independent and run side by side (same results
        // and same error order as one after the other).
        let (r0, r1) = rayon::join(
            || make_solved(px, cut, s, &raw, &base, &base_st, &[], p, ctl),
            || make_solved(px, cut, s, &raw, &base, &base_st, &all, p, ctl),
        );
        let c0 = r0?;
        let e0 = c0.3;
        let c1 = r1?;
        diag.e1 = Some(c1.3);
        if guard(c1.3, e0) {
            diag.loop_set = LoopSet::All;
            (c1, e0)
        } else {
            drop(c1);
            let rings: Vec<u32> = all
                .iter()
                .copied()
                .filter(|&i| raw.loops[i as usize].ring)
                .collect();
            if has_rows(&rings) {
                let c2 = make_solved(px, cut, s, &raw, &base, &base_st, &rings, p, ctl)?;
                diag.e2 = Some(c2.3);
                if guard(c2.3, e0) {
                    diag.loop_set = LoopSet::Rings;
                    (c2, e0)
                } else {
                    (c0, e0)
                }
            } else {
                (c0, e0)
            }
        }
    };
    ctl.progress(Stage::Param, 0.4);
    let (sys, mut sol, mut y, _) = chosen;
    let mut nu = scale_to_count(px, &sys, &mut y, nc)?;
    // Stiffening.
    let uv = sys.corners(&y);
    let folds0 = count_folds(px, &uv);
    diag.fold_history.push(folds0);
    diag.kept_folds.push(folds0);
    let mut best = (y.clone(), folds0, nu);
    let mut prev = folds0;
    let mut w = vec![1.0f64; nf];
    let mut cur_uv = uv;
    for _ in 0..p.stiff_iters {
        if best.1 == 0 {
            break;
        }
        ctl.check()?;
        let folded = folded_faces(fr, m, &cur_uv);
        let mut wn = w.clone();
        for fi in 0..nf {
            let j = jacobian(fr, m, fi as u32, &cur_uv[fi]);
            let (s1, s2) = singular_values(&j);
            let h = s.h_face[fi];
            let mut d = (s1 * h - 1.0).abs() + (s2 * h - 1.0).abs();
            if folded[fi] {
                d += 10.0;
            }
            if !d.is_finite() {
                d = 15.0;
            }
            wn[fi] += d.min(5.0);
        }
        // One Jacobi average over the face 1-ring (edge neighbours).
        for fi in 0..nf {
            let mut sum = wn[fi];
            let mut cnt = 1.0;
            for k in 0..3 {
                let tw = t.twin[3 * fi + k];
                if tw != NONE {
                    sum += wn[(tw / 3) as usize];
                    cnt += 1.0;
                }
            }
            w[fi] = sum / cnt;
        }
        let mean = w.iter().sum::<f64>() / nf.max(1) as f64;
        for x in w.iter_mut() {
            *x = aq_core::tol::clamp_weight(*x, mean);
        }
        let mut yn = sol.solve(&w, p.penalty_w, ctl)?;
        let nun = scale_to_count(px, &sys, &mut yn, nc)?;
        cur_uv = sys.corners(&yn);
        let fo = count_folds(px, &cur_uv);
        diag.fold_history.push(fo);
        if fo < best.1 {
            best = (yn.clone(), fo, nun);
        }
        diag.kept_folds.push(best.1);
        if fo == 0 || fo >= prev {
            break;
        }
        prev = fo;
    }
    let (mut y, flips_after_stiff, nu_best) = best;
    nu = nu_best;
    ctl.progress(Stage::Param, 0.7);
    // Injectivity repair.
    if p.repair && flips_after_stiff > 0 {
        let inp = RepairInput {
            px,
            cut,
            index4: &f.index4,
            theta_combed: &cut.theta_combed,
            h_face: &s.h_face,
            rep_face: &raw.rep_face,
        };
        diag.repair = repair::repair(&inp, &sys, &mut y, p.repair_max_grow, ctl)?;
        {
            // WP-16: folds next to cones (skipped by the region repair in M1) and cone fans
            // that wind a full turn short; cheap when there are none.
            let (moved, mut left) = repair::untangle(&inp, &sys, &mut y, UNTANGLE_SWEEPS, ctl)?;
            diag.repair.untangled = moved;
            // WP-16 round 2: untangle fixes cone windings, after which the cone-chart region
            // repair can take the folds left next to cones; one more round of both.
            if left > 0 {
                let r2 = repair::repair(&inp, &sys, &mut y, p.repair_max_grow, ctl)?;
                diag.repair.repaired += r2.repaired;
                diag.repair.regions += r2.regions;
                if r2.repaired > 0 {
                    let (moved2, left2) =
                        repair::untangle(&inp, &sys, &mut y, UNTANGLE_SWEEPS, ctl)?;
                    diag.repair.untangled += moved2;
                    left = left2;
                } else {
                    left = r2.folds_after;
                }
            }
            diag.repair.folds_after = left;
        }
    }
    ctl.progress(Stage::Param, 0.9);
    let uv_float = sys.corners(&y);
    let n_pred_f = n_pred(m, fr, &uv_float);
    let e_final = sol.asm.energy(&y, None);
    // Dyadic snap and exact reconstruction.
    let (uv, trans) = exact_output(px, cut, &sys, &y)?;
    let folded = folded_faces(fr, m, &uv);
    let n_fold = folded.iter().filter(|&&b| b).count() as u32;
    diag.n_pred_snapped = n_pred(m, fr, &uv);
    diag.uv_float = uv_float;
    let st = &sys.stats;
    let report = ParamReport {
        n_free: sys.n_free as u32,
        rows: st.rows,
        redundant: st.redundant,
        inconsistent: st.inconsistent,
        penalty_rows: st.penalties.len() as u32,
        loops_exact: sys.loops_exact.len() as u32,
        loops_dropped: (loops.len() - sys.loops_exact.len()) as u32,
        e0,
        e_final,
        flips_initial: folds0,
        flips_after_stiff,
        folds_after_repair: n_fold,
        n_pred: n_pred_f,
        scale: nu,
    };
    let mut loops_exact = sys.loops_exact.clone();
    loops_exact.sort_unstable();
    let sp = SeamlessParam {
        uv,
        trans,
        is_cut: cut.is_cut.clone(),
        cones: f.cones(),
        he_family: raw.he_family.clone(),
        loops_exact,
        folded,
        report,
    };
    sp.validate(m, t)?;
    for &li in &sp.loops_exact {
        let l = &loops[li as usize];
        let c = loop_holonomy(&sp.trans, &l.hes);
        let k =
            aq_core::cross::raw_to_combed_branch(l.branch, cut.comb_offset[l.start_face as usize]);
        let perp = if k.is_multiple_of(2) { c.t[1] } else { c.t[0] };
        if c.r != 0 || perp != 0.0 {
            return Err(AqError::invariant(format!(
                "exact loop {li} does not close exactly: {c:?}"
            )));
        }
    }
    ctl.progress(Stage::Param, 1.0);
    Ok((sp, diag))
}

/// Composition of the transitions crossed by a dual path (start chart to end chart).
pub fn loop_holonomy(trans: &[Transition], hes: &[u32]) -> Transition {
    hes.iter()
        .fold(Transition::IDENTITY, |c, &h| c.compose(&trans[h as usize]))
}

/// Sum of `c * q` (exact) as a dyadic f64, or `Numerical` at or above `bound`.
fn eval_exact(ix: &[u32], cs: &[i64], q: &[i64], bound: f64) -> Result<f64, AqError> {
    let mut acc: i128 = 0;
    for (&i, &c) in ix.iter().zip(cs) {
        acc += c as i128 * q[i as usize] as i128;
    }
    let lim = (bound * DYADIC_SCALE) as i128;
    if acc.abs() >= lim {
        return Err(AqError::numerical(format!(
            "aq_param: |uv| or |t| above {bound}"
        )));
    }
    Ok(acc as f64 * DYADIC_QUANTUM + 0.0)
}

/// Dyadic snap of the free values and exact evaluation of every corner and translation.
fn exact_output(
    px: &ProxyRef,
    cut: &CutGraph,
    sys: &System,
    y: &[f64],
) -> Result<(Vec<[V2; 3]>, Vec<Transition>), AqError> {
    let nf = px.mesh.tris.len();
    let mut q: Vec<i64> = Vec::with_capacity(y.len());
    for &x in y {
        if !(x.is_finite() && x.abs() < UV_BOUND) {
            return Err(AqError::numerical(format!(
                "aq_param: free value {x} out of bound"
            )));
        }
        q.push((x * DYADIC_SCALE).round() as i64);
    }
    let mut uv = vec![[[0.0; 2]; 3]; nf];
    for f in 0..nf {
        for k in 0..3 {
            for c in 0..2 {
                let (ix, cs) = sys.slot(2 * (3 * f + k) + c);
                uv[f][k][c] = eval_exact(ix, cs, &q, UV_BOUND)?;
            }
        }
    }
    let mut tr = Vec::with_capacity(cut.arcs.len());
    for a in 0..cut.arcs.len() {
        let mut tt = [0.0; 2];
        for k in 0..2 {
            let (ix, cs) = sys.t_terms(a, k);
            tt[k] = eval_exact(ix, cs, &q, 2.0 * UV_BOUND)?;
        }
        tr.push(tt);
    }
    let nh = 3 * nf;
    let mut trans = vec![Transition::IDENTITY; nh];
    for h in 0..nh {
        if !cut.is_cut[h] {
            continue;
        }
        let (a, side) = cut.he_arc[h];
        if a == NONE || a as usize >= tr.len() {
            return Err(AqError::invalid_input(format!(
                "aq_param: cut halfedge {h} has no arc"
            )));
        }
        let ta = Transition {
            r: cut.arcs[a as usize].r % 4,
            t: tr[a as usize],
        };
        trans[h] = if side > 0 { ta } else { ta.inverse() };
    }
    Ok((uv, trans))
}

/// det J of every face (C11) for per-corner uv (diagnostics).
pub fn face_dets(px: &ProxyRef, uv: &[[V2; 3]]) -> Vec<f64> {
    (0..px.mesh.tris.len())
        .map(|f| det2(&jacobian(px.frames, px.mesh, f as u32, &uv[f])))
        .collect()
}

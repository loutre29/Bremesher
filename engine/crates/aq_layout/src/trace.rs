//! Simultaneous exact motorcycles (DESIGN 6.10 "Simultaneous motorcycles").
//!
//! Time is the parametric distance travelled (|du| or |dv|, invariant under transitions).
//! A trace places its whole chord in a face when it enters the face; every collision found
//! against the pieces already placed there becomes a `Stop` event at the arrival time of the
//! loser. Events are processed in (time, sequence) order and a `Stop` is validated when it is
//! processed: the other curve must still cover the point (it may have been truncated before
//! reaching it in the meantime, in which case the stop is void and the trace goes on). This
//! keeps the result exact without the "no deeper rollback" assumption of the design text.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use aq_core::tol::snap_dyadic;
use aq_core::{AqError, Ctl, NONE, SeamlessParam, Stage};

use crate::ctx::Ctx;
use crate::curves::{chain_verts, loop_occupancy};
use crate::geom::{Line, Pt, aval, chord_from_edge, chord_from_vertex, cmp_a, cmp_travel, on_edge_of};
use crate::{Curve, Emitter, LayoutIssue, LayoutParams, Piece, ProxyView};

/// Counters of one tracing run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceStats {
    pub events: u64,
    pub pieces: u64,
    /// Stop events voided because the other curve was truncated before the point.
    pub voided: u64,
}

#[derive(Clone, Copy, Debug)]
enum EvK {
    Adv { tr: u32, n: u32 },
    Stop { tr: u32, pi: u32, pt: Pt, oc: u32, oj: u32, opt: Pt, head: bool },
}

#[derive(Clone, Copy, Debug)]
struct Ev {
    t: f64,
    seq: u64,
    k: EvK,
}

impl PartialEq for Ev {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Ev {}
impl PartialOrd for Ev {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ev {
    /// Reversed so that the max-heap pops the earliest (time, seq).
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then(o.seq.cmp(&self.seq))
    }
}

/// A piece of another curve seen in the chart of the face being checked.
#[derive(Clone, Copy, Debug)]
struct View {
    line: Line,
    s: Pt,
    e: Pt,
    t_in: f64,
    x_in: f64,
    /// 255 when the piece lives in this face; else the local edge to cross to reach it.
    back: u8,
}

struct Sim<'a, 'b> {
    cx: &'a Ctx<'b>,
    curves: Vec<Curve>,
    dead: Vec<bool>,
    base: u32,
    face_reg: Vec<Vec<(u32, u32)>>,
    edge_reg: BTreeMap<u32, Vec<(u32, u32)>>,
    stop_vert: Vec<bool>,
    heap: BinaryHeap<Ev>,
    seq: u64,
    now: f64,
    max_len: Vec<f64>,
    cap: usize,
    stats: TraceStats,
}

fn fatal(msg: String) -> LayoutIssue {
    LayoutIssue::Fatal(AqError::invariant(msg).at(Stage::Layout))
}

impl<'a, 'b> Sim<'a, 'b> {
    fn register(&mut self, c: u32, j: u32) {
        let pc = self.curves[c as usize].pieces[j as usize];
        self.face_reg[pc.face as usize].push((c, j));
        if pc.on_edge != 255 {
            let key = self.cx.canon(3 * pc.face + pc.on_edge as u32);
            self.edge_reg.entry(key).or_default().push((c, j));
        }
        self.stats.pieces += 1;
    }

    fn push_ev(&mut self, t: f64, k: EvK) {
        let t = if t < self.now { self.now } else { t };
        self.seq += 1;
        self.heap.push(Ev { t, seq: self.seq, k });
    }

    fn is_static(&self, c: u32) -> bool {
        c < self.base
    }

    /// Pieces of other curves that share face f with piece (me, pi), mapped into f's chart.
    fn views(&self, me: u32, pi: u32, only: Option<u32>) -> Vec<(u32, u32, View)> {
        let pc = self.curves[me as usize].pieces[pi as usize];
        let f = pc.face;
        let tri = self.cx.tri(f);
        let mut out = Vec::new();
        for &(c, j) in &self.face_reg[f as usize] {
            if (c, j) == (me, pi) || only.is_some_and(|o| o != c) {
                continue;
            }
            let Some(q) = self.curves[c as usize].pieces.get(j as usize) else { continue };
            out.push((c, j, View { line: q.line, s: q.start, e: q.end, t_in: q.t_in, x_in: q.x_in, back: 255 }));
        }
        if pc.on_edge != 255 {
            let key = self.cx.canon(3 * f + pc.on_edge as u32);
            if let Some(list) = self.edge_reg.get(&key) {
                for &(c, j) in list {
                    if (c, j) == (me, pi) || only.is_some_and(|o| o != c) {
                        continue;
                    }
                    let Some(q) = self.curves[c as usize].pieces.get(j as usize) else { continue };
                    if q.face == f {
                        continue;
                    }
                    let (Some((f2, kf, l2, s2)), Some((_, _, _, e2))) = (
                        self.cx.cross(q.face, q.on_edge, q.line, q.start),
                        self.cx.cross(q.face, q.on_edge, q.line, q.end),
                    ) else {
                        continue;
                    };
                    if f2 != f {
                        continue;
                    }
                    let x_in = aval(tri, l2, s2);
                    out.push((c, j, View { line: l2, s: s2, e: e2, t_in: q.t_in, x_in, back: kf }));
                }
            }
        }
        out
    }

    fn to_other(&self, f: u32, v: &View, p: Pt) -> Pt {
        if v.back == 255 {
            return p;
        }
        match self.cx.cross(f, v.back, v.line, p) {
            Some((_, _, _, q)) => q,
            None => p,
        }
    }

    fn collide(&mut self, me: u32, pi: u32, only: Option<u32>) {
        let pc = self.curves[me as usize].pieces[pi as usize];
        for (c, j, v) in self.views(me, pi, only) {
            self.interact(me, pi, &pc, c, j, &v);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sched_stop(&mut self, t: f64, tr: u32, pi: u32, pt: Pt, oc: u32, oj: u32, opt: Pt, head: bool) {
        self.push_ev(t, EvK::Stop { tr, pi, pt, oc, oj, opt, head });
    }

    fn interact(&mut self, me: u32, pi: u32, p: &Piece, c: u32, j: u32, v: &View) {
        let f = p.face;
        let tri = *self.cx.tri(f);
        let other_fixed = self.is_static(c) || self.dead[c as usize];
        let t_me = |x: Pt| p.t_in + (aval(&tri, p.line, x) - p.x_in).abs();
        let t_v = |x: Pt| v.t_in + (aval(&tri, v.line, x) - v.x_in).abs();
        if p.line.a() == v.line.a() {
            if p.line.c != v.line.c || c == me {
                return;
            }
            let l = p.line;
            let (lo_p, hi_p) = if cmp_a(&tri, l, p.start, p.end) == Ordering::Greater { (p.end, p.start) } else { (p.start, p.end) };
            let (lo_v, hi_v) = if cmp_a(&tri, l, v.s, v.e) == Ordering::Greater { (v.e, v.s) } else { (v.s, v.e) };
            let ov_lo = if cmp_a(&tri, l, lo_p, lo_v) == Ordering::Less { lo_v } else { lo_p };
            let ov_hi = if cmp_a(&tri, l, hi_p, hi_v) == Ordering::Greater { hi_v } else { hi_p };
            if cmp_a(&tri, l, ov_lo, ov_hi) == Ordering::Greater {
                return;
            }
            let first = if l.s() > 0.0 { ov_lo } else { ov_hi };
            if p.line.d == v.line.d || other_fixed {
                let o = self.to_other(f, v, first);
                self.sched_stop(t_me(first), me, pi, first, c, j, o, false);
                return;
            }
            // Head-on: both stop where their times meet, clamped to the overlap.
            let (xq, tq, xs, ts) = (p.x_in, p.t_in, v.x_in, v.t_in);
            let m = if l.s() > 0.0 { 0.5 * (ts - tq + xq + xs) } else { 0.5 * (tq - ts + xq + xs) };
            let mut mp = Pt::X(snap_dyadic(m));
            if cmp_a(&tri, l, mp, ov_lo) != Ordering::Greater {
                mp = ov_lo;
            } else if cmp_a(&tri, l, mp, ov_hi) != Ordering::Less {
                mp = ov_hi;
            }
            let mo = self.to_other(f, v, mp);
            self.sched_stop(t_me(mp), me, pi, mp, c, j, mo, true);
            self.sched_stop(t_v(mp), c, j, mo, me, pi, mp, true);
            return;
        }
        // Transverse crossing at (v.c along my axis, my c along v's axis).
        let mut pm = Pt::X(v.line.c);
        if cmp_travel(&tri, p.line, p.start, pm) == Ordering::Greater || cmp_travel(&tri, p.line, pm, p.end) == Ordering::Greater {
            return;
        }
        if cmp_a(&tri, p.line, pm, p.start) == Ordering::Equal {
            pm = p.start;
        } else if cmp_a(&tri, p.line, pm, p.end) == Ordering::Equal {
            pm = p.end;
        }
        let mut pv = Pt::X(p.line.c);
        if cmp_travel(&tri, v.line, v.s, pv) == Ordering::Greater || cmp_travel(&tri, v.line, pv, v.e) == Ordering::Greater {
            return;
        }
        if cmp_a(&tri, v.line, pv, v.s) == Ordering::Equal {
            pv = v.s;
        } else if cmp_a(&tri, v.line, pv, v.e) == Ordering::Equal {
            pv = v.e;
        }
        let tm = t_me(pm);
        let tv = t_v(pv);
        if tm == 0.0 && tv == 0.0 {
            return;
        }
        let po = self.to_other(f, v, pv);
        if c == me {
            if tm > tv {
                self.sched_stop(tm, me, pi, pm, NONE, 0, pm, false);
            }
            return;
        }
        if other_fixed || tv < tm || (tv == tm && me > c) {
            self.sched_stop(tm, me, pi, pm, c, j, po, false);
        } else {
            self.sched_stop(tv, c, j, po, me, pi, pm, false);
        }
    }

    fn covers(&self, c: u32, j: u32, pt: Pt) -> bool {
        if self.is_static(c) {
            return true;
        }
        let cv = &self.curves[c as usize];
        let Some(pc) = cv.pieces.get(j as usize) else { return false };
        if self.dead[c as usize] && j as usize + 1 == cv.pieces.len() {
            let tri = self.cx.tri(pc.face);
            return cmp_travel(tri, pc.line, pc.start, pt) != Ordering::Greater
                && cmp_travel(tri, pc.line, pt, pc.end) != Ordering::Greater;
        }
        true
    }

    fn advance(&mut self, tr: u32, n: u32) -> Result<(), LayoutIssue> {
        let ti = tr as usize;
        if self.dead[ti] || self.curves[ti].pieces.len() != n as usize {
            return Ok(());
        }
        let Some(&last) = self.curves[ti].pieces.last() else { return Ok(()) };
        let f = last.face;
        let tri = *self.cx.tri(f);
        let stop = match last.end {
            Pt::V(k) => self.stop_vert[self.cx.vid(f, k) as usize],
            Pt::E(k) => self.cx.constraint_he[(3 * f + k as u32) as usize],
            Pt::X(_) => true,
        };
        if stop {
            self.dead[ti] = true;
            return Ok(());
        }
        let t_exit = last.time_at(&tri, last.end);
        let k = tr - self.base;
        if t_exit > self.max_len[k as usize] || n as usize > self.cap {
            return Err(LayoutIssue::Runaway(k));
        }
        let Some((g, kg, l2, _)) = self.cx.cross(f, last.exit, last.line, last.end) else {
            self.dead[ti] = true;
            return Ok(());
        };
        let tg = *self.cx.tri(g);
        let Some(ch) = chord_from_edge(&tg, l2, kg) else {
            return Err(fatal(format!("trace {k} lost its line entering face {g}")));
        };
        let pc = Piece {
            face: g,
            line: l2,
            start: ch.p0,
            end: ch.p1,
            entry: kg,
            exit: ch.exit,
            on_edge: on_edge_of(ch.p0, ch.p1),
            t_in: t_exit,
            x_in: aval(&tg, l2, ch.p0),
        };
        self.curves[ti].pieces.push(pc);
        let pi = n;
        self.register(tr, pi);
        self.collide(tr, pi, None);
        let te = pc.time_at(&tg, pc.end);
        self.push_ev(te, EvK::Adv { tr, n: n + 1 });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn stop(&mut self, tr: u32, pi: u32, pt: Pt, oc: u32, oj: u32, opt: Pt, head: bool) {
        let ti = tr as usize;
        if self.dead[ti] || pi as usize >= self.curves[ti].pieces.len() {
            return;
        }
        if oc != NONE && !self.covers(oc, oj, opt) {
            self.stats.voided += 1;
            if head {
                self.collide(tr, pi, Some(oc));
            }
            return;
        }
        let pcs = &mut self.curves[ti].pieces;
        pcs.truncate(pi as usize + 1);
        if let Some(l) = pcs.last_mut() {
            l.end = pt;
        }
        self.dead[ti] = true;
    }
}

/// Runs the motorcycles of `emitters` against the constraint `curves` (crate entry).
pub(crate) fn trace_curves(
    cx: &Ctx,
    curves: &[Curve],
    emitters: &[Emitter],
    p: &LayoutParams,
    ctl: &Ctl,
) -> Result<(Vec<Curve>, TraceStats), LayoutIssue> {
    let nf = cx.n_faces();
    let nv = cx.m.pos.len();
    let mut stop_vert = chain_verts(cx);
    for (v, &b) in cx.t.is_bnd_vert.iter().enumerate() {
        if b {
            stop_vert[v] = true;
        }
    }
    for &v in cx.cone_index.keys() {
        stop_vert[v as usize] = true;
    }
    for &(v, _, _) in &loop_occupancy(cx, curves) {
        stop_vert[v as usize] = true;
    }
    for c in curves {
        for pc in &c.pieces {
            for q in [pc.start, pc.end] {
                if let Pt::V(k) = q {
                    stop_vert[cx.vid(pc.face, k) as usize] = true;
                }
            }
        }
    }
    for e in emitters {
        if (e.vert as usize) < nv {
            stop_vert[e.vert as usize] = true;
        }
    }
    let mut face_reg = Vec::new();
    face_reg.try_reserve_exact(nf).map_err(|_| LayoutIssue::Fatal(AqError::budget("layout face registry").at(Stage::Layout)))?;
    face_reg.resize_with(nf, Vec::new);
    let base = curves.len() as u32;
    let mut sim = Sim {
        cx,
        curves: curves.to_vec(),
        dead: vec![false; curves.len()],
        base,
        face_reg,
        edge_reg: BTreeMap::new(),
        stop_vert,
        heap: BinaryHeap::new(),
        seq: 0,
        now: 0.0,
        max_len: Vec::new(),
        cap: 8 * nf + 64,
        stats: TraceStats::default(),
    };
    for c in 0..base {
        for j in 0..sim.curves[c as usize].pieces.len() as u32 {
            sim.register(c, j);
        }
    }
    for (i, em) in emitters.iter().enumerate() {
        if (em.face as usize) >= nf {
            return Err(fatal(format!("emitter {i} face out of range")));
        }
        let Some(kv) = cx.corner_of(em.face, em.vert) else {
            return Err(fatal(format!("emitter {i}: vertex {} not on face {}", em.vert, em.face)));
        };
        let tri = *cx.tri(em.face);
        let q = 1 - (em.dir & 1) as usize;
        let line = Line { d: em.dir % 4, c: tri[kv as usize][q] };
        let Some(ch) = chord_from_vertex(&tri, line, kv) else {
            return Err(fatal(format!("emitter {i}: ray {} does not enter face {}", em.dir, em.face)));
        };
        let pc = Piece {
            face: em.face,
            line,
            start: ch.p0,
            end: ch.p1,
            entry: 255,
            exit: ch.exit,
            on_edge: on_edge_of(ch.p0, ch.p1),
            t_in: 0.0,
            x_in: aval(&tri, line, ch.p0),
        };
        let tr = base + i as u32;
        sim.curves.push(Curve { kind: em.kind, chain: NONE, src: i as u32, pieces: vec![pc], closed: false });
        sim.dead.push(false);
        sim.max_len.push(cx.max_len(em.face, p.max_trace_len));
        sim.register(tr, 0);
        sim.collide(tr, 0, None);
        let te = pc.time_at(&tri, pc.end);
        sim.push_ev(te, EvK::Adv { tr, n: 1 });
    }
    while let Some(ev) = sim.heap.pop() {
        sim.stats.events += 1;
        if sim.stats.events.is_multiple_of(1024) {
            ctl.check().map_err(|e| LayoutIssue::Fatal(e.at(Stage::Layout)))?;
        }
        sim.now = ev.t;
        match ev.k {
            EvK::Adv { tr, n } => sim.advance(tr, n)?,
            EvK::Stop { tr, pi, pt, oc, oj, opt, head } => sim.stop(tr, pi, pt, oc, oj, opt, head),
        }
    }
    let stats = sim.stats;
    let traces = sim.curves.split_off(base as usize);
    Ok((traces, stats))
}

/// DESIGN 6.10 `trace_all`: the traces of `emitters` (in emitter order) against the
/// constraint curves `curves` (chains have no pieces; loops are obstacles).
pub fn trace_all(
    px: &ProxyView,
    prm: &SeamlessParam,
    curves: &[Curve],
    emitters: &[Emitter],
    p: &LayoutParams,
    ctl: &Ctl,
) -> Result<Vec<Curve>, LayoutIssue> {
    let cx = Ctx::new(px, prm).map_err(|e| LayoutIssue::Fatal(e.at(Stage::Layout)))?;
    trace_curves(&cx, curves, emitters, p, ctl).map(|(t, _)| t)
}

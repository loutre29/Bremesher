//! # aq_loops: S5 closed streamline loops (DESIGN 6.7, WP-09)
//!
//! [`find_loops`] finds closed field lines of the cross field (rings around limbs and tori,
//! loops around eye sockets). aq_param forces their transversal translation to zero and
//! aq_layout pre-inserts them as rings.
//!
//! * **Seeds:** greedy farthest-point sampling over the proxy vertices (geodesic distance on
//!   edges, measured in local h), keeping interior, non-cone vertices whose incident faces have
//!   a mean confidence of at least `min_conf` and that have a non-cone incident face, until
//!   every eligible vertex is closer than `stride` to a seed. The seed rng (C15) picks the
//!   first seed; the farthest-point order is kept (it is the clustering tie-break, see below;
//!   CHANGES.md, WP-09). A seed starts at the centroid of its lowest-id non-cone incident face.
//! * **Tracing** ([`trace_streamline`]): straight inside each face along branch k
//!   (`theta_f + k pi/2`), exit edge = smallest positive ray parameter (entry edge excluded),
//!   new branch `k' = (k - p[he]) mod 4` from the matching (C7), the point re-expressed on the
//!   twin edge. An exit within 1e-9 h of a vertex is redone with the ray moved by a few 1e-9 h
//!   toward its CCW (left) side. A trace stops at a boundary, a hard chain (Hard or GuideHard),
//!   before entering a cone face (a face incident to a vertex with `index4 != 0`), or at
//!   `max_len`.
//! * **Closure:** once the trace has left the seed face's 1-ring (the faces sharing a vertex
//!   with it), every segment inside that 1-ring is tested: the closing dual path is the
//!   shortest dual path inside the 1-ring back to the seed face, the branch carried along it
//!   must equal the starting branch, and the distance from the seed to the segment must be
//!   below `close_tol` h. The best (closest) segment of that visit closes the loop.
//! * **Turning** ([`turning_check`]): `T = sum delta` over the closed dual cycle is the signed
//!   geodesic turning of the field line (C8). The cycle splits the vertices of its component
//!   into cells (connected through edges it does not cross). A cycle with two cells whose
//!   winding numbers differ by 1 is *separating*: Gauss-Bonnet on its left side D predicts
//!   `T = 2pi chi(D) - K(D)` (K = angle defects, `pi - omega` on the boundary), and the loop is
//!   rejected when `|T - prediction| > 60 degrees`. This accepts flat eye and mouth loops
//!   (T near 2pi) and rejects tight cycles around cones. A *non-separating* cycle (one cell, no
//!   edge crossed twice: torus meridians and parallels) has no Gauss-Bonnet prediction and is
//!   accepted at any turning. Any other cycle (winding 2 or more, self-overlap) is rejected.
//!   The check runs only for loops that pass clustering. See CHANGES.md, WP-09.
//! * **Ring flag:** `length < 1.5 * 2pi / mean_along(|k1|)` (length-weighted mean, quad units).
//! * **Clustering:** rings first, then shortest first, loops of equal length (within 5%) in
//!   seed order; a loop is dropped when at least 80% of its length lies within `cluster` h
//!   (3D distance) of an accepted loop of the same family. The family is compared at the
//!   nearest point of each accepted loop (tangents within 45 degrees), so a crossing loop of
//!   the other family never counts.
//!
//! The functions take a [`ProxyRef`] (borrowed mesh, topology, frames and features), because
//! the aq_prep `Proxy` type is not visible from this crate (DESIGN 3.2: aq_loops depends on
//! aq_core only). See CHANGES.md, WP-09.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::{FRAC_PI_2, TAU};

use aq_core::cross::map_branch;
use aq_core::math;
use aq_core::{
    AqError, CrossField, Ctl, CurvField, FeatureKind, FeatureSet, Frames, LoopCand, NONE, SizeField,
    SplitMix64, Stage, SurfacePoint, Topology, TriMesh, V2, V3,
};
use rayon::prelude::*;

/// Borrowed view of a proxy: everything S5 and S6 read from an aq_prep `Proxy`.
#[derive(Clone, Copy, Debug)]
pub struct ProxyRef<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub frames: &'a Frames,
    pub feats: &'a FeatureSet,
}

impl<'a> ProxyRef<'a> {
    pub fn new(mesh: &'a TriMesh, topo: &'a Topology, frames: &'a Frames, feats: &'a FeatureSet) -> Self {
        ProxyRef { mesh, topo, frames, feats }
    }
}

/// S5 parameters (DESIGN 13: stride 2h, max_len 40h, close_tol 0.3h, min_conf 0.3, cluster 6h).
/// Lengths are in units of the local size h.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopParams {
    pub stride: f64,
    pub max_len: f64,
    pub close_tol: f64,
    pub min_conf: f64,
    pub cluster: f64,
    pub seed: u64,
}

impl Default for LoopParams {
    fn default() -> Self {
        LoopParams { stride: 2.0, max_len: 40.0, close_tol: 0.3, min_conf: 0.3, cluster: 6.0, seed: 0 }
    }
}

/// Clustering order: loops whose lengths agree within this factor of the shortest loop of
/// their group are ordered by seed rank.
pub const LENGTH_GROUP: f64 = 1.05;

/// Spacing of the clustering samples along a loop, in h.
pub const SAMPLE_STEP: f64 = 0.5;

/// The turning bound of DESIGN 6.7 (60 degrees): the largest accepted difference between the
/// turning of a separating loop and its Gauss-Bonnet prediction.
pub const MAX_TURNING_EXCESS: f64 = 60.0 * std::f64::consts::PI / 180.0;

/// Why a streamline stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StopReason {
    #[default]
    MaxLen,
    Boundary,
    HardChain,
    ConeFace,
    /// No exit edge (degenerate face or direction), or the step guard.
    Degenerate,
    /// Invalid start (face out of range, start in a cone face, bad branch).
    Invalid,
}

/// A traced streamline. Segment i runs from `points[i]` to `points[i + 1]` inside `faces[i]`
/// along raw branch `branches[i]`; `hes[i]` is the halfedge crossed from `faces[i]` into
/// `faces[i + 1]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Streamline {
    pub start: SurfacePoint,
    pub faces: Vec<u32>,
    pub branches: Vec<u8>,
    pub hes: Vec<u32>,
    pub points: Vec<V3>,
    /// Geometric length (mesh units).
    pub length: f64,
    pub stop: StopReason,
    /// The halfedge at which the trace stopped (boundary, hard chain, entry of a cone face), or
    /// NONE.
    pub stop_he: u32,
}

/// One accepted loop with its diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopInfo {
    pub cand: LoopCand,
    /// Distance from the seed to the closing segment, in h of the seed face.
    pub closure_err: f64,
    /// Signed geodesic turning `sum delta` over the cycle (radians).
    pub turning: f64,
    /// Gauss-Bonnet prediction of `turning` (`2pi chi(D) - K(D)` of the left side D) for a
    /// separating cycle; equal to `turning` for a non-separating one.
    pub expected_turning: f64,
    /// `|turning - expected_turning|` (radians); 0 for a non-separating cycle.
    pub turning_excess: f64,
    /// Separating or non-separating (overlapping cycles are never accepted).
    pub cycle: CycleKind,
    /// Length-weighted mean |k1| along the loop.
    pub mean_k1: f64,
    /// Seed vertex and its rank in the farthest-point order.
    pub seed_vert: u32,
    pub seed_rank: u32,
    /// Clustering samples about every [`SAMPLE_STEP`] h: (segment midpoint, unit tangent,
    /// length represented).
    pub samples: Vec<(V3, V3, f64)>,
}

/// Counters of one [`find_loops_report`] run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopReport {
    pub seeds: u32,
    pub traces: u32,
    pub closed: u32,
    /// Separating loops whose turning differs from Gauss-Bonnet by more than 60 degrees.
    pub rejected_turning: u32,
    /// Loops whose cycle winds 2 or more times or overlaps itself.
    pub rejected_overlap: u32,
    pub rejected_cluster: u32,
    pub accepted: u32,
}

fn bad(msg: impl Into<String>) -> AqError {
    AqError::invalid_input(msg).at(Stage::Loops)
}

/// Barycentric coordinates of the 2D point p in the triangle q.
fn bary_of(q: &[V2; 3], p: V2) -> V3 {
    let area2 = math::cross2(math::sub2(q[1], q[0]), math::sub2(q[2], q[0]));
    if area2.abs() < 1e-300 {
        return [1.0 / 3.0; 3];
    }
    let mut l = [0.0; 3];
    for i in 0..3 {
        l[i] = math::cross2(math::sub2(q[(i + 1) % 3], p), math::sub2(q[(i + 2) % 3], p)) / area2;
    }
    l
}

fn clamp_bary(l: V3) -> V3 {
    let c = [l[0].max(0.0), l[1].max(0.0), l[2].max(0.0)];
    let s = c[0] + c[1] + c[2];
    if s > 0.0 && s.is_finite() { [c[0] / s, c[1] / s, c[2] / s] } else { [1.0 / 3.0; 3] }
}

fn p2_of(q: &[V2; 3], l: V3) -> V2 {
    [
        l[0] * q[0][0] + l[1] * q[1][0] + l[2] * q[2][0],
        l[0] * q[0][1] + l[1] * q[1][1] + l[2] * q[2][1],
    ]
}

/// Exit of the ray from bary `l` along the 2D direction `d` in triangle q, skipping the edge
/// opposite vertex `skip` (3 = none). Returns (vertex i opposite the exit edge, exit bary).
fn exit_ray(q: &[V2; 3], l: V3, d: V2, skip: usize) -> Option<(usize, V3)> {
    let area2 = math::cross2(math::sub2(q[1], q[0]), math::sub2(q[2], q[0]));
    if area2.abs().is_nan() || area2.abs() <= 1e-300 {
        return None;
    }
    let mut dl = [0.0; 3];
    for i in 0..3 {
        dl[i] = math::cross2(d, math::sub2(q[(i + 1) % 3], q[(i + 2) % 3])) / area2;
    }
    let mut best: Option<(usize, f64)> = None;
    for i in 0..3 {
        if i == skip || dl[i].is_nan() || dl[i] >= 0.0 {
            continue;
        }
        let s = (-l[i] / dl[i]).max(0.0);
        if best.is_none_or(|(_, bs)| s < bs) {
            best = Some((i, s));
        }
    }
    let (i, s) = best?;
    let mut e = [l[0] + s * dl[0], l[1] + s * dl[1], l[2] + s * dl[2]];
    e[i] = 0.0; // exactly on the exit edge (clamp_bary keeps the zero)
    Some((i, clamp_bary(e)))
}

/// Per-face "incident to a cone vertex" flags.
fn cone_faces(m: &TriMesh, f: &CrossField) -> Vec<bool> {
    m.tris
        .iter()
        .map(|t| t.iter().any(|&v| f.index4.get(v as usize).copied().unwrap_or(0) != 0))
        .collect()
}

fn is_hard(feats: &FeatureSet, h: u32) -> bool {
    matches!(
        feats.he_kind.get(h as usize).copied().unwrap_or(FeatureKind::None),
        FeatureKind::Hard | FeatureKind::GuideHard
    )
}

/// Streamline tracer state.
struct Tracer<'a> {
    px: &'a ProxyRef<'a>,
    f: &'a CrossField,
    cone: &'a [bool],
    face: u32,
    bary: V3,
    branch: u8,
    /// Local index (0..3) of the entry halfedge in `face`, 3 at the start.
    entry: usize,
}

enum Step {
    /// Moved from `a` to `b` in the previous face and crossed `he` into the current one.
    Crossed { he: u32, a: V3, b: V3 },
    /// Moved from `a` to `b` and stopped at `he` (NONE when there was no exit).
    Stopped { reason: StopReason, he: u32, a: V3, b: V3 },
}

impl Tracer<'_> {
    fn pos3(&self, face: u32, l: V3) -> V3 {
        let [a, b, c] = self.px.mesh.tri_pos(face);
        math::bary_point(a, b, c, l)
    }

    fn step(&mut self, h_local: f64) -> Step {
        let (m, t, fr) = (self.px.mesh, self.px.topo, self.px.frames);
        let fc = self.face;
        let a3 = self.pos3(fc, self.bary);
        let q = fr.face_coords(m, fc);
        let ang = self.f.theta[fc as usize] + (self.branch % 4) as f64 * FRAC_PI_2;
        let d = [ang.cos(), ang.sin()];
        let skip = if self.entry < 3 { (self.entry + 2) % 3 } else { 3 };
        let Some((mut i, mut e)) = exit_ray(&q, self.bary, d, skip) else {
            return Step::Stopped { reason: StopReason::Degenerate, he: NONE, a: a3, b: a3 };
        };
        // Vertex hit: move the ray toward its CCW (left) side and redo the exit.
        let elen = {
            let (u, w) = (q[(i + 1) % 3], q[(i + 2) % 3]);
            math::sub2(w, u)[0].hypot(math::sub2(w, u)[1])
        };
        if e[(i + 1) % 3].min(e[(i + 2) % 3]) * elen < 1e-9 * h_local {
            let p = p2_of(&q, self.bary);
            let eps = 4e-9 * h_local;
            let p2 = [p[0] - eps * d[1], p[1] + eps * d[0]];
            let l2 = clamp_bary(bary_of(&q, p2));
            if let Some((i2, e2)) = exit_ray(&q, l2, d, skip) {
                i = i2;
                e = e2;
            }
        }
        let b3 = self.pos3(fc, e);
        let k = (i + 1) % 3; // exit halfedge local index: from vertex k to vertex k + 1
        let he = 3 * fc + k as u32;
        let tw = t.twin[he as usize];
        if tw == NONE {
            return Step::Stopped { reason: StopReason::Boundary, he, a: a3, b: b3 };
        }
        if is_hard(self.px.feats, he) {
            return Step::Stopped { reason: StopReason::HardChain, he, a: a3, b: b3 };
        }
        let g = tw / 3;
        if self.cone[g as usize] {
            return Step::Stopped { reason: StopReason::ConeFace, he, a: a3, b: b3 };
        }
        // Re-express the exit point on the twin edge of g.
        let kt = (tw % 3) as usize; // twin runs from head(he) = vertex k+1 of f to vertex k
        let mut nb = [0.0; 3];
        nb[kt] = e[(k + 1) % 3];
        nb[(kt + 1) % 3] = e[k];
        self.face = g;
        self.bary = clamp_bary(nb);
        self.branch = map_branch(self.branch, self.f.p[he as usize]);
        self.entry = kt;
        Step::Crossed { he, a: a3, b: b3 }
    }
}

fn valid_start(px: &ProxyRef, f: &CrossField, from: &SurfacePoint, branch: u8) -> bool {
    let nf = px.mesh.tris.len();
    from.validate(nf).is_ok()
        && branch < 4
        && f.theta.len() == nf
        && f.p.len() == 3 * nf
        && f.index4.len() == px.mesh.pos.len()
        && px.topo.twin.len() == 3 * nf
        && px.frames.e1.len() == nf
        && px.feats.he_kind.len() == 3 * nf
}

/// Traces a streamline of branch `branch` (raw theta, C6) from `from`, forward, until a
/// boundary, a hard chain, a cone face or `max_len` (mesh units, h = 1). Never panics: an
/// invalid start gives an empty streamline with `StopReason::Invalid`.
pub fn trace_streamline(px: &ProxyRef, f: &CrossField, from: SurfacePoint, branch: u8, max_len: f64) -> Streamline {
    let mut sl = Streamline { start: from, stop_he: NONE, ..Default::default() };
    if !valid_start(px, f, &from, branch) {
        sl.stop = StopReason::Invalid;
        return sl;
    }
    let cone = cone_faces(px.mesh, f);
    if cone[from.face as usize] {
        sl.stop = StopReason::Invalid;
        return sl;
    }
    trace_into(px, f, &cone, from, branch, max_len, &|_| 1.0, &mut sl);
    sl
}

/// Tracing loop shared by [`trace_streamline`]; `hf` gives the local h of a face, `max_len`
/// is in units of h.
#[allow(clippy::too_many_arguments)]
fn trace_into(
    px: &ProxyRef,
    f: &CrossField,
    cone: &[bool],
    from: SurfacePoint,
    branch: u8,
    max_len: f64,
    hf: &dyn Fn(u32) -> f64,
    sl: &mut Streamline,
) {
    let mut tr = Tracer { px, f, cone, face: from.face, bary: clamp_bary(from.bary), branch, entry: 3 };
    sl.faces.push(from.face);
    sl.branches.push(branch);
    sl.points.push(tr.pos3(from.face, tr.bary));
    let mut len_h = 0.0;
    let guard = 4 * px.mesh.tris.len() + 64;
    for _ in 0..guard {
        let fc = tr.face;
        let h = hf(fc);
        match tr.step(h) {
            Step::Crossed { he, a, b } => {
                let l = math::dist(a, b);
                if len_h + l / h >= max_len {
                    let s = ((max_len - len_h) * h / l.max(1e-300)).clamp(0.0, 1.0);
                    sl.points.push(math::lerp(a, b, s));
                    sl.length += l * s;
                    sl.stop = StopReason::MaxLen;
                    return;
                }
                len_h += l / h;
                sl.length += l;
                sl.points.push(b);
                sl.hes.push(he);
                sl.faces.push(tr.face);
                sl.branches.push(tr.branch);
            }
            Step::Stopped { reason, he, a, b } => {
                let l = math::dist(a, b);
                if len_h + l / h >= max_len {
                    let s = ((max_len - len_h) * h / l.max(1e-300)).clamp(0.0, 1.0);
                    sl.points.push(math::lerp(a, b, s));
                    sl.length += l * s;
                    sl.stop = StopReason::MaxLen;
                    return;
                }
                sl.length += l;
                sl.points.push(b);
                sl.stop = reason;
                sl.stop_he = he;
                return;
            }
        }
    }
    sl.stop = StopReason::Degenerate;
}

/// Sums over the crossed halfedges of a dual cycle: `(T, H)` with `T = sum delta` (the signed
/// geodesic turning of the field line, C8) and `H = sum kappa` (the transport, C5). For a
/// cycle that returns to its branch, `T + H` is a multiple of 2pi, so H says nothing about T
/// beyond that: [`turning_check`] uses Gauss-Bonnet instead.
pub fn loop_turning(px: &ProxyRef, f: &CrossField, hes: &[u32]) -> (f64, f64) {
    let mut tt = 0.0;
    let mut hh = 0.0;
    for &h in hes {
        tt += f.delta.get(h as usize).copied().unwrap_or(0.0);
        hh += px.frames.kappa.get(h as usize).copied().unwrap_or(0.0);
    }
    (tt, hh)
}

/// Topology of a closed dual cycle in its component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleKind {
    /// Two cells whose winding numbers differ by 1: the cycle bounds its left side.
    Separating,
    /// One cell and no edge crossed twice in the same direction (a torus meridian).
    NonSeparating,
    /// Anything else: winding 2 or more, or a cycle that overlaps itself.
    Overlapping,
}

/// Result of [`turning_check`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurningCheck {
    pub kind: CycleKind,
    /// `sum delta` over the cycle.
    pub turning: f64,
    /// Gauss-Bonnet prediction of `turning` (separating), `turning` itself (non-separating),
    /// NaN (overlapping).
    pub expected: f64,
    /// `|turning - expected|`: 0 for non-separating, infinite for overlapping.
    pub excess: f64,
    /// Separating only: the side the prediction was computed on (the side whose flood
    /// finished first), whether it is the left side, its Euler characteristic and its
    /// curvature `sum of angle defects`. The prediction is
    /// `+-(2pi side_chi - side_k)`, + for the left side.
    pub side_left: bool,
    pub side_chi: i64,
    pub side_k: f64,
}

/// Gauss-Bonnet test of a closed dual cycle `hes` (halfedges crossed in order, LoopCand
/// format).
///
/// Net crossings per edge define a winding number w on the vertices: crossing he from
/// face(he) to face(twin he) is a CCW step about head(he), so `w(head) = w(tail) + 1`. The
/// vertices are flooded through the edges with no net crossing (all cells grown together, one
/// vertex at a time, stopped when one cell is left, so a separating cycle costs about twice
/// its smaller side and a non-separating one a full sweep of its component). The exact
/// identity `T = sum_v w(v) ((pi/2) index4(v) - K(v))` (C9) makes the excess of a separating
/// cycle `(pi/2) |index4(D) - 4 chi(D)|`: 0 for a closed field line around D (Poincare-Hopf
/// on D), a multiple of 90 degrees otherwise.
pub fn turning_check(px: &ProxyRef, f: &CrossField, hes: &[u32], ctl: &Ctl) -> Result<TurningCheck, AqError> {
    let (m, t, fr) = (px.mesh, px.topo, px.frames);
    let nv = m.pos.len();
    let nh = 3 * m.tris.len();
    if t.twin.len() != nh
        || t.vert_he.len() != nv
        || t.is_bnd_vert.len() != nv
        || f.delta.len() != nh
        || fr.corner_angle.len() != nh
    {
        return Err(bad("aq_loops: turning_check inputs do not match the mesh"));
    }
    let (tt, _) = loop_turning(px, f, hes);
    let overlapping = TurningCheck {
        kind: CycleKind::Overlapping,
        turning: tt,
        expected: f64::NAN,
        excess: f64::INFINITY,
        side_left: false,
        side_chi: 0,
        side_k: 0.0,
    };
    // Net crossings per canonical edge (positive = along the canonical halfedge).
    let mut net: Vec<(u32, i32)> = Vec::with_capacity(hes.len());
    for &h in hes {
        let tw = t.twin.get(h as usize).copied().unwrap_or(NONE);
        if tw == NONE {
            return Err(bad("aq_loops: a loop cycle crosses a boundary or invalid halfedge"));
        }
        net.push(if h < tw { (h, 1) } else { (tw, -1) });
    }
    net.sort_unstable();
    let mut walls: Vec<(u32, i32)> = Vec::with_capacity(net.len());
    for (h, s) in net {
        match walls.last_mut() {
            Some(last) if last.0 == h => last.1 += s,
            _ => walls.push((h, s)),
        }
    }
    walls.retain(|e| e.1 != 0);
    if walls.is_empty() || walls.iter().any(|e| e.1.abs() >= 2) {
        return Ok(overlapping);
    }
    let is_wall = |h: u32| {
        let tw = t.twin[h as usize];
        let c = if tw == NONE || h < tw { h } else { tw };
        walls.binary_search_by_key(&c, |e| e.0).is_ok()
    };
    // Cell flood. Labels are union-find ids; `pending` counts the queued vertices per root.
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let g = parent[parent[x as usize] as usize];
            parent[x as usize] = g;
            x = g;
        }
        x
    }
    let mut label = vec![NONE; nv];
    let mut parent: Vec<u32> = Vec::new();
    let mut pending: Vec<i64> = Vec::new();
    let mut queue: Vec<u32> = Vec::new();
    for &(h, _) in &walls {
        for v in [t.tail(m, h), t.head(m, h)] {
            if label[v as usize] == NONE {
                let id = parent.len() as u32;
                parent.push(id);
                pending.push(1);
                label[v as usize] = id;
                queue.push(v);
            }
        }
    }
    // Only wall endpoints have wall edges.
    let mut wall_verts: Vec<u32> = queue.clone();
    wall_verts.sort_unstable();
    let mut unfinished = parent.len();
    let mut qi = 0usize;
    while qi < queue.len() && unfinished > 1 {
        if qi % 1024 == 1023 {
            ctl.check().map_err(|e| e.at(Stage::Loops))?;
        }
        let u = queue[qi];
        qi += 1;
        let mut r = find(&mut parent, label[u as usize]);
        pending[r as usize] -= 1;
        let u_wall = wall_verts.binary_search(&u).is_ok();
        // The heads of the outgoing halfedges are all the neighbours of an interior vertex; a
        // boundary vertex also has the tail of its incoming boundary edge.
        let n_e = if t.is_bnd_vert[u as usize] { 2 } else { 1 };
        for h in t.fan_out(u) {
            // Edge (u, head h) is halfedge h; edge (tail prev h, u) is prev h.
            let p = Topology::prev(h);
            for (e, w) in [(h, t.head(m, h)), (p, t.tail(m, p))].into_iter().take(n_e) {
                if u_wall && is_wall(e) {
                    continue;
                }
                let lw = label[w as usize];
                if lw == r {
                    continue;
                }
                if lw == NONE {
                    label[w as usize] = r;
                    pending[r as usize] += 1;
                    queue.push(w);
                } else {
                    let rw = find(&mut parent, lw);
                    if rw != r {
                        // Two growing cells meet (a finished cell has no unlabeled
                        // neighbour left, so it never meets another one).
                        let (a, b) = (r.min(rw), r.max(rw));
                        parent[b as usize] = a;
                        pending[a as usize] += pending[b as usize];
                        pending[b as usize] = 0;
                        unfinished -= 1;
                        r = a;
                    }
                }
            }
        }
        if pending[r as usize] == 0 {
            unfinished -= 1;
        }
    }
    // Cells of the wall vertices and their winding numbers.
    let root: Vec<u32> = (0..parent.len() as u32).map(|x| find(&mut parent, x)).collect();
    let root_of = |v: u32| root[label[v as usize] as usize];
    let mut cells: Vec<u32> = walls.iter().flat_map(|&(h, _)| [root_of(t.tail(m, h)), root_of(t.head(m, h))]).collect();
    cells.sort_unstable();
    cells.dedup();
    let cell_of = |v: u32| cells.binary_search(&root_of(v)).unwrap_or(0);
    let mut w: Vec<Option<i64>> = vec![None; cells.len()];
    w[0] = Some(0);
    let mut consistent = true;
    let mut changed = true;
    while changed && consistent {
        changed = false;
        for &(h, n) in &walls {
            let (ct, ch) = (cell_of(t.tail(m, h)), cell_of(t.head(m, h)));
            match (w[ct], w[ch]) {
                (Some(a), Some(b)) => consistent &= b - a == n as i64,
                (Some(a), None) => {
                    w[ch] = Some(a + n as i64);
                    changed = true;
                }
                (None, Some(b)) => {
                    w[ct] = Some(b - n as i64);
                    changed = true;
                }
                (None, None) => {}
            }
        }
    }
    if !consistent {
        // Not null-homologous. A simple such cycle leaves its complement connected.
        if cells.len() == 1 {
            return Ok(TurningCheck {
                kind: CycleKind::NonSeparating,
                turning: tt,
                expected: tt,
                excess: 0.0,
                side_left: false,
                side_chi: 0,
                side_k: 0.0,
            });
        }
        return Ok(overlapping);
    }
    let ws: Vec<i64> = w.iter().flatten().copied().collect();
    let (wmin, wmax) = (ws.iter().copied().min().unwrap_or(0), ws.iter().copied().max().unwrap_or(0));
    if cells.len() != 2 || ws.len() != 2 || wmax - wmin != 1 {
        return Ok(overlapping);
    }
    // Gauss-Bonnet on a finished side (all its vertices are labeled).
    let (left, right) = if w[0] == Some(wmax) { (cells[0], cells[1]) } else { (cells[1], cells[0]) };
    let side_left = pending[left as usize] == 0;
    let side = if side_left { left } else { right };
    let in_side = |v: u32| {
        let l = label[v as usize];
        l != NONE && root[l as usize] == side
    };
    let (mut nverts, mut nedges, mut nfaces) = (0i64, 0i64, 0i64);
    let mut side_k = 0.0;
    for &u in &queue {
        if !in_side(u) {
            continue;
        }
        nverts += 1;
        side_k += fr.angle_defect(t, u);
        // Every edge is counted once, from the tail of its canonical halfedge (a boundary
        // halfedge is canonical and outgoing from its tail).
        for h in t.fan_out(u) {
            if t.is_canonical(h) && in_side(t.head(m, h)) {
                nedges += 1;
            }
            if h % 3 == 0 && m.tris[(h / 3) as usize].iter().all(|&x| in_side(x)) {
                nfaces += 1;
            }
        }
    }
    let side_chi = nverts - nedges + nfaces;
    let gb = TAU * side_chi as f64 - side_k;
    let expected = if side_left { gb } else { -gb };
    Ok(TurningCheck {
        kind: CycleKind::Separating,
        turning: tt,
        expected,
        excess: (tt - expected).abs(),
        side_left,
        side_chi,
        side_k,
    })
}

/// Faces sharing a vertex with face f, sorted.
fn one_ring_faces(px: &ProxyRef, f: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(16);
    for &v in &px.mesh.tris[f as usize] {
        for h in px.topo.fan_out(v) {
            out.push(h / 3);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Shortest dual path from `from` to `to` inside the sorted face set `zone` (BFS, interior
/// edges, halfedges in order). None when unreachable.
fn zone_path(px: &ProxyRef, zone: &[u32], from: u32, to: u32) -> Option<Vec<u32>> {
    if from == to {
        return Some(Vec::new());
    }
    let n = zone.len();
    let idx = |f: u32| zone.binary_search(&f).ok();
    let mut prev: Vec<u32> = vec![NONE; n]; // halfedge used to reach the face
    let mut seen = vec![false; n];
    let s = idx(from)?;
    seen[s] = true;
    let mut queue = vec![from];
    let mut qi = 0;
    while qi < queue.len() {
        let fc = queue[qi];
        qi += 1;
        for k in 0..3 {
            let h = 3 * fc + k;
            let tw = px.topo.twin[h as usize];
            if tw == NONE {
                continue;
            }
            let g = tw / 3;
            let Some(gi) = idx(g) else { continue };
            if seen[gi] {
                continue;
            }
            seen[gi] = true;
            prev[gi] = h;
            if g == to {
                let mut path = Vec::new();
                let mut cur = g;
                while cur != from {
                    let ci = idx(cur)?;
                    let ph = prev[ci];
                    path.push(ph);
                    cur = ph / 3;
                }
                path.reverse();
                return Some(path);
            }
            queue.push(g);
        }
    }
    None
}

fn point_seg_dist(p: V3, a: V3, b: V3) -> (f64, f64) {
    let ab = math::sub(b, a);
    let l2 = math::norm2(ab);
    let s = if l2 > 0.0 { (math::dot(math::sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (math::dist(p, math::lerp(a, b, s)), s)
}

/// Context shared by all loop traces.
struct Ctx<'a> {
    px: &'a ProxyRef<'a>,
    f: &'a CrossField,
    c: &'a CurvField,
    s: &'a SizeField,
    p: &'a LoopParams,
    cone: Vec<bool>,
}

enum Outcome {
    None,
    Loop(Box<LoopInfo>),
}

struct Best {
    dist: f64,
    seg: usize,
    s: f64,
    path: Vec<u32>,
}

/// Traces one loop candidate from seed vertex v (start face f0) along branch k0.
fn trace_loop(cx: &Ctx, v: u32, rank: u32, f0: u32, k0: u8) -> Outcome {
    let px = cx.px;
    let hf = |f: u32| cx.s.h_face[f as usize];
    let from = SurfacePoint { face: f0, bary: [1.0 / 3.0; 3] };
    let zone = one_ring_faces(px, f0);
    let in_zone = |f: u32| zone.binary_search(&f).is_ok();
    let h0 = hf(f0);
    let mut tr = Tracer { px, f: cx.f, cone: &cx.cone, face: f0, bary: from.bary, branch: k0, entry: 3 };
    let seed_p = tr.pos3(f0, from.bary);
    let mut faces = vec![f0];
    let mut hes: Vec<u32> = Vec::new();
    let mut seg_pts: Vec<(V3, V3)> = Vec::new(); // segment i in faces[i]
    let mut len_h = 0.0;
    let mut left = false;
    let mut best: Option<Best> = None;
    let guard = 4 * px.mesh.tris.len() + 64;
    for _ in 0..guard {
        let fc = tr.face;
        let bc = tr.branch;
        let h = hf(fc);
        let step = tr.step(h);
        let (a, b, stop) = match step {
            Step::Crossed { a, b, .. } => (a, b, false),
            Step::Stopped { a, b, .. } => (a, b, true),
        };
        let seg = seg_pts.len();
        seg_pts.push((a, b));
        if !in_zone(fc) {
            left = true;
            if best.is_some() {
                break;
            }
        } else if left && let Some(path) = zone_path(px, &zone, fc, f0) {
            // The branch carried back to the seed face must be the starting one.
            let kb = path.iter().fold(bc, |k, &ph| map_branch(k, cx.f.p[ph as usize]));
            if kb == k0 {
                let (dist, s) = point_seg_dist(seed_p, a, b);
                let dist = dist / h0;
                if dist < cx.p.close_tol && best.as_ref().is_none_or(|bb| dist < bb.dist) {
                    best = Some(Best { dist, seg, s, path });
                }
            }
        }
        if stop {
            break;
        }
        let l = math::dist(a, b);
        len_h += l / h;
        if len_h >= cx.p.max_len {
            break;
        }
        if let Step::Crossed { he, .. } = step {
            hes.push(he);
            faces.push(tr.face);
        }
    }
    let Some(best) = best else { return Outcome::None };
    // Closed dual cycle: crossings before the closing segment, then the closing path.
    let mut cyc: Vec<u32> = hes[..best.seg].to_vec();
    cyc.extend_from_slice(&best.path);
    if cyc.is_empty() {
        return Outcome::None;
    }
    // The Gauss-Bonnet test (turning_check) needs a flood over the mesh, so it runs later, only
    // for the loops that pass clustering.
    let (turning, _) = loop_turning(px, cx.f, &cyc);
    if !turning.is_finite() {
        return Outcome::None;
    }
    let mut length = 0.0;
    let mut k1w = 0.0;
    // Clustering samples about every SAMPLE_STEP h: the midpoint and tangent of the segment
    // that completes a run, weighted by the run length.
    let mut samples = Vec::with_capacity(best.seg / 2 + 2);
    let mut run = 0.0;
    for (i, &(a, b)) in seg_pts.iter().enumerate().take(best.seg + 1) {
        let b = if i == best.seg { math::lerp(a, b, best.s) } else { b };
        let l = math::dist(a, b);
        length += l;
        k1w += l * cx.c.k1[faces[i] as usize].abs();
        run += l;
        if l > 0.0 && (run >= SAMPLE_STEP * h0 || i == best.seg) {
            let tng = math::scale(math::sub(b, a), 1.0 / l);
            samples.push((math::lerp(a, b, 0.5), tng, run));
            run = 0.0;
        }
    }
    let close_l = best.dist * h0;
    length += close_l;
    let mean_k1 = if length > 0.0 { k1w / (length - close_l).max(1e-300) } else { 0.0 };
    let ring = mean_k1 > 0.0 && length < 1.5 * TAU / mean_k1;
    if !(length.is_finite() && length > 0.0) {
        return Outcome::None;
    }
    let cand = LoopCand { hes: cyc, start_face: f0, seed: from, branch: k0, length, ring };
    if cand.validate(px.mesh, px.topo).is_err() {
        return Outcome::None;
    }
    Outcome::Loop(Box::new(LoopInfo {
        cand,
        closure_err: best.dist,
        turning,
        expected_turning: turning,
        turning_excess: 0.0,
        cycle: CycleKind::NonSeparating,
        mean_k1,
        seed_vert: v,
        seed_rank: rank,
        samples,
    }))
}

fn check_inputs(px: &ProxyRef, f: &CrossField, c: &CurvField, s: &SizeField, p: &LoopParams) -> Result<(), AqError> {
    let nf = px.mesh.tris.len();
    let nh = 3 * nf;
    let nv = px.mesh.pos.len();
    if px.topo.twin.len() != nh || px.topo.vert_he.len() != nv || px.topo.is_bnd_vert.len() != nv {
        return Err(bad("aq_loops: topology does not match the mesh"));
    }
    if px.frames.e1.len() != nf || px.frames.kappa.len() != nh {
        return Err(bad("aq_loops: frames do not match the mesh"));
    }
    if px.feats.he_kind.len() != nh {
        return Err(bad("aq_loops: feature set does not match the mesh"));
    }
    if f.theta.len() != nf || f.p.len() != nh || f.delta.len() != nh || f.index4.len() != nv {
        return Err(bad("aq_loops: cross field does not match the mesh"));
    }
    if c.conf.len() != nf || c.k1.len() != nf {
        return Err(bad("aq_loops: curvature field does not match the mesh"));
    }
    if s.h_face.len() != nf || s.h_vert.len() != nv {
        return Err(bad("aq_loops: size field does not match the mesh"));
    }
    if s.h_face.iter().chain(&s.h_vert).any(|&h| !(h.is_finite() && h > 0.0)) {
        return Err(bad("aq_loops: size field must be finite and positive"));
    }
    let pos = |x: f64| x.is_finite() && x > 0.0;
    if !(pos(p.stride) && pos(p.max_len) && pos(p.close_tol) && pos(p.cluster) && p.min_conf.is_finite()) {
        return Err(bad("aq_loops: invalid LoopParams"));
    }
    Ok(())
}

/// Neighbour vertices of v (with repeats).
fn neighbours(px: &ProxyRef, v: u32, out: &mut Vec<u32>) {
    out.clear();
    for h in px.topo.fan_out(v) {
        out.push(px.topo.head(px.mesh, h));
        out.push(px.topo.tail(px.mesh, aq_core::Topology::prev(h)));
    }
}

/// Greedy farthest-point sampling of the eligible vertices (geodesic on edges, in h).
///
/// The first seed is the eligible vertex picked by the seed rng (C15); every later seed is the
/// eligible vertex farthest from all previous ones (ties to the lowest id). Stops when every
/// eligible vertex is closer than `stride`. The seeds are returned in this order.
fn fps_seeds(
    px: &ProxyRef,
    eligible: &[bool],
    hv: &[f64],
    stride: f64,
    seed: u64,
    ctl: &Ctl,
) -> Result<Vec<u32>, AqError> {
    let nv = px.mesh.pos.len();
    let mut dist = vec![f64::INFINITY; nv];
    let elig: Vec<u32> = (0..nv as u32).filter(|&v| eligible[v as usize]).collect();
    let mut global: BinaryHeap<(u64, Reverse<u32>)> =
        elig.iter().map(|&v| (f64::INFINITY.to_bits(), Reverse(v))).collect();
    if !elig.is_empty() {
        let first = elig[SplitMix64::new(seed).below(elig.len() as u64) as usize];
        global.push((u64::MAX, Reverse(first)));
    }
    let mut local: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    let mut nb = Vec::new();
    let mut seeds = Vec::new();
    let mut pops = 0usize;
    while let Some((dbits, Reverse(v))) = global.pop() {
        pops += 1;
        if pops.is_multiple_of(1024) {
            ctl.check().map_err(|e| e.at(Stage::Loops))?;
        }
        let first = dbits == u64::MAX && seeds.is_empty();
        if !first && dbits != dist[v as usize].to_bits() {
            continue;
        }
        if !first && f64::from_bits(dbits) < stride {
            break;
        }
        seeds.push(v);
        dist[v as usize] = 0.0;
        local.clear();
        local.push(Reverse((0.0f64.to_bits(), v)));
        while let Some(Reverse((ub, u))) = local.pop() {
            // The first seed of a component sweeps all of it: check here too (3.4).
            pops += 1;
            if pops.is_multiple_of(1024) {
                ctl.check().map_err(|e| e.at(Stage::Loops))?;
            }
            if ub != dist[u as usize].to_bits() {
                continue;
            }
            let du = f64::from_bits(ub);
            neighbours(px, u, &mut nb);
            for &w in &nb {
                let l = math::dist(px.mesh.pos[u as usize], px.mesh.pos[w as usize]);
                let nd = du + l / (0.5 * (hv[u as usize] + hv[w as usize]));
                if nd < dist[w as usize] {
                    dist[w as usize] = nd;
                    local.push(Reverse((nd.to_bits(), w)));
                    if eligible[w as usize] {
                        global.push((nd.to_bits(), Reverse(w)));
                    }
                }
            }
        }
    }
    Ok(seeds)
}

/// Accepted clustering samples per grid cell (point, tangent, accepted loop id). Only looked
/// up by key, never iterated, so the hash order cannot leak into the result.
type SampleGrid = HashMap<(i64, i64, i64), Vec<(V3, V3, u32)>>;

/// S5 (DESIGN 6.7): closed streamline loops, rings first, then shortest first.
pub fn find_loops(
    px: &ProxyRef,
    f: &CrossField,
    c: &CurvField,
    s: &SizeField,
    p: &LoopParams,
    ctl: &Ctl,
) -> Result<Vec<LoopCand>, AqError> {
    Ok(find_loops_report(px, f, c, s, p, ctl)?.0.into_iter().map(|l| l.cand).collect())
}

/// [`find_loops`] with the per-loop diagnostics and counters.
pub fn find_loops_report(
    px: &ProxyRef,
    f: &CrossField,
    c: &CurvField,
    s: &SizeField,
    p: &LoopParams,
    ctl: &Ctl,
) -> Result<(Vec<LoopInfo>, LoopReport), AqError> {
    check_inputs(px, f, c, s, p)?;
    let (m, t) = (px.mesh, px.topo);
    let nv = m.pos.len();
    let cone = cone_faces(m, f);
    // Eligible seeds and their start faces.
    let mut start_face = vec![NONE; nv];
    let mut eligible = vec![false; nv];
    for v in 0..nv as u32 {
        if t.vert_he[v as usize] == NONE || t.is_bnd_vert[v as usize] || f.index4[v as usize] != 0 {
            continue;
        }
        let mut sum = 0.0;
        let mut n = 0usize;
        let mut sf = NONE;
        for h in t.fan_out(v) {
            let fc = h / 3;
            sum += c.conf[fc as usize];
            n += 1;
            if !cone[fc as usize] && fc < sf {
                sf = fc;
            }
        }
        if n > 0 && sum / n as f64 >= p.min_conf && sf != NONE {
            eligible[v as usize] = true;
            start_face[v as usize] = sf;
        }
    }
    let seeds = fps_seeds(px, &eligible, &s.h_vert, p.stride, p.seed, ctl)?;
    let mut report = LoopReport { seeds: seeds.len() as u32, ..Default::default() };
    let cx = Ctx { px, f, c, s, p, cone };
    // Trace in parallel; results are collected in seed order, so the output does not depend
    // on the thread count.
    let outcomes: Vec<Result<[Outcome; 2], AqError>> = seeds
        .par_iter()
        .enumerate()
        .map(|(rank, &v)| {
            ctl.check().map_err(|e| e.at(Stage::Loops))?;
            let f0 = start_face[v as usize];
            let rank = rank as u32;
            Ok([trace_loop(&cx, v, rank, f0, 0), trace_loop(&cx, v, rank, f0, 1)])
        })
        .collect();
    let mut cands: Vec<LoopInfo> = Vec::new();
    for o in outcomes {
        for x in o? {
            report.traces += 1;
            match x {
                Outcome::None => {}
                Outcome::Loop(l) => {
                    report.closed += 1;
                    cands.push(*l);
                }
            }
        }
    }
    ctl.progress(Stage::Loops, 0.8);
    // Clustering order: rings first, then shortest first. Loops whose lengths agree within
    // LENGTH_GROUP of the shortest loop of their group are ordered by seed rank (the
    // farthest-point order spreads equal loops apart), then by branch.
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (&cands[a].cand, &cands[b].cand);
        y.ring.cmp(&x.ring).then(x.length.total_cmp(&y.length)).then(a.cmp(&b))
    });
    let mut grouped: Vec<usize> = Vec::with_capacity(order.len());
    let mut gs = 0usize;
    while gs < order.len() {
        let l0 = &cands[order[gs]].cand;
        let mut ge = gs + 1;
        while ge < order.len()
            && cands[order[ge]].cand.ring == l0.ring
            && cands[order[ge]].cand.length <= LENGTH_GROUP * l0.length
        {
            ge += 1;
        }
        let mut g: Vec<usize> = order[gs..ge].to_vec();
        g.sort_by_key(|&i| (cands[i].seed_rank, cands[i].cand.branch));
        grouped.extend(g);
        gs = ge;
    }
    let hmax = s.h_face.iter().fold(0.0f64, |a, &b| a.max(b));
    let cell = p.cluster * hmax;
    let key = |q: V3| -> (i64, i64, i64) {
        ((q[0] / cell).floor() as i64, (q[1] / cell).floor() as i64, (q[2] / cell).floor() as i64)
    };
    let cos45 = std::f64::consts::FRAC_1_SQRT_2;
    // Accepted samples by grid cell: (point, tangent, accepted loop id), in insertion order.
    let mut grid: SampleGrid = HashMap::new();
    let mut accepted: Vec<LoopInfo> = Vec::new();
    // Per accepted loop near the current sample: (id, squared distance, tangent).
    let mut nearest: Vec<(u32, f64, V3)> = Vec::new();
    for (n, &ci) in grouped.iter().enumerate() {
        if n % 64 == 0 {
            ctl.check().map_err(|e| e.at(Stage::Loops))?;
        }
        let l = &cands[ci];
        let total: f64 = l.samples.iter().map(|x| x.2).sum();
        let r = p.cluster * s.h_face[l.cand.start_face as usize].min(hmax);
        let r2 = r * r;
        let mut near = 0.0;
        for &(q, tg, w) in &l.samples {
            // The nearest sample of every accepted loop within r decides its family.
            nearest.clear();
            let (kx, ky, kz) = key(q);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(v) = grid.get(&(kx + dx, ky + dy, kz + dz)) else { continue };
                        for &(x, t2, id) in v {
                            let d2 = math::dist2(x, q);
                            if d2 > r2 {
                                continue;
                            }
                            match nearest.iter_mut().find(|e| e.0 == id) {
                                Some(e) => {
                                    if d2 < e.1 {
                                        *e = (id, d2, t2);
                                    }
                                }
                                None => nearest.push((id, d2, t2)),
                            }
                        }
                    }
                }
            }
            if nearest.iter().any(|e| math::dot(e.2, tg).abs() >= cos45) {
                near += w;
            }
        }
        if total > 0.0 && near >= 0.8 * total {
            report.rejected_cluster += 1;
            continue;
        }
        let tc = turning_check(px, f, &l.cand.hes, ctl)?;
        match tc.kind {
            CycleKind::Overlapping => {
                report.rejected_overlap += 1;
                continue;
            }
            CycleKind::Separating if tc.excess.is_nan() || tc.excess > MAX_TURNING_EXCESS => {
                report.rejected_turning += 1;
                continue;
            }
            _ => {}
        }
        let id = accepted.len() as u32;
        for &(q, tg, _) in &l.samples {
            grid.entry(key(q)).or_default().push((q, tg, id));
        }
        let mut l = l.clone();
        l.expected_turning = tc.expected;
        l.turning_excess = tc.excess;
        l.cycle = tc.kind;
        accepted.push(l);
    }
    report.accepted = accepted.len() as u32;
    ctl.progress(Stage::Loops, 1.0);
    Ok((accepted, report))
}

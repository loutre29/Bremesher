//! DESIGN 6.6 steps 3 to 6: path edits, high-index split, push-off and pair cancellation,
//! plus the WP-23 (M2) pole steps: corner and boundary merge, same-sign repulsion, the
//! energy-guarded far cancellation and the merge of -1 pairs into -2 cones.
//!
//! A path edit adds s to the matching of every halfedge `v_i -> v_{i+1}` of a vertex path:
//! delta of that halfedge drops by `s pi/2`, so index4(v_i) gains s and index4(v_{i+1}) loses
//! s; along the path these cancel and only `index4(v0) += s`, `index4(vk) -= s` remain.
//!
//! Every edit made by this module keeps the true residual inside (-pi, pi) (an edge is used
//! only while `|delta - s pi/2| < 0.8 pi`), because aq_core reduces delta modulo 2pi
//! (CHANGES.md, WP-00 item 2): past pi the stored delta would no longer describe the edit.

use std::collections::BTreeSet;
use std::f64::consts::{FRAC_PI_2, PI};

use aq_core::alloc::try_vec;
use aq_core::cross::delta_edge;
use aq_core::{AqError, CrossField, Ctl, NONE, SizeField, Stage, delta_from_p, index4_all};

use crate::graph::{Dij, Graph};
use crate::resolve::{Resolver, face_energies};
use crate::setup::{Setup, snap};
use crate::{CHAIN_CLEARANCE, FieldParams, FieldReport, KEEP_OUT, LOW_CONF, MAX_EDIT_DELTA, PUSH_TARGET, ProxyRef, bad, check_field_sizes};

/// Largest |delta - s pi/2| allowed on an edge before an edit.
const EDIT_LIMIT: f64 = 0.8 * PI;
/// Push-off searches at most this many h from the cone.
const PUSH_RADIUS: f64 = 20.0;
/// Push-off and split alternate at most this many times.
const PUSH_SPLIT_ROUNDS: usize = 4;
/// Same-sign repulsion distance in h (DESIGN 6.6 M2: 0.7 h).
const REPEL_DIST: f64 = 0.7;
/// Rounds of same-sign repulsion.
const REPEL_ROUNDS: usize = 3;
/// Corner merge radius in h (DESIGN 6.6 step 6, M2: 0.5 h).
const CORNER_MERGE: f64 = 0.5;
/// M2 integration: a merge into a boundary vertex is allowed only when the quad corner it
/// leaves, `(2 - index4) * 90` degrees in the parametrization, stays within this ratio of the
/// vertex's surface angle (either way). A slit (about 360 degrees, index4 -2) that took two
/// +1 cones became a flat 180 degree corner and folded S7 next to it on retopo_c.
const BOUNDARY_MERGE_RATIO: f64 = 1.5;

/// [`try_edit_path`] that ignores an invalid path (the field is then left unchanged), with
/// the DESIGN 6.6 signature.
pub fn edit_path(px: &ProxyRef, f: &mut CrossField, path: &[u32], s: i8) {
    let _ = try_edit_path(px, f, path, s);
}

/// Path edit (DESIGN 6.6 step 4): for each i, the halfedge `he_i` from `path[i]` to
/// `path[i+1]` gets `p = (p + s) mod 4` (its twin `(4 - p) mod 4`) and delta is recomputed;
/// then `index4(path[0]) += s` and `index4(path[k]) -= s`.
///
/// So `edit_path([a, .., b], -1)` moves a quarter of positive index from a to b, and
/// `edit_path([a(-1), .., b(+1)], +1)` cancels the pair. Every consecutive pair must be the
/// ends of an interior edge (`InvalidInput` otherwise, nothing changed). The index update is
/// exact as long as each edited residual stays inside (-pi, pi); callers keep it there.
pub fn try_edit_path(px: &ProxyRef, f: &mut CrossField, path: &[u32], s: i8) -> Result<(), AqError> {
    check_field_sizes(px, f)?;
    if path.len() < 2 {
        return Ok(());
    }
    let m = px.mesh;
    let t = px.topo;
    let nv = m.pos.len() as u32;
    let mut hes = Vec::with_capacity(path.len() - 1);
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a >= nv || b >= nv {
            return Err(bad(format!("edit_path: vertex out of range ({a}, {b})")));
        }
        let Some(h) = t.find_he(m, a, b) else {
            return Err(bad(format!("edit_path: no halfedge from {a} to {b}")));
        };
        if t.twin[h as usize] == NONE {
            return Err(bad(format!("edit_path: edge ({a}, {b}) is a boundary edge")));
        }
        hes.push(h);
    }
    let (v0, vk) = (path[0] as usize, path[path.len() - 1] as usize);
    let s16 = s as i16;
    if v0 != vk {
        let a = f.index4[v0] as i16 + s16;
        let b = f.index4[vk] as i16 - s16;
        if !(i8::MIN as i16..=i8::MAX as i16).contains(&a) || !(i8::MIN as i16..=i8::MAX as i16).contains(&b) {
            return Err(bad("edit_path: index4 out of range"));
        }
        f.index4[v0] = a as i8;
        f.index4[vk] = b as i8;
    }
    let sm = (s as i32).rem_euclid(4) as u8;
    for h in hes {
        let tw = t.twin[h as usize];
        f.p[h as usize] = (f.p[h as usize] + sm) % 4;
        f.p[tw as usize] = (4 - f.p[h as usize]) % 4;
        let c = h.min(tw) as usize;
        let ct = h.max(tw) as usize;
        let d = delta_edge(f.theta[c / 3], f.theta[ct / 3], px.frames.kappa[c], f.p[c]);
        f.delta[c] = d;
        f.delta[ct] = -d;
    }
    Ok(())
}

/// Shared state of the edit steps.
struct Ctx<'a> {
    px: &'a ProxyRef<'a>,
    st: &'a Setup,
    h: &'a [f64],
    g: Graph,
    dij: Dij,
    dij2: Dij,
    ctl: &'a Ctl<'a>,
}

impl Ctx<'_> {
    /// Non-chain interior vertex: the only vertices a path may enter.
    fn allowed(&self, v: u32) -> bool {
        !self.st.chain_vert[v as usize] && !self.px.topo.is_bnd_vert[v as usize]
    }

    fn corner(&self, v: u32) -> bool {
        self.px.feats.vert_corner[v as usize]
    }

    /// True when v lies within `KEEP_OUT` h of a keep-out point: no WP-23 edit there.
    fn kept_out(&self, p: &FieldParams, v: u32) -> bool {
        let q = self.px.mesh.pos[v as usize];
        let r = KEEP_OUT * self.h[v as usize];
        p.keep_out.iter().any(|&k| aq_core::math::dist(k, q) < r)
    }
}

/// True when halfedge he (interior) can take an edit of sign s.
fn edge_ok(px: &ProxyRef, f: &CrossField, he: u32, s: i8) -> bool {
    he != NONE && px.topo.twin[he as usize] != NONE && (f.delta[he as usize] - s as f64 * FRAC_PI_2).abs() < EDIT_LIMIT
}

/// Path of halfedges of a vertex path (NONE where missing).
fn path_hes(px: &ProxyRef, path: &[u32]) -> Vec<u32> {
    path.windows(2).map(|w| px.topo.find_he(px.mesh, w[0], w[1]).unwrap_or(NONE)).collect()
}

/// Steps 3 to 7 on f, ending with a re-solve. f must be consistent (delta from theta and p).
pub(crate) fn simplify(
    px: &ProxyRef,
    st: &Setup,
    sz: &SizeField,
    f: &mut CrossField,
    p: &FieldParams,
    ctl: &Ctl,
) -> Result<FieldReport, AqError> {
    let nv = px.mesh.pos.len();
    // Hard faces on their constraint edge, delta and index consistent with theta and p.
    for fc in 0..px.mesh.tris.len() {
        if st.hard[fc] {
            f.theta[fc] = snap(f.theta[fc], st.edge_angle[fc]);
        }
    }
    f.delta = delta_from_p(px.mesh, px.topo, px.frames, &f.theta, &f.p);
    f.index4 = index4_all(px.mesh, px.topo, px.frames, &f.delta).map_err(|e| e.at(Stage::Field))?;
    let mut cx = Ctx { px, st, h: &sz.h_vert, g: Graph::new(px)?, dij: Dij::new(nv)?, dij2: Dij::new(nv)?, ctl };
    let mut rs = Resolver::new(px, st, f, Some(ctl))?;
    let mut rep = FieldReport::default();
    ctl.check()?;

    let n = split_spikes(&mut cx, f, p)? + split_high(&mut cx, f)?;
    rep.split += n;
    if n > 0 {
        rs.solve(px, st, f)?;
    }
    // WP-23 (DESIGN 6.6 step 6, M2): cones next to an interior feature corner merge into it,
    // before the push-off would move them away from the chains.
    let (nc, nb) = merge_corners(&mut cx, f, p)?;
    rep.corner_merged += nc;
    rep.boundary_merged += nb;
    if nc + nb > 0 {
        rs.solve(px, st, f)?;
    }
    ctl.progress(Stage::Field, 0.5);
    // Push-off; a split it makes necessary may move a quarter next to a chain again, so the
    // push-off runs again after such a split (bounded).
    for _ in 0..PUSH_SPLIT_ROUNDS {
        let n = push_off(&mut cx, f, p)?;
        rep.pushed += n;
        if n == 0 {
            break;
        }
        ctl.check()?;
        rs.solve(px, st, f)?;
        let n2 = split_high(&mut cx, f)?;
        rep.split += n2;
        if n2 == 0 {
            break;
        }
        ctl.check()?;
        rs.solve(px, st, f)?;
    }
    // WP-16 round 2: same-sign repulsion (DESIGN 6.6 M2) before the cancellation.
    for _ in 0..REPEL_ROUNDS {
        let n = repel_same_sign(&mut cx, f)?;
        rep.pushed += n;
        if n == 0 {
            break;
        }
        ctl.check()?;
        rs.solve(px, st, f)?;
    }
    ctl.progress(Stage::Field, 0.6);
    cancel_pairs(&mut cx, &mut rs, f, p, &mut rep)?;
    if p.far_cancel > p.cancel_dist.max(p.cancel_dist_lowconf) {
        ctl.progress(Stage::Field, 0.75);
        cancel_far(&mut cx, &mut rs, f, p, &mut rep)?;
    }
    if p.merge_neg > 0.0 {
        ctl.progress(Stage::Field, 0.85);
        merge_negative(&mut cx, &mut rs, f, p, &mut rep)?;
    }
    ctl.progress(Stage::Field, 0.9);
    rs.solve(px, st, f)?;
    rep.cones_final = f.index4.iter().filter(|&&i| i != 0).count() as u32;
    rep.max_abs_delta = f.delta.iter().fold(0.0f64, |a, d| a.max(d.abs()));
    Ok(rep)
}

/// Step 3: moves quarters of index away from interior non-corner vertices with |index4| >= 2.
fn split_high(cx: &mut Ctx, f: &mut CrossField) -> Result<u32, AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let cap = 4 * nv + 64;
    let mut moves = 0u32;
    while let Some(v) = (0..nv as u32)
        .find(|&v| f.index4[v as usize].abs() >= 2 && !px.topo.is_bnd_vert[v as usize] && !cx.corner(v))
    {
        if moves as usize >= cap {
            break;
        }
        cx.ctl.check()?;
        let s: i8 = if f.index4[v as usize] > 0 { -1 } else { 1 };
        let sources: Vec<u32> = (0..nv as u32).filter(|&u| u != v && f.index4[u as usize] != 0).collect();
        let dist = if sources.is_empty() { None } else { Some(cx.dij.distances(&cx.g, &sources, Some(cx.ctl))?) };
        let mut best: Option<(f64, u32)> = None;
        for (w, he, _) in cx.g.neighbors(v) {
            if !cx.allowed(w) || !edge_ok(px, f, he, s) {
                continue;
            }
            let score = dist.as_ref().map_or(0.0, |d| d[w as usize]);
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, w));
            }
        }
        let Some((_, w)) = best else { break };
        try_edit_path(px, f, &[v, w], s)?;
        moves += 1;
    }
    Ok(moves)
}

/// WP-16 addition to step 3 (CHANGES.md): a boundary or feature corner with index4 >= 2 has
/// zero (or fewer) quarter turns in its sector, a spike tip that no quad corner can take (S8
/// then finds no 4-corner patch there and S7 folds next to it). Each extra quarter above +1
/// moves along a path edit into the interior, to the nearest non-chain vertex at least
/// 0.75 h from every chain (the push-off target), so the corner keeps one quarter turn
/// (index4 +1, a squeezed quad corner) and a valence 3 pole appears next to the tip.
fn split_spikes(cx: &mut Ctx, f: &mut CrossField, p: &FieldParams) -> Result<u32, AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let spikes: Vec<u32> = (0..nv as u32)
        .filter(|&v| f.index4[v as usize] >= 2 && (px.topo.is_bnd_vert[v as usize] || cx.corner(v)))
        .collect();
    if spikes.is_empty() {
        return Ok(0);
    }
    let chain: Vec<u32> = (0..nv as u32).filter(|&v| cx.st.chain_vert[v as usize] || px.topo.is_bnd_vert[v as usize]).collect();
    let d_ch = cx.dij2.distances(&cx.g, &chain, Some(cx.ctl))?;
    let _ = p;
    let mut moves = 0u32;
    for v in spikes {
        let hv = cx.h[v as usize];
        let mut guard = 0;
        while f.index4[v as usize] >= 2 && guard < 8 {
            guard += 1;
            cx.ctl.check()?;
            let s: i8 = -1;
            let st = cx.st;
            let h = cx.h;
            let ff: &CrossField = f;
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, s);
            let mut stop = |w: u32, _: f64| w != v && d_ch[w as usize] >= PUSH_TARGET * h[w as usize] && ff.index4[w as usize] <= 0;
            let found = cx.dij.run(&cx.g, &[v], PUSH_RADIUS * hv, &can, &eok, &mut stop, Some(cx.ctl))?;
            let Some(w) = found else { break };
            let path = cx.dij.path_to(w);
            try_edit_path(px, f, &path, s)?;
            moves += 1;
        }
    }
    Ok(moves)
}

/// Step 6: cones within `sym_clearance` h of a Symmetry chain or 0.35 h of another chain (the
/// boundary included) move, one quarter at a time, to the nearest non-chain interior vertex at
/// least 0.75 h from every chain that is not a cone of the same sign. Corners keep their index.
fn push_off(cx: &mut Ctx, f: &mut CrossField, p: &FieldParams) -> Result<u32, AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let sym: Vec<u32> = (0..nv as u32).filter(|&v| cx.st.sym_vert[v as usize]).collect();
    let oth: Vec<u32> = (0..nv as u32).filter(|&v| cx.st.other_vert[v as usize]).collect();
    if sym.is_empty() && oth.is_empty() {
        return Ok(0);
    }
    let d_sym = if sym.is_empty() { vec![f64::INFINITY; nv] } else { cx.dij.distances(&cx.g, &sym, Some(cx.ctl))? };
    let d_oth = if oth.is_empty() { vec![f64::INFINITY; nv] } else { cx.dij.distances(&cx.g, &oth, Some(cx.ctl))? };
    let cones: Vec<u32> = (0..nv as u32).filter(|&v| f.index4[v as usize] != 0 && !cx.corner(v)).collect();
    let mut moves = 0u32;
    for v in cones {
        let hv = cx.h[v as usize];
        if !(d_sym[v as usize] < p.sym_clearance * hv || d_oth[v as usize] < CHAIN_CLEARANCE * hv) {
            continue;
        }
        let mut guard = 0;
        while f.index4[v as usize] != 0 && guard < 16 {
            guard += 1;
            cx.ctl.check()?;
            let sign = f.index4[v as usize].signum();
            let s = -sign;
            let st = cx.st;
            let h = cx.h;
            let ff: &CrossField = f;
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, s);
            let mut stop = |w: u32, _: f64| {
                w != v
                    && d_sym[w as usize].min(d_oth[w as usize]) >= PUSH_TARGET * h[w as usize]
                    && ff.index4[w as usize].signum() != sign
            };
            let found = cx.dij.run(&cx.g, &[v], PUSH_RADIUS * hv, &can, &eok, &mut stop, Some(cx.ctl))?;
            let Some(w) = found else { break };
            let path = cx.dij.path_to(w);
            try_edit_path(px, f, &path, s)?;
            moves += 1;
        }
    }
    Ok(moves)
}

/// WP-16 round 2, DESIGN 6.6 M2 same-sign repulsion pulled into M1 (CHANGES.md): two cones of
/// the same sign closer than 0.7 h (the Knoppel solve often spreads a double singularity over
/// two adjacent proxy vertices) leave S7 folds next to them that no repair can open. The one
/// with the lower confidence (mean conf of its faces; ties: the first in vertex order) moves
/// along a path edit to the nearest free vertex at least 0.7 h from every other cone. Corners,
/// chain and boundary vertices never move.
fn repel_same_sign(cx: &mut Ctx, f: &mut CrossField) -> Result<u32, AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let st = cx.st;
    let free_cone = |v: u32, f: &CrossField| {
        f.index4[v as usize] != 0 && !px.feats.vert_corner[v as usize] && !st.chain_vert[v as usize] && !px.topo.is_bnd_vert[v as usize]
    };
    let vconf = |v: u32| {
        let (mut a, mut c) = (0.0, 0.0);
        for h in px.topo.fan_out(v) {
            let fc = (h / 3) as usize;
            a += px.frames.area[fc];
            c += px.frames.area[fc] * st.conf_eff[fc];
        }
        if a > 0.0 { c / a } else { 0.0 }
    };
    let cones: Vec<u32> = (0..nv as u32).filter(|&v| free_cone(v, f)).collect();
    let mut moves = 0u32;
    for &v in &cones {
        cx.ctl.check()?;
        if !free_cone(v, f) {
            continue;
        }
        let sign = f.index4[v as usize].signum();
        let hv = cx.h[v as usize];
        let near = {
            let ff: &CrossField = f;
            cx.dij.run(&cx.g, &[v], REPEL_DIST * hv, &|_| true, &|_, _, _| true, &mut |w, _| w != v && ff.index4[w as usize].signum() == sign, Some(cx.ctl))?
        };
        let Some(u) = near else { continue };
        let mover = if free_cone(u, f) && vconf(u) < vconf(v) { u } else { v };
        if !free_cone(mover, f) {
            continue;
        }
        let hm = cx.h[mover as usize];
        let others: Vec<u32> = (0..nv as u32).filter(|&w| w != mover && f.index4[w as usize] != 0).collect();
        cx.dij2.run(&cx.g, &others, 2.0 * REPEL_DIST * hm, &|_| true, &|_, _, _| true, &mut |_, _| false, Some(cx.ctl))?;
        let s: i8 = -sign;
        let h = cx.h;
        let ff: &CrossField = f;
        let d2 = &cx.dij2;
        let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
        let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, s);
        let mut stop = |w: u32, _: f64| w != mover && ff.index4[w as usize] == 0 && d2.dist_of(w) >= REPEL_DIST * h[w as usize];
        let found = cx.dij.run(&cx.g, &[mover], PUSH_RADIUS * hm, &can, &eok, &mut stop, Some(cx.ctl))?;
        let Some(w) = found else { continue };
        let path = cx.dij.path_to(w);
        try_edit_path(px, f, &path, s)?;
        moves += 1;
    }
    Ok(moves)
}

/// WP-23, DESIGN 6.6 step 6 (M2): a free cone (index4 +-1, off chains and the boundary) moves
/// by a path edit whose inner vertices are free into
/// - an interior feature corner within `CORNER_MERGE` h when the corner's merged |index4|
///   stays <= 1 (DESIGN: the pole then sits on the feature corner instead of next to it, and a
///   -1 cone next to a +1 corner cancels with it);
/// - a boundary vertex within `p.boundary_merge` h whose index4 has the opposite sign (WP-23
///   extension, CHANGES.md): an interior valence 3 pole next to a boundary notch (index4 -1, a
///   boundary vertex of valence 4) or a valence 5 pole next to a boundary spike leave two
///   irregular vertices where the merged boundary vertex leaves one or none, at the price of a
///   wider or narrower quad corner on the boundary.
///
/// The nearest target wins. Returns the number of merges into feature corners and into
/// boundary vertices.
fn merge_corners(cx: &mut Ctx, f: &mut CrossField, p: &FieldParams) -> Result<(u32, u32), AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let st = cx.st;
    let is_corner = |v: u32| px.feats.vert_corner[v as usize] && !px.topo.is_bnd_vert[v as usize];
    let is_bnd = |v: u32| px.topo.is_bnd_vert[v as usize];
    let rb = p.boundary_merge;
    if !(0..nv as u32).any(|v| is_corner(v) || (rb > 0.0 && is_bnd(v) && f.index4[v as usize] != 0)) {
        return Ok((0, 0));
    }
    let cones: Vec<u32> = (0..nv as u32)
        .filter(|&v| f.index4[v as usize].abs() == 1 && !st.chain_vert[v as usize] && !px.topo.is_bnd_vert[v as usize] && !cx.corner(v))
        .collect();
    let kov = kept_out_all(cx, p);
    let ko = |v: u32| kov.get(v as usize).copied().unwrap_or(false);
    let mut merges = (0u32, 0u32);
    for v in cones {
        cx.ctl.check()?;
        let sign = f.index4[v as usize];
        if sign.abs() != 1 {
            continue;
        }
        let s = -sign;
        let hv = cx.h[v as usize];
        let ff: &CrossField = f;
        let out_v = ko(v);
        let target = |w: u32, d: f64| {
            let i = ff.index4[w as usize];
            (is_corner(w) && d <= CORNER_MERGE * hv && (i + sign).abs() <= 1)
                || (is_bnd(w) && d <= rb * hv && (i + sign).abs() < i.abs() && bnd_corner_ok(px, w, i + sign) && !out_v && !ko(w))
        };
        let can = |w: u32| !st.chain_vert[w as usize] || is_corner(w) || is_bnd(w);
        // Paths never cross a chain or the boundary: only their last vertex may be one.
        let eok = |a: u32, _: u32, he: u32| (a == v || (!st.chain_vert[a as usize] && !is_bnd(a))) && edge_ok(px, ff, he, s);
        let mut stop = |w: u32, d: f64| w != v && target(w, d);
        let found = cx.dij.run(&cx.g, &[v], CORNER_MERGE.max(rb) * hv, &can, &eok, &mut stop, Some(cx.ctl))?;
        let Some(w) = found else { continue };
        let path = cx.dij.path_to(w);
        try_edit_path(px, f, &path, s)?;
        if is_bnd(w) {
            merges.1 += 1;
        } else {
            merges.0 += 1;
        }
    }
    Ok(merges)
}

/// [`Ctx::kept_out`] for every vertex (empty when there is no keep-out point).
fn kept_out_all(cx: &Ctx, p: &FieldParams) -> Vec<bool> {
    if p.keep_out.is_empty() {
        return Vec::new();
    }
    (0..cx.px.mesh.pos.len() as u32).map(|v| cx.kept_out(p, v)).collect()
}

/// True when boundary vertex w with index4 `i` keeps a parametric quad corner of
/// `(2 - i) * 90` degrees within [`BOUNDARY_MERGE_RATIO`] of its surface angle.
fn bnd_corner_ok(px: &ProxyRef, w: u32, i: i8) -> bool {
    let k = 2 - i as i32;
    if k < 1 {
        return false;
    }
    let omega: f64 = px.topo.fan_out(w).map(|h| px.frames.corner_angle[h as usize]).sum();
    let uv = k as f64 * FRAC_PI_2;
    omega > 0.0 && uv.max(omega) <= BOUNDARY_MERGE_RATIO * uv.min(omega)
}

/// Area-weighted mean confidence of the faces touching the vertices within `r` of `mid`.
fn disc_conf(cx: &mut Ctx, mid: u32, r: f64) -> Result<f64, AqError> {
    let px = cx.px;
    cx.dij2.run(&cx.g, &[mid], r, &|_| true, &|_, _, _| true, &mut |_, _| false, Some(cx.ctl))?;
    let mut faces = BTreeSet::new();
    for &v in cx.dij2.touched() {
        if cx.dij2.is_settled(v) {
            for h in px.topo.fan_out(v) {
                faces.insert(h / 3);
            }
        }
    }
    let (mut a, mut ac) = (0.0, 0.0);
    for f in faces {
        let ar = px.frames.area[f as usize];
        a += ar;
        ac += ar * cx.st.conf_eff[f as usize];
    }
    Ok(if a > 0.0 { ac / a } else { 0.0 })
}

struct Applied {
    pair: (u32, u32),
    path: Vec<u32>,
}

/// Step 5: greedy cancellation of close +1/-1 pairs, the |delta| <= pi/3 check and revert.
fn cancel_pairs(cx: &mut Ctx, rs: &mut Resolver, f: &mut CrossField, p: &FieldParams, rep: &mut FieldReport) -> Result<(), AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let dmax = p.cancel_dist.max(p.cancel_dist_lowconf);
    let mut rejected: BTreeSet<(u32, u32)> = BTreeSet::new();
    for _round in 0..p.max_cancel_rounds {
        cx.ctl.check()?;
        let eligible = |v: u32, f: &CrossField| {
            !px.feats.vert_corner[v as usize]
                && !cx.st.chain_vert[v as usize]
                && !px.topo.is_bnd_vert[v as usize]
                && f.index4[v as usize].abs() == 1
        };
        let minus: Vec<u32> = (0..nv as u32).filter(|&v| eligible(v, f) && f.index4[v as usize] == -1).collect();
        let mut is_plus = try_vec(nv, false)?;
        let mut any_plus = false;
        for v in 0..nv as u32 {
            if eligible(v, f) && f.index4[v as usize] == 1 {
                is_plus[v as usize] = true;
                any_plus = true;
            }
        }
        if minus.is_empty() || !any_plus {
            break;
        }
        // Candidate pairs.
        let mut cands: Vec<(u64, u32, u32)> = Vec::new();
        for &a in &minus {
            cx.ctl.check()?;
            let ha = cx.h[a as usize];
            let (st, ff) = (cx.st, &*f);
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, 1);
            cx.dij.run(&cx.g, &[a], 2.0 * dmax * ha, &can, &eok, &mut |_, _| false, Some(cx.ctl))?;
            let reached: Vec<(u32, f64)> = cx
                .dij
                .touched()
                .iter()
                .filter(|&&b| is_plus[b as usize] && cx.dij.is_settled(b))
                .map(|&b| (b, cx.dij.dist[b as usize]))
                .collect();
            for (b, d) in reached {
                if rejected.contains(&(a, b)) {
                    continue;
                }
                let hbar = 0.5 * (ha + cx.h[b as usize]);
                let ok = if d < p.cancel_dist * hbar {
                    true
                } else if d < p.cancel_dist_lowconf * hbar {
                    cx.ctl.check()?;
                    let path = cx.dij.path_to(b);
                    let mid = path.iter().copied().find(|&u| cx.dij.dist[u as usize] >= 0.5 * d).unwrap_or(b);
                    disc_conf(cx, mid, d)? < LOW_CONF
                } else {
                    false
                };
                if ok {
                    cands.push((d.to_bits(), a, b));
                }
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort_unstable();
        let mut used = BTreeSet::new();
        let mut selected = Vec::new();
        for &(db, a, b) in &cands {
            if used.contains(&a) || used.contains(&b) {
                continue;
            }
            used.insert(a);
            used.insert(b);
            selected.push((f64::from_bits(db), a, b));
        }
        // Apply the edits, each on a path found in the current state.
        let snapshot = f.clone();
        let mut applied: Vec<Applied> = Vec::new();
        for &(d, a, b) in &selected {
            cx.ctl.check()?;
            let ha = cx.h[a as usize];
            let (st, ff) = (cx.st, &*f);
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, 1);
            let bound = (1.5 * d).max(2.0 * dmax * ha);
            let found = cx.dij.run(&cx.g, &[a], bound, &can, &eok, &mut |w, _| w == b, Some(cx.ctl))?;
            if found != Some(b) {
                continue;
            }
            let path = cx.dij.path_to(b);
            try_edit_path(px, f, &path, 1)?;
            applied.push(Applied { pair: (a, b), path });
        }
        if applied.is_empty() {
            break;
        }
        cx.ctl.check()?;
        rs.solve(px, cx.st, f)?;
        // Revert the pairs whose edited edges keep |delta| > pi/3: restore the state before
        // this round, re-apply the kept pairs, re-solve; repeat until every kept path passes.
        loop {
            let mut keep = Vec::new();
            let mut dropped = 0u32;
            for ap in applied.drain(..) {
                let bad_path = path_hes(px, &ap.path)
                    .iter()
                    .any(|&h| h == NONE || f.delta[h as usize].abs() > MAX_EDIT_DELTA + 1e-12);
                if bad_path {
                    rejected.insert(ap.pair);
                    dropped += 1;
                } else {
                    keep.push(ap);
                }
            }
            if dropped == 0 {
                applied = keep;
                break;
            }
            rep.reverted += dropped;
            *f = snapshot.clone();
            for ap in keep {
                let ok = path_hes(px, &ap.path).iter().all(|&h| edge_ok(px, f, h, 1));
                if ok {
                    try_edit_path(px, f, &ap.path, 1)?;
                    applied.push(ap);
                } else {
                    rejected.insert(ap.pair);
                    rep.reverted += 1;
                }
            }
            if applied.is_empty() {
                break;
            }
            cx.ctl.check()?;
            rs.solve(px, cx.st, f)?;
        }
        rep.cancelled += applied.len() as u32;
    }
    Ok(())
}

/// Rounds of the energy-guarded far cancellation (WP-23).
const FAR_ROUNDS: usize = 4;
/// Radius in h of the Gaussian curvature ball of a far-cancellation cone (WP-23).
const FAR_K_RADIUS: f64 = 1.5;
/// A cone whose ball carries more than this many quarter turns of its own sign is geometric.
const FAR_K_TOL: f64 = 1.5;
/// Far cancellation only where the mean coherent confidence of the disc is below this. The
/// step 5 low-confidence threshold is 0.2; 0.25 (the upper end of the curvature confidence
/// gate) lets the K-free pairs of retopo_b go (discs 0.14 to 0.22) and keeps the strand pairs
/// of sculpt_cylinder (discs 0.27), whose cancellation raised edge CV and angle deviation
/// above QuadriFlow's although it lowered the field energy (CHANGES.md, WP-23).
const FAR_CONF: f64 = 0.25;
/// M2 integration (WP-20 + WP-23): a far pair whose cones have sizes h more than this ratio
/// apart is not cancelled. With the thickness cap such a pair straddles the edge of a capped
/// rim (suzanne's ears: h 0.79 against 1.05); cancelling it raised suzanne_sub2's edge CV from
/// 0.158 to 0.217. 1.15 keeps every far pair of a uniform size field.
const FAR_SIZE_RATIO: f64 = 1.15;

/// WP-23 (M2, CHANGES.md): energy-guarded cancellation of +1/-1 pairs farther apart than the
/// step 5 distances, up to `far_cancel` h. Step 5 accepts a pair on distance alone; past
/// 1.2 h (2.5 h in low confidence) a pair is cancelled only when the field pays for it.
/// Candidates: both cones free (index4 +-1, off corners, chains and the boundary), neither
/// geometric (more than `FAR_K_TOL` quarter turns of Gaussian curvature of its own sign within
/// `FAR_K_RADIUS` h), joined by a free path of at most 1.5 d, and the disc of radius d around
/// the path midpoint of mean confidence below `FAR_CONF` (a coherent target is kept). The
/// pairs of a round (greedy by increasing distance, each cone once) are edited together and
/// re-solved; the change of the step 7 energy is split over the pairs by a Voronoi partition
/// of the faces around their edit paths, and a pair is reverted when its share exceeds
/// `far_cancel_energy` or an edge of its path keeps |delta| > pi/3 (the step 5 check), with
/// the revert loop of step 5. Where alignment is weak or incoherent, the pair's own
/// smoothness energy is released, so the share is often negative there.
fn cancel_far(cx: &mut Ctx, rs: &mut Resolver, f: &mut CrossField, p: &FieldParams, rep: &mut FieldReport) -> Result<(), AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let dmax = p.far_cancel;
    let mut rejected: BTreeSet<(u32, u32)> = BTreeSet::new();
    for _round in 0..FAR_ROUNDS {
        cx.ctl.check()?;
        let eligible = |v: u32, f: &CrossField| {
            !px.feats.vert_corner[v as usize]
                && !cx.st.chain_vert[v as usize]
                && !px.topo.is_bnd_vert[v as usize]
                && f.index4[v as usize].abs() == 1
        };
        let minus: Vec<u32> = (0..nv as u32).filter(|&v| eligible(v, f) && f.index4[v as usize] == -1).collect();
        let mut is_plus = try_vec(nv, false)?;
        let mut any_plus = false;
        for v in 0..nv as u32 {
            if eligible(v, f) && f.index4[v as usize] == 1 {
                is_plus[v as usize] = true;
                any_plus = true;
            }
        }
        if minus.is_empty() || !any_plus {
            break;
        }
        // Geometric cones (a ball of 1.5 h around them carries more than FAR_K_TOL quarter
        // turns of Gaussian curvature of their own sign) are never cancelled on energy alone.
        let mut geometric = try_vec(nv, false)?;
        for v in 0..nv as u32 {
            let i = f.index4[v as usize];
            if eligible(v, f) {
                let k = k_ball(cx, v, FAR_K_RADIUS)? / FRAC_PI_2;
                geometric[v as usize] = k * i as f64 > FAR_K_TOL;
            }
        }
        let kov = kept_out_all(cx, p);
        let ko = |v: u32| kov.get(v as usize).copied().unwrap_or(false);
        let mut cands: Vec<(u64, u32, u32)> = Vec::new();
        for &a in &minus {
            cx.ctl.check()?;
            if geometric[a as usize] || ko(a) {
                continue;
            }
            let ha = cx.h[a as usize];
            let (st, ff) = (cx.st, &*f);
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, 1);
            cx.dij.run(&cx.g, &[a], 1.5 * dmax * ha, &can, &eok, &mut |_, _| false, Some(cx.ctl))?;
            for &b in cx.dij.touched() {
                if !is_plus[b as usize] || geometric[b as usize] || ko(b) || !cx.dij.is_settled(b) || rejected.contains(&(a, b)) {
                    continue;
                }
                let d = cx.dij.dist[b as usize];
                if d < dmax * 0.5 * (ha + cx.h[b as usize]) {
                    cands.push((d.to_bits(), a, b));
                }
            }
        }
        // Low confidence only: the disc of radius d around the path midpoint (the step 5
        // rule) must have a mean coherent confidence below FAR_CONF.
        let mut kept = Vec::with_capacity(cands.len());
        for &(db, a, b) in &cands {
            cx.ctl.check()?;
            let d = f64::from_bits(db);
            let (st, ff) = (cx.st, &*f);
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, 1);
            let found = cx.dij.run(&cx.g, &[a], 1.5 * d + 1e-9, &can, &eok, &mut |w, _| w == b, Some(cx.ctl))?;
            if found != Some(b) {
                continue;
            }
            let path = cx.dij.path_to(b);
            let mid = path.iter().copied().find(|&u| cx.dij.dist[u as usize] >= 0.5 * d).unwrap_or(b);
            // M2 integration: a pair whose cones sit at sizes more than FAR_SIZE_RATIO apart
            // spans a size transition (a thickness cap), and is kept.
            let (ha, hb) = (cx.h[a as usize], cx.h[b as usize]);
            if ha.max(hb) <= FAR_SIZE_RATIO * ha.min(hb) && disc_conf(cx, mid, d)? < FAR_CONF {
                kept.push((db, a, b));
            }
        }
        let mut cands = kept;
        if cands.is_empty() {
            break;
        }
        cands.sort_unstable();
        let mut used = BTreeSet::new();
        let mut selected = Vec::new();
        for &(db, a, b) in &cands {
            if used.contains(&a) || used.contains(&b) {
                continue;
            }
            used.insert(a);
            used.insert(b);
            selected.push((f64::from_bits(db), a, b));
        }
        let snapshot = f.clone();
        let e0 = face_energies(px, cx.st, f);
        let mut applied: Vec<Applied> = Vec::new();
        for &(d, a, b) in &selected {
            cx.ctl.check()?;
            let ha = cx.h[a as usize];
            let (st, ff) = (cx.st, &*f);
            let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
            let eok = |_: u32, _: u32, he: u32| edge_ok(px, ff, he, 1);
            let bound = (1.5 * d).max(1.5 * dmax * ha);
            let found = cx.dij.run(&cx.g, &[a], bound, &can, &eok, &mut |w, _| w == b, Some(cx.ctl))?;
            if found != Some(b) {
                continue;
            }
            let path = cx.dij.path_to(b);
            try_edit_path(px, f, &path, 1)?;
            applied.push(Applied { pair: (a, b), path });
        }
        if applied.is_empty() {
            break;
        }
        cx.ctl.check()?;
        rs.solve(px, cx.st, f)?;
        loop {
            let e1 = face_energies(px, cx.st, f);
            let share = pair_energy_shares(cx, &applied, &e0, &e1)?;
            let mut keep = Vec::new();
            let mut dropped = 0u32;
            for (ap, de) in applied.drain(..).zip(share) {
                let bad_path = path_hes(px, &ap.path)
                    .iter()
                    .any(|&h| h == NONE || f.delta[h as usize].abs() > MAX_EDIT_DELTA + 1e-12);
                if bad_path || de > p.far_cancel_energy {
                    rejected.insert(ap.pair);
                    dropped += 1;
                } else {
                    keep.push(ap);
                }
            }
            if dropped == 0 {
                applied = keep;
                break;
            }
            rep.reverted += dropped;
            *f = snapshot.clone();
            for ap in keep {
                if path_hes(px, &ap.path).iter().all(|&h| edge_ok(px, f, h, 1)) {
                    try_edit_path(px, f, &ap.path, 1)?;
                    applied.push(ap);
                } else {
                    rejected.insert(ap.pair);
                    rep.reverted += 1;
                }
            }
            cx.ctl.check()?;
            rs.solve(px, cx.st, f)?;
            if applied.is_empty() {
                break;
            }
        }
        rep.cancelled += applied.len() as u32;
        rep.far_cancelled += applied.len() as u32;
    }
    Ok(())
}

/// Energy change per applied pair: every face goes to the pair whose edit path is nearest
/// (multi-source Dijkstra from all path vertices; the face's first corner decides), and the
/// pair's share is the sum of `after - before` over its faces.
fn pair_energy_shares(cx: &mut Ctx, applied: &[Applied], e0: &[f64], e1: &[f64]) -> Result<Vec<f64>, AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let mut label = try_vec(nv, u32::MAX)?;
    let mut sources = Vec::new();
    for (i, ap) in applied.iter().enumerate() {
        for &v in &ap.path {
            if label[v as usize] == u32::MAX {
                label[v as usize] = i as u32;
                sources.push(v);
            }
        }
    }
    cx.dij2.run(&cx.g, &sources, f64::INFINITY, &|_| true, &|_, _, _| true, &mut |_, _| false, Some(cx.ctl))?;
    let mut chain = Vec::new();
    for &v in cx.dij2.touched() {
        let mut cur = v;
        chain.clear();
        while label[cur as usize] == u32::MAX {
            chain.push(cur);
            let pr = cx.dij2.pred[cur as usize];
            if pr == NONE {
                break;
            }
            cur = pr;
        }
        let l = label[cur as usize];
        for &u in &chain {
            label[u as usize] = l;
        }
    }
    let mut share = vec![0.0; applied.len()];
    for (fc, tri) in px.mesh.tris.iter().enumerate() {
        let l = label[tri[0] as usize];
        if l != u32::MAX {
            share[l as usize] += e1[fc] - e0[fc];
        }
    }
    Ok(share)
}

/// Angle defect (integrated Gaussian curvature) of the interior vertices within `r` h of v.
fn k_ball(cx: &mut Ctx, v: u32, r: f64) -> Result<f64, AqError> {
    let px = cx.px;
    let hv = cx.h[v as usize];
    cx.dij2.run(&cx.g, &[v], r * hv, &|_| true, &|_, _, _| true, &mut |_, _| false, Some(cx.ctl))?;
    let mut k = 0.0;
    for &u in cx.dij2.touched() {
        if cx.dij2.is_settled(u) && !px.topo.is_bnd_vert[u as usize] {
            k += 2.0 * PI - px.frames.angle_sum(px.topo, u);
        }
    }
    Ok(k)
}

/// WP-23 (M2, CHANGES.md): two free -1 cones (off corners, chains and the boundary) joined by
/// a free path shorter than `merge_neg` h merge into one -2 cone, an interior valence 6 pole,
/// at the vertex of the path nearest its middle: one irregular vertex instead of two. Saddles
/// whose Gaussian curvature needs two negative quarters (the palm between fingers, strand
/// roots) otherwise leave two valence 5 poles side by side, which the same-sign repulsion
/// keeps 0.7 h apart. Runs after the cancellation, so a -1 cone that can cancel with a +1 cone
/// never merges. Positive cones never merge (a +2 interior vertex is a valence 2 vertex).
/// Pairs are greedy by increasing distance, each cone once; the round is re-solved and a pair
/// whose path keeps an edge with |delta| > pi/3 is reverted (the step 5 check). One round: a
/// -2 cone never merges again.
fn merge_negative(cx: &mut Ctx, rs: &mut Resolver, f: &mut CrossField, p: &FieldParams, rep: &mut FieldReport) -> Result<(), AqError> {
    let px = cx.px;
    let nv = px.mesh.pos.len();
    let st = cx.st;
    let free = |v: u32| !px.feats.vert_corner[v as usize] && !st.chain_vert[v as usize] && !px.topo.is_bnd_vert[v as usize];
    let minus: Vec<u32> = (0..nv as u32).filter(|&v| free(v) && f.index4[v as usize] == -1).collect();
    if minus.len() < 2 {
        return Ok(());
    }
    let kov = kept_out_all(cx, p);
    let ko = |v: u32| kov.get(v as usize).copied().unwrap_or(false);
    let mut cands: Vec<(u64, u32, u32)> = Vec::new();
    for &a in &minus {
        cx.ctl.check()?;
        if ko(a) {
            continue;
        }
        let ha = cx.h[a as usize];
        let ff: &CrossField = f;
        let can = |w: u32| !st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize];
        cx.dij.run(&cx.g, &[a], p.merge_neg * ha, &can, &|_, _, _| true, &mut |_, _| false, Some(cx.ctl))?;
        for &b in cx.dij.touched() {
            if b <= a || !cx.dij.is_settled(b) || ff.index4[b as usize] != -1 || !free(b) || ko(b) {
                continue;
            }
            let d = cx.dij.dist[b as usize];
            if d < p.merge_neg * 0.5 * (ha + cx.h[b as usize]) {
                cands.push((d.to_bits(), a, b));
            }
        }
    }
    if cands.is_empty() {
        return Ok(());
    }
    cands.sort_unstable();
    let mut used = BTreeSet::new();
    let snapshot = f.clone();
    let mut applied: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for &(db, a, b) in &cands {
        if used.contains(&a) || used.contains(&b) {
            continue;
        }
        cx.ctl.check()?;
        let d = f64::from_bits(db);
        let ff: &CrossField = f;
        // Inner vertices free and without index, so the path carries no other cone.
        let can = |w: u32| w == b || (!st.chain_vert[w as usize] && !px.topo.is_bnd_vert[w as usize] && ff.index4[w as usize] == 0);
        let found = cx.dij.run(&cx.g, &[a], 1.5 * d + 1e-9, &can, &|_, _, _| true, &mut |w, _| w == b, Some(cx.ctl))?;
        if found != Some(b) {
            continue;
        }
        let path = cx.dij.path_to(b);
        let dm = cx.dij.dist[b as usize];
        let mid = path.iter().position(|&u| cx.dij.dist[u as usize] >= 0.5 * dm).unwrap_or(path.len() - 1);
        // a -> m and b -> m, each with s = +1: a and b reach 0, m reaches -2.
        let pa: Vec<u32> = path[..=mid].to_vec();
        let pb: Vec<u32> = path[mid..].iter().rev().copied().collect();
        let ok = |q: &[u32]| path_hes(px, q).iter().all(|&h| edge_ok(px, ff, h, 1));
        if !ok(&pa) || !ok(&pb) {
            continue;
        }
        try_edit_path(px, f, &pa, 1)?;
        try_edit_path(px, f, &pb, 1)?;
        used.insert(a);
        used.insert(b);
        applied.push((pa, pb));
    }
    if applied.is_empty() {
        return Ok(());
    }
    cx.ctl.check()?;
    rs.solve(px, st, f)?;
    loop {
        let bad_path = |q: &[u32], f: &CrossField| {
            path_hes(px, q).iter().any(|&h| h == NONE || f.delta[h as usize].abs() > MAX_EDIT_DELTA + 1e-12)
        };
        let (keep, drop): (Vec<_>, Vec<_>) = applied.drain(..).partition(|(pa, pb)| !bad_path(pa, f) && !bad_path(pb, f));
        if drop.is_empty() {
            applied = keep;
            break;
        }
        rep.reverted += drop.len() as u32;
        *f = snapshot.clone();
        for (pa, pb) in keep {
            let ok = |q: &[u32], f: &CrossField| path_hes(px, q).iter().all(|&h| edge_ok(px, f, h, 1));
            if ok(&pa, f) {
                try_edit_path(px, f, &pa, 1)?;
                if ok(&pb, f) {
                    try_edit_path(px, f, &pb, 1)?;
                    applied.push((pa, pb));
                    continue;
                }
                try_edit_path(px, f, &pa, -1)?;
            }
            rep.reverted += 1;
        }
        cx.ctl.check()?;
        rs.solve(px, st, f)?;
        if applied.is_empty() {
            break;
        }
    }
    rep.merged_neg += applied.len() as u32;
    Ok(())
}

//! Constraint curves (chains and exact loops) and emitters (DESIGN 6.10).

use std::cmp::Ordering;
use std::collections::BTreeSet;

use aq_core::exact::orient2d;
use aq_core::math::{cross2, dot2, sub2};
use aq_core::tol::snap_dyadic;
use aq_core::{AqError, Ctl, LoopCand, NONE, SeamlessParam, Stage, Topology, Transition};

use crate::ctx::Ctx;
use crate::geom::{DIR_ALL, Line, Pt, aval, chord_free, chord_from_edge, dir_vec, on_edge_of};
use crate::{Curve, CurveKind, Emitter, LayoutParams, Piece, ProxyView, Seed};

/// One curve per constraint chain, in chain id order (curve id = position).
pub fn chain_curves(cx: &Ctx) -> Vec<Curve> {
    let mut out = Vec::new();
    for (i, ch) in cx.feats.chains.iter().enumerate() {
        if cx.chain_curve[i] != NONE {
            out.push(Curve { kind: CurveKind::Chain, chain: i as u32, src: i as u32, pieces: Vec::new(), closed: ch.closed });
        }
    }
    out
}

/// Traces every exact loop (`prm.loops_exact`) as an iso-line through its snapped seed,
/// until it closes exactly. Loops that hit the boundary, exceed the length cap, have an
/// inexact holonomy or duplicate an earlier loop are dropped.
pub fn trace_loops(cx: &Ctx, loops: &[LoopCand], p: &LayoutParams, ctl: &Ctl) -> Result<Vec<Curve>, AqError> {
    let nf = cx.n_faces();
    let mut idx: Vec<u32> = cx.prm.loops_exact.clone();
    idx.sort_unstable();
    idx.dedup();
    let mut seen: BTreeSet<(u32, usize, u64)> = BTreeSet::new();
    let mut out = Vec::new();
    for li in idx {
        ctl.check()?;
        let Some(lc) = loops.get(li as usize) else { continue };
        if (lc.seed.face as usize) >= nf || lc.hes.is_empty() {
            continue;
        }
        let mut tau = Transition::IDENTITY;
        if lc.hes.iter().any(|&h| (h as usize) >= 3 * nf) {
            continue;
        }
        for &h in &lc.hes {
            tau = tau.compose(&cx.prm.trans[h as usize]);
        }
        if !tau.r.is_multiple_of(4) {
            continue;
        }
        let a: u8 = match [tau.t[0] != 0.0, tau.t[1] != 0.0] {
            [true, false] => 0,
            [false, true] => 1,
            _ => continue,
        };
        let f0 = lc.seed.face;
        let tri = cx.tri(f0);
        let b = lc.seed.bary;
        let pq = |i: usize| b[0] * tri[0][i] + b[1] * tri[1][i] + b[2] * tri[2][i];
        let line0 = Line { d: a, c: snap_dyadic(pq(1 - a as usize)) };
        let Some(ch0) = chord_free(tri, line0) else { continue };
        let maxl = cx.max_len(f0, p.max_trace_len);
        let mut pieces: Vec<Piece> = Vec::new();
        let (mut f, mut l, mut ch) = (f0, line0, ch0);
        let mut t = 0.0;
        let closed = loop {
            let tr = cx.tri(f);
            let x_in = aval(tr, l, ch.p0);
            pieces.push(Piece {
                face: f,
                line: l,
                start: ch.p0,
                end: ch.p1,
                entry: ch.entry,
                exit: ch.exit,
                on_edge: on_edge_of(ch.p0, ch.p1),
                t_in: t,
                x_in,
            });
            t += (aval(tr, l, ch.p1) - x_in).abs();
            if t > maxl || pieces.len() > 8 * nf + 64 {
                break false;
            }
            if pieces.len().is_multiple_of(1024) {
                ctl.check()?;
            }
            let Some((g, kg, l2, _)) = cx.cross(f, ch.exit, l, ch.p1) else { break false };
            if g == f0 && l2 == line0 && kg == ch0.entry {
                break true;
            }
            let Some(c2) = chord_from_edge(cx.tri(g), l2, kg) else { break false };
            f = g;
            l = l2;
            ch = c2;
        };
        if !closed {
            continue;
        }
        let key = |pc: &Piece| (pc.face, pc.line.a(), (pc.line.c + 0.0).to_bits());
        if pieces.iter().any(|pc| seen.contains(&key(pc))) {
            continue;
        }
        for pc in &pieces {
            seen.insert(key(pc));
        }
        out.push(Curve { kind: CurveKind::Loop, chain: NONE, src: li, pieces, closed: true });
    }
    Ok(out)
}

/// A candidate ray at a vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ray {
    pub face: u32,
    pub d: u8,
    pub occ: bool,
}

/// Rays (vertex, owner face, direction) already used by an exact loop.
pub(crate) fn loop_occupancy(cx: &Ctx, curves: &[Curve]) -> BTreeSet<(u32, u32, u8)> {
    let mut occ = BTreeSet::new();
    for c in curves.iter().filter(|c| c.kind == CurveKind::Loop) {
        for pc in &c.pieces {
            let tri = cx.tri(pc.face);
            if pc.is_zero(tri) {
                continue;
            }
            if let Pt::V(k) = pc.start {
                occ.insert((cx.vid(pc.face, k), pc.face, pc.line.d));
            }
            if let Pt::V(k) = pc.end {
                let v = cx.vid(pc.face, k);
                let back = (pc.line.d + 2) % 4;
                if pc.on_edge != 255 {
                    let e = 3 * pc.face + pc.on_edge as u32;
                    let tw = cx.t.twin[e as usize];
                    if tw != NONE {
                        occ.insert((v, tw / 3, cx.map_dir(e, back)));
                    }
                } else {
                    occ.insert((v, pc.face, back));
                }
            }
        }
    }
    occ
}

/// The axis rays at vertex v in CCW order with the CCW ownership rule: a face owns the
/// directions in `[e_a, e_b)` of its corner (plus `e_b` at the end of an open fan).
pub(crate) fn vertex_rays(cx: &Ctx, v: u32, occ: &BTreeSet<(u32, u32, u8)>) -> Result<(Vec<Ray>, bool), AqError> {
    let fan: Vec<u32> = cx.t.fan_out(v).collect();
    let open = cx.t.is_bnd_vert.get(v as usize).copied().unwrap_or(false);
    let mut rays = Vec::new();
    for (i, &h) in fan.iter().enumerate() {
        let f = h / 3;
        let k = (h % 3) as usize;
        if cx.orient[f as usize] <= 0 {
            return Err(AqError::invariant(format!("emitter vertex {v} touches folded or degenerate face {f}")));
        }
        let t = cx.tri(f);
        let ea = sub2(t[(k + 1) % 3], t[k]);
        let eb = sub2(t[(k + 2) % 3], t[k]);
        let last_open = open && i + 1 == fan.len();
        let mut loc: Vec<(u8, u8)> = Vec::new();
        for d in 0..4u8 {
            let u = dir_vec(d);
            let along_a = cross2(ea, u) == 0.0 && dot2(ea, u) > 0.0;
            let along_b = cross2(eb, u) == 0.0 && dot2(eb, u) > 0.0;
            let inside = orient2d([0.0, 0.0], ea, u) > 0.0 && orient2d([0.0, 0.0], u, eb) > 0.0;
            if along_a {
                loc.push((d, 0));
            } else if inside {
                loc.push((d, 1));
            } else if along_b && last_open {
                loc.push((d, 2));
            }
        }
        loc.sort_by(|x, y| {
            x.1.cmp(&y.1).then_with(|| {
                if x.1 == 1 && y.1 == 1 {
                    let o = orient2d([0.0, 0.0], dir_vec(x.0), dir_vec(y.0));
                    if o > 0.0 { Ordering::Less } else { Ordering::Greater }
                } else {
                    Ordering::Equal
                }
            })
        });
        for (d, w) in loc {
            let o = (w == 0 && cx.constraint_he[h as usize]) || w == 2 || occ.contains(&(v, f, d));
            rays.push(Ray { face: f, d, occ: o });
        }
    }
    Ok((rays, open))
}

/// Finds the ray of v given in the chart of `face` by walking the fan.
fn locate_ray(cx: &Ctx, v: u32, face: u32, dir: u8, rays: &[Ray], open: bool) -> Option<Ray> {
    let fan: Vec<u32> = cx.t.fan_out(v).collect();
    let i0 = fan.iter().position(|&h| h / 3 == face)?;
    let find = |f: u32, d: u8| rays.iter().find(|r| r.face == f && r.d == d).copied();
    let mut idx = i0;
    let mut d = dir % 4;
    for _ in 0..fan.len() {
        let h = fan[idx];
        if let Some(r) = find(h / 3, d) {
            return Some(r);
        }
        let e = Topology::prev(h);
        if cx.t.twin[e as usize] == NONE {
            break;
        }
        d = cx.map_dir(e, d);
        idx += 1;
        if idx >= fan.len() {
            if open {
                break;
            }
            idx = 0;
        }
    }
    if open {
        let mut idx = i0;
        let mut d = dir % 4;
        while idx > 0 {
            let h = fan[idx];
            d = cx.map_dir(h, d);
            idx -= 1;
            if let Some(r) = find(fan[idx] / 3, d) {
                return Some(r);
            }
        }
    }
    None
}

/// Vertices with an incident constraint halfedge.
pub(crate) fn chain_verts(cx: &Ctx) -> Vec<bool> {
    let mut is_cv = vec![false; cx.m.pos.len()];
    for (h, &c) in cx.constraint_he.iter().enumerate() {
        if c {
            let tri = cx.m.tris[h / 3];
            is_cv[tri[h % 3] as usize] = true;
            is_cv[tri[(h % 3 + 1) % 3] as usize] = true;
        }
    }
    is_cv
}

/// Chain corners and open chain ends of the constraint chains: (corners, ends).
pub(crate) fn chain_corners_ends(cx: &Ctx) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut corners = BTreeSet::new();
    let mut ends = BTreeSet::new();
    let m = cx.m;
    for (v, &c) in cx.feats.vert_corner.iter().enumerate() {
        if c {
            corners.insert(v as u32);
        }
    }
    for (i, ch) in cx.feats.chains.iter().enumerate() {
        if cx.chain_curve[i] == NONE || ch.hes.is_empty() {
            continue;
        }
        for &ci in &ch.corners {
            if let Some(&h) = ch.hes.get(ci as usize) {
                corners.insert(cx.t.tail(m, h));
            }
        }
        if !ch.closed
            && let (Some(&h0), Some(&h1)) = (ch.hes.first(), ch.hes.last()) {
                ends.insert(cx.t.tail(m, h0));
                ends.insert(cx.t.head(m, h1));
            }
    }
    (corners, ends)
}

/// Emitters (DESIGN 6.10): interior cones emit every free axis ray (their count must be
/// `4 - index4`); chain corners, chain cones and open chain ends emit the minimal rays
/// (offsets 2, 4, ... quarter turns inside each sector between occupied rays); seeds emit
/// their ray or every free ray.
pub fn emitters(cx: &Ctx, curves: &[Curve], seeds: &[Seed]) -> Result<Vec<Emitter>, AqError> {
    let occ = loop_occupancy(cx, curves);
    let is_cv = chain_verts(cx);
    let (corners, ends) = chain_corners_ends(cx);
    let mut cand: BTreeSet<u32> = cx.cone_index.keys().copied().collect();
    cand.extend(corners.iter().copied());
    cand.extend(ends.iter().copied());
    let mut out = Vec::new();
    let mut used: BTreeSet<(u32, u32, u8)> = BTreeSet::new();
    let mut push = |out: &mut Vec<Emitter>, v: u32, r: Ray, kind: CurveKind| {
        if !r.occ && used.insert((v, r.face, r.d)) {
            out.push(Emitter { vert: v, face: r.face, dir: r.d, kind });
        }
    };
    for &v in &cand {
        let (rays, open) = vertex_rays(cx, v, &occ)?;
        let m = rays.len();
        if m == 0 {
            continue;
        }
        if is_cv[v as usize] {
            let occ_idx: Vec<usize> = (0..m).filter(|&i| rays[i].occ).collect();
            if occ_idx.is_empty() {
                continue;
            }
            let npairs = if open { occ_idx.len().saturating_sub(1) } else { occ_idx.len() };
            for pi in 0..npairs {
                let i = occ_idx[pi];
                let j = occ_idx[(pi + 1) % occ_idx.len()];
                let n = if open {
                    j - i
                } else {
                    let d = (j + m - i) % m;
                    if d == 0 { m } else { d }
                };
                let mut off = 2;
                while off < n {
                    push(&mut out, v, rays[(i + off) % m], CurveKind::Separatrix);
                    off += 2;
                }
            }
        } else {
            let idx4 = cx.cone_index.get(&v).copied().unwrap_or(0) as i32;
            if m as i32 != 4 - idx4 {
                return Err(AqError::invariant(format!(
                    "cone {v} of index4 {idx4} has {m} axis rays instead of {}",
                    4 - idx4
                )));
            }
            for r in &rays {
                push(&mut out, v, *r, CurveKind::Separatrix);
            }
        }
    }
    for s in seeds {
        if (s.vert as usize) >= cx.m.pos.len() {
            continue;
        }
        let (rays, open) = vertex_rays(cx, s.vert, &occ)?;
        if s.dir == DIR_ALL {
            for r in &rays {
                push(&mut out, s.vert, *r, CurveKind::SeedTrace);
            }
        } else if let Some(r) = locate_ray(cx, s.vert, s.face, s.dir, &rays, open) {
            push(&mut out, s.vert, r, CurveKind::SeedTrace);
        }
    }
    Ok(out)
}

/// Public entry: the constraint curves (chains, then exact loops) of a proxy and param.
pub fn constraint_curves(
    px: &ProxyView,
    prm: &SeamlessParam,
    loops: &[LoopCand],
    p: &LayoutParams,
    ctl: &Ctl,
) -> Result<Vec<Curve>, AqError> {
    let st = |e: AqError| e.at(Stage::Layout);
    let cx = Ctx::new(px, prm).map_err(st)?;
    let mut c = chain_curves(&cx);
    c.extend(trace_loops(&cx, loops, p, ctl).map_err(st)?);
    Ok(c)
}

/// Public entry: the emitters for constraint curves `curves` and seeds.
pub fn emitters_for(px: &ProxyView, prm: &SeamlessParam, curves: &[Curve], seeds: &[Seed]) -> Result<Vec<Emitter>, AqError> {
    let st = |e: AqError| e.at(Stage::Layout);
    let cx = Ctx::new(px, prm).map_err(st)?;
    emitters(&cx, curves, seeds).map_err(st)
}

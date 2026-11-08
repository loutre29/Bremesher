//! Embedding of the curves into W' (DESIGN 6.10 "Embedding").
//!
//! * Edge points (curve crossings of proxy edges) are kept per canonical proxy edge, sorted
//!   exactly along the edge and merged when equal, so both faces of an edge share them.
//! * Inside a face, the pieces and the split face edges form a planar graph whose directed
//!   edges carry exact direction vectors (axis vectors or face edge vectors). Cells are
//!   extracted by the angular next-edge rule and triangulated by ear clipping on strict
//!   corners (corners are known combinatorially: two sides on different carriers), with no
//!   Steiner points.
//! * W' uv: proxy corners are exact; an edge point keeps its iso coordinate exactly and gets
//!   the other one snapped to 2^-24 in the canonical chart, then mapped exactly to the twin
//!   chart, so the inherited transitions hold bitwise; face points are exact.
//! * 3D positions come from linear interpolation on the proxy face.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use aq_core::exact::orient2d;
use aq_core::tol::snap_dyadic;
use aq_core::{AqError, NONE, SeamlessParam, Stage, Topology, Transition, TriMesh, V2, V3, math};
use rayon::prelude::*;

use crate::ctx::Ctx;
use crate::geom::{Line, Pt, cmp_a, cmp_angle};
use crate::{Curve, CurveKind, ProxyView};

/// W' with curve labels (input of [`crate::extract`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedded {
    pub wmesh: TriMesh,
    pub wtopo: Topology,
    pub wuv: Vec<[V2; 3]>,
    pub wtrans: Vec<Transition>,
    pub parent_face: Vec<u32>,
    /// Per W' halfedge: id of the curve it lies on, or NONE.
    pub he_label: Vec<u32>,
    /// W' vertices below this id are the proxy vertices (same ids).
    pub n_proxy_verts: u32,
    pub curve_kind: Vec<CurveKind>,
    pub curve_chain: Vec<u32>,
    pub curve_closed: Vec<bool>,
    /// Per curve: W' vertices of the first and last point of a trace (NONE for constraints).
    pub curve_ends: Vec<[u32; 2]>,
}

#[derive(Clone, Copy, Debug)]
struct EKey {
    q: u8,
    c: f64,
}

fn sgn(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Exact order of two edge points of edge A->B by their position along the edge.
fn cmp_ekey(a: V2, b: V2, k1: &EKey, k2: &EKey) -> Ordering {
    if k1.q == k2.q {
        let q = k1.q as usize;
        return (sgn(k1.c - k2.c) * sgn(b[q] - a[q])).cmp(&0);
    }
    let (kv, ku) = if k1.q == 1 { (k1, k2) } else { (k2, k1) };
    let s = sgn(orient2d(a, b, [ku.c, kv.c])) * sgn(b[1] - a[1]) * sgn(b[0] - a[0]);
    let s = if k1.q == 1 { s } else { -s };
    s.cmp(&0)
}

struct EdgePts {
    keys: Vec<EKey>,
    ex: Vec<[Option<f64>; 2]>,
    /// uv in the canonical chart.
    uv: Vec<V2>,
    vid0: u32,
}

struct Shared<'a, 'b> {
    cx: &'a Ctx<'b>,
    edges: BTreeMap<u32, EdgePts>,
    labels: BTreeMap<u32, Vec<u32>>,
}

fn emb_err(msg: String) -> AqError {
    AqError::numerical(msg).at(Stage::Layout)
}

impl Shared<'_, '_> {
    fn ends(&self, h0: u32) -> (V2, V2) {
        let t = self.cx.tri(h0 / 3);
        let k = (h0 % 3) as usize;
        (t[k], t[(k + 1) % 3])
    }

    /// Canonical key of an edge point given in face f on local edge k.
    fn key_of(&self, f: u32, k: u8, l: Line, p: Pt) -> Option<(u32, EKey)> {
        let (q, c) = match p {
            Pt::E(_) => (l.q() as u8, l.c),
            Pt::X(x) => (l.a() as u8, x),
            Pt::V(_) => return None,
        };
        let h = 3 * f + k as u32;
        let h0 = self.cx.canon(h);
        if h0 == h {
            return Some((h, EKey { q, c: c + 0.0 }));
        }
        let lq = Line { d: 1 - q, c };
        let m = lq.map(&self.cx.prm.trans[h as usize]);
        Some((h0, EKey { q: m.q() as u8, c: m.c + 0.0 }))
    }

    fn lookup(&self, h0: u32, key: &EKey) -> Option<usize> {
        let ep = self.edges.get(&h0)?;
        let (a, b) = self.ends(h0);
        ep.keys.binary_search_by(|probe| cmp_ekey(a, b, probe, key)).ok()
    }

    fn base_label(&self, h: u32) -> u32 {
        let cx = self.cx;
        if !cx.constraint_he[h as usize] {
            return NONE;
        }
        let ch = cx.feats.he_chain[h as usize];
        if ch != NONE && (ch as usize) < cx.chain_curve.len() { cx.chain_curve[ch as usize] } else { NONE }
    }

    fn npts(&self, h0: u32) -> usize {
        self.edges.get(&h0).map_or(0, |e| e.keys.len())
    }

    /// Label of canonical sub-segment s of edge h0.
    fn seg_label(&self, h0: u32, s: usize) -> u32 {
        match self.labels.get(&h0) {
            Some(v) => v.get(s).copied().unwrap_or(NONE),
            None => self.base_label(h0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VRef {
    G(u32),
    L(u32),
}

#[derive(Clone, Copy, Debug)]
struct Node {
    r: VRef,
    uv: V2,
    ex: [Option<f64>; 2],
    mask: u8,
}

struct GEdge {
    a: usize,
    b: usize,
    dir: V2,
    carrier: u32,
    label: u32,
    bnd: bool,
}

struct FaceOut {
    interior: Vec<V2>,
    tris: Vec<([VRef; 3], [V2; 3], [u32; 3], [u8; 3])>,
}

fn share_line(a: &Node, b: &Node) -> bool {
    (a.ex[0].is_some() && a.ex[0] == b.ex[0]) || (a.ex[1].is_some() && a.ex[1] == b.ex[1]) || (a.mask & b.mask) != 0
}

/// Ear clipping of a convex cell given as (node, carrier of the edge to the next node).
fn triangulate(poly: &[(usize, u32)], nodes: &[Node]) -> Option<Vec<[usize; 3]>> {
    let mut p: Vec<(usize, u32)> = poly.to_vec();
    let mut out = Vec::new();
    let mut nc = 1u32 << 30;
    while p.len() > 3 {
        let n = p.len();
        let mut pick = None;
        'outer: for pass in 0..2 {
            for i in 0..n {
                let pr = (i + n - 1) % n;
                let nx = (i + 1) % n;
                if p[pr].1 == p[i].1 || p[(pr + n - 1) % n].1 == p[nx].1 {
                    continue;
                }
                if pass == 0 && share_line(&nodes[p[pr].0], &nodes[p[nx].0]) {
                    continue;
                }
                pick = Some(i);
                break 'outer;
            }
        }
        let i = pick?;
        let pr = (i + n - 1) % n;
        let nx = (i + 1) % n;
        out.push([p[pr].0, p[i].0, p[nx].0]);
        p[pr].1 = nc;
        nc += 1;
        p.remove(i);
    }
    if p.len() != 3 || p[0].1 == p[1].1 || p[1].1 == p[2].1 || p[2].1 == p[0].1 {
        return None;
    }
    out.push([p[0].0, p[1].0, p[2].0]);
    Some(out)
}

fn exact_axis_map(tr: &Transition, ex: [Option<f64>; 2], uv: V2) -> [Option<f64>; 2] {
    let mut o = [None, None];
    for (i, e) in ex.iter().enumerate() {
        if e.is_some() {
            let j = if tr.r % 2 == 1 { 1 - i } else { i };
            o[j] = Some(uv[j]);
        }
    }
    o
}

/// Splits face f by its pieces and the edge points of its edges.
fn process_face(sh: &Shared, f: u32, pieces: &[(u32, crate::Piece)]) -> Result<FaceOut, AqError> {
    let cx = sh.cx;
    let tri = *cx.tri(f);
    let orient = cx.orient[f as usize];
    let mut nodes: Vec<Node> = Vec::new();
    for k in 0..3usize {
        nodes.push(Node {
            r: VRef::G(cx.vid(f, k as u8)),
            uv: tri[k],
            ex: [Some(tri[k][0]), Some(tri[k][1])],
            mask: (1 << k) | (1 << ((k + 2) % 3)),
        });
    }
    // Boundary nodes per local edge, ordered from corner k to corner k+1.
    let mut bnodes: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut bsegs: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for k in 0..3u32 {
        let h = 3 * f + k;
        let h0 = cx.canon(h);
        let fwd = h0 == h;
        let n = sh.npts(h0);
        let mut ids = Vec::with_capacity(n);
        if let Some(ep) = sh.edges.get(&h0) {
            for i in 0..n {
                let ci = if fwd { i } else { n - 1 - i };
                let (uv, ex) = if fwd {
                    (ep.uv[ci], ep.ex[ci])
                } else {
                    let tr = &cx.prm.trans[h0 as usize];
                    let u2 = tr.apply(ep.uv[ci]);
                    (u2, exact_axis_map(tr, ep.ex[ci], u2))
                };
                ids.push(nodes.len());
                nodes.push(Node { r: VRef::G(ep.vid0 + ci as u32), uv, ex, mask: 1 << k });
            }
        }
        let mut segs = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let s = if fwd { i } else { n - i };
            segs.push(sh.seg_label(h0, s));
        }
        bnodes[k as usize] = ids;
        bsegs[k as usize] = segs;
    }
    let mut edges: Vec<GEdge> = Vec::new();
    let mut emap: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let refl = |d: V2| if orient < 0 { [d[0], -d[1]] } else { d };
    let mut add_edge = |edges: &mut Vec<GEdge>, a: usize, b: usize, dir: V2, carrier: u32, label: u32, bnd: bool| {
        if a == b {
            return;
        }
        let key = (a.min(b), a.max(b));
        if let Some(&i) = emap.get(&key) {
            if edges[i].label == NONE {
                edges[i].label = label;
            }
            return;
        }
        emap.insert(key, edges.len());
        edges.push(GEdge { a, b, dir: refl(dir), carrier, label, bnd });
    };
    for k in 0..3usize {
        let mut seq = vec![k];
        seq.extend(bnodes[k].iter().copied());
        seq.push((k + 1) % 3);
        let dir = math::sub2(tri[(k + 1) % 3], tri[k]);
        for (i, w) in seq.windows(2).enumerate() {
            add_edge(&mut edges, w[0], w[1], dir, k as u32, bsegs[k][i], true);
        }
    }
    let n_bnd_edges = edges.len();
    // Interior points and piece sub-segments.
    let mut interior: Vec<V2> = Vec::new();
    let mut imap: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    let mut lines: Vec<(usize, f64)> = Vec::new();
    for (pi, &(cid, p)) in pieces.iter().enumerate() {
        let l = p.line;
        let mut pts: Vec<Pt> = vec![p.start, p.end];
        for (qi, &(_, q)) in pieces.iter().enumerate() {
            if qi == pi {
                continue;
            }
            if q.line.a() != l.a() {
                let mut pm = Pt::X(q.line.c);
                let inside_p = cmp_a(&tri, l, pm, lo(&tri, &p)) != Ordering::Less && cmp_a(&tri, l, pm, hi(&tri, &p)) != Ordering::Greater;
                let pv = Pt::X(l.c);
                let inside_q = cmp_a(&tri, q.line, pv, lo(&tri, &q)) != Ordering::Less
                    && cmp_a(&tri, q.line, pv, hi(&tri, &q)) != Ordering::Greater;
                if inside_p && inside_q {
                    if cmp_a(&tri, l, pm, p.start) == Ordering::Equal {
                        pm = p.start;
                    } else if cmp_a(&tri, l, pm, p.end) == Ordering::Equal {
                        pm = p.end;
                    }
                    pts.push(pm);
                }
            } else if q.line.c == l.c {
                for e in [q.start, q.end] {
                    if cmp_a(&tri, l, e, lo(&tri, &p)) != Ordering::Less && cmp_a(&tri, l, e, hi(&tri, &p)) != Ordering::Greater {
                        pts.push(e);
                    }
                }
            }
        }
        pts.sort_by(|x, y| cmp_a(&tri, l, *x, *y));
        pts.dedup_by(|x, y| cmp_a(&tri, l, *x, *y) == Ordering::Equal);
        let mut ids = Vec::with_capacity(pts.len());
        for pt in &pts {
            let id = match *pt {
                Pt::V(k) => k as usize,
                Pt::E(k) => {
                    let (h0, key) = sh.key_of(f, k, l, *pt).ok_or_else(|| emb_err("edge key".into()))?;
                    let idx = sh.lookup(h0, &key).ok_or_else(|| emb_err(format!("edge point missing on edge {h0}")))?;
                    let fwd = h0 == 3 * f + k as u32;
                    let n = sh.npts(h0);
                    let li = if fwd { idx } else { n - 1 - idx };
                    bnodes[k as usize][li]
                }
                Pt::X(x) => {
                    let uv = l.point(x);
                    let key = ((uv[0] + 0.0).to_bits(), (uv[1] + 0.0).to_bits());
                    match imap.get(&key) {
                        Some(&i) => i,
                        None => {
                            if orient == 0 {
                                return Err(emb_err(format!("curve point inside degenerate face {f}")));
                            }
                            let i = nodes.len();
                            imap.insert(key, i);
                            nodes.push(Node { r: VRef::L(interior.len() as u32), uv, ex: [Some(uv[0]), Some(uv[1])], mask: 0 });
                            interior.push(uv);
                            i
                        }
                    }
                }
            };
            ids.push(id);
        }
        let li = match lines.iter().position(|&(a, c)| a == l.a() && c == l.c) {
            Some(i) => i,
            None => {
                lines.push((l.a(), l.c));
                lines.len() - 1
            }
        };
        let mut dir = [0.0, 0.0];
        dir[l.a()] = 1.0;
        for w in ids.windows(2) {
            add_edge(&mut edges, w[0], w[1], dir, 3 + li as u32, cid, false);
        }
    }
    let mut polys: Vec<Vec<(usize, u32)>> = Vec::new();
    if edges.len() == n_bnd_edges {
        let mut poly = Vec::new();
        for e in &edges {
            poly.push((e.a, e.carrier));
        }
        polys.push(poly);
    } else {
        if orient == 0 {
            return Err(emb_err(format!("curves cross degenerate face {f}")));
        }
        let nn = nodes.len();
        let nhe = 2 * edges.len();
        let from = |he: usize| if he.is_multiple_of(2) { edges[he / 2].a } else { edges[he / 2].b };
        let to = |he: usize| if he.is_multiple_of(2) { edges[he / 2].b } else { edges[he / 2].a };
        let hdir = |he: usize| {
            let d = edges[he / 2].dir;
            if he.is_multiple_of(2) { d } else { [-d[0], -d[1]] }
        };
        let mut outs: Vec<Vec<usize>> = vec![Vec::new(); nn];
        for he in 0..nhe {
            outs[from(he)].push(he);
        }
        let mut pos = vec![0usize; nhe];
        for o in outs.iter_mut() {
            o.sort_by(|&x, &y| cmp_angle(hdir(x), hdir(y)));
            for (i, &he) in o.iter().enumerate() {
                pos[he] = i;
            }
        }
        let next = |he: usize| {
            let b = to(he);
            let tw = he ^ 1;
            let o = &outs[b];
            o[(pos[tw] + o.len() - 1) % o.len()]
        };
        let mut seen = vec![false; nhe];
        let mut n_cycles = 0usize;
        for start in 0..nhe {
            if seen[start] {
                continue;
            }
            n_cycles += 1;
            let mut cyc = Vec::new();
            let mut he = start;
            let mut outer = false;
            loop {
                if seen[he] {
                    break;
                }
                seen[he] = true;
                if edges[he / 2].bnd && he % 2 == 1 {
                    outer = true;
                }
                cyc.push((from(he), edges[he / 2].carrier));
                he = next(he);
                if cyc.len() > nhe {
                    return Err(emb_err(format!("cell walk does not close in face {f}")));
                }
            }
            if he != start {
                return Err(emb_err(format!("inconsistent cell walk in face {f}")));
            }
            if !outer {
                polys.push(cyc);
            }
        }
        let used: usize = outs.iter().filter(|o| !o.is_empty()).count();
        if used as i64 - edges.len() as i64 + n_cycles as i64 != 2 {
            return Err(emb_err(format!("curve graph of face {f} is not connected")));
        }
    }
    let mut out = FaceOut { interior, tris: Vec::new() };
    for poly in &polys {
        let mut vs: Vec<usize> = poly.iter().map(|x| x.0).collect();
        vs.sort_unstable();
        if vs.windows(2).any(|w| w[0] == w[1]) {
            return Err(emb_err(format!("dangling curve in face {f}")));
        }
        let ts = triangulate(poly, &nodes).ok_or_else(|| emb_err(format!("cell triangulation failed in face {f}")))?;
        for t in ts {
            let mut labs = [NONE; 3];
            let mut pe = [255u8; 3];
            for j in 0..3 {
                let (a, b) = (t[j], t[(j + 1) % 3]);
                if let Some(&ei) = emap.get(&(a.min(b), a.max(b))) {
                    labs[j] = edges[ei].label;
                }
                let common = nodes[a].mask & nodes[b].mask;
                if common != 0 {
                    pe[j] = common.trailing_zeros() as u8;
                }
            }
            out.tris.push(([nodes[t[0]].r, nodes[t[1]].r, nodes[t[2]].r], [nodes[t[0]].uv, nodes[t[1]].uv, nodes[t[2]].uv], labs, pe));
        }
    }
    Ok(out)
}

fn lo(tri: &[V2; 3], p: &crate::Piece) -> Pt {
    if cmp_a(tri, p.line, p.start, p.end) == Ordering::Greater { p.end } else { p.start }
}
fn hi(tri: &[V2; 3], p: &crate::Piece) -> Pt {
    if cmp_a(tri, p.line, p.start, p.end) == Ordering::Greater { p.start } else { p.end }
}

/// Builds W' from all curves (chains first, as produced by the layout driver).
pub(crate) fn embed_ctx(cx: &Ctx, curves: &[Curve]) -> Result<Embedded, AqError> {
    let m = cx.m;
    let nv = m.pos.len() as u32;
    let nf = cx.n_faces();
    let mut sh = Shared { cx, edges: BTreeMap::new(), labels: BTreeMap::new() };
    // 1. Edge point keys.
    let mut raw: BTreeMap<u32, Vec<EKey>> = BTreeMap::new();
    let mut face_pieces: BTreeMap<u32, Vec<(u32, crate::Piece)>> = BTreeMap::new();
    for (ci, c) in curves.iter().enumerate() {
        for p in &c.pieces {
            let tri = cx.tri(p.face);
            if p.is_zero(tri) {
                continue;
            }
            for pt in [p.start, p.end] {
                let k = match pt {
                    Pt::E(k) => Some(k),
                    Pt::X(_) if p.on_edge != 255 => Some(p.on_edge),
                    _ => None,
                };
                if let Some(k) = k
                    && let Some((h0, key)) = sh.key_of(p.face, k, p.line, pt) {
                        raw.entry(h0).or_default().push(key);
                    }
            }
            if p.on_edge == 255 {
                face_pieces.entry(p.face).or_default().push((ci as u32, *p));
            }
        }
    }
    let mut next_vid = nv;
    let mut epos: Vec<V3> = Vec::new();
    for (h0, mut keys) in raw {
        let (a, b) = sh.ends(h0);
        keys.sort_by(|x, y| cmp_ekey(a, b, x, y));
        let mut merged: Vec<EKey> = Vec::new();
        let mut ex: Vec<[Option<f64>; 2]> = Vec::new();
        for k in keys {
            if let Some(last) = merged.last()
                && cmp_ekey(a, b, last, &k) == Ordering::Equal {
                    if let Some(e) = ex.last_mut() {
                        e[k.q as usize] = Some(k.c);
                    }
                    continue;
                }
            let mut e = [None, None];
            e[k.q as usize] = Some(k.c);
            merged.push(k);
            ex.push(e);
        }
        let tail = m.pos[m.tris[(h0 / 3) as usize][(h0 % 3) as usize] as usize];
        let head = m.pos[m.tris[(h0 / 3) as usize][((h0 % 3 + 1) % 3) as usize] as usize];
        let mut uvs = Vec::with_capacity(merged.len());
        for (k, e) in merged.iter().zip(ex.iter_mut()) {
            let q = k.q as usize;
            let lam = ((k.c - a[q]) / (b[q] - a[q])).clamp(0.0, 1.0);
            let mut uv = [0.0; 2];
            for i in 0..2 {
                uv[i] = match e[i] {
                    Some(x) => x,
                    None if a[i] == b[i] => {
                        e[i] = Some(a[i]);
                        a[i]
                    }
                    None => snap_dyadic(a[i] + lam * (b[i] - a[i])),
                };
            }
            uvs.push(uv);
            epos.push(math::lerp(tail, head, lam));
        }
        let n = merged.len() as u32;
        sh.edges.insert(h0, EdgePts { keys: merged, ex, uv: uvs, vid0: next_vid });
        next_vid += n;
    }
    // 2. On-edge labels.
    for (ci, c) in curves.iter().enumerate() {
        for p in &c.pieces {
            if p.on_edge == 255 || p.is_zero(cx.tri(p.face)) {
                continue;
            }
            let h = 3 * p.face + p.on_edge as u32;
            let h0 = cx.canon(h);
            let n = sh.npts(h0);
            let tail0 = m.tris[(h0 / 3) as usize][(h0 % 3) as usize];
            let mut posn = [0usize; 2];
            for (i, pt) in [p.start, p.end].into_iter().enumerate() {
                posn[i] = match pt {
                    Pt::V(k) => {
                        if cx.vid(p.face, k) == tail0 {
                            0
                        } else {
                            n + 1
                        }
                    }
                    _ => {
                        let (hh, key) = sh.key_of(p.face, p.on_edge, p.line, pt).ok_or_else(|| emb_err("edge key".into()))?;
                        sh.lookup(hh, &key).ok_or_else(|| emb_err(format!("on-edge point missing on edge {hh}")))? + 1
                    }
                };
            }
            let (lo, hi) = (posn[0].min(posn[1]), posn[0].max(posn[1]));
            let base = sh.base_label(h0);
            let lab = sh.labels.entry(h0).or_insert_with(|| vec![base; n + 1]);
            for s in lo..hi {
                if let Some(x) = lab.get_mut(s)
                    && *x == NONE {
                        *x = ci as u32;
                    }
            }
        }
    }
    // 3. Faces.
    let mut work: Vec<u32> = Vec::new();
    for f in 0..nf as u32 {
        let touched = face_pieces.contains_key(&f) || (0..3).any(|k| sh.edges.contains_key(&cx.canon(3 * f + k)));
        if touched {
            work.push(f);
        }
    }
    let empty: Vec<(u32, crate::Piece)> = Vec::new();
    let outs: Vec<Result<FaceOut, AqError>> = work
        .par_iter()
        .map(|&f| process_face(&sh, f, face_pieces.get(&f).unwrap_or(&empty)))
        .collect();
    let mut face_out: BTreeMap<u32, FaceOut> = BTreeMap::new();
    for (f, o) in work.iter().zip(outs) {
        face_out.insert(*f, o?);
    }
    // 4. Assemble.
    let mut pos: Vec<V3> = Vec::new();
    pos.try_reserve(next_vid as usize + 16).map_err(|_| AqError::budget("W' vertices").at(Stage::Layout))?;
    pos.extend_from_slice(&m.pos);
    pos.extend(epos);
    let mut ibase: BTreeMap<u32, u32> = BTreeMap::new();
    for (&f, o) in &face_out {
        ibase.insert(f, pos.len() as u32);
        let tri = cx.tri(f);
        let [p0, p1, p2] = m.tri_pos(f);
        for uv in &o.interior {
            let d = math::cross2(math::sub2(tri[1], tri[0]), math::sub2(tri[2], tri[0]));
            let w1 = math::cross2(math::sub2(*uv, tri[0]), math::sub2(tri[2], tri[0])) / d;
            let w2 = math::cross2(math::sub2(tri[1], tri[0]), math::sub2(*uv, tri[0])) / d;
            pos.push(math::bary_point(p0, p1, p2, [1.0 - w1 - w2, w1, w2]));
        }
    }
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut wuv: Vec<[V2; 3]> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    let mut he_label: Vec<u32> = Vec::new();
    let mut wtrans: Vec<Transition> = Vec::new();
    for f in 0..nf as u32 {
        let res = |r: VRef| match r {
            VRef::G(v) => v,
            VRef::L(i) => ibase.get(&f).copied().unwrap_or(0) + i,
        };
        match face_out.get(&f) {
            Some(o) => {
                for (rs, uvs, labs, pe) in &o.tris {
                    tris.push([res(rs[0]), res(rs[1]), res(rs[2])]);
                    wuv.push(*uvs);
                    parent.push(f);
                    for j in 0..3 {
                        he_label.push(labs[j]);
                        wtrans.push(if pe[j] != 255 { cx.prm.trans[(3 * f + pe[j] as u32) as usize] } else { Transition::IDENTITY });
                    }
                }
            }
            None => {
                tris.push(m.tris[f as usize]);
                wuv.push(*cx.tri(f));
                parent.push(f);
                for k in 0..3u32 {
                    let h0 = cx.canon(3 * f + k);
                    he_label.push(sh.seg_label(h0, 0));
                    wtrans.push(cx.prm.trans[(3 * f + k) as usize]);
                }
            }
        }
    }
    // 5. Trace end vertices.
    let mut curve_ends = vec![[NONE; 2]; curves.len()];
    for (ci, c) in curves.iter().enumerate() {
        if c.kind.is_constraint() {
            continue;
        }
        let (Some(p0), Some(p1)) = (c.pieces.first(), c.pieces.last()) else { continue };
        let mut e = [NONE; 2];
        for (i, (p, pt)) in [(p0, p0.start), (p1, p1.end)].into_iter().enumerate() {
            e[i] = match pt {
                Pt::V(k) => cx.vid(p.face, k),
                Pt::E(k) => resolve_edge(&sh, p.face, k, p.line, pt).unwrap_or(NONE),
                Pt::X(x) => {
                    if p.on_edge != 255 {
                        resolve_edge(&sh, p.face, p.on_edge, p.line, pt).unwrap_or(NONE)
                    } else {
                        let uv = p.line.point(x);
                        match (face_out.get(&p.face), ibase.get(&p.face)) {
                            (Some(o), Some(&b)) => o.interior.iter().position(|q| *q == uv).map_or(NONE, |i| b + i as u32),
                            _ => NONE,
                        }
                    }
                }
            };
        }
        curve_ends[ci] = e;
    }
    let wmesh = TriMesh { pos, tris };
    let wtopo = Topology::build(&wmesh).map_err(|e| e.at(Stage::Layout))?;
    Ok(Embedded {
        wmesh,
        wtopo,
        wuv,
        wtrans,
        parent_face: parent,
        he_label,
        n_proxy_verts: nv,
        curve_kind: curves.iter().map(|c| c.kind).collect(),
        curve_chain: curves.iter().map(|c| c.chain).collect(),
        curve_closed: curves.iter().map(|c| c.closed).collect(),
        curve_ends,
    })
}

fn resolve_edge(sh: &Shared, f: u32, k: u8, l: Line, pt: Pt) -> Option<u32> {
    let (h0, key) = sh.key_of(f, k, l, pt)?;
    let i = sh.lookup(h0, &key)?;
    Some(sh.edges.get(&h0)?.vid0 + i as u32)
}

/// DESIGN 6.10 `embed`: W' of the curves (chains first, then loops and traces, as built by
/// [`crate::curves::constraint_curves`] followed by [`crate::trace_all`]).
pub fn embed(px: &ProxyView, prm: &SeamlessParam, curves: &[Curve]) -> Result<Embedded, AqError> {
    let cx = Ctx::new(px, prm).map_err(|e| e.at(Stage::Layout))?;
    embed_ctx(&cx, curves)
}

//! Local isotropic remesher of the SF path (Botsch-Kobbelt 2004 at a constant length).
//!
//! `aq_prep::build_proxy` clamps every proxy to at least `MIN_PROXY_TRIS` triangles, which is
//! about 4000 SF quads. Below that, and for the debris batch (no proxy each, DESIGN 7.1), SF
//! uses this remesher on the component itself (CHANGES.md, WP-04). It splits edges longer than
//! 4/3 l, collapses edges shorter than 4/5 l, flips toward valence 6 (4 on the boundary),
//! smooths tangentially and projects, for 6 iterations.
//!
//! Chains (feature and boundary halfedges): a chain vertex is never moved; a regular chain
//! vertex (chain degree 2, one chain id, not a corner) may only be removed by collapsing one
//! of its chain edges; corners and chain junctions are frozen; an edge joining two chain
//! vertices that is not a chain edge is never collapsed; chain edges are never flipped. A
//! split of a chain edge keeps both halves on the chain.

use std::cmp::Ordering;

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::math;
use aq_core::{AqError, Ctl, ErrKind, FeatureKind, NONE, Projector, Stage, Topology, TriMesh, V3};
use aq_prep::ComponentMesh;

use crate::{SfTri, ferr, project_onto};

const SPLIT_HI: f64 = 4.0 / 3.0;
const COLLAPSE_LO: f64 = 4.0 / 5.0;
const SMOOTH_LAMBDA: f64 = 0.5;
const MIN_ANGLE_DEG: f64 = 10.0;
/// cos 60 degrees: largest normal turn of a face in a collapse.
const COS_COLLAPSE_TURN: f64 = 0.5;
/// cos 30 degrees: largest dihedral created by a flip (unless the old one was larger).
const COS_FLIP_DIHEDRAL: f64 = 0.866_025_403_784_438_6;
/// A new edge may also be up to this factor longer than the longest edge it replaces.
const GROWTH: f64 = 1.02;
/// Upper bound on the walk around one vertex (corruption guard).
const MAX_FAN: usize = 4096;

#[inline]
fn next(h: u32) -> u32 {
    3 * (h / 3) + (h % 3 + 1) % 3
}

#[inline]
fn prev(h: u32) -> u32 {
    3 * (h / 3) + (h % 3 + 2) % 3
}

fn be(e: AqError) -> AqError {
    e.at(Stage::Fallback)
}

fn tri_normal(a: V3, b: V3, c: V3) -> V3 {
    math::cross(math::sub(b, a), math::sub(c, a))
}

fn min_angle(a: V3, b: V3, c: V3) -> f64 {
    let ang = |p: V3, q: V3, r: V3| math::angle_between(math::sub(q, p), math::sub(r, p));
    ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
}

struct DMesh {
    pos: Vec<V3>,
    tri: Vec<[u32; 3]>,
    twin: Vec<u32>,
    hchain: Vec<u32>,
    alive: Vec<bool>,
    /// An outgoing halfedge of a live face, NONE for a removed vertex.
    vhe: Vec<u32>,
    /// Never removed and never moved.
    frozen: Vec<bool>,
    /// Has at least one chain (or boundary) edge: never moved.
    onchain: Vec<bool>,
    hint: Vec<u32>,
    stamp: Vec<u32>,
    round: u32,
    n_alive: usize,
}

/// Scratch buffers of the vertex walks.
#[derive(Default)]
struct Scr {
    fa: Vec<u32>,
    fb: Vec<u32>,
    na: Vec<u32>,
    nb: Vec<u32>,
    tmp: Vec<u32>,
}

impl DMesh {
    #[inline]
    fn tail(&self, h: u32) -> u32 {
        self.tri[(h / 3) as usize][(h % 3) as usize]
    }

    #[inline]
    fn head(&self, h: u32) -> u32 {
        self.tri[(h / 3) as usize][((h % 3 + 1) % 3) as usize]
    }

    #[inline]
    fn is_feat(&self, h: u32) -> bool {
        self.twin[h as usize] == NONE || self.hchain[h as usize] != NONE
    }

    fn he_ok(&self, h: u32, v: u32) -> bool {
        h != NONE && (h as usize) < self.twin.len() && self.alive[(h / 3) as usize] && self.tail(h) == v
    }

    /// Outgoing halfedges of v (live faces). Returns (ok, on_boundary).
    fn fan(&self, v: u32, out: &mut Vec<u32>) -> (bool, bool) {
        out.clear();
        let h0 = self.vhe[v as usize];
        if !self.he_ok(h0, v) {
            return (false, false);
        }
        out.push(h0);
        let mut h = h0;
        let mut open = false;
        loop {
            let t = self.twin[prev(h) as usize];
            if t == NONE {
                open = true;
                break;
            }
            if t == h0 {
                break;
            }
            out.push(t);
            h = t;
            if out.len() > MAX_FAN {
                return (false, false);
            }
        }
        if open {
            let mut h = h0;
            loop {
                let t = self.twin[h as usize];
                if t == NONE {
                    break;
                }
                let n = next(t);
                out.push(n);
                h = n;
                if out.len() > MAX_FAN {
                    return (false, false);
                }
            }
        }
        (true, open)
    }

    /// Sorted neighbor vertices of v from its fan.
    fn neighbors(&self, fan: &[u32], out: &mut Vec<u32>) {
        out.clear();
        for &h in fan {
            out.push(self.head(h));
            out.push(self.tail(prev(h)));
        }
        out.sort_unstable();
        out.dedup();
    }

    fn vnormal(&self, fan: &[u32]) -> V3 {
        let mut n = [0.0; 3];
        for &h in fan {
            let [a, b, c] = self.tri[(h / 3) as usize];
            n = math::add(n, tri_normal(self.pos[a as usize], self.pos[b as usize], self.pos[c as usize]));
        }
        n
    }

    fn reseat(&mut self, v: u32, cands: &[u32]) {
        if v == NONE || self.he_ok(self.vhe[v as usize], v) {
            return;
        }
        for &c in cands {
            if c != NONE && self.he_ok(c, v) {
                self.vhe[v as usize] = c;
                return;
            }
        }
        // The vertex lost its last face (cannot happen under the link and valence rules).
        self.vhe[v as usize] = NONE;
    }

    /// Checks a collapse of the edge of h removing `a` into `b` at position p.
    fn can_collapse(&self, h: u32, a: u32, b: u32, p: V3, l: f64, s: &mut Scr) -> bool {
        if self.frozen[a as usize] {
            return false;
        }
        let feat = self.is_feat(h);
        if self.onchain[a as usize] && !feat {
            return false;
        }
        if !feat && self.onchain[a as usize] && self.onchain[b as usize] {
            return false;
        }
        let tw = self.twin[h as usize];
        // Two chain edges merging into one.
        if self.is_feat(prev(h)) && self.is_feat(next(h)) {
            return false;
        }
        if tw != NONE && self.is_feat(prev(tw)) && self.is_feat(next(tw)) {
            return false;
        }
        let c = self.head(next(h));
        let d = if tw == NONE { NONE } else { self.head(next(tw)) };
        if c == d {
            return false;
        }
        let (oka, bnd_a) = self.fan(a, &mut s.fa);
        let (okb, bnd_b) = self.fan(b, &mut s.fb);
        if !oka || !okb {
            return false;
        }
        // An interior edge between two boundary vertices would pinch the surface.
        if tw != NONE && bnd_a && bnd_b {
            return false;
        }
        let fa = std::mem::take(&mut s.fa);
        let fb = std::mem::take(&mut s.fb);
        self.neighbors(&fa, &mut s.na);
        self.neighbors(&fb, &mut s.nb);
        let ok = self.collapse_checks(h, a, b, c, d, p, l, &fa, &fb, bnd_a || bnd_b, s);
        s.fa = fa;
        s.fb = fb;
        ok
    }

    fn collapse_checks(
        &self,
        h: u32,
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        p: V3,
        l: f64,
        fa: &[u32],
        fb: &[u32],
        bnd_ab: bool,
        s: &mut Scr,
    ) -> bool {
        // Link condition: common neighbors are exactly the apexes.
        s.tmp.clear();
        let (mut i, mut j) = (0, 0);
        while i < s.na.len() && j < s.nb.len() {
            match s.na[i].cmp(&s.nb[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    s.tmp.push(s.na[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        let mut apex = vec![c];
        if d != NONE {
            apex.push(d);
        }
        apex.sort_unstable();
        if s.tmp != apex {
            return false;
        }
        // Valences after the collapse.
        let merged = s.na.len() + s.nb.len() - 2 - apex.len();
        if merged < if bnd_ab { 2 } else { 3 } {
            return false;
        }
        let mut fan = Vec::new();
        for &x in &apex {
            let (ok, bnd) = self.fan(x, &mut fan);
            if !ok {
                return false;
            }
            let mut nb = Vec::new();
            self.neighbors(&fan, &mut nb);
            if nb.len() < if bnd { 3 } else { 4 } {
                return false;
            }
        }
        // Geometry.
        let f0 = h / 3;
        let tw = self.twin[h as usize];
        let f1 = if tw == NONE { NONE } else { tw / 3 };
        let pa = self.pos[a as usize];
        let pb = self.pos[b as usize];
        let moved_b = p != pb;
        let mut old_min = f64::INFINITY;
        let mut old_max_len: f64 = 0.0;
        for &x in s.na.iter() {
            old_max_len = old_max_len.max(math::dist(pa, self.pos[x as usize]));
        }
        for &x in s.nb.iter() {
            old_max_len = old_max_len.max(math::dist(pb, self.pos[x as usize]));
        }
        let max_len = (SPLIT_HI * l).max(GROWTH * old_max_len);
        for &x in s.na.iter() {
            if x != b && math::dist(p, self.pos[x as usize]) > max_len {
                return false;
            }
        }
        if moved_b {
            for &x in s.nb.iter() {
                if x != a && math::dist(p, self.pos[x as usize]) > max_len {
                    return false;
                }
            }
        }
        let mut new_min = f64::INFINITY;
        let faces = fa.iter().chain(fb.iter());
        for &hh in faces {
            let f = hh / 3;
            if f == f0 || f == f1 {
                let [x, y, z] = self.tri[f as usize];
                old_min = old_min.min(min_angle(self.pos[x as usize], self.pos[y as usize], self.pos[z as usize]));
                continue;
            }
            let t = self.tri[f as usize];
            let po = [self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]];
            let mut pn = po;
            let mut changed = false;
            for k in 0..3 {
                if t[k] == a || (moved_b && t[k] == b) {
                    pn[k] = p;
                    changed = true;
                }
            }
            old_min = old_min.min(min_angle(po[0], po[1], po[2]));
            if !changed {
                continue;
            }
            let no = tri_normal(po[0], po[1], po[2]);
            let nn = tri_normal(pn[0], pn[1], pn[2]);
            let (lo, ln) = (math::norm(no), math::norm(nn));
            if !(ln > 1e-14 * l * l) {
                return false;
            }
            if lo > 0.0 && !(math::dot(no, nn) >= COS_COLLAPSE_TURN * lo * ln) {
                return false;
            }
            new_min = new_min.min(min_angle(pn[0], pn[1], pn[2]));
        }
        let lim = MIN_ANGLE_DEG.to_radians();
        if new_min < lim && new_min < old_min {
            return false;
        }
        true
    }

    /// Performs the collapse checked by `can_collapse`.
    fn do_collapse(&mut self, h: u32, a: u32, b: u32, p: V3, hint: u32, s: &mut Scr) {
        let tw = self.twin[h as usize];
        let f0 = h / 3;
        let f1 = if tw == NONE { NONE } else { tw / 3 };
        let c = self.head(next(h));
        let d = if tw == NONE { NONE } else { self.head(next(tw)) };
        let mut fa = std::mem::take(&mut s.fa);
        self.fan(a, &mut fa);
        for &hh in &fa {
            let f = hh / 3;
            if f == f0 || f == f1 {
                continue;
            }
            for k in 0..3 {
                if self.tri[f as usize][k] == a {
                    self.tri[f as usize][k] = b;
                }
            }
        }
        s.fa = fa;
        let mut cands = [NONE; 8];
        let mut merge = |dm: &mut DMesh, hp: u32, hn: u32, slot: usize| {
            let x = dm.twin[hp as usize];
            let y = dm.twin[hn as usize];
            let cx = dm.hchain[hp as usize];
            let cy = dm.hchain[hn as usize];
            let ch = if cx != NONE { cx } else { cy };
            if x != NONE {
                dm.twin[x as usize] = y;
                dm.hchain[x as usize] = ch;
            }
            if y != NONE {
                dm.twin[y as usize] = x;
                dm.hchain[y as usize] = ch;
            }
            cands[slot] = x;
            cands[slot + 1] = y;
            cands[slot + 2] = if x != NONE { next(x) } else { NONE };
            cands[slot + 3] = if y != NONE { next(y) } else { NONE };
        };
        merge(self, prev(h), next(h), 0);
        if tw != NONE {
            merge(self, prev(tw), next(tw), 4);
        }
        self.alive[f0 as usize] = false;
        self.n_alive -= 1;
        if f1 != NONE {
            self.alive[f1 as usize] = false;
            self.n_alive -= 1;
        }
        self.pos[b as usize] = p;
        self.hint[b as usize] = hint;
        self.vhe[a as usize] = NONE;
        for v in [b, c, d] {
            self.reseat(v, &cands);
        }
    }

    /// Splits the edge of h at p (new vertex). Returns the new vertex.
    fn do_split(&mut self, h: u32, p: V3, hint: u32) -> u32 {
        let tw = self.twin[h as usize];
        let f0 = h / 3;
        let a = self.tail(h);
        let b = self.head(h);
        let c = self.head(next(h));
        let ch = self.hchain[h as usize];
        let (tw_bc, ch_bc) = (self.twin[next(h) as usize], self.hchain[next(h) as usize]);
        let (tw_ca, ch_ca) = (self.twin[prev(h) as usize], self.hchain[prev(h) as usize]);
        let m = self.pos.len() as u32;
        self.pos.push(p);
        self.hint.push(hint);
        self.frozen.push(false);
        self.onchain.push(self.is_feat(h));
        self.stamp.push(self.round);
        self.vhe.push(NONE);
        let set = |dm: &mut DMesh, f: u32, t: [u32; 3]| {
            dm.tri[f as usize] = t;
        };
        let new_face = |dm: &mut DMesh| -> u32 {
            let f = dm.tri.len() as u32;
            dm.tri.push([0; 3]);
            dm.twin.extend_from_slice(&[NONE; 3]);
            dm.hchain.extend_from_slice(&[NONE; 3]);
            dm.alive.push(true);
            dm.n_alive += 1;
            f
        };
        let link = |dm: &mut DMesh, x: u32, y: u32| {
            if x != NONE {
                dm.twin[x as usize] = y;
            }
            if y != NONE {
                dm.twin[y as usize] = x;
            }
        };
        let g0 = new_face(self);
        // f0 = (a, m, c): a->m, m->c, c->a. g0 = (m, b, c): m->b, b->c, c->m.
        set(self, f0, [a, m, c]);
        set(self, g0, [m, b, c]);
        let (e_am, e_mc, e_ca) = (3 * f0, 3 * f0 + 1, 3 * f0 + 2);
        let (e_mb, e_bc, e_cm) = (3 * g0, 3 * g0 + 1, 3 * g0 + 2);
        for (e, c_) in [(e_am, ch), (e_mc, NONE), (e_ca, ch_ca), (e_mb, ch), (e_bc, ch_bc), (e_cm, NONE)] {
            self.hchain[e as usize] = c_;
        }
        link(self, e_mc, e_cm);
        link(self, e_ca, tw_ca);
        link(self, e_bc, tw_bc);
        if tw == NONE {
            self.twin[e_am as usize] = NONE;
            self.twin[e_mb as usize] = NONE;
        } else {
            let f1 = tw / 3;
            let d = self.head(next(tw));
            let (tw_ad, ch_ad) = (self.twin[next(tw) as usize], self.hchain[next(tw) as usize]);
            let (tw_db, ch_db) = (self.twin[prev(tw) as usize], self.hchain[prev(tw) as usize]);
            let g1 = new_face(self);
            // f1 = (m, a, d): m->a, a->d, d->m. g1 = (b, m, d): b->m, m->d, d->b.
            set(self, f1, [m, a, d]);
            set(self, g1, [b, m, d]);
            let (e_ma, e_ad, e_dm) = (3 * f1, 3 * f1 + 1, 3 * f1 + 2);
            let (e_bm, e_md, e_db) = (3 * g1, 3 * g1 + 1, 3 * g1 + 2);
            for (e, c_) in [(e_ma, ch), (e_ad, ch_ad), (e_dm, NONE), (e_bm, ch), (e_md, NONE), (e_db, ch_db)] {
                self.hchain[e as usize] = c_;
            }
            link(self, e_am, e_ma);
            link(self, e_mb, e_bm);
            link(self, e_dm, e_md);
            link(self, e_ad, tw_ad);
            link(self, e_db, tw_db);
            self.vhe[d as usize] = NONE;
            self.reseat(d, &[e_dm, e_db]);
        }
        // Any outgoing halfedge will do, also on the boundary: the fan walk goes both ways.
        self.vhe[m as usize] = e_mb;
        self.vhe[a as usize] = e_am;
        self.vhe[b as usize] = e_bc;
        self.vhe[c as usize] = e_ca;
        m
    }
}

/// The local remesher, kept between runs so that the count feedback of the SF path can
/// continue from the current mesh with a corrected length.
pub(crate) struct Remesher {
    dm: DMesh,
    s: Scr,
    chi: i64,
    chain_kind: Vec<FeatureKind>,
    area: f64,
}

impl Remesher {
    /// Starts from a component; projection hints are faces of `proj.mesh` (the component's own
    /// faces when the projector is over it, else the BVH closest face).
    pub(crate) fn from_component(src: &ComponentMesh, proj: &Projector) -> Result<Self, AqError> {
        let m = &src.mesh;
        let t = &src.topo;
        let same = std::ptr::eq(proj.mesh, m) || (proj.mesh.tris.len() == m.tris.len() && proj.mesh.pos == m.pos);
        let mut hint = try_vec(m.pos.len(), 0u32).map_err(be)?;
        for v in 0..m.pos.len() {
            let h = t.vert_he[v];
            hint[v] = if same && h != NONE { h / 3 } else { proj.bvh.closest(m.pos[v]).0.face };
        }
        let kinds = src.feats.chains.iter().map(|c| c.kind).collect();
        Self::build(&m.pos, &m.tris, &t.twin, &src.feats.he_chain, &src.feats.vert_corner, hint, kinds, src.chi)
    }

    /// Starts from a triangle mesh of the SF path (the aq_prep proxy).
    pub(crate) fn from_tri(t: &SfTri, chi: i64) -> Result<Self, AqError> {
        Self::build(&t.pos, &t.tris, &t.twin, &t.hchain, &t.vcorner, t.vhint.clone(), t.chain_kind.clone(), chi)
    }

    fn build(
        pos: &[V3],
        tris: &[[u32; 3]],
        twin: &[u32],
        hchain: &[u32],
        corner: &[bool],
        hint: Vec<u32>,
        chain_kind: Vec<FeatureKind>,
        chi: i64,
    ) -> Result<Self, AqError> {
        let nf = tris.len();
        let nv = pos.len();
        if twin.len() != 3 * nf || hchain.len() != 3 * nf || corner.len() != nv || hint.len() != nv {
            return Err(ferr(ErrKind::Internal, "SF remesh: inconsistent input arrays"));
        }
        let mut dm = DMesh {
            pos: try_with_capacity(nv).map_err(be)?,
            tri: try_with_capacity(nf).map_err(be)?,
            twin: try_with_capacity(3 * nf).map_err(be)?,
            hchain: try_with_capacity(3 * nf).map_err(be)?,
            alive: try_vec(nf, true).map_err(be)?,
            vhe: try_vec(nv, NONE).map_err(be)?,
            frozen: try_vec(nv, false).map_err(be)?,
            onchain: try_vec(nv, false).map_err(be)?,
            hint,
            stamp: try_vec(nv, 0u32).map_err(be)?,
            round: 0,
            n_alive: nf,
        };
        dm.pos.extend_from_slice(pos);
        dm.tri.extend_from_slice(tris);
        dm.twin.extend_from_slice(twin);
        dm.hchain.extend_from_slice(hchain);
        for (f, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let v = t[k] as usize;
                if v >= nv {
                    return Err(ferr(ErrKind::Internal, "SF remesh: index out of range"));
                }
                if dm.vhe[v] == NONE {
                    dm.vhe[v] = (3 * f + k) as u32;
                }
            }
        }
        // Chain classes: degree and ids of chain edges per vertex.
        let mut deg = try_vec(nv, 0u32).map_err(be)?;
        let mut cid = try_vec(nv, NONE).map_err(be)?;
        let mut mixed = try_vec(nv, false).map_err(be)?;
        for h in 0..3 * nf as u32 {
            let tw = dm.twin[h as usize];
            if !dm.is_feat(h) || (tw != NONE && tw < h) {
                continue;
            }
            let c = dm.hchain[h as usize];
            for v in [dm.tail(h), dm.head(h)] {
                let v = v as usize;
                deg[v] += 1;
                if cid[v] == NONE && deg[v] == 1 {
                    cid[v] = c;
                } else if cid[v] != c {
                    mixed[v] = true;
                }
            }
        }
        for v in 0..nv {
            dm.onchain[v] = deg[v] > 0;
            dm.frozen[v] = corner[v] || (deg[v] > 0 && (deg[v] != 2 || mixed[v])) || dm.vhe[v] == NONE;
        }
        let mut area = 0.0;
        for t in tris {
            area += 0.5 * math::norm(tri_normal(pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]));
        }
        Ok(Remesher { dm, s: Scr::default(), chi, chain_kind, area })
    }

    /// `iters` Botsch-Kobbelt iterations at the constant length l.
    pub(crate) fn run(&mut self, proj: &Projector, l: f64, iters: usize, ctl: &Ctl) -> Result<(), AqError> {
        if !(l.is_finite() && l > 0.0) {
            return Err(ferr(ErrKind::InvalidInput, format!("SF remesh: length {l}")));
        }
        let expected = self.area / (0.25 * 3f64.sqrt() * l * l);
        let cap = 8 * self.dm.n_alive.max(expected as usize) + 64;
        for _ in 0..iters {
            ctl.check()?;
            split_pass(&mut self.dm, proj, l, cap, &mut self.s, ctl)?;
            collapse_pass(&mut self.dm, proj, l, &mut self.s, ctl)?;
            flip_pass(&mut self.dm, &mut self.s, ctl)?;
            smooth_pass(&mut self.dm, proj, l, &mut self.s, ctl)?;
        }
        Ok(())
    }

    /// The current mesh, compacted and checked (manifold, one component, chi).
    pub(crate) fn result(&self) -> Result<SfTri, AqError> {
        let mut out = compact(&self.dm, self.chi)?;
        out.chain_kind = self.chain_kind.clone();
        Ok(out)
    }
}

/// Iterations of the first local remesh.
pub(crate) const ITERS: usize = 6;

fn split_pass(dm: &mut DMesh, proj: &Projector, l: f64, cap: usize, s: &mut Scr, ctl: &Ctl) -> Result<(), AqError> {
    for _ in 0..16 {
        dm.round += 1;
        let mut cand: Vec<(f64, u32)> = Vec::new();
        for h in 0..dm.twin.len() as u32 {
            if !dm.alive[(h / 3) as usize] {
                continue;
            }
            let tw = dm.twin[h as usize];
            if tw != NONE && tw < h {
                continue;
            }
            let len = math::dist(dm.pos[dm.tail(h) as usize], dm.pos[dm.head(h) as usize]);
            if len > SPLIT_HI * l {
                cand.push((len, h));
            }
        }
        if cand.is_empty() {
            break;
        }
        cand.sort_unstable_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
        let mut n = 0usize;
        for (i, &(_, h)) in cand.iter().enumerate() {
            if i % 1024 == 0 {
                ctl.check()?;
            }
            if dm.n_alive >= cap {
                break;
            }
            let a = dm.tail(h);
            let b = dm.head(h);
            let r = dm.round;
            if dm.stamp[a as usize] == r || dm.stamp[b as usize] == r {
                continue;
            }
            let tw = dm.twin[h as usize];
            let [x, y, z] = dm.tri[(h / 3) as usize];
            let mut nr = tri_normal(dm.pos[x as usize], dm.pos[y as usize], dm.pos[z as usize]);
            if tw != NONE {
                let [x, y, z] = dm.tri[(tw / 3) as usize];
                nr = math::add(nr, tri_normal(dm.pos[x as usize], dm.pos[y as usize], dm.pos[z as usize]));
            }
            let mid = math::scale(math::add(dm.pos[a as usize], dm.pos[b as usize]), 0.5);
            let (p, hint) = match project_onto(proj, mid, dm.hint[a as usize], nr) {
                Some((sp, q)) => (q, sp.face),
                None => (mid, dm.hint[a as usize]),
            };
            let c = dm.head(next(h));
            let d = if tw == NONE { NONE } else { dm.head(next(tw)) };
            dm.do_split(h, p, hint);
            for v in [a, b, c, d] {
                if v != NONE {
                    dm.stamp[v as usize] = r;
                }
            }
            n += 1;
        }
        let _ = s;
        if n == 0 {
            break;
        }
    }
    Ok(())
}

fn collapse_pass(dm: &mut DMesh, proj: &Projector, l: f64, s: &mut Scr, ctl: &Ctl) -> Result<(), AqError> {
    for _ in 0..200 {
        dm.round += 1;
        let mut cand: Vec<(f64, u32)> = Vec::new();
        for h in 0..dm.twin.len() as u32 {
            if !dm.alive[(h / 3) as usize] {
                continue;
            }
            let tw = dm.twin[h as usize];
            if tw != NONE && tw < h {
                continue;
            }
            let len = math::dist(dm.pos[dm.tail(h) as usize], dm.pos[dm.head(h) as usize]);
            if len < COLLAPSE_LO * l {
                cand.push((len, h));
            }
        }
        if cand.is_empty() {
            break;
        }
        cand.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
        let mut n = 0usize;
        for (i, &(_, h)) in cand.iter().enumerate() {
            if i % 1024 == 0 {
                ctl.check()?;
            }
            if !dm.alive[(h / 3) as usize] {
                continue;
            }
            let a = dm.tail(h);
            let b = dm.head(h);
            let r = dm.round;
            if dm.stamp[a as usize] == r || dm.stamp[b as usize] == r {
                continue;
            }
            // Options in order: (removed, kept, position, hint).
            let mut opts: Vec<(u32, u32, V3, u32)> = Vec::with_capacity(3);
            let (fa, fb) = (dm.onchain[a as usize], dm.onchain[b as usize]);
            if !fa && !fb {
                let mid = math::scale(math::add(dm.pos[a as usize], dm.pos[b as usize]), 0.5);
                dm.fan(a, &mut s.fa);
                let nr = dm.vnormal(&s.fa);
                if let Some((sp, q)) = proj.project(mid, dm.hint[a as usize], nr) {
                    opts.push((a, b, q, sp.face));
                }
            }
            opts.push((a, b, dm.pos[b as usize], dm.hint[b as usize]));
            opts.push((b, a, dm.pos[a as usize], dm.hint[a as usize]));
            for (x, y, p, hint) in opts {
                if dm.can_collapse(h, x, y, p, l, s) {
                    dm.do_collapse(h, x, y, p, hint, s);
                    // Stamp the 1-ring of the kept vertex.
                    dm.stamp[x as usize] = r;
                    let mut fan = std::mem::take(&mut s.fb);
                    dm.fan(y, &mut fan);
                    dm.stamp[y as usize] = r;
                    for &hh in &fan {
                        let u = dm.head(hh);
                        let w = dm.tail(prev(hh));
                        dm.stamp[u as usize] = r;
                        dm.stamp[w as usize] = r;
                    }
                    s.fb = fan;
                    n += 1;
                    break;
                }
            }
        }
        if n == 0 {
            break;
        }
    }
    Ok(())
}

fn flip_pass(dm: &mut DMesh, s: &mut Scr, ctl: &Ctl) -> Result<(), AqError> {
    let nv = dm.pos.len();
    let mut val = vec![0i32; nv];
    let mut bnd = vec![false; nv];
    let mut fan = Vec::new();
    let mut nb = Vec::new();
    for v in 0..nv as u32 {
        if dm.vhe[v as usize] == NONE {
            continue;
        }
        let (ok, b) = dm.fan(v, &mut fan);
        if ok {
            dm.neighbors(&fan, &mut nb);
            val[v as usize] = nb.len() as i32;
            bnd[v as usize] = b;
        }
    }
    let target = |v: u32, bnd: &[bool]| if bnd[v as usize] { 4 } else { 6 };
    for h in 0..dm.twin.len() as u32 {
        if h % 4096 == 0 {
            ctl.check()?;
        }
        if !dm.alive[(h / 3) as usize] {
            continue;
        }
        let tw = dm.twin[h as usize];
        if tw == NONE || tw < h || dm.hchain[h as usize] != NONE {
            continue;
        }
        let a = dm.tail(h);
        let b = dm.head(h);
        let c = dm.head(next(h));
        let d = dm.head(next(tw));
        if c == d {
            continue;
        }
        let dev = |x: i32, t: i32| (x - t) * (x - t);
        let (ta, tb, tc, td) = (target(a, &bnd), target(b, &bnd), target(c, &bnd), target(d, &bnd));
        let (va, vb, vc, vd) = (val[a as usize], val[b as usize], val[c as usize], val[d as usize]);
        let before = dev(va, ta) + dev(vb, tb) + dev(vc, tc) + dev(vd, td);
        let after = dev(va - 1, ta) + dev(vb - 1, tb) + dev(vc + 1, tc) + dev(vd + 1, td);
        if after >= before || va - 1 < if bnd[a as usize] { 2 } else { 3 } || vb - 1 < if bnd[b as usize] { 2 } else { 3 } {
            continue;
        }
        // c - d must not be an edge already.
        let (ok, _) = dm.fan(c, &mut fan);
        if !ok {
            continue;
        }
        dm.neighbors(&fan, &mut nb);
        if nb.binary_search(&d).is_ok() {
            continue;
        }
        let (pa, pb, pc, pd) = (dm.pos[a as usize], dm.pos[b as usize], dm.pos[c as usize], dm.pos[d as usize]);
        let n0 = tri_normal(pa, pb, pc);
        let n1 = tri_normal(pb, pa, pd);
        let m0 = tri_normal(pc, pa, pd);
        let m1 = tri_normal(pd, pb, pc);
        let navg = math::add(n0, n1);
        let (l0, l1, k0, k1, la) = (math::norm(n0), math::norm(n1), math::norm(m0), math::norm(m1), math::norm(navg));
        if !(k0 > 0.0 && k1 > 0.0 && la > 0.0) {
            continue;
        }
        if !(math::dot(m0, navg) > 0.5 * k0 * la && math::dot(m1, navg) > 0.5 * k1 * la) {
            continue;
        }
        let old_cos = if l0 > 0.0 && l1 > 0.0 { math::dot(n0, n1) / (l0 * l1) } else { -1.0 };
        let new_cos = math::dot(m0, m1) / (k0 * k1);
        if new_cos < COS_FLIP_DIHEDRAL && new_cos < old_cos {
            continue;
        }
        if min_angle(pc, pa, pd).min(min_angle(pd, pb, pc)) < min_angle(pa, pb, pc).min(min_angle(pb, pa, pd)) * 0.5 {
            continue;
        }
        flip(dm, h);
        val[a as usize] -= 1;
        val[b as usize] -= 1;
        val[c as usize] += 1;
        val[d as usize] += 1;
    }
    let _ = s;
    Ok(())
}

/// Flips the interior edge of h: (a, b, c) + (b, a, d) -> (c, a, d) + (d, b, c).
fn flip(dm: &mut DMesh, h: u32) {
    let tw = dm.twin[h as usize];
    let (f0, f1) = (h / 3, tw / 3);
    let a = dm.tail(h);
    let b = dm.head(h);
    let c = dm.head(next(h));
    let d = dm.head(next(tw));
    let (t_bc, c_bc) = (dm.twin[next(h) as usize], dm.hchain[next(h) as usize]);
    let (t_ca, c_ca) = (dm.twin[prev(h) as usize], dm.hchain[prev(h) as usize]);
    let (t_ad, c_ad) = (dm.twin[next(tw) as usize], dm.hchain[next(tw) as usize]);
    let (t_db, c_db) = (dm.twin[prev(tw) as usize], dm.hchain[prev(tw) as usize]);
    dm.tri[f0 as usize] = [c, a, d];
    dm.tri[f1 as usize] = [d, b, c];
    let (e_ca, e_ad, e_dc) = (3 * f0, 3 * f0 + 1, 3 * f0 + 2);
    let (e_db, e_bc, e_cd) = (3 * f1, 3 * f1 + 1, 3 * f1 + 2);
    let mut link = |x: u32, y: u32, ch: u32| {
        dm.twin[x as usize] = y;
        dm.hchain[x as usize] = ch;
        if y != NONE {
            dm.twin[y as usize] = x;
        }
    };
    link(e_ca, t_ca, c_ca);
    link(e_ad, t_ad, c_ad);
    link(e_db, t_db, c_db);
    link(e_bc, t_bc, c_bc);
    link(e_dc, e_cd, NONE);
    dm.hchain[e_cd as usize] = NONE;
    dm.vhe[a as usize] = e_ad;
    dm.vhe[b as usize] = e_bc;
    dm.vhe[c as usize] = e_ca;
    dm.vhe[d as usize] = e_db;
}

fn smooth_pass(dm: &mut DMesh, proj: &Projector, l: f64, s: &mut Scr, ctl: &Ctl) -> Result<(), AqError> {
    let nv = dm.pos.len();
    for v in 0..nv as u32 {
        if v % 1024 == 0 {
            ctl.check()?;
        }
        if dm.onchain[v as usize] || dm.frozen[v as usize] || dm.vhe[v as usize] == NONE {
            continue;
        }
        let (ok, open) = dm.fan(v, &mut s.fa);
        if !ok || open {
            continue;
        }
        let mut acc = [0.0; 3];
        let mut w = 0.0;
        let mut n = [0.0; 3];
        for &h in &s.fa {
            let [x, y, z] = dm.tri[(h / 3) as usize];
            let (px, py, pz) = (dm.pos[x as usize], dm.pos[y as usize], dm.pos[z as usize]);
            let fnrm = tri_normal(px, py, pz);
            let a = 0.5 * math::norm(fnrm);
            let cen = math::scale(math::add(math::add(px, py), pz), 1.0 / 3.0);
            acc = math::add(acc, math::scale(cen, a));
            w += a;
            n = math::add(n, fnrm);
        }
        let Some(nu) = math::normalize(n) else { continue };
        if !(w > 0.0) {
            continue;
        }
        let p = dm.pos[v as usize];
        let mut dlt = math::sub(math::scale(acc, 1.0 / w), p);
        dlt = math::sub(dlt, math::scale(nu, math::dot(dlt, nu)));
        let cand = math::add(p, math::scale(dlt, SMOOTH_LAMBDA));
        let Some((sp, q)) = proj.project(cand, dm.hint[v as usize], nu) else { continue };
        // Guard: no incident face may turn by more than 60 degrees or degenerate.
        let mut good = true;
        for &h in &s.fa {
            let t = dm.tri[(h / 3) as usize];
            let po = [dm.pos[t[0] as usize], dm.pos[t[1] as usize], dm.pos[t[2] as usize]];
            let mut pn = po;
            for k in 0..3 {
                if t[k] == v {
                    pn[k] = q;
                }
            }
            let no = tri_normal(po[0], po[1], po[2]);
            let nn = tri_normal(pn[0], pn[1], pn[2]);
            let (lo, ln) = (math::norm(no), math::norm(nn));
            if !(ln > 1e-14 * l * l) || (lo > 0.0 && !(math::dot(no, nn) >= COS_COLLAPSE_TURN * lo * ln)) {
                good = false;
                break;
            }
        }
        if good {
            dm.pos[v as usize] = q;
            dm.hint[v as usize] = sp.face;
        }
    }
    Ok(())
}

/// Live faces and referenced vertices, renumbered in order; checks manifoldness and chi.
fn compact(dm: &DMesh, chi: i64) -> Result<SfTri, AqError> {
    let nf = dm.tri.len();
    let nv = dm.pos.len();
    let mut fmap = try_vec(nf, NONE).map_err(be)?;
    let mut vmap = try_vec(nv, NONE).map_err(be)?;
    let mut n_f = 0u32;
    for f in 0..nf {
        if dm.alive[f] {
            fmap[f] = n_f;
            n_f += 1;
            for &v in &dm.tri[f] {
                vmap[v as usize] = 0;
            }
        }
    }
    let mut n_v = 0u32;
    for v in 0..nv {
        if vmap[v] != NONE {
            vmap[v] = n_v;
            n_v += 1;
        }
    }
    let mut out = SfTri {
        pos: try_with_capacity(n_v as usize).map_err(be)?,
        tris: try_with_capacity(n_f as usize).map_err(be)?,
        twin: try_with_capacity(3 * n_f as usize).map_err(be)?,
        hchain: try_with_capacity(3 * n_f as usize).map_err(be)?,
        chain_kind: Vec::new(),
        vcorner: try_with_capacity(n_v as usize).map_err(be)?,
        vhint: try_with_capacity(n_v as usize).map_err(be)?,
    };
    for v in 0..nv {
        if vmap[v] != NONE {
            out.pos.push(dm.pos[v]);
            out.vcorner.push(dm.frozen[v] && dm.onchain[v]);
            out.vhint.push(dm.hint[v]);
        }
    }
    for f in 0..nf {
        if !dm.alive[f] {
            continue;
        }
        let t = dm.tri[f];
        out.tris.push([vmap[t[0] as usize], vmap[t[1] as usize], vmap[t[2] as usize]]);
        for k in 0..3 {
            let h = 3 * f + k;
            let tw = dm.twin[h];
            out.twin.push(if tw == NONE { NONE } else { 3 * fmap[(tw / 3) as usize] + tw % 3 });
            out.hchain.push(dm.hchain[h]);
        }
    }
    // Manifold check and twins agree with a fresh topology.
    let tm = TriMesh { pos: out.pos.clone(), tris: out.tris.clone() };
    let topo = Topology::build(&tm).map_err(|e| ferr(ErrKind::Topology, format!("SF remesh: {}", e.msg)))?;
    if topo.twin != out.twin {
        return Err(ferr(ErrKind::Internal, "SF remesh: twin table out of sync"));
    }
    if topo.n_comp != 1 || topo.euler_char(&tm, 0) != chi {
        return Err(ferr(
            ErrKind::Topology,
            format!("SF remesh changed the topology (components {}, chi {} != {chi})", topo.n_comp, topo.euler_char(&tm, 0)),
        ));
    }
    Ok(out)
}

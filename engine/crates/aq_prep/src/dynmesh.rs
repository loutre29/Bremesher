//! Editable triangle mesh with the implicit halfedges of C2 (`he = 3f + k`), used by the
//! proxy remesher and the feature isolation. Dead faces are `[NONE; 3]`; `compact` renumbers.
//!
//! Every topological edit goes through [`DynMesh::rewire`], which replaces a set of faces by
//! new faces and re-establishes twins and feature labels from the region boundary. The edit
//! is validated before anything is written, so a refused edit leaves the mesh unchanged.

use aq_core::math;
use aq_core::alloc::{try_reserve, try_vec, try_with_capacity};
use aq_core::{AqError, FeatureKind, NONE, Topology, TriMesh, V3};

/// Per-vertex attributes.
#[derive(Debug, Default)]
pub(crate) struct DynMesh {
    pub pos: Vec<V3>,
    pub tris: Vec<[u32; 3]>,
    pub twin: Vec<u32>,
    /// One outgoing halfedge per live vertex (any), NONE for a dead vertex.
    pub vhe: Vec<u32>,
    /// Per halfedge feature kind (both halves of an interior edge).
    pub kind: Vec<FeatureKind>,
    /// Per halfedge source chain id (both halves), NONE off chains.
    pub cid: Vec<u32>,
    pub vdead: Vec<bool>,
    /// Corner vertices: never moved, never removed.
    pub frozen: Vec<bool>,
    /// Source vertex id of vertices copied from the source (NONE for new vertices).
    pub vorig: Vec<u32>,
    /// (chain, arc length) of non-frozen chain vertices, (NONE, 0) otherwise.
    pub vchain: Vec<(u32, f64)>,
    /// Source tracking: face of the projector mesh and barycentric coordinates.
    pub vsrc_f: Vec<u32>,
    pub vsrc_b: Vec<V3>,
    /// Target edge length at the vertex.
    pub lv: Vec<f64>,
}

/// Limit for fan walks (guards against corrupted connectivity).
const FAN_LIMIT: usize = 1 << 16;

/// Next halfedge in its face (C2, [`Topology::next`]).
#[inline]
pub(crate) fn next(h: u32) -> u32 {
    Topology::next(h)
}
/// Previous halfedge in its face (C2, [`Topology::prev`]).
#[inline]
pub(crate) fn prev(h: u32) -> u32 {
    Topology::prev(h)
}

/// Fallible copy of a slice (DESIGN 3.4: buffers over 16 MB fail with `ErrKind::Budget`).
pub(crate) fn try_copy<T: Clone>(src: &[T]) -> Result<Vec<T>, AqError> {
    let mut v = try_with_capacity(src.len())?;
    v.extend_from_slice(src);
    Ok(v)
}

/// An edit refused by [`DynMesh::rewire`].
#[derive(Debug)]
pub(crate) struct EditRefused;

/// New vertex data.
pub(crate) struct NewVert {
    pub pos: V3,
    pub chain: (u32, f64),
    pub src_f: u32,
    pub src_b: V3,
    pub lv: f64,
}

impl DynMesh {
    pub fn from_mesh(m: &TriMesh, t: &Topology, kind: Vec<FeatureKind>, cid: Vec<u32>) -> Result<DynMesh, AqError> {
        let nv = m.pos.len();
        let mut vorig = try_with_capacity(nv)?;
        vorig.extend(0..nv as u32);
        let mut vdead = try_with_capacity(nv)?;
        vdead.extend(t.vert_he.iter().map(|&h| h == NONE));
        Ok(DynMesh {
            pos: try_copy(&m.pos)?,
            tris: try_copy(&m.tris)?,
            twin: try_copy(&t.twin)?,
            vhe: try_copy(&t.vert_he)?,
            kind,
            cid,
            vdead,
            frozen: try_vec(nv, false)?,
            vorig,
            vchain: try_vec(nv, (NONE, 0.0))?,
            vsrc_f: try_vec(nv, NONE)?,
            vsrc_b: try_vec(nv, [0.0; 3])?,
            lv: try_vec(nv, 1.0)?,
        })
    }

    /// Fallible deep copy (the per-iteration snapshot of the remesher).
    pub fn try_clone(&self) -> Result<DynMesh, AqError> {
        Ok(DynMesh {
            pos: try_copy(&self.pos)?,
            tris: try_copy(&self.tris)?,
            twin: try_copy(&self.twin)?,
            vhe: try_copy(&self.vhe)?,
            kind: try_copy(&self.kind)?,
            cid: try_copy(&self.cid)?,
            vdead: try_copy(&self.vdead)?,
            frozen: try_copy(&self.frozen)?,
            vorig: try_copy(&self.vorig)?,
            vchain: try_copy(&self.vchain)?,
            vsrc_f: try_copy(&self.vsrc_f)?,
            vsrc_b: try_copy(&self.vsrc_b)?,
            lv: try_copy(&self.lv)?,
        })
    }

    /// Reserves room for `nv` more vertices and `nf` more faces (fallible, DESIGN 3.4).
    pub fn try_reserve(&mut self, nv: usize, nf: usize) -> Result<(), AqError> {
        try_reserve(&mut self.pos, nv)?;
        try_reserve(&mut self.vhe, nv)?;
        try_reserve(&mut self.vdead, nv)?;
        try_reserve(&mut self.frozen, nv)?;
        try_reserve(&mut self.vorig, nv)?;
        try_reserve(&mut self.vchain, nv)?;
        try_reserve(&mut self.vsrc_f, nv)?;
        try_reserve(&mut self.vsrc_b, nv)?;
        try_reserve(&mut self.lv, nv)?;
        try_reserve(&mut self.tris, nf)?;
        try_reserve(&mut self.twin, 3 * nf)?;
        try_reserve(&mut self.kind, 3 * nf)?;
        try_reserve(&mut self.cid, 3 * nf)
    }

    /// Number of live faces.
    pub fn live_faces(&self) -> usize {
        self.tris.iter().filter(|t| t[0] != NONE).count()
    }

    #[inline]
    pub fn alive(&self, f: u32) -> bool {
        self.tris[f as usize][0] != NONE
    }
    #[inline]
    pub fn tail(&self, h: u32) -> u32 {
        self.tris[(h / 3) as usize][(h % 3) as usize]
    }
    #[inline]
    pub fn head(&self, h: u32) -> u32 {
        self.tris[(h / 3) as usize][((h % 3 + 1) % 3) as usize]
    }
    #[inline]
    pub fn is_chain(&self, h: u32) -> bool {
        self.kind[h as usize] != FeatureKind::None
    }

    /// Outgoing halfedges of v in CCW order, starting at the CW-most one. Returns true when v
    /// is a boundary vertex.
    pub fn fan(&self, v: u32, out: &mut Vec<u32>) -> bool {
        out.clear();
        let h0 = self.vhe[v as usize];
        if h0 == NONE {
            return false;
        }
        let mut h = h0;
        let mut bnd = false;
        let mut steps = 0;
        loop {
            let t = self.twin[h as usize];
            if t == NONE {
                bnd = true;
                break;
            }
            let p = next(t);
            if p == h0 {
                break;
            }
            h = p;
            steps += 1;
            if steps > FAN_LIMIT {
                break;
            }
        }
        let start = if bnd { h } else { h0 };
        let mut h = start;
        loop {
            out.push(h);
            let n = self.twin[prev(h) as usize];
            if n == NONE || n == start || out.len() > FAN_LIMIT {
                break;
            }
            h = n;
        }
        bnd
    }

    /// True when v is on the boundary.
    pub fn is_bnd_vert(&self, v: u32) -> bool {
        let h0 = self.vhe[v as usize];
        if h0 == NONE {
            return false;
        }
        let mut h = h0;
        for _ in 0..FAN_LIMIT {
            let t = self.twin[h as usize];
            if t == NONE {
                return true;
            }
            h = next(t);
            if h == h0 {
                return false;
            }
        }
        false
    }

    /// Neighbour vertices of v (sorted, deduplicated); `fan` receives the fan. Returns the
    /// boundary flag.
    pub fn neighbors(&self, v: u32, fan: &mut Vec<u32>, out: &mut Vec<u32>) -> bool {
        let bnd = self.fan(v, fan);
        out.clear();
        for &h in fan.iter() {
            out.push(self.head(h));
        }
        if bnd && let Some(&last) = fan.last() {
            out.push(self.tail(prev(last)));
        }
        out.sort_unstable();
        out.dedup();
        bnd
    }

    /// Number of chain edges at v.
    pub fn chain_degree(&self, v: u32, fan: &mut Vec<u32>) -> u32 {
        let bnd = self.fan(v, fan);
        let mut d = fan.iter().filter(|&&h| self.is_chain(h)).count() as u32;
        if bnd
            && let Some(&last) = fan.last()
            && self.is_chain(prev(last))
        {
            d += 1;
        }
        d
    }

    /// The halfedge from a to b, or NONE.
    pub fn find_he(&self, a: u32, b: u32) -> u32 {
        let h0 = self.vhe[a as usize];
        if h0 == NONE {
            return NONE;
        }
        // CCW from h0 until the end of the fan or back to h0.
        let mut h = h0;
        for _ in 0..FAN_LIMIT {
            if self.head(h) == b {
                return h;
            }
            let n = self.twin[prev(h) as usize];
            if n == NONE || n == h0 {
                break;
            }
            h = n;
        }
        // CW from h0 (open fans).
        let mut h = h0;
        for _ in 0..FAN_LIMIT {
            let t = self.twin[h as usize];
            if t == NONE {
                break;
            }
            h = next(t);
            if h == h0 {
                break;
            }
            if self.head(h) == b {
                return h;
            }
        }
        NONE
    }

    /// Unnormalized normal of face f.
    pub fn face_cross(&self, f: u32) -> V3 {
        let t = self.tris[f as usize];
        let (a, b, c) = (self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]);
        math::cross(math::sub(b, a), math::sub(c, a))
    }

    pub fn add_vertex(&mut self, nv: NewVert) -> u32 {
        let id = self.pos.len() as u32;
        self.pos.push(nv.pos);
        self.vhe.push(NONE);
        self.vdead.push(false);
        self.frozen.push(false);
        self.vorig.push(NONE);
        self.vchain.push(nv.chain);
        self.vsrc_f.push(nv.src_f);
        self.vsrc_b.push(nv.src_b);
        self.lv.push(nv.lv);
        id
    }

    /// Replaces the faces `old` by `new_tris`.
    ///
    /// * Halfedges of `old` whose twin is outside `old` (or NONE) form the region boundary;
    ///   each must be matched by a new halfedge with the same direction (it takes over the
    ///   outside twin and the labels), except the boundary halfedges listed in `consumed`.
    /// * Other new halfedges pair with their reversed new halfedge, or are boundary when listed
    ///   in `new_bnd`.
    /// * Labels of new interior edges: `overrides` (unordered vertex pair), else the old
    ///   interior edge with the same pair after mapping vertex `remap.0` to `remap.1`, else
    ///   none. Overrides also apply to boundary-matched and new boundary halfedges (and to the
    ///   outside twin).
    ///
    /// Returns the new face ids (old slots are reused first).
    pub fn rewire(
        &mut self,
        old: &[u32],
        new_tris: &[[u32; 3]],
        remap: (u32, u32),
        overrides: &[(u32, u32, FeatureKind, u32)],
        new_bnd: &[(u32, u32)],
        consumed: &[(u32, u32)],
    ) -> Result<Vec<u32>, EditRefused> {
        let map = |v: u32| if v == remap.0 { remap.1 } else { v };
        struct Bnd {
            u: u32,
            v: u32,
            ou: u32,
            ov: u32,
            ext: u32,
            kind: FeatureKind,
            cid: u32,
            used: bool,
        }
        let mut bnd: Vec<Bnd> = Vec::with_capacity(3 * old.len());
        let mut inter: Vec<(u32, u32, FeatureKind, u32)> = Vec::new();
        for &f in old {
            if !self.alive(f) {
                return Err(EditRefused);
            }
            for k in 0..3 {
                let h = 3 * f + k;
                let tw = self.twin[h as usize];
                let (u, v) = (self.tail(h), self.head(h));
                if tw != NONE && old.contains(&(tw / 3)) {
                    if h < tw {
                        let (x, y) = (map(u), map(v));
                        inter.push((x.min(y), x.max(y), self.kind[h as usize], self.cid[h as usize]));
                    }
                } else {
                    bnd.push(Bnd {
                        u: map(u),
                        v: map(v),
                        ou: u,
                        ov: v,
                        ext: tw,
                        kind: self.kind[h as usize],
                        cid: self.cid[h as usize],
                        used: false,
                    });
                }
            }
        }
        let nn = new_tris.len();
        // Local matching: Ok(b) boundary index, Err(local partner) or NONE (new boundary).
        enum M {
            Bnd(usize),
            Pair(usize),
            Open,
        }
        let mut mt: Vec<M> = Vec::with_capacity(3 * nn);
        let he_uv = |li: usize| -> (u32, u32) {
            let t = new_tris[li / 3];
            (t[li % 3], t[(li % 3 + 1) % 3])
        };
        for t in new_tris {
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                return Err(EditRefused);
            }
        }
        for li in 0..3 * nn {
            let (u, v) = he_uv(li);
            if let Some(bi) = bnd.iter().position(|b| b.u == u && b.v == v && !b.used) {
                bnd[bi].used = true;
                mt.push(M::Bnd(bi));
                continue;
            }
            let mut n_partner = 0usize;
            let mut partner = 0usize;
            let mut same = 0usize;
            for lj in 0..3 * nn {
                let e = he_uv(lj);
                if e == (v, u) {
                    n_partner += 1;
                    partner = lj;
                } else if e == (u, v) {
                    same += 1;
                }
            }
            if n_partner == 1 && same == 1 {
                mt.push(M::Pair(partner));
            } else if n_partner == 0 && same == 1 && new_bnd.contains(&(u, v)) {
                mt.push(M::Open);
            } else {
                return Err(EditRefused);
            }
        }
        for b in &bnd {
            if !b.used && !(b.ext == NONE && consumed.contains(&(b.ou, b.ov))) {
                return Err(EditRefused);
            }
        }
        for li in 0..3 * nn {
            if let M::Pair(lj) = mt[li]
                && !matches!(mt[lj], M::Pair(x) if x == li)
            {
                return Err(EditRefused);
            }
        }
        // Commit.
        let mut ids: Vec<u32> = Vec::with_capacity(nn);
        for i in 0..nn {
            if i < old.len() {
                ids.push(old[i]);
            } else {
                let f = self.tris.len() as u32;
                self.tris.push([NONE; 3]);
                self.twin.extend_from_slice(&[NONE; 3]);
                self.kind.extend_from_slice(&[FeatureKind::None; 3]);
                self.cid.extend_from_slice(&[NONE; 3]);
                ids.push(f);
            }
        }
        for &f in old.iter().skip(nn) {
            self.tris[f as usize] = [NONE; 3];
            for k in 0..3 {
                let h = (3 * f + k) as usize;
                self.twin[h] = NONE;
                self.kind[h] = FeatureKind::None;
                self.cid[h] = NONE;
            }
        }
        for i in 0..nn {
            self.tris[ids[i] as usize] = new_tris[i];
        }
        let gh = |li: usize| -> u32 { 3 * ids[li / 3] + (li % 3) as u32 };
        let lookup = |x: u32, y: u32| -> Option<(FeatureKind, u32)> {
            let (a, b) = (x.min(y), x.max(y));
            overrides.iter().find(|o| o.0.min(o.1) == a && o.0.max(o.1) == b).map(|o| (o.2, o.3))
        };
        for li in 0..3 * nn {
            let g = gh(li);
            let (u, v) = he_uv(li);
            match mt[li] {
                M::Bnd(bi) => {
                    let b = &bnd[bi];
                    self.twin[g as usize] = b.ext;
                    if b.ext != NONE {
                        self.twin[b.ext as usize] = g;
                    }
                    let (k, c) = lookup(u, v).unwrap_or((b.kind, b.cid));
                    self.kind[g as usize] = k;
                    self.cid[g as usize] = c;
                    if b.ext != NONE {
                        self.kind[b.ext as usize] = k;
                        self.cid[b.ext as usize] = c;
                    }
                }
                M::Pair(lj) => {
                    self.twin[g as usize] = gh(lj);
                    let (k, c) = lookup(u, v)
                        .or_else(|| {
                            let (a, b) = (u.min(v), u.max(v));
                            inter.iter().find(|e| e.0 == a && e.1 == b).map(|e| (e.2, e.3))
                        })
                        .unwrap_or((FeatureKind::None, NONE));
                    self.kind[g as usize] = k;
                    self.cid[g as usize] = c;
                }
                M::Open => {
                    self.twin[g as usize] = NONE;
                    let (k, c) = lookup(u, v).unwrap_or((FeatureKind::Boundary, NONE));
                    self.kind[g as usize] = k;
                    self.cid[g as usize] = c;
                }
            }
            self.vhe[u as usize] = g;
        }
        Ok(ids)
    }

    /// Removes dead faces and dead or unreferenced vertices, preserving order. Returns the
    /// vertex map old -> new (NONE for removed). Allocations are fallible (`ErrKind::Budget`);
    /// on an error the mesh is unchanged.
    pub fn compact(&mut self) -> Result<Vec<u32>, AqError> {
        let nf = self.tris.len();
        let mut fmap = try_vec(nf, NONE)?;
        let mut nfn = 0u32;
        for f in 0..nf {
            if self.tris[f][0] != NONE {
                fmap[f] = nfn;
                nfn += 1;
            }
        }
        let nv = self.pos.len();
        let mut used = try_vec(nv, false)?;
        for t in &self.tris {
            if t[0] != NONE {
                for &v in t {
                    used[v as usize] = true;
                }
            }
        }
        let mut vmap = try_vec(nv, NONE)?;
        let mut nvn = 0u32;
        for v in 0..nv {
            if used[v] && !self.vdead[v] {
                vmap[v] = nvn;
                nvn += 1;
            }
        }
        let hmap = |h: u32| -> u32 { if h == NONE { NONE } else { 3 * fmap[(h / 3) as usize] + h % 3 } };
        let mut tris = try_with_capacity(nfn as usize)?;
        let mut twin = try_with_capacity(3 * nfn as usize)?;
        let mut kind = try_with_capacity(3 * nfn as usize)?;
        let mut cid = try_with_capacity(3 * nfn as usize)?;
        for f in 0..nf {
            if fmap[f] == NONE {
                continue;
            }
            let t = self.tris[f];
            tris.push([vmap[t[0] as usize], vmap[t[1] as usize], vmap[t[2] as usize]]);
            for k in 0..3 {
                let h = 3 * f + k;
                twin.push(hmap(self.twin[h]));
                kind.push(self.kind[h]);
                cid.push(self.cid[h]);
            }
        }
        fn keep<T: Copy>(old: &[T], vmap: &[u32], n: usize) -> Result<Vec<T>, AqError> {
            let mut out = try_with_capacity(n)?;
            out.extend(old.iter().zip(vmap).filter(|(_, m)| **m != NONE).map(|(x, _)| *x));
            Ok(out)
        }
        let n = nvn as usize;
        let pos = keep(&self.pos, &vmap, n)?;
        let frozen = keep(&self.frozen, &vmap, n)?;
        let vorig = keep(&self.vorig, &vmap, n)?;
        let vchain = keep(&self.vchain, &vmap, n)?;
        let vsrc_f = keep(&self.vsrc_f, &vmap, n)?;
        let vsrc_b = keep(&self.vsrc_b, &vmap, n)?;
        let lv = keep(&self.lv, &vmap, n)?;
        let mut vhe = keep(&self.vhe, &vmap, n)?;
        for h in vhe.iter_mut() {
            *h = hmap(*h);
        }
        let vdead = try_vec(n, false)?;
        self.pos = pos;
        self.frozen = frozen;
        self.vorig = vorig;
        self.vchain = vchain;
        self.vsrc_f = vsrc_f;
        self.vsrc_b = vsrc_b;
        self.lv = lv;
        self.vhe = vhe;
        self.vdead = vdead;
        self.tris = tris;
        self.twin = twin;
        self.kind = kind;
        self.cid = cid;
        // Every vhe must point at an outgoing halfedge of its vertex (rewire keeps it so).
        for v in 0..n {
            let h = self.vhe[v];
            if h == NONE || self.tail(h) != v as u32 {
                self.vhe[v] = NONE;
            }
        }
        if self.vhe.contains(&NONE) {
            for f in 0..self.tris.len() {
                for k in 0..3 {
                    let v = self.tris[f][k] as usize;
                    if self.vhe[v] == NONE {
                        self.vhe[v] = (3 * f + k) as u32;
                    }
                }
            }
        }
        Ok(vmap)
    }

    /// The live mesh (call [`DynMesh::compact`] first).
    pub fn to_trimesh(&self) -> Result<TriMesh, AqError> {
        Ok(TriMesh { pos: try_copy(&self.pos)?, tris: try_copy(&self.tris)? })
    }

    /// Splits edge (a, b) with a new vertex. Both new halves of a chain edge keep its labels.
    pub fn split_edge(&mut self, a: u32, b: u32, nv: NewVert) -> Result<u32, EditRefused> {
        let (a, b, h) = {
            let h = self.find_he(a, b);
            if h != NONE {
                (a, b, h)
            } else {
                let h = self.find_he(b, a);
                if h == NONE {
                    return Err(EditRefused);
                }
                (b, a, h)
            }
        };
        let f = h / 3;
        let c = self.tail(prev(h));
        let t = self.twin[h as usize];
        let (k, ci) = (self.kind[h as usize], self.cid[h as usize]);
        let m = self.pos.len() as u32;
        let mut old = vec![f];
        let mut new = vec![[a, m, c], [m, b, c]];
        let ov = [(a, m, k, ci), (m, b, k, ci)];
        let (nb, cons): (Vec<(u32, u32)>, Vec<(u32, u32)>) = if t == NONE {
            (vec![(a, m), (m, b)], vec![(a, b)])
        } else {
            let g = t / 3;
            let d = self.tail(prev(t));
            old.push(g);
            new.push([b, m, d]);
            new.push([m, a, d]);
            (Vec::new(), Vec::new())
        };
        // Reserve the vertex id before rewiring (rewire only reads ids).
        self.add_vertex(nv);
        match self.rewire(&old, &new, (NONE, NONE), &ov, &nb, &cons) {
            Ok(_) => Ok(m),
            Err(e) => {
                self.vdead[m as usize] = true;
                Err(e)
            }
        }
    }

    /// Inserts a vertex inside face f (1-to-3 split).
    pub fn insert_in_face(&mut self, f: u32, nv: NewVert) -> Result<u32, EditRefused> {
        if !self.alive(f) {
            return Err(EditRefused);
        }
        let [a, b, c] = self.tris[f as usize];
        let m = self.add_vertex(nv);
        let new = [[a, b, m], [b, c, m], [c, a, m]];
        match self.rewire(&[f], &new, (NONE, NONE), &[], &[], &[]) {
            Ok(_) => Ok(m),
            Err(e) => {
                self.vdead[m as usize] = true;
                Err(e)
            }
        }
    }

    /// Flips the interior edge of halfedge h: (a, b, c), (b, a, d) become (c, a, d), (c, d, b).
    pub fn flip(&mut self, h: u32) -> Result<(), EditRefused> {
        let t = self.twin[h as usize];
        if t == NONE {
            return Err(EditRefused);
        }
        let (a, b) = (self.tail(h), self.head(h));
        let c = self.tail(prev(h));
        let d = self.tail(prev(t));
        if c == d {
            return Err(EditRefused);
        }
        self.rewire(&[h / 3, t / 3], &[[c, a, d], [c, d, b]], (NONE, NONE), &[], &[], &[]).map(|_| ())
    }

    /// Collapses edge (a, b) into b (a is removed; b keeps its position). No validity check
    /// beyond the rewire matching: the caller checks the link condition and the geometry.
    /// `merged` gives the labels of the side edges that absorb a chain edge of a.
    pub fn collapse(&mut self, a: u32, b: u32, merged: &[(u32, u32, FeatureKind, u32)], fan: &mut Vec<u32>) -> Result<(), EditRefused> {
        self.fan(a, fan);
        let mut old: Vec<u32> = fan.iter().map(|&h| h / 3).collect();
        old.sort_unstable();
        old.dedup();
        let mut new: Vec<[u32; 3]> = Vec::with_capacity(old.len());
        for &f in &old {
            let t = self.tris[f as usize];
            if t.contains(&b) {
                continue;
            }
            new.push(t.map(|v| if v == a { b } else { v }));
        }
        self.rewire(&old, &new, (a, b), merged, &[], &[(a, b), (b, a)])?;
        self.vdead[a as usize] = true;
        self.vhe[a as usize] = NONE;
        Ok(())
    }
}

/// Minimum interior angle of a triangle (radians).
/// The smallest angle is the one opposite the shortest side.
pub(crate) fn min_angle(a: V3, b: V3, c: V3) -> f64 {
    let (lab, lbc, lca) = (math::norm2(math::sub(b, a)), math::norm2(math::sub(c, b)), math::norm2(math::sub(a, c)));
    let (o, p, q) = if lab <= lbc && lab <= lca {
        (c, a, b)
    } else if lbc <= lca {
        (a, b, c)
    } else {
        (b, c, a)
    };
    math::angle_between(math::sub(p, o), math::sub(q, o))
}

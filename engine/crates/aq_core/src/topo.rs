//! C2, C3: implicit halfedges, twins, fans, components, boundary loops.
//!
//! * `he = 3f + k` runs from `tail = tris[f][k]` to `head = tris[f][(k+1)%3]`;
//!   `face(he) = he/3`, `next(he) = 3f + (k+1)%3`, `prev(he) = 3f + (k+2)%3`.
//! * `twin[he]` is `NONE` on a boundary.
//! * Corner `c = 3f + k` sits at vertex `tris[f][k]` (the tail of halfedge c).
//! * Canonical halfedge: `twin == NONE || he < twin`. Per-edge quantities are computed on the
//!   canonical halfedge and derived for the twin, never recomputed.
//! * `vert_he[v]` is one outgoing halfedge; for a boundary vertex it is the outgoing boundary
//!   halfedge, the CW-most one. `ccw_out(he) = twin(prev(he))` is the CCW successor
//!   (`NONE` at the end of an open fan).
//! * `H_in(v)` = all interior halfedges whose head is v, in CCW order. Crossing `he` from
//!   `face(he)` to `face(twin he)` is one CCW step around `head(he)`.

use crate::error::AqError;
use crate::mesh::{NONE, TriMesh};

/// Halfedge connectivity of a manifold, consistently oriented `TriMesh`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Topology {
    /// Per halfedge, the opposite halfedge or `NONE` on the boundary.
    pub twin: Vec<u32>,
    /// Per vertex, the CW-most outgoing halfedge (`NONE` for an isolated vertex).
    pub vert_he: Vec<u32>,
    /// Per face, the connected component id (components numbered by their lowest face).
    pub face_comp: Vec<u32>,
    pub n_comp: u32,
    /// Boundary halfedges in loop order; each loop starts at its lowest halfedge id and the
    /// loops are sorted by that id.
    pub boundary_loops: Vec<Vec<u32>>,
    pub is_bnd_vert: Vec<bool>,
}

/// Outgoing halfedges of a vertex in CCW order (C3).
#[derive(Clone, Debug)]
pub struct FanIter<'a> {
    twin: &'a [u32],
    start: u32,
    cur: u32,
    steps: usize,
}

impl Iterator for FanIter<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.cur == NONE || self.steps > self.twin.len() {
            return None;
        }
        let h = self.cur;
        self.steps += 1;
        let n = self.twin.get(Topology::prev(h) as usize).copied().unwrap_or(NONE);
        self.cur = if n == self.start { NONE } else { n };
        Some(h)
    }
}

/// `H_in(v)`: interior halfedges with head v, in CCW order (C3).
#[derive(Clone, Debug)]
pub struct InIter<'a> {
    fan: FanIter<'a>,
}

impl Iterator for InIter<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        loop {
            let h = self.fan.next()?;
            let p = Topology::prev(h);
            if self.fan.twin.get(p as usize).copied().unwrap_or(NONE) != NONE {
                return Some(p);
            }
        }
    }
}

impl Topology {
    /// Builds the connectivity. Requires a manifold, consistently oriented mesh:
    /// * an index out of range or a repeated index in a face gives `InvalidInput`;
    /// * a directed edge (tail, head) that appears twice (inconsistent orientation or an edge
    ///   with 3+ faces) gives `Topology`;
    /// * a non-manifold vertex (two fans) gives `Topology`.
    ///
    /// Isolated vertices are allowed (`vert_he = NONE`, in no component).
    pub fn build(m: &TriMesh) -> Result<Self, AqError> {
        let nv = m.pos.len();
        let nf = m.tris.len();
        if nv >= NONE as usize || nf.saturating_mul(3) >= NONE as usize {
            return Err(AqError::invalid_input("mesh too large for u32 ids"));
        }
        let nh = 3 * nf;
        for (f, t) in m.tris.iter().enumerate() {
            for &v in t {
                if v as usize >= nv {
                    return Err(AqError::invalid_input(format!(
                        "face {f} index {v} out of range ({nv} vertices)"
                    )));
                }
            }
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                return Err(AqError::invalid_input(format!("face {f} has a repeated index")));
            }
        }
        // Directed edges sorted by (tail, head).
        let mut keys: Vec<(u32, u32, u32)> = crate::alloc::try_with_capacity(nh)?;
        for (f, t) in m.tris.iter().enumerate() {
            for k in 0..3 {
                keys.push((t[k], t[(k + 1) % 3], (3 * f + k) as u32));
            }
        }
        keys.sort_unstable();
        for w in keys.windows(2) {
            if w[0].0 == w[1].0 && w[0].1 == w[1].1 {
                return Err(AqError::topology(format!(
                    "directed edge ({}, {}) appears twice (halfedges {} and {}): inconsistent \
                     orientation or non-manifold edge",
                    w[0].0, w[0].1, w[0].2, w[1].2
                )));
            }
        }
        let mut twin = crate::alloc::try_vec(nh, NONE)?;
        for &(a, b, h) in &keys {
            if let Ok(i) = keys.binary_search_by(|e| (e.0, e.1).cmp(&(b, a))) {
                twin[h as usize] = keys[i].2;
            }
        }
        // vert_he: the outgoing boundary halfedge if any, else the lowest outgoing id.
        let mut vert_he = crate::alloc::try_vec(nv, NONE)?;
        let mut out_count = crate::alloc::try_vec(nv, 0u32)?;
        let mut i = 0;
        while i < keys.len() {
            let v = keys[i].0;
            let mut j = i;
            let mut bnd = NONE;
            let mut bnd_count = 0;
            let mut lowest = NONE;
            while j < keys.len() && keys[j].0 == v {
                let h = keys[j].2;
                lowest = lowest.min(h);
                if twin[h as usize] == NONE {
                    bnd_count += 1;
                    bnd = bnd.min(h);
                }
                j += 1;
            }
            if bnd_count > 1 {
                return Err(AqError::topology(format!(
                    "vertex {v} has {bnd_count} outgoing boundary halfedges (non-manifold vertex)"
                )));
            }
            vert_he[v as usize] = if bnd != NONE { bnd } else { lowest };
            out_count[v as usize] = (j - i) as u32;
            i = j;
        }
        let mut is_bnd_vert = crate::alloc::try_vec(nv, false)?;
        for v in 0..nv {
            let h0 = vert_he[v];
            if h0 == NONE {
                continue;
            }
            is_bnd_vert[v] = twin[h0 as usize] == NONE;
            // The fan walk must reach every outgoing halfedge (one fan per vertex).
            let fan = FanIter { twin: &twin, start: h0, cur: h0, steps: 0 };
            let count = fan.count();
            if count != out_count[v] as usize {
                return Err(AqError::topology(format!(
                    "vertex {v} is non-manifold: fan reaches {count} of {} outgoing halfedges",
                    out_count[v]
                )));
            }
        }
        // Components by BFS over faces (explicit queue), numbered by lowest face id.
        let mut face_comp = crate::alloc::try_vec(nf, NONE)?;
        let mut n_comp = 0u32;
        let mut queue: Vec<u32> = Vec::new();
        for f0 in 0..nf {
            if face_comp[f0] != NONE {
                continue;
            }
            face_comp[f0] = n_comp;
            queue.clear();
            queue.push(f0 as u32);
            let mut qi = 0;
            while qi < queue.len() {
                let f = queue[qi];
                qi += 1;
                for k in 0..3 {
                    let t = twin[(3 * f + k) as usize];
                    if t != NONE {
                        let g = (t / 3) as usize;
                        if face_comp[g] == NONE {
                            face_comp[g] = n_comp;
                            queue.push(g as u32);
                        }
                    }
                }
            }
            n_comp += 1;
        }
        // Boundary loops: from a boundary halfedge, the next one is the outgoing boundary
        // halfedge of its head (unique on a manifold vertex).
        let mut boundary_loops = Vec::new();
        let mut seen = crate::alloc::try_vec(nh, false)?;
        for h0 in 0..nh {
            if twin[h0] != NONE || seen[h0] {
                continue;
            }
            let mut lp = Vec::new();
            let mut h = h0 as u32;
            loop {
                if seen[h as usize] {
                    return Err(AqError::topology(format!(
                        "boundary loop from halfedge {h0} does not close"
                    )));
                }
                seen[h as usize] = true;
                lp.push(h);
                let head = m.tris[(h / 3) as usize][((h % 3 + 1) % 3) as usize];
                let nxt = vert_he[head as usize];
                if nxt == NONE || twin[nxt as usize] != NONE {
                    return Err(AqError::topology(format!(
                        "boundary halfedge {h} has no boundary successor"
                    )));
                }
                if nxt == h0 as u32 {
                    break;
                }
                h = nxt;
            }
            boundary_loops.push(lp);
        }
        Ok(Topology { twin, vert_he, face_comp, n_comp, boundary_loops, is_bnd_vert })
    }

    #[inline]
    pub fn face(h: u32) -> u32 {
        h / 3
    }
    #[inline]
    pub fn next(h: u32) -> u32 {
        h - h % 3 + (h % 3 + 1) % 3
    }
    #[inline]
    pub fn prev(h: u32) -> u32 {
        h - h % 3 + (h % 3 + 2) % 3
    }
    /// Tail vertex of h (= the vertex of corner h).
    #[inline]
    pub fn tail(&self, m: &TriMesh, h: u32) -> u32 {
        m.tris[(h / 3) as usize][(h % 3) as usize]
    }
    #[inline]
    pub fn head(&self, m: &TriMesh, h: u32) -> u32 {
        m.tris[(h / 3) as usize][((h % 3 + 1) % 3) as usize]
    }
    #[inline]
    pub fn twin_of(&self, h: u32) -> u32 {
        self.twin[h as usize]
    }
    #[inline]
    pub fn is_boundary_he(&self, h: u32) -> bool {
        self.twin[h as usize] == NONE
    }
    /// C2: `twin == NONE || he < twin`.
    #[inline]
    pub fn is_canonical(&self, h: u32) -> bool {
        let t = self.twin[h as usize];
        t == NONE || h < t
    }
    /// C3: CCW successor of an outgoing halfedge, `twin(prev(he))`, NONE at the end of an
    /// open fan.
    #[inline]
    pub fn ccw_out(&self, h: u32) -> u32 {
        self.twin[Self::prev(h) as usize]
    }
    /// C3: CW successor of an outgoing halfedge, `next(twin(he))`, NONE at the start of an
    /// open fan.
    #[inline]
    pub fn cw_out(&self, h: u32) -> u32 {
        let t = self.twin[h as usize];
        if t == NONE { NONE } else { Self::next(t) }
    }
    /// Outgoing halfedges of v in CCW order, starting at `vert_he[v]` (C3).
    pub fn fan_out(&self, v: u32) -> FanIter<'_> {
        let h0 = self.vert_he.get(v as usize).copied().unwrap_or(NONE);
        FanIter { twin: &self.twin, start: h0, cur: h0, steps: 0 }
    }
    /// `H_in(v)`: interior halfedges with head v, in CCW order (C3). For each outgoing
    /// halfedge `o` of the fan, `prev(o)` is incoming, and crossing it leads to
    /// `face(ccw_out(o))`.
    pub fn incoming(&self, _m: &TriMesh, v: u32) -> InIter<'_> {
        InIter { fan: self.fan_out(v) }
    }
    /// Number of faces incident to v.
    pub fn valence_faces(&self, v: u32) -> usize {
        self.fan_out(v).count()
    }
    /// Vertex-edge valence of v (number of incident edges).
    pub fn valence(&self, v: u32) -> usize {
        let n = self.fan_out(v).count();
        if self.is_bnd_vert.get(v as usize).copied().unwrap_or(false) { n + 1 } else { n }
    }
    /// The halfedge from a to b, if any.
    pub fn find_he(&self, m: &TriMesh, a: u32, b: u32) -> Option<u32> {
        self.fan_out(a).find(|&h| self.head(m, h) == b)
    }
    /// Euler characteristic V - E + F of component `comp`.
    pub fn euler_char(&self, _m: &TriMesh, comp: u32) -> i64 {
        let mut f = 0i64;
        let mut e = 0i64;
        for (fi, &c) in self.face_comp.iter().enumerate() {
            if c != comp {
                continue;
            }
            f += 1;
            for k in 0..3 {
                if self.is_canonical((3 * fi + k) as u32) {
                    e += 1;
                }
            }
        }
        let v = self
            .vert_he
            .iter()
            .filter(|&&h| h != NONE && self.face_comp[(h / 3) as usize] == comp)
            .count() as i64;
        v - e + f
    }
    /// Number of boundary loops of component `comp`.
    pub fn n_boundary_loops(&self, comp: u32) -> usize {
        self.boundary_loops
            .iter()
            .filter(|l| l.first().is_some_and(|&h| self.face_comp[(h / 3) as usize] == comp))
            .count()
    }
    /// Checks the C2/C3 identities against `m`: sizes, `twin(twin h) = h`, twins reversed,
    /// `vert_he` outgoing and CW-most, boundary loops closed.
    pub fn validate(&self, m: &TriMesh) -> Result<(), AqError> {
        use crate::error::ensure;
        let nh = 3 * m.tris.len();
        ensure(self.twin.len() == nh, || "Topology.twin size".into())?;
        ensure(self.vert_he.len() == m.pos.len(), || "Topology.vert_he size".into())?;
        ensure(self.face_comp.len() == m.tris.len(), || "Topology.face_comp size".into())?;
        ensure(self.is_bnd_vert.len() == m.pos.len(), || "Topology.is_bnd_vert size".into())?;
        for h in 0..nh as u32 {
            let t = self.twin[h as usize];
            if t == NONE {
                continue;
            }
            ensure((t as usize) < nh && t != h, || format!("twin of {h} out of range"))?;
            ensure(self.twin[t as usize] == h, || format!("twin(twin({h})) != {h}"))?;
            ensure(
                self.tail(m, h) == self.head(m, t) && self.head(m, h) == self.tail(m, t),
                || format!("twin of {h} is not reversed"),
            )?;
            ensure(self.face_comp[(h / 3) as usize] == self.face_comp[(t / 3) as usize], || {
                format!("twin of {h} in another component")
            })?;
        }
        for (v, &h) in self.vert_he.iter().enumerate() {
            if h == NONE {
                continue;
            }
            ensure((h as usize) < nh && self.tail(m, h) == v as u32, || {
                format!("vert_he[{v}] is not outgoing")
            })?;
            ensure(self.is_bnd_vert[v] == (self.twin[h as usize] == NONE), || {
                format!("vert_he[{v}] is not the CW-most (boundary) halfedge")
            })?;
            if self.is_bnd_vert[v] {
                ensure(self.cw_out(h) == NONE, || format!("vertex {v}: boundary fan start"))?;
            }
        }
        for lp in &self.boundary_loops {
            for (i, &h) in lp.iter().enumerate() {
                let n = lp[(i + 1) % lp.len()];
                ensure(self.twin[h as usize] == NONE, || format!("loop he {h} not boundary"))?;
                ensure(self.head(m, h) == self.tail(m, n), || format!("loop broken at {h}"))?;
            }
        }
        Ok(())
    }
}

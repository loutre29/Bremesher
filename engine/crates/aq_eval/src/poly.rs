//! Polygon mesh helpers shared by the metrics: sanitized face lists, the edge census of
//! `tools/quad_stats.py` (unique undirected edges with their face counts, boundary and
//! valence per vertex) and a polygon halfedge structure for the loop walker.

use aq_core::{NONE, QuadMesh, TriMesh};

/// Faces of q with a valid CSR range and in-range vertex ids (all of them for a mesh that
/// passes `QuadMesh::validate`). Malformed faces are skipped instead of panicking.
pub(crate) fn valid_faces(q: &QuadMesh) -> Vec<&[u32]> {
    let nv = q.pos.len();
    let mut out = Vec::with_capacity(q.n_faces());
    for f in 0..q.n_faces() {
        let (a, b) = (q.face_off[f] as usize, q.face_off[f + 1] as usize);
        if let Some(fv) = q.face_verts.get(a..b)
            && !fv.is_empty()
            && fv.iter().all(|&v| (v as usize) < nv)
        {
            out.push(fv);
        }
    }
    out
}

/// Fan triangulation of every valid face of q from its first vertex, (v0, vi, vi+1).
/// Positions are shared with q (same vertex ids).
pub fn fan_triangulate(q: &QuadMesh) -> TriMesh {
    let mut tris = Vec::with_capacity(q.face_verts.len());
    for fv in valid_faces(q) {
        for i in 1..fv.len().saturating_sub(1) {
            tris.push([fv[0], fv[i], fv[i + 1]]);
        }
    }
    TriMesh { pos: q.pos.clone(), tris }
}

/// A polygon mesh whose faces are the triangles of m (vflag and face_comp 0).
pub fn poly_from_tris(m: &TriMesh) -> QuadMesh {
    let mut q = QuadMesh::new();
    q.pos = m.pos.clone();
    q.vflag = vec![0; m.pos.len()];
    for t in &m.tris {
        q.push_face(t, 0);
    }
    q
}

/// The edge census of quad_stats.py.
pub(crate) struct EdgeCensus {
    /// Unique undirected edges (min, max) in first-seen order (the iteration order of the
    /// script's `edge_faces` dict), with the number of face sides on them.
    pub edges: Vec<([u32; 2], u32)>,
    /// Per vertex: number of unique edges (0 for a vertex used by no face).
    pub valence: Vec<u32>,
    /// Per vertex: endpoint of an edge used by exactly one face side.
    pub boundary: Vec<bool>,
    /// Per vertex: endpoint of an edge used by more than two face sides.
    pub nonmanifold: Vec<bool>,
}

pub(crate) fn edge_census(nv: usize, faces: &[&[u32]]) -> EdgeCensus {
    let mut all: Vec<(u32, u32, u32)> = Vec::new();
    let mut seq = 0u32;
    for fv in faces {
        let n = fv.len();
        for i in 0..n {
            let (a, b) = (fv[i], fv[(i + 1) % n]);
            all.push((a.min(b), a.max(b), seq));
            seq += 1;
        }
    }
    all.sort_unstable();
    let mut uniq: Vec<(u32, [u32; 2], u32)> = Vec::new();
    for &(a, b, s) in &all {
        match uniq.last_mut() {
            Some(last) if last.1 == [a, b] => last.2 += 1,
            _ => uniq.push((s, [a, b], 1)),
        }
    }
    uniq.sort_unstable_by_key(|e| e.0);
    let mut valence = vec![0u32; nv];
    let mut boundary = vec![false; nv];
    let mut nonmanifold = vec![false; nv];
    for &(_, [a, b], c) in &uniq {
        valence[a as usize] += 1;
        if a != b {
            valence[b as usize] += 1;
        }
        if c == 1 {
            boundary[a as usize] = true;
            boundary[b as usize] = true;
        }
        if c > 2 {
            nonmanifold[a as usize] = true;
            nonmanifold[b as usize] = true;
        }
    }
    EdgeCensus { edges: uniq.into_iter().map(|(_, e, c)| (e, c)).collect(), valence, boundary, nonmanifold }
}

/// quad_stats.py irregular rule: interior valence != 4, boundary valence not in {2, 3}.
pub(crate) fn is_irregular(valence: u32, boundary: bool) -> bool {
    if boundary { !(valence == 2 || valence == 3) } else { valence != 4 }
}

/// Polygon halfedge structure over the valid faces. Halfedge i runs from `tail[i]` to
/// `head[i]` inside face `face[i]`. `twin` is NONE on the boundary and on every edge that is
/// not a clean manifold, consistently oriented pair.
pub(crate) struct PolyTopo {
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub next: Vec<u32>,
    pub prev: Vec<u32>,
    pub twin: Vec<u32>,
    /// Undirected edge id per halfedge.
    pub edge: Vec<u32>,
    pub n_edges: usize,
    /// Per edge: number of face sides on it.
    pub edge_count: Vec<u32>,
    /// Per edge: lowest halfedge on it.
    pub edge_he: Vec<u32>,
    pub census: EdgeCensus,
}

impl PolyTopo {
    pub fn build(q: &QuadMesh) -> PolyTopo {
        let faces = valid_faces(q);
        let census = edge_census(q.pos.len(), &faces);
        let mut tail = Vec::new();
        let mut head = Vec::new();
        let mut next = Vec::new();
        let mut prev = Vec::new();
        for fv in &faces {
            let base = tail.len() as u32;
            let n = fv.len() as u32;
            for i in 0..n {
                tail.push(fv[i as usize]);
                head.push(fv[((i + 1) % n) as usize]);
                next.push(base + (i + 1) % n);
                prev.push(base + (i + n - 1) % n);
            }
        }
        let nh = tail.len();
        let mut und: Vec<(u32, u32, u32)> =
            (0..nh).map(|h| (tail[h].min(head[h]), tail[h].max(head[h]), h as u32)).collect();
        und.sort_unstable();
        let mut edge = vec![NONE; nh];
        let mut edge_count = Vec::new();
        let mut edge_he = Vec::new();
        let mut twin = vec![NONE; nh];
        let mut i = 0;
        while i < und.len() {
            let mut j = i;
            while j < und.len() && und[j].0 == und[i].0 && und[j].1 == und[i].1 {
                j += 1;
            }
            let e = edge_count.len() as u32;
            edge_count.push((j - i) as u32);
            edge_he.push(und[i].2);
            for k in &und[i..j] {
                edge[k.2 as usize] = e;
            }
            if j - i == 2 {
                let (h0, h1) = (und[i].2 as usize, und[i + 1].2 as usize);
                if tail[h0] == head[h1] && head[h0] == tail[h1] && tail[h0] != head[h0] {
                    twin[h0] = h1 as u32;
                    twin[h1] = h0 as u32;
                }
            }
            i = j;
        }
        PolyTopo { tail, head, next, prev, twin, edge, n_edges: edge_count.len(), edge_count, edge_he, census }
    }

    /// True for a vertex where an edge line runs straight through: interior, valence 4, and a
    /// closed fan of exactly 4 clean edges.
    pub fn regular_interior(&self, v: u32, h_in: u32) -> bool {
        let vi = v as usize;
        if self.census.boundary[vi] || self.census.nonmanifold[vi] || self.census.valence[vi] != 4 {
            return false;
        }
        // Rotate around v starting from the outgoing twin of h_in.
        let Some(&o0) = self.twin.get(h_in as usize) else { return false };
        if o0 == NONE {
            return false;
        }
        let mut o = o0;
        for _ in 0..4 {
            let t = self.twin[self.prev[o as usize] as usize];
            if t == NONE {
                return false;
            }
            o = t;
        }
        o == o0
    }

    /// Regular boundary vertex: boundary, valence 3, not touching a non-manifold edge.
    pub fn regular_boundary(&self, v: u32) -> bool {
        let vi = v as usize;
        self.census.boundary[vi] && !self.census.nonmanifold[vi] && self.census.valence[vi] == 3
    }
}

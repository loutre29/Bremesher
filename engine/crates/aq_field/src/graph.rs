//! Vertex graph of the mesh and a reusable Dijkstra workspace (geodesic distances on edges,
//! DESIGN 6.6 steps 3 to 6). Ties are broken by the lowest vertex id, so every search is
//! deterministic.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use aq_core::alloc::try_vec;
use aq_core::math;
use aq_core::{AqError, Ctl, NONE};

use crate::ProxyRef;

/// CSR adjacency: for vertex v, its neighbours sorted by id, the halfedge v -> neighbour
/// (`NONE` when only the reverse boundary halfedge exists) and the edge length.
pub(crate) struct Graph {
    off: Vec<u32>,
    nb: Vec<u32>,
    he: Vec<u32>,
    len: Vec<f64>,
}

impl Graph {
    pub fn new(px: &ProxyRef) -> Result<Self, AqError> {
        let m = px.mesh;
        let t = px.topo;
        let nv = m.pos.len();
        let nh = 3 * m.tris.len();
        let mut ent: Vec<(u32, u32, u32)> = aq_core::alloc::try_with_capacity(nh + nh / 8)?;
        for h in 0..nh as u32 {
            let (a, b) = (t.tail(m, h), t.head(m, h));
            ent.push((a, b, h));
            if t.twin[h as usize] == NONE {
                ent.push((b, a, NONE));
            }
        }
        ent.sort_unstable();
        let mut off = try_vec(nv + 1, 0u32)?;
        for e in &ent {
            off[e.0 as usize + 1] += 1;
        }
        for v in 0..nv {
            off[v + 1] += off[v];
        }
        let nb = ent.iter().map(|e| e.1).collect();
        let he = ent.iter().map(|e| e.2).collect();
        let len = ent.iter().map(|e| math::dist(m.pos[e.0 as usize], m.pos[e.1 as usize])).collect();
        Ok(Graph { off, nb, he, len })
    }

    /// (neighbour, halfedge v -> neighbour or NONE, length).
    pub fn neighbors(&self, v: u32) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        let (a, b) = (self.off[v as usize] as usize, self.off[v as usize + 1] as usize);
        (a..b).map(move |i| (self.nb[i], self.he[i], self.len[i]))
    }
}

/// Dijkstra workspace, reset in O(touched).
pub(crate) struct Dij {
    pub dist: Vec<f64>,
    pub pred: Vec<u32>,
    settled: Vec<bool>,
    touched: Vec<u32>,
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl Dij {
    pub fn new(nv: usize) -> Result<Self, AqError> {
        Ok(Dij {
            dist: try_vec(nv, f64::INFINITY)?,
            pred: try_vec(nv, NONE)?,
            settled: try_vec(nv, false)?,
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        })
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            self.dist[v as usize] = f64::INFINITY;
            self.pred[v as usize] = NONE;
            self.settled[v as usize] = false;
        }
        self.touched.clear();
        self.heap.clear();
    }

    /// Multi-source search. A source is always settled; another vertex is entered only when
    /// `can_enter` accepts it and the edge passes `edge_ok(from, to, he)`. Vertices farther than
    /// `bound` are not reached. `stop` is called on every settled vertex (sources included) and
    /// ends the search when it returns true; that vertex is returned.
    pub fn run(
        &mut self,
        g: &Graph,
        sources: &[u32],
        bound: f64,
        can_enter: &dyn Fn(u32) -> bool,
        edge_ok: &dyn Fn(u32, u32, u32) -> bool,
        stop: &mut dyn FnMut(u32, f64) -> bool,
        ctl: Option<&Ctl>,
    ) -> Result<Option<u32>, AqError> {
        self.reset();
        for &s in sources {
            if (s as usize) < self.dist.len() && self.dist[s as usize] != 0.0 {
                self.dist[s as usize] = 0.0;
                self.touched.push(s);
                self.heap.push(Reverse((0f64.to_bits(), s)));
            }
        }
        let mut pops = 0usize;
        while let Some(Reverse((db, v))) = self.heap.pop() {
            if self.settled[v as usize] {
                continue;
            }
            let d = f64::from_bits(db);
            self.settled[v as usize] = true;
            pops += 1;
            if pops.is_multiple_of(1024)
                && let Some(c) = ctl
            {
                c.check()?;
            }
            if stop(v, d) {
                return Ok(Some(v));
            }
            for (w, he, l) in g.neighbors(v) {
                if self.settled[w as usize] || !can_enter(w) || !edge_ok(v, w, he) {
                    continue;
                }
                let nd = d + l;
                if nd > bound || nd >= self.dist[w as usize] {
                    continue;
                }
                if self.dist[w as usize].is_infinite() {
                    self.touched.push(w);
                }
                self.dist[w as usize] = nd;
                self.pred[w as usize] = v;
                self.heap.push(Reverse((nd.to_bits(), w)));
            }
        }
        Ok(None)
    }

    /// Full multi-source distances over the whole graph (no restriction, no bound).
    pub fn distances(&mut self, g: &Graph, sources: &[u32], ctl: Option<&Ctl>) -> Result<Vec<f64>, AqError> {
        self.run(g, sources, f64::INFINITY, &|_| true, &|_, _, _| true, &mut |_, _| false, ctl)?;
        let d = self.dist.clone();
        Ok(d)
    }

    /// The vertex path from the source to v (settled by the last run), source first.
    pub fn path_to(&self, v: u32) -> Vec<u32> {
        let mut path = vec![v];
        let mut cur = v;
        let mut guard = 0usize;
        while self.pred[cur as usize] != NONE && guard <= self.pred.len() {
            cur = self.pred[cur as usize];
            path.push(cur);
            guard += 1;
        }
        path.reverse();
        path
    }

    /// Vertices reached by the last run (settled or not), in reach order.
    pub fn touched(&self) -> &[u32] {
        &self.touched
    }

    /// Distance of v found by the last run (infinite when not reached).
    pub fn dist_of(&self, v: u32) -> f64 {
        self.dist.get(v as usize).copied().unwrap_or(f64::INFINITY)
    }

    /// True when v was settled by the last run.
    pub fn is_settled(&self, v: u32) -> bool {
        self.settled[v as usize]
    }
}

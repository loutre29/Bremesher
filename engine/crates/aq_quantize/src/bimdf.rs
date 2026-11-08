//! Bi-MDF solver (DESIGN 6.11 steps 1 to 3).
//!
//! 1. Min-cost flow on the double cover by successive shortest paths (Dijkstra with
//!    potentials, unit or capacity-scaled augmentations). Its symmetrization is an optimal
//!    half-integral solution.
//! 2. Parity repair: the half-integral arcs are decomposed into closed walks and rounded with
//!    alternating +-1/2 changes; an inconsistent walk leaves a +-1 imbalance at its start.
//! 3. Integer repair: every remaining imbalance is removed by cheapest paths in the double
//!    cover (+-1 strip moves) to a compensating node or to the ground.
//! 4. Polish (an addition, CHANGES.md WP-12): improving closed walks of the double cover are
//!    applied while one exists (negative cycle cancelling on the true marginal costs).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use aq_core::{AqError, Ctl, ErrKind, NONE, TMesh};

use crate::graph::{ArcCost, CGraph, N_MAX, err};

/// Total excess above which the double cover flow uses capacity scaling (DESIGN 6.11).
const SCALING_EXCESS: i64 = 2000;
/// Reduced costs above `-EPS` count as non-negative.
const EPS: f64 = 1e-9;
/// Dijkstra runs between two `ctl.check()` calls.
const CHECK_EVERY: u32 = 256;
/// Upper bound on polish rounds per solve.
const MAX_POLISH: usize = 64;
/// Polish work budget per solve, in passes over the residual edges.
const POLISH_WORK: usize = 48;

#[derive(Clone, Copy, Debug)]
struct Item {
    d: f64,
    v: u32,
}

impl PartialEq for Item {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    // Min-heap on (d, v).
    fn cmp(&self, o: &Self) -> Ordering {
        o.d.total_cmp(&self.d).then(o.v.cmp(&self.v))
    }
}

/// Reusable Dijkstra workspace over the double cover.
pub(crate) struct Dijkstra {
    dist: Vec<f64>,
    seen: Vec<u32>,
    done: Vec<u32>,
    pred: Vec<u32>,
    stamp: u32,
    heap: BinaryHeap<Item>,
    settled: Vec<u32>,
    runs: u32,
}

impl Dijkstra {
    pub fn new(n: usize) -> Self {
        Dijkstra {
            dist: vec![0.0; n],
            seen: vec![0; n],
            done: vec![0; n],
            pred: vec![NONE; n],
            stamp: 0,
            heap: BinaryHeap::new(),
            settled: Vec::new(),
            runs: 0,
        }
    }

    fn fit(&mut self, n: usize) {
        if self.dist.len() != n {
            *self = Dijkstra::new(n);
        }
    }

    /// Shortest path from the nearest of `srcs` to the nearest node accepted by `is_target`
    /// (the caller makes sure no source is a target). `cost(copy, backward)` gives the
    /// (clamped non-negative) cost of a residual edge, or None when it does not exist.
    fn run(
        &mut self,
        g: &CGraph,
        srcs: &[u32],
        mut cost: impl FnMut(usize, bool) -> Option<f64>,
        is_target: impl Fn(u32) -> bool,
    ) -> Option<(u32, f64)> {
        if self.stamp == u32::MAX {
            self.seen.iter_mut().for_each(|x| *x = 0);
            self.done.iter_mut().for_each(|x| *x = 0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.runs = self.runs.wrapping_add(1);
        let st = self.stamp;
        self.settled.clear();
        self.heap.clear();
        for &src in srcs {
            let s = src as usize;
            self.dist[s] = 0.0;
            self.seen[s] = st;
            self.pred[s] = NONE;
            self.heap.push(Item { d: 0.0, v: src });
        }
        while let Some(Item { d, v }) = self.heap.pop() {
            let vu = v as usize;
            if self.done[vu] == st || d > self.dist[vu] {
                continue;
            }
            self.done[vu] = st;
            self.settled.push(v);
            if is_target(v) {
                return Some((v, d));
            }
            for &e in &g.adj[g.adj_off[vu] as usize..g.adj_off[vu + 1] as usize] {
                let c = (e >> 1) as usize;
                let back = e & 1 == 1;
                let w = if back { g.ctail[c] } else { g.chead[c] } as usize;
                if self.done[w] == st {
                    continue;
                }
                let Some(rc) = cost(c, back) else { continue };
                let nd = d + rc.max(0.0);
                if self.seen[w] != st || nd < self.dist[w] {
                    self.seen[w] = st;
                    self.dist[w] = nd;
                    self.pred[w] = e;
                    self.heap.push(Item { d: nd, v: w as u32 });
                }
            }
        }
        None
    }

    /// Source and residual edges (copy, backward), in path order, of the last path to `t`.
    fn path(&self, g: &CGraph, t: u32) -> (u32, Vec<(usize, bool)>) {
        let mut out = Vec::new();
        let mut v = t;
        loop {
            let e = self.pred[v as usize];
            if e == NONE || out.len() > self.pred.len() {
                break;
            }
            let c = (e >> 1) as usize;
            let back = e & 1 == 1;
            out.push((c, back));
            v = if back { g.chead[c] } else { g.ctail[c] };
        }
        out.reverse();
        (v, out)
    }

    fn tick(&self, ctl: Option<&Ctl>) -> Result<(), AqError> {
        if self.runs.is_multiple_of(CHECK_EVERY)
            && let Some(c) = ctl
        {
            c.check()?;
        }
        Ok(())
    }
}

/// Solves the quantization for lower bounds `lb` at scale `sigma`.
pub(crate) fn solve(
    t: &TMesh,
    g: &CGraph,
    sigma: f64,
    lb: &[u32],
    ctl: Option<&Ctl>,
    dij: &mut Dijkstra,
) -> Result<Vec<u32>, AqError> {
    if lb.len() != g.na {
        return Err(err(ErrKind::InvalidInput, format!("lb has {} entries for {} arcs", lb.len(), g.na)));
    }
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(err(ErrKind::InvalidInput, format!("sigma {sigma} must be positive")));
    }
    if lb.iter().any(|&l| l as i64 > N_MAX) {
        return Err(err(ErrKind::InvalidInput, "lb above the solver range"));
    }
    dij.fit(g.dn());
    let cost: Vec<ArcCost> = t.arcs.iter().map(|a| ArcCost::new(a, sigma)).collect();
    let mut n: Vec<i64> = (0..g.na).map(|a| cost[a].argmin(lb[a])).collect();
    let x = flow(g, &cost, lb, ctl, dij)?;
    // Step 2: twice the half-integral solution.
    let mut n2 = vec![0i64; g.na];
    for a in 0..g.na {
        let c = g.arc_copy[a];
        if c != NONE {
            n2[a] = x[c as usize] + x[c as usize + 1];
        }
    }
    parity_round(g, &cost, &mut n2);
    for a in 0..g.na {
        if g.arc_copy[a] != NONE {
            n[a] = n2[a] / 2;
        }
    }
    integer_repair(g, &cost, lb, &mut n, ctl, dij)?;
    polish(g, &cost, lb, &mut n, ctl)?;
    // Final exact check.
    let b = g.balances(&n);
    if let Some(q) = b.iter().position(|&x| x != 0) {
        return Err(err(ErrKind::Internal, format!("quantization leaves node {q} unbalanced")));
    }
    let mut out = Vec::with_capacity(g.na);
    for a in 0..g.na {
        if n[a] < lb[a] as i64 || n[a] > N_MAX {
            return Err(err(ErrKind::Internal, format!("arc {a} quantized to {} outside its bounds", n[a])));
        }
        out.push(n[a] as u32);
    }
    Ok(out)
}

/// Step 1: min-cost flow on the double cover. Returns the flow per copy.
///
/// Primal-dual successive shortest paths: a multi-source Dijkstra (reduced costs, early stop
/// at the first target) updates the potentials, then every admissible path (zero reduced
/// cost) from a source to a target is augmented by DFS before the next Dijkstra. Every
/// admissible path is a shortest path, so this is the same algorithm as one augmentation per
/// Dijkstra, with far fewer searches. Stage A routes excess to deficits or to the ground,
/// stage B feeds the remaining deficits from the ground. Capacity scaling (delta-phases) is
/// used when the total excess is above 2000.
fn flow(g: &CGraph, cost: &[ArcCost], lb: &[u32], ctl: Option<&Ctl>, dij: &mut Dijkstra) -> Result<Vec<i64>, AqError> {
    let nc = g.ctail.len();
    let dn = g.dn();
    let dg = g.dg();
    let half = |c: usize, x: i64| 0.5 * cost[g.carc[c] as usize].f(x as f64);
    let clb = |c: usize| lb[g.carc[c] as usize] as i64;
    let mut st = FlowState {
        x: (0..nc).map(|c| cost[g.carc[c] as usize].argmin(lb[g.carc[c] as usize])).collect(),
        pi: vec![0.0f64; dn],
        exc: vec![0i64; dn],
        delta: 1,
    };
    for c in 0..nc {
        st.exc[g.chead[c] as usize] += st.x[c];
        st.exc[g.ctail[c] as usize] -= st.x[c];
    }
    let total: i64 = (0..dn).filter(|&v| v != dg as usize).map(|v| st.exc[v].max(0)).sum();
    if total > SCALING_EXCESS {
        let m = (0..dn).filter(|&v| v != dg as usize).map(|v| st.exc[v].abs()).max().unwrap_or(1).max(1);
        st.delta = 1i64 << (63 - m.leading_zeros());
    }
    let mut dfs = Dfs::new(dn);
    let mut srcs: Vec<u32> = Vec::new();
    loop {
        let delta = st.delta;
        // Restore non-negative reduced costs of the delta-residual edges.
        for c in 0..nc {
            let (tl, hd) = (g.ctail[c] as usize, g.chead[c] as usize);
            while st.x[c] + delta <= N_MAX
                && half(c, st.x[c] + delta) - half(c, st.x[c]) + st.pi[tl] - st.pi[hd] < -EPS
            {
                st.x[c] += delta;
                st.exc[tl] -= delta;
                st.exc[hd] += delta;
            }
            while st.x[c] - delta >= clb(c)
                && half(c, st.x[c] - delta) - half(c, st.x[c]) + st.pi[hd] - st.pi[tl] < -EPS
            {
                st.x[c] -= delta;
                st.exc[tl] += delta;
                st.exc[hd] -= delta;
            }
        }
        for stage_b in [false, true] {
            loop {
                srcs.clear();
                if stage_b {
                    if (0..dn).any(|v| v != dg as usize && st.exc[v] <= -delta) {
                        srcs.push(dg);
                    }
                } else {
                    srcs.extend((0..dn as u32).filter(|&v| v != dg && st.exc[v as usize] >= delta));
                }
                if srcs.is_empty() {
                    break;
                }
                let r = {
                    let s = &st;
                    dij.run(
                        g,
                        &srcs,
                        |c, back| rc(g, cost, lb, &s.x, &s.pi, delta, c, back),
                        |v| s.is_target(v, dg, stage_b),
                    )
                };
                dij.tick(ctl)?;
                let Some((tg, d)) = r else { break };
                for &v in &dij.settled {
                    let dv = dij.dist[v as usize];
                    if dv < d {
                        st.pi[v as usize] -= d - dv;
                    }
                }
                let (s0, path) = dij.path(g, tg);
                st.push(g, s0, tg, &path);
                // Every other admissible path is a shortest path too.
                dfs.augment_all(g, cost, lb, &mut st, &srcs, stage_b);
            }
        }
        if delta == 1 {
            break;
        }
        st.delta /= 2;
    }
    if let Some(v) = (0..dn).find(|&v| v != dg as usize && st.exc[v] != 0) {
        return Err(err(
            ErrKind::Infeasible,
            format!("no feasible quantization (double cover node {v} keeps excess {})", st.exc[v]),
        ));
    }
    Ok(st.x)
}

/// Flow, potentials and excesses of the double cover during step 1.
struct FlowState {
    x: Vec<i64>,
    pi: Vec<f64>,
    exc: Vec<i64>,
    delta: i64,
}

impl FlowState {
    #[inline]
    fn is_target(&self, v: u32, dg: u32, stage_b: bool) -> bool {
        if v == dg { !stage_b } else { self.exc[v as usize] <= -self.delta }
    }

    #[inline]
    fn is_source(&self, v: u32, dg: u32, stage_b: bool) -> bool {
        if v == dg { stage_b } else { !stage_b && self.exc[v as usize] >= self.delta }
    }

    /// Pushes delta along `path` from `s` to `t`.
    fn push(&mut self, g: &CGraph, s: u32, t: u32, path: &[(usize, bool)]) {
        let delta = self.delta;
        for &(c, back) in path {
            self.x[c] += if back { -delta } else { delta };
        }
        let dg = g.dg();
        if s != dg {
            self.exc[s as usize] -= delta;
        }
        if t != dg {
            self.exc[t as usize] += delta;
        }
    }
}

/// Admissible reduced costs are at most this (absolute; potentials are O(objective)).
const ADM_EPS: f64 = 1e-9;

/// DFS workspace for augmenting along admissible edges.
struct Dfs {
    ptr: Vec<u32>,
    dead: Vec<bool>,
    on: Vec<bool>,
    stack: Vec<u32>,
    edges: Vec<(usize, bool)>,
}

impl Dfs {
    fn new(n: usize) -> Self {
        Dfs { ptr: vec![0; n], dead: vec![false; n], on: vec![false; n], stack: Vec::new(), edges: Vec::new() }
    }

    /// Augments delta along admissible paths from the sources until none is left (current-arc
    /// pointers and dead marks keep one call linear in the number of residual edges, plus the
    /// path lengths).
    fn augment_all(&mut self, g: &CGraph, cost: &[ArcCost], lb: &[u32], st: &mut FlowState, srcs: &[u32], stage_b: bool) {
        let dg = g.dg();
        let delta = st.delta;
        for v in 0..self.ptr.len() {
            self.ptr[v] = g.adj_off[v];
            self.dead[v] = false;
        }
        for &s in srcs {
            while st.is_source(s, dg, stage_b) && !self.dead[s as usize] {
                self.stack.clear();
                self.edges.clear();
                self.stack.push(s);
                self.on[s as usize] = true;
                let mut found = false;
                while let Some(&u) = self.stack.last() {
                    if u != s && st.is_target(u, dg, stage_b) {
                        found = true;
                        break;
                    }
                    let uu = u as usize;
                    let end = g.adj_off[uu + 1];
                    let mut next: Option<(u32, usize, bool)> = None;
                    while self.ptr[uu] < end {
                        let e = g.adj[self.ptr[uu] as usize];
                        let c = (e >> 1) as usize;
                        let back = e & 1 == 1;
                        let w = if back { g.ctail[c] } else { g.chead[c] };
                        let wu = w as usize;
                        if !self.dead[wu]
                            && !self.on[wu]
                            && rc(g, cost, lb, &st.x, &st.pi, delta, c, back).is_some_and(|r| r <= ADM_EPS)
                        {
                            next = Some((w, c, back));
                            break;
                        }
                        self.ptr[uu] += 1;
                    }
                    match next {
                        Some((w, c, back)) => {
                            self.on[w as usize] = true;
                            self.stack.push(w);
                            self.edges.push((c, back));
                        }
                        None => {
                            self.dead[uu] = true;
                            self.on[uu] = false;
                            self.stack.pop();
                            self.edges.pop();
                        }
                    }
                }
                for &v in &self.stack {
                    self.on[v as usize] = false;
                }
                if !found {
                    break;
                }
                let t = self.stack.last().copied().unwrap_or(s);
                let path = std::mem::take(&mut self.edges);
                st.push(g, s, t, &path);
                self.edges = path;
            }
        }
    }
}

/// Reduced cost of a delta-residual edge of the double cover, or None if it does not exist.
#[allow(clippy::too_many_arguments)]
#[inline]
fn rc(g: &CGraph, cost: &[ArcCost], lb: &[u32], x: &[i64], pi: &[f64], delta: i64, c: usize, back: bool) -> Option<f64> {
    let a = g.carc[c] as usize;
    let xc = x[c];
    let (tl, hd) = (g.ctail[c] as usize, g.chead[c] as usize);
    let f = |v: i64| 0.5 * cost[a].f(v as f64);
    if back {
        if xc - delta < lb[a] as i64 {
            return None;
        }
        Some(f(xc - delta) - f(xc) + pi[hd] - pi[tl])
    } else {
        if xc + delta > N_MAX {
            return None;
        }
        Some(f(xc + delta) - f(xc) + pi[tl] - pi[hd])
    }
}

/// Step 2: rounds every half-integral arc (odd `n2`) by +-1 along closed walks.
fn parity_round(g: &CGraph, cost: &[ArcCost], n2: &mut [i64]) {
    let nw = g.nq as usize + 1; // walk graph nodes: constraint nodes and G = nq
    let odd: Vec<usize> = (0..g.na).filter(|&a| g.arc_copy[a] != NONE && n2[a] % 2 != 0).collect();
    if odd.is_empty() {
        return;
    }
    let ends = |a: usize| g.ends[a].unwrap_or([(0, 1), (0, 1)]);
    let mut deg = vec![0u32; nw + 1];
    for &a in &odd {
        for (q, _) in ends(a) {
            deg[q as usize] += 1;
        }
    }
    let mut off = vec![0u32; nw + 1];
    for v in 0..nw {
        off[v + 1] = off[v] + deg[v];
    }
    let mut fill = off.clone();
    let mut adj = vec![(0u32, 0u8); off[nw] as usize];
    for &a in &odd {
        for (e, (q, _)) in ends(a).iter().enumerate() {
            adj[fill[*q as usize] as usize] = (a as u32, e as u8);
            fill[*q as usize] += 1;
        }
    }
    let mut used = vec![false; g.na];
    let mut ptr: Vec<u32> = off[..nw].to_vec();
    let gq = g.nq as usize;
    let starts = std::iter::once(gq).chain(0..gq);
    let mut walk: Vec<(usize, usize)> = Vec::new();
    for start in starts {
        loop {
            walk.clear();
            let mut cur = start;
            loop {
                while ptr[cur] < off[cur + 1] && used[adj[ptr[cur] as usize].0 as usize] {
                    ptr[cur] += 1;
                }
                if ptr[cur] == off[cur + 1] {
                    break;
                }
                let (a, e) = adj[ptr[cur] as usize];
                used[a as usize] = true;
                walk.push((a as usize, e as usize));
                cur = ends(a as usize)[1 - e as usize].0 as usize;
            }
            if walk.is_empty() {
                break;
            }
            // Alternating half changes: delta_{i+1} = -sigma_in sigma_out delta_i.
            let mut d = Vec::with_capacity(walk.len());
            let mut di: i64 = 1;
            for (i, &(a, e)) in walk.iter().enumerate() {
                if i > 0 {
                    let (pa, pe) = walk[i - 1];
                    let s_in = ends(pa)[1 - pe].1 as i64;
                    let s_out = ends(a)[e].1 as i64;
                    di *= -s_in * s_out;
                }
                d.push(di);
            }
            let (mut cp, mut cm) = (0.0, 0.0);
            for (&(a, _), &di) in walk.iter().zip(&d) {
                cp += cost[a].f((n2[a] + di) as f64 * 0.5);
                cm += cost[a].f((n2[a] - di) as f64 * 0.5);
            }
            let sgn = if cp <= cm { 1 } else { -1 };
            for (&(a, _), &di) in walk.iter().zip(&d) {
                n2[a] += sgn * di;
            }
        }
    }
}

/// Cost of moving arc `a` from `n` by `+1` (forward) or `-1` (backward), or None.
#[inline]
fn step_cost(cost: &[ArcCost], lb: &[u32], n: &[i64], a: usize, back: bool) -> Option<f64> {
    let x = n[a];
    if back {
        if x - 1 < lb[a] as i64 {
            return None;
        }
        Some(cost[a].f((x - 1) as f64) - cost[a].f(x as f64))
    } else {
        if x + 1 > N_MAX {
            return None;
        }
        Some(cost[a].f((x + 1) as f64) - cost[a].f(x as f64))
    }
}

/// Net change per arc of a double cover path (sorted by arc).
fn net_change(g: &CGraph, path: &[(usize, bool)]) -> Vec<(usize, i64)> {
    let mut ch: Vec<(usize, i64)> = path.iter().map(|&(c, back)| (g.carc[c] as usize, if back { -1 } else { 1 })).collect();
    ch.sort_unstable();
    let mut out: Vec<(usize, i64)> = Vec::with_capacity(ch.len());
    for (a, d) in ch {
        match out.last_mut() {
            Some(l) if l.0 == a => l.1 += d,
            _ => out.push((a, d)),
        }
    }
    out.retain(|&(_, d)| d != 0);
    out
}

fn apply_change(g: &CGraph, n: &mut [i64], b: &mut [i64], ch: &[(usize, i64)]) {
    for &(a, d) in ch {
        n[a] += d;
        if let Some(e) = g.ends[a] {
            for (q, s) in e {
                if q != g.nq {
                    b[q as usize] += s as i64 * d;
                }
            }
        }
    }
}

/// Step 3: removes every constraint imbalance with +-1 moves along cheapest double cover
/// paths from the imbalanced node to a compensating node or to the ground.
fn integer_repair(
    g: &CGraph,
    cost: &[ArcCost],
    lb: &[u32],
    n: &mut [i64],
    ctl: Option<&Ctl>,
    dij: &mut Dijkstra,
) -> Result<(), AqError> {
    let mut b = g.balances(n);
    if b.iter().all(|&x| x == 0) {
        return Ok(());
    }
    let cap = 4 * g.na.max(1);
    let mut moves = 0usize;
    let mut blocked = vec![false; g.na];
    let dg = g.dg();
    for q in 0..g.nq {
        while b[q as usize] != 0 {
            moves += 1;
            if moves > cap {
                return Err(err(ErrKind::Infeasible, format!("integer repair exceeded {cap} moves")));
            }
            let bq = b[q as usize];
            let src = g.dnode(q, if bq > 0 { -1 } else { 1 });
            let mut applied = false;
            for _attempt in 0..g.na.max(1) {
                let r = {
                    let (nn, bb, bl) = (&*n, &b, &blocked);
                    dij.run(
                        g,
                        &[src],
                        |c, back| {
                            let a = g.carc[c] as usize;
                            if back && bl[a] {
                                return None;
                            }
                            step_cost(cost, lb, nn, a, back)
                        },
                        |v| {
                            if v == dg {
                                return true;
                            }
                            let q2 = v / 2;
                            let sv: i64 = if v % 2 == 0 { 1 } else { -1 };
                            let b2 = bb[q2 as usize];
                            b2 != 0 && b2.signum() == sv && (q2 != q || b2.abs() >= 2)
                        },
                    )
                };
                dij.tick(ctl)?;
                let Some((tg, _)) = r else {
                    return Err(err(ErrKind::Infeasible, format!("constraint node {q} cannot be balanced")));
                };
                let ch = net_change(g, &dij.path(g, tg).1);
                if let Some(&(a, _)) = ch.iter().find(|&&(a, d)| n[a] + d < lb[a] as i64) {
                    blocked[a] = true;
                    continue;
                }
                apply_change(g, n, &mut b, &ch);
                applied = true;
                break;
            }
            if !applied {
                return Err(err(ErrKind::Infeasible, format!("constraint node {q} cannot be balanced within bounds")));
            }
            blocked.iter_mut().for_each(|x| *x = false);
        }
    }
    Ok(())
}

/// Step 4 (polish): applies improving closed walks of the double cover. A closed walk keeps
/// every constraint balanced, so each applied walk is a feasible move that lowers the true
/// objective. Negative cycles of the marginal costs are found with a FIFO Bellman-Ford from
/// a virtual source joined to every node, checking the predecessor graph for a cycle after
/// every pass.
fn polish(
    g: &CGraph,
    cost: &[ArcCost],
    lb: &[u32],
    n: &mut [i64],
    ctl: Option<&Ctl>,
) -> Result<(), AqError> {
    let dn = g.dn();
    if g.ctail.is_empty() {
        return Ok(());
    }
    let mut b = vec![0i64; g.nq as usize];
    let mut dist = vec![0.0f64; dn];
    let mut pred = vec![NONE; dn];
    let mut mark = vec![0u32; dn];
    // Work budget in edge relaxations: the polish never costs more than a few flow solves.
    let mut work = POLISH_WORK * g.adj.len() + 4096;
    for _round in 0..MAX_POLISH {
        dist.iter_mut().for_each(|d| *d = 0.0);
        pred.iter_mut().for_each(|p| *p = NONE);
        let mut found: Option<u32> = None;
        for _pass in 0..dn {
            if work < g.adj.len() {
                return Ok(());
            }
            work -= g.adj.len();
            if let Some(c) = ctl {
                c.check()?;
            }
            let mut changed = false;
            for v in 0..dn {
                let dv = dist[v];
                for &e in &g.adj[g.adj_off[v] as usize..g.adj_off[v + 1] as usize] {
                    let c = (e >> 1) as usize;
                    let back = e & 1 == 1;
                    let w = if back { g.ctail[c] } else { g.chead[c] } as usize;
                    let Some(sc) = step_cost(cost, lb, n, g.carc[c] as usize, back) else { continue };
                    let nd = dv + sc;
                    if nd < dist[w] - 1e-9 {
                        dist[w] = nd;
                        pred[w] = e;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            // Cycle in the predecessor graph?
            mark.iter_mut().for_each(|m| *m = 0);
            'outer: for s in 0..dn {
                if mark[s] != 0 {
                    continue;
                }
                let mut v = s;
                let tag = s as u32 + 1;
                while mark[v] == 0 {
                    mark[v] = tag;
                    let e = pred[v];
                    if e == NONE {
                        break;
                    }
                    let c = (e >> 1) as usize;
                    v = if e & 1 == 1 { g.chead[c] } else { g.ctail[c] } as usize;
                    if mark[v] == tag {
                        found = Some(v as u32);
                        break 'outer;
                    }
                }
            }
            if found.is_some() {
                break;
            }
        }
        let Some(start) = found else { return Ok(()) };
        // Extract the cycle.
        let mut cyc: Vec<(usize, bool)> = Vec::new();
        let mut v = start;
        loop {
            let e = pred[v as usize];
            let c = (e >> 1) as usize;
            let back = e & 1 == 1;
            cyc.push((c, back));
            v = if back { g.chead[c] } else { g.ctail[c] };
            if v == start || cyc.len() > dn {
                break;
            }
        }
        let ch = net_change(g, &cyc);
        if ch.is_empty() || ch.iter().any(|&(a, d)| n[a] + d < lb[a] as i64 || n[a] + d > N_MAX) {
            return Ok(());
        }
        let gain: f64 = ch.iter().map(|&(a, d)| cost[a].f((n[a] + d) as f64) - cost[a].f(n[a] as f64)).sum();
        if gain >= -1e-12 {
            return Ok(());
        }
        apply_change(g, n, &mut b, &ch);
        debug_assert!(b.iter().all(|&x| x == 0));
    }
    Ok(())
}

//! Arc costs (DESIGN 6.11 table) and the bi-directed constraint graph with its double cover.

use aq_core::{AqError, ArcKind, ErrKind, NONE, Stage, TArc, TMesh};

/// Largest quantized length the solver accepts for one arc (guards integer overflow).
pub(crate) const N_MAX: i64 = 1 << 30;

pub(crate) fn err(kind: ErrKind, msg: impl Into<String>) -> AqError {
    AqError::new(Stage::Quantize, kind, msg)
}

/// Weight of an arc (DESIGN 6.11 table): separatrix and seed traces 1; feature, boundary,
/// guide and loop arcs 4; symmetry 8; wrap arcs 8.
pub fn arc_weight(a: &TArc) -> f64 {
    if a.wrap {
        return 8.0;
    }
    match a.kind {
        ArcKind::Separatrix | ArcKind::SeedTrace => 1.0,
        ArcKind::Feature | ArcKind::Boundary | ArcKind::Guide | ArcKind::Loop => 4.0,
        ArcKind::Symmetry => 8.0,
    }
}

/// Lower bound of an arc from the DESIGN 6.11 table at scale `sigma`.
pub fn table_lb(a: &TArc, sigma: f64, allow_collapse: bool) -> u32 {
    if a.wrap {
        return 0;
    }
    match a.kind {
        ArcKind::Separatrix | ArcKind::SeedTrace => u32::from(!allow_collapse),
        ArcKind::Feature | ArcKind::Boundary | ArcKind::Guide | ArcKind::Loop => {
            u32::from(sigma * a.len >= 0.5)
        }
        ArcKind::Symmetry => 1,
    }
}

/// Target length of an arc at scale `sigma` (0 for a wrap arc).
pub fn arc_target(a: &TArc, sigma: f64) -> f64 {
    if a.wrap { 0.0 } else { sigma * a.len }
}

/// Convex cost `f_a(n) = weight (n - target)^2 / max(sigma l, 0.5)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArcCost {
    pub w: f64,
    pub target: f64,
    pub inv_d: f64,
}

impl ArcCost {
    pub fn new(a: &TArc, sigma: f64) -> Self {
        let d = (sigma * a.len).max(0.5);
        ArcCost { w: arc_weight(a), target: arc_target(a, sigma), inv_d: 1.0 / d }
    }

    #[inline]
    pub fn f(&self, n: f64) -> f64 {
        let d = n - self.target;
        self.w * d * d * self.inv_d
    }

    /// Unconstrained integer minimizer clamped to `[lb, N_MAX]` (ties round up).
    pub fn argmin(&self, lb: u32) -> i64 {
        let r = (self.target + 0.5).floor();
        let r = if r.is_finite() { r.clamp(0.0, N_MAX as f64) as i64 } else { 0 };
        r.max(lb as i64)
    }
}

/// The objective `sum_a f_a(n_a)` of DESIGN 6.11 at scale `sigma`.
pub fn objective(t: &TMesh, sigma: f64, n: &[u32]) -> f64 {
    t.arcs.iter().zip(n).map(|(a, &x)| ArcCost::new(a, sigma).f(x as f64)).sum()
}

/// One end of a bi-directed edge: constraint node q (or the ground `g`) and its sign.
pub(crate) type End = (u32, i8);

/// The bi-directed constraint graph of a T-mesh and its double cover.
///
/// Constraint nodes: `q = 2 P + axis`; the ground node G is `q = nq`. Double cover nodes:
/// `2 q` (sign +) and `2 q + 1` (sign -) for `q < nq`, and one merged ground node `2 nq`
/// (G has a free balance, so its two copies are the same node).
pub(crate) struct CGraph {
    pub na: usize,
    pub nq: u32,
    /// Per arc: its two ends, or None for a free arc (no constraint).
    pub ends: Vec<Option<[End; 2]>>,
    /// Per arc: first of its two double cover copies, or NONE for a free arc.
    pub arc_copy: Vec<u32>,
    /// Per copy: tail, head (double cover node ids) and arc.
    pub ctail: Vec<u32>,
    pub chead: Vec<u32>,
    pub carc: Vec<u32>,
    /// CSR adjacency over double cover nodes; entry = `copy * 2 + dir` with dir 0 for an
    /// outgoing copy (forward residual) and 1 for an incoming copy (backward residual).
    pub adj_off: Vec<u32>,
    pub adj: Vec<u32>,
}

impl CGraph {
    pub fn dn(&self) -> usize {
        2 * self.nq as usize + 1
    }

    pub fn dg(&self) -> u32 {
        2 * self.nq
    }

    pub fn dnode(&self, q: u32, s: i8) -> u32 {
        if q == self.nq { 2 * self.nq } else { 2 * q + u32::from(s < 0) }
    }

    /// Builds the graph. Errors: an arc with more than 2 occurrences (`InvalidInput`).
    pub fn new(t: &TMesh) -> Result<Self, AqError> {
        let na = t.arcs.len();
        let nq = u32::try_from(2 * t.patches.len())
            .map_err(|_| err(ErrKind::Budget, "too many patches"))?;
        let mut occ: Vec<Vec<End>> = vec![Vec::new(); na];
        for (p, pa) in t.patches.iter().enumerate() {
            for (k, side) in pa.sides.iter().enumerate() {
                let q = 2 * p as u32 + (k % 2) as u32;
                let s: i8 = if k < 2 { 1 } else { -1 };
                for &(a, _) in side {
                    let o = occ
                        .get_mut(a as usize)
                        .ok_or_else(|| err(ErrKind::InvalidInput, format!("arc {a} out of range")))?;
                    o.push((q, s));
                }
            }
        }
        let mut ends = Vec::with_capacity(na);
        for (a, o) in occ.iter().enumerate() {
            let e = match o.len() {
                0 => None,
                1 => Some([o[0], (nq, -o[0].1)]),
                2 => {
                    if o[0].0 == o[1].0 && o[0].1 == -o[1].1 {
                        None
                    } else {
                        Some([o[0], o[1]])
                    }
                }
                m => {
                    return Err(err(ErrKind::InvalidInput, format!("arc {a} occurs {m} times on patch sides")));
                }
            };
            ends.push(e);
        }
        let mut g = CGraph {
            na,
            nq,
            ends,
            arc_copy: vec![NONE; na],
            ctail: Vec::new(),
            chead: Vec::new(),
            carc: Vec::new(),
            adj_off: Vec::new(),
            adj: Vec::new(),
        };
        for a in 0..na {
            if let Some([(u, su), (v, sv)]) = g.ends[a] {
                g.arc_copy[a] = g.ctail.len() as u32;
                let (t1, h1) = (g.dnode(u, su), g.dnode(v, -sv));
                let (t2, h2) = (g.dnode(v, sv), g.dnode(u, -su));
                g.ctail.extend([t1, t2]);
                g.chead.extend([h1, h2]);
                g.carc.extend([a as u32, a as u32]);
            }
        }
        let dn = g.dn();
        let mut deg = vec![0u32; dn + 1];
        for c in 0..g.ctail.len() {
            deg[g.ctail[c] as usize] += 1;
            deg[g.chead[c] as usize] += 1;
        }
        let mut off = vec![0u32; dn + 1];
        for v in 0..dn {
            off[v + 1] = off[v] + deg[v];
        }
        let mut fill = off.clone();
        let mut adj = vec![0u32; off[dn] as usize];
        for c in 0..g.ctail.len() {
            let (tl, hd) = (g.ctail[c] as usize, g.chead[c] as usize);
            adj[fill[tl] as usize] = 2 * c as u32;
            fill[tl] += 1;
            adj[fill[hd] as usize] = 2 * c as u32 + 1;
            fill[hd] += 1;
        }
        g.adj_off = off;
        g.adj = adj;
        Ok(g)
    }

    /// Constraint balances `b[q] = sum sigma n` over the non-ground ends.
    pub fn balances(&self, n: &[i64]) -> Vec<i64> {
        let mut b = vec![0i64; self.nq as usize];
        for (a, e) in self.ends.iter().enumerate() {
            if let Some(e) = e {
                for &(q, s) in e {
                    if q != self.nq {
                        b[q as usize] += s as i64 * n[a];
                    }
                }
            }
        }
        b
    }
}

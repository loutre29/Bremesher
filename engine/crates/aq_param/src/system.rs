//! Variables, corner expressions and exact constraint elimination (DESIGN 6.9 "Variables",
//! "Constraints").
//!
//! Scalar variable ids: the rep wedge of vertex v owns `2v` (u) and `2v + 1` (v); cut arc a
//! owns `2 n_verts + 2a` and `2 n_verts + 2a + 1` (its translation `t_a`, side A, C10). So every
//! translation id is above every vertex id, and "prefer translations, then the newest
//! variable" is simply "prefer the highest id".
//!
//! [`Raw`] holds everything that does not depend on the elimination: the corner expressions
//! over raw variables, the closure, chain and loop rows and the gauge vertices. [`System`] is
//! one elimination variant (a loop selection): corner expressions substituted over the free
//! variables, penalty rows and the translation expressions.

use aq_core::cross::{edge_family, raw_to_combed_branch};
use aq_core::param::FAMILY_NONE;
use aq_core::{AqError, Ctl, CutGraph, FeatureKind, LoopCand, NONE, Topology};
use aq_linalg::{Elim, RowOutcome};

use crate::ProxyRef;
use crate::expr::{self, Lin, Lin2, SymTr};

/// One loop closure row (DESIGN 6.9 constraint 3).
#[derive(Clone, Debug)]
pub(crate) struct LoopRow {
    /// Index into the LoopCand slice.
    pub idx: u32,
    pub ring: bool,
    /// The transversal translation row over raw t variables; `None` when the rotation part of
    /// the loop's holonomy is not the identity (the loop is skipped).
    pub row: Option<Lin>,
}

/// Row statistics of one elimination.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RowStats {
    pub rows: u32,
    pub redundant: u32,
    pub inconsistent: u32,
    /// Raw rows that became penalties.
    pub penalties: Vec<Lin>,
}

/// Adds a homogeneous row to `el` after substituting the current expressions and dividing by
/// the gcd of its coefficients (an equivalent integer row; `Elim` itself only looks for +-1
/// coefficients, so `2 x + 2 y = 0` would otherwise be `NoUnitPivot`). See CHANGES.md: the
/// last closure of a genus-0 surface with index +-1 cones only reduces to such a row.
pub(crate) fn add_row_gcd(el: &mut Elim, row: &Lin, prefer: &dyn Fn(u32) -> i32) -> RowOutcome {
    let Some(mut sub) = el.substitute(row) else {
        return RowOutcome::NoUnitPivot;
    };
    if sub.is_empty() {
        return RowOutcome::Redundant;
    }
    let g = sub.iter().fold(0u64, |g, &(_, c)| gcd(g, c.unsigned_abs()));
    if g > 1 {
        for e in sub.iter_mut() {
            e.1 /= g as i64;
        }
    }
    el.add_row(&sub, prefer)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

impl RowStats {
    fn record(&mut self, out: RowOutcome, row: &Lin) {
        self.rows += 1;
        match out {
            RowOutcome::Eliminated { .. } => {}
            RowOutcome::Redundant => self.redundant += 1,
            RowOutcome::Inconsistent => self.inconsistent += 1,
            RowOutcome::NoUnitPivot => self.penalties.push(row.clone()),
        }
    }
}

/// Everything that does not depend on the elimination.
#[derive(Clone, Debug)]
pub(crate) struct Raw {
    pub nv: usize,
    pub n_arcs: usize,
    pub n_vars: usize,
    /// First translation variable id (`2 nv`).
    pub t_base: u32,
    /// CSR over the slots `2c + k` (corner c, coordinate k): raw expression of the corner.
    pub slot_off: Vec<u32>,
    pub slot_terms: Vec<(u32, i64)>,
    /// Non-trivial closure rows (2 per interior vertex whose fan crosses the cut).
    pub closure: Vec<Lin>,
    /// Chain iso rows.
    pub chain: Vec<Lin>,
    pub loops: Vec<LoopRow>,
    /// Lowest vertex of each component (gauge).
    pub gauge: Vec<u32>,
    /// Per halfedge: 0 U, 1 V, FAMILY_NONE.
    pub he_family: Vec<u8>,
    /// Per vertex: the face of its rep wedge (its lowest-index face), NONE if isolated.
    pub rep_face: Vec<u32>,
}

impl Raw {
    pub fn slot(&self, s: usize) -> &[(u32, i64)] {
        &self.slot_terms[self.slot_off[s] as usize..self.slot_off[s + 1] as usize]
    }

    pub fn t_var(&self, arc: usize, k: usize) -> u32 {
        self.t_base + (2 * arc + k) as u32
    }

    /// Pivot preference for closure and loop rows: translations first, then the newest
    /// variable (the highest id).
    pub fn prefer_t(v: u32) -> i32 {
        v.min(i32::MAX as u32) as i32
    }

    /// Pivot preference for chain rows: vertex variables first (newest first), then
    /// translations. Keeps translations free of chain vertices (see CHANGES.md).
    pub fn prefer_x(&self) -> impl Fn(u32) -> i32 + '_ {
        let tb = self.t_base;
        move |v: u32| {
            if v < tb {
                v.min(i32::MAX as u32) as i32
            } else {
                i32::MIN / 2 + (v - tb).min(i32::MAX as u32 / 4) as i32
            }
        }
    }
}

/// Symbolic transition of halfedge h (crossing from face(h) to face(twin h)); identity off the
/// cut.
fn he_transition(raw_t_base: u32, cut: &CutGraph, h: u32) -> Result<Option<(u8, Lin2)>, AqError> {
    if !cut.is_cut[h as usize] {
        return Ok(None);
    }
    let (a, side) = cut.he_arc[h as usize];
    if a == NONE || a as usize >= cut.arcs.len() {
        return Err(AqError::invalid_input(format!(
            "cut halfedge {h} belongs to no arc"
        )));
    }
    let ra = cut.arcs[a as usize].r % 4;
    let ta = expr::var2(raw_t_base + 2 * a);
    if side > 0 {
        Ok(Some((ra, ta)))
    } else {
        // Side B: the inverse of side A, r' = (4 - r) mod 4, t' = -R(r) t_a (C10).
        let rt = expr::rot(ra, &ta)?;
        Ok(Some((
            (4 - ra) % 4,
            [expr::neg(&rt[0])?, expr::neg(&rt[1])?],
        )))
    }
}

/// Kinds whose chains get iso rows in M1 (Guide chains are soft field alignment only).
pub(crate) fn iso_kind(k: FeatureKind) -> bool {
    matches!(
        k,
        FeatureKind::Hard | FeatureKind::GuideHard | FeatureKind::Boundary | FeatureKind::Symmetry
    )
}

impl Raw {
    pub fn build(
        px: &ProxyRef,
        cut: &CutGraph,
        loops: &[LoopCand],
        ctl: &Ctl,
    ) -> Result<Raw, AqError> {
        let (m, t, fr, fs) = (px.mesh, px.topo, px.frames, px.feats);
        let nv = m.pos.len();
        let nf = m.tris.len();
        let nh = 3 * nf;
        let n_arcs = cut.arcs.len();
        let n_vars = 2 * nv + 2 * n_arcs;
        if n_vars >= (i32::MAX / 2) as usize {
            return Err(AqError::budget(format!(
                "aq_param: {n_vars} variables is too many"
            )));
        }
        let t_base = (2 * nv) as u32;
        // Corner expressions by fan walks.
        let mut corner: Vec<Lin2> = aq_core::alloc::try_vec(nh, [Vec::new(), Vec::new()])?;
        let mut assigned = aq_core::alloc::try_vec(nh, false)?;
        let mut rep_face = aq_core::alloc::try_vec(nv, NONE)?;
        let mut closure: Vec<Lin> = Vec::new();
        let mut fan: Vec<u32> = Vec::new();
        let mut comp: Vec<SymTr> = Vec::new();
        for v in 0..nv as u32 {
            if v % 4096 == 0 {
                ctl.check()?;
            }
            fan.clear();
            fan.extend(t.fan_out(v));
            let n = fan.len();
            if n == 0 {
                continue;
            }
            let Some(i0) = (0..n).min_by_key(|&i| fan[i] / 3) else {
                continue;
            };
            rep_face[v as usize] = fan[i0] / 3;
            comp.clear();
            comp.resize(n, SymTr::identity());
            let step = |c: &SymTr, h: u32| -> Result<SymTr, AqError> {
                match he_transition(t_base, cut, h)? {
                    None => Ok(c.clone()),
                    Some((r, tt)) => c.then(r, &tt),
                }
            };
            if t.is_bnd_vert[v as usize] {
                for i in i0..n - 1 {
                    comp[i + 1] = step(&comp[i], Topology::prev(fan[i]))?;
                }
                for i in (1..=i0).rev() {
                    comp[i - 1] = step(&comp[i], fan[i])?;
                }
            } else {
                for k in 1..=n {
                    let i = (i0 + k - 1) % n;
                    let nx = (i0 + k) % n;
                    let c = step(&comp[i], Topology::prev(fan[i]))?;
                    if nx == i0 {
                        // Closure: x = R(-r) x + tau, i.e. x - (R(-r) x + tau) = 0.
                        let x = expr::var2(2 * v);
                        let y = c.apply(&x)?;
                        for k2 in 0..2 {
                            let row = expr::sub(&x[k2], &y[k2])?;
                            if !row.is_empty() {
                                closure.push(row);
                            }
                        }
                    } else {
                        comp[nx] = c;
                    }
                }
            }
            let x = expr::var2(2 * v);
            for i in 0..n {
                let c = fan[i] as usize;
                if assigned[c] {
                    return Err(AqError::invalid_input(format!(
                        "corner {c} reached twice by the fan walks"
                    )));
                }
                assigned[c] = true;
                corner[c] = comp[i].apply(&x)?;
            }
        }
        if let Some(c) = assigned.iter().position(|a| !a) {
            return Err(AqError::invalid_input(format!(
                "corner {c} is not reached by its vertex fan"
            )));
        }
        let mut slot_off: Vec<u32> = aq_core::alloc::try_with_capacity(2 * nh + 1)?;
        let mut slot_terms: Vec<(u32, i64)> = aq_core::alloc::try_with_capacity(2 * nh + nh / 2)?;
        slot_off.push(0);
        for c in corner.iter() {
            for k in 0..2 {
                slot_terms.extend_from_slice(&c[k]);
                if slot_terms.len() >= u32::MAX as usize {
                    return Err(AqError::budget("aq_param: corner expressions too large"));
                }
                slot_off.push(slot_terms.len() as u32);
            }
        }
        drop(corner);
        let slot = |s: usize| -> &[(u32, i64)] {
            &slot_terms[slot_off[s] as usize..slot_off[s + 1] as usize]
        };
        // Chain iso rows and families.
        let mut he_family = aq_core::alloc::try_vec(nh, FAMILY_NONE)?;
        let mut chain: Vec<Lin> = Vec::new();
        for c in &fs.chains {
            if !iso_kind(c.kind) {
                continue;
            }
            for &h in &c.hes {
                if h as usize >= nh {
                    return Err(AqError::invalid_input(format!(
                        "chain halfedge {h} out of range"
                    )));
                }
                let f = (h / 3) as usize;
                let fam = edge_family(fr, m, h, cut.theta_combed[f]);
                he_family[h as usize] = fam;
                let tw = t.twin[h as usize];
                if tw != NONE {
                    he_family[tw as usize] = fam ^ (cut.r[h as usize] & 1);
                }
                // U edge lies on a v-iso line: constrain coordinate v (index 1); V: u.
                let q = if fam == 0 { 1 } else { 0 };
                let a = slot(2 * h as usize + q).to_vec();
                let b = slot(2 * Topology::next(h) as usize + q).to_vec();
                let row = expr::sub(&a, &b)?;
                if !row.is_empty() {
                    chain.push(row);
                }
            }
        }
        // Loop rows.
        let mut lrows = Vec::with_capacity(loops.len());
        for (i, l) in loops.iter().enumerate() {
            l.validate(m, t)
                .map_err(|e| AqError::invalid_input(format!("loop {i}: {}", e.msg)))?;
            let mut s = SymTr::identity();
            for &h in &l.hes {
                if let Some((r, tt)) = he_transition(t_base, cut, h)? {
                    s = s.then(r, &tt)?;
                }
            }
            let row = if !s.r.is_multiple_of(4) {
                None
            } else {
                let k = raw_to_combed_branch(l.branch, cut.comb_offset[l.start_face as usize]);
                // Even combed branch: the loop runs along u, so its v translation must vanish.
                Some(if k.is_multiple_of(2) {
                    s.t[1].clone()
                } else {
                    s.t[0].clone()
                })
            };
            lrows.push(LoopRow {
                idx: i as u32,
                ring: l.ring,
                row,
            });
        }
        // Gauge: the first corner of the lowest face of each component. WP-16 round 2: this was
        // the lowest vertex; a local repair that encloses the pinned vertex (the cone-chart
        // region repair) cannot give the same result as one where it moves, so the gauge now
        // follows the faces and not the vertex numbering (gauge_invariance_genus2).
        let mut gauge = vec![NONE; t.n_comp as usize];
        for (fi, tri) in m.tris.iter().enumerate() {
            let c = t.face_comp[fi] as usize;
            if c < gauge.len() && gauge[c] == NONE {
                gauge[c] = tri[0];
            }
        }
        gauge.retain(|&v| v != NONE);
        Ok(Raw {
            nv,
            n_arcs,
            n_vars,
            t_base,
            slot_off,
            slot_terms,
            closure,
            chain,
            loops: lrows,
            gauge,
            he_family,
            rep_face,
        })
    }

    /// Base elimination: closure rows then chain rows (DESIGN 6.9 constraints 1 and 2).
    pub fn base_elim(&self, ctl: &Ctl) -> Result<(Elim, RowStats), AqError> {
        let mut el = Elim::try_new(self.n_vars)?;
        let mut st = RowStats::default();
        for (i, row) in self.closure.iter().enumerate() {
            if i % 1024 == 0 {
                ctl.check()?;
            }
            let out = add_row_gcd(&mut el, row, &Raw::prefer_t);
            st.record(out, row);
        }
        let px = self.prefer_x();
        for (i, row) in self.chain.iter().enumerate() {
            if i % 1024 == 0 {
                ctl.check()?;
            }
            let out = add_row_gcd(&mut el, row, &px);
            st.record(out, row);
        }
        Ok((el, st))
    }
}

/// One elimination variant, ready to assemble.
#[derive(Clone, Debug)]
pub struct System {
    pub(crate) elim: Elim,
    /// Number of free variables (unknowns of the solve).
    pub n_free: usize,
    /// CSR over the slots `2c + k`: corner expression over free indices.
    pub(crate) s_off: Vec<u32>,
    pub(crate) s_idx: Vec<u32>,
    pub(crate) s_c: Vec<i64>,
    /// Penalty rows over free indices.
    pub(crate) pen: Vec<Vec<(u32, i64)>>,
    /// Per arc and coordinate (`2a + k`): translation over free indices.
    pub(crate) t_off: Vec<u32>,
    pub(crate) t_idx: Vec<u32>,
    pub(crate) t_c: Vec<i64>,
    /// Per vertex: free indices of its rep variables (NONE when not free).
    pub(crate) rep_free: Vec<[u32; 2]>,
    /// Per free index: used by the corners of more than one vertex, by a penalty row or by a
    /// translation expression.
    pub(crate) shared: Vec<bool>,
    pub(crate) stats: RowStats,
    /// Loops whose row was Eliminated or Redundant.
    pub(crate) loops_exact: Vec<u32>,
    /// Selected loops that were not exact (holonomy, penalty or inconsistent).
    pub(crate) loops_not_exact: u32,
}

impl System {
    /// Builds a variant: the base elimination plus the selected loop rows, the gauge, then
    /// every free variable used by no corner and no penalty row is pinned to zero.
    pub(crate) fn build(
        px: &ProxyRef,
        raw: &Raw,
        base: &Elim,
        base_stats: &RowStats,
        loop_sel: &[u32],
        ctl: &Ctl,
    ) -> Result<System, AqError> {
        let mut el = base.clone();
        let mut st = base_stats.clone();
        let mut loops_exact = Vec::new();
        let mut not_exact = 0u32;
        for &li in loop_sel {
            let Some(lr) = raw.loops.get(li as usize) else {
                continue;
            };
            let Some(row) = &lr.row else {
                not_exact += 1;
                continue;
            };
            if row.is_empty() {
                loops_exact.push(lr.idx);
                continue;
            }
            let out = add_row_gcd(&mut el, row, &Raw::prefer_t);
            st.record(out, row);
            match out {
                RowOutcome::Eliminated { .. } | RowOutcome::Redundant => loops_exact.push(lr.idx),
                _ => not_exact += 1,
            }
        }
        for &g in &raw.gauge {
            for k in 0..2 {
                let var = 2 * g + k;
                if !el.is_pinned(var) {
                    st.rows += 1;
                    match el.pin(var, true) {
                        RowOutcome::Redundant => st.redundant += 1,
                        RowOutcome::Inconsistent => st.inconsistent += 1,
                        RowOutcome::NoUnitPivot => st.penalties.push(vec![(var, 1)]),
                        RowOutcome::Eliminated { .. } => {}
                    }
                }
            }
        }
        ctl.check()?;
        // Substitute corners and penalty rows (over variable ids).
        let n_slots = raw.slot_off.len() - 1;
        let mut sub_off: Vec<u32> = aq_core::alloc::try_with_capacity(n_slots + 1)?;
        let mut sub_terms: Vec<(u32, i64)> =
            aq_core::alloc::try_with_capacity(raw.slot_terms.len())?;
        sub_off.push(0);
        let ovf = || AqError::numerical("aq_param: integer overflow while substituting");
        for s in 0..n_slots {
            if s % 65536 == 0 {
                ctl.check()?;
            }
            let e = el.substitute(raw.slot(s)).ok_or_else(ovf)?;
            sub_terms.extend_from_slice(&e);
            if sub_terms.len() >= u32::MAX as usize {
                return Err(AqError::budget(
                    "aq_param: substituted expressions too large",
                ));
            }
            sub_off.push(sub_terms.len() as u32);
        }
        let mut pen_sub: Vec<Lin> = Vec::with_capacity(st.penalties.len());
        for r in &st.penalties {
            let e = el.substitute(r).ok_or_else(ovf)?;
            if !e.is_empty() {
                pen_sub.push(e);
            }
        }
        // Pin unused free variables.
        let mut used = aq_core::alloc::try_vec(raw.n_vars, false)?;
        for &(v, _) in sub_terms.iter().chain(pen_sub.iter().flatten()) {
            used[v as usize] = true;
        }
        for v in 0..raw.n_vars as u32 {
            if !used[v as usize] && el.is_free(v) {
                el.pin(v, true);
            }
        }
        let n_free = el.n_free();
        let fidx = |v: u32| el.free_index(v).unwrap_or(NONE);
        let mut s_idx: Vec<u32> = aq_core::alloc::try_with_capacity(sub_terms.len())?;
        let mut s_c: Vec<i64> = aq_core::alloc::try_with_capacity(sub_terms.len())?;
        for &(v, c) in &sub_terms {
            let i = fidx(v);
            if i == NONE {
                return Err(AqError::internal(format!(
                    "aq_param: variable {v} in a corner is not free"
                )));
            }
            s_idx.push(i);
            s_c.push(c);
        }
        let pen: Vec<Vec<(u32, i64)>> = pen_sub
            .iter()
            .map(|r| r.iter().map(|&(v, c)| (fidx(v), c)).collect())
            .collect();
        if pen.iter().flatten().any(|e| e.0 == NONE) {
            return Err(AqError::internal(
                "aq_param: penalty row over a non-free variable",
            ));
        }
        let mut t_off = vec![0u32];
        let mut t_idx = Vec::new();
        let mut t_c = Vec::new();
        for a in 0..raw.n_arcs {
            for k in 0..2 {
                for &(v, c) in el.expr(raw.t_var(a, k)) {
                    let i = fidx(v);
                    if i == NONE {
                        return Err(AqError::internal(
                            "aq_param: translation over a non-free variable",
                        ));
                    }
                    t_idx.push(i);
                    t_c.push(c);
                }
                t_off.push(t_idx.len() as u32);
            }
        }
        let mut rep_free = aq_core::alloc::try_vec(raw.nv, [NONE; 2])?;
        for (v, rf) in rep_free.iter_mut().enumerate() {
            for k in 0..2 {
                let var = (2 * v + k) as u32;
                if el.is_free(var) {
                    rf[k] = fidx(var);
                }
            }
        }
        // Sharing of free variables.
        let mut owner = aq_core::alloc::try_vec(n_free, NONE)?;
        let mut shared = aq_core::alloc::try_vec(n_free, false)?;
        let m = px.mesh;
        for (f, tri) in m.tris.iter().enumerate() {
            for (k, &v) in tri.iter().enumerate() {
                let c = 3 * f + k;
                for s in [2 * c, 2 * c + 1] {
                    for &i in &s_idx[sub_off[s] as usize..sub_off[s + 1] as usize] {
                        let o = &mut owner[i as usize];
                        if *o == NONE {
                            *o = v;
                        } else if *o != v {
                            shared[i as usize] = true;
                        }
                    }
                }
            }
        }
        for &(i, _) in pen.iter().flatten() {
            shared[i as usize] = true;
        }
        for &i in &t_idx {
            shared[i as usize] = true;
        }
        Ok(System {
            elim: el,
            n_free,
            s_off: sub_off,
            s_idx,
            s_c,
            pen,
            t_off,
            t_idx,
            t_c,
            rep_free,
            shared,
            stats: st,
            loops_exact,
            loops_not_exact: not_exact,
        })
    }

    /// Terms of slot `2c + k` over free indices.
    #[inline]
    pub(crate) fn slot(&self, s: usize) -> (&[u32], &[i64]) {
        let (a, b) = (self.s_off[s] as usize, self.s_off[s + 1] as usize);
        (&self.s_idx[a..b], &self.s_c[a..b])
    }

    /// Translation `t_a` coordinate k over free indices.
    #[inline]
    pub(crate) fn t_terms(&self, a: usize, k: usize) -> (&[u32], &[i64]) {
        let s = 2 * a + k;
        let (lo, hi) = (self.t_off[s] as usize, self.t_off[s + 1] as usize);
        (&self.t_idx[lo..hi], &self.t_c[lo..hi])
    }

    /// Per face corner uv (f64) at the free values y.
    pub fn corners(&self, y: &[f64]) -> Vec<[[f64; 2]; 3]> {
        let nf = (self.s_off.len() - 1) / 6;
        let mut out = vec![[[0.0; 2]; 3]; nf];
        for (f, o) in out.iter_mut().enumerate() {
            for k in 0..3 {
                for q in 0..2 {
                    let (ix, cs) = self.slot(2 * (3 * f + k) + q);
                    o[k][q] = ix
                        .iter()
                        .zip(cs)
                        .map(|(&i, &c)| c as f64 * y[i as usize])
                        .sum::<f64>()
                        + 0.0;
                }
            }
        }
        out
    }

    /// Translation of arc a at the free values y (f64).
    pub fn translation(&self, a: usize, y: &[f64]) -> [f64; 2] {
        let mut t = [0.0; 2];
        for (k, tk) in t.iter_mut().enumerate() {
            let (ix, cs) = self.t_terms(a, k);
            *tk = ix
                .iter()
                .zip(cs)
                .map(|(&i, &c)| c as f64 * y[i as usize])
                .sum::<f64>()
                + 0.0;
        }
        t
    }

    /// True when the rep variables of v are both free and used only by the corners of v, so
    /// they can be moved without changing anything outside the fan of v.
    pub fn movable(&self, v: u32) -> bool {
        let Some(r) = self.rep_free.get(v as usize) else {
            return false;
        };
        r.iter().all(|&i| i != NONE && !self.shared[i as usize])
    }

    /// The elimination state (expressions over variable ids).
    pub fn elim(&self) -> &Elim {
        &self.elim
    }

    /// Selected loops whose row was not exact (holonomy, penalty or inconsistent).
    pub fn loops_not_exact(&self) -> u32 {
        self.loops_not_exact
    }

    /// Free indices of the rep variables of v (NONE when eliminated or pinned).
    pub fn rep_free(&self, v: u32) -> [u32; 2] {
        self.rep_free.get(v as usize).copied().unwrap_or([NONE; 2])
    }
}

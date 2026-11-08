//! S3 sizing (DESIGN 6.5 "Sizing"), all in quad units.
//!
//! * **Base:** `g(v) = clamp((1 + kappa_hat/kappa_ref)^(-alpha) * rho(v)^(-1/2), 1/4, 4)` with
//!   `alpha = 0.7 * adaptive` and kappa_ref the area-weighted median of kappa_hat. With
//!   adaptive = 0 and no density, g = 1 exactly.
//! * **Tube cap:** at a vertex of a face with `conf > 0.5` and relative anisotropy `> 0.6`,
//!   `h <= 2pi / (ring_min |k1|)` (the smallest bound over such faces).
//! * **Gradation** (Persson): `log h(y) <= log h(x) + grad |xy| / h(x)` on every proxy edge,
//!   enforced by a Dijkstra pass in increasing h ([`grade`]).
//! * **Normalization:** the caps and the gradation are absolute (quad units), so they are
//!   applied *after* the scale: `h = Grade(min(s g, cap))`, and s is found by a safeguarded
//!   root search so that `n_pred = sum_f A_f / h_f^2 = N` (see CHANGES.md, WP-07). A final
//!   rescale by `sqrt(n/N)` (a factor within about 1e-14 of 1) makes the count exact to
//!   rounding. Without active caps or gradation this is the plain `s = sqrt(sum A/g^2 / N)`.
//! * **Infeasible caps:** when the caps alone need more than N quads (a whole component that
//!   is a thin tube), the count wins: the capped, graded shape is scaled up uniformly (and
//!   regraded) until it gives N, and [`SizingReport::cap_relax`] reports the factor.
//! * `h_face` is the mean of the 3 vertex values (exactly the common value when they agree).
//! * **Thickness cap (M2, WP-20):** with `thickness = true`, `h <= max(pi d / ring_min, f)`
//!   with d the shape diameter ([`crate::thick`]). The floor `f >= G_MIN` is the budget: the
//!   smallest value for which the thickness caps spend at most `thick_budget * N` quads beyond
//!   the field without them ([`thickness_floor`], at least `thick_floor_min`), raised to the
//!   engine's global value (`thick_floor`) when that is higher. A floor trims the thinnest parts first, which keeps the edge length
//!   spread smallest for a given budget. See CHANGES.md, WP-20.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f64::consts::{PI, TAU};

use aq_core::math;
use aq_core::{AqError, Ctl, CurvField, ErrKind, SizeField, Stage};

use crate::curv::anisotropy;
use crate::{ProxyRef, bad, check_proxy, restamp};

/// Parameters of [`sizing`] (DESIGN 6.5 API).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizingParams {
    /// Target quad count N_c of the component.
    pub n_target: f64,
    /// `adaptive_size` in [0, 1]; alpha = 0.7 adaptive. M1 runs with 0.
    pub adaptive: f64,
    /// Minimum quads around a tube (tube cap); 0 disables the cap.
    pub ring_min: u32,
    /// Gradation rate (Persson), 0.25 by default.
    pub grad: f64,
    /// Thickness cap (M2, WP-20): `h <= max(pi d / ring_min, floor)` with d the shape diameter.
    pub thickness: bool,
    /// Largest share of `n_target` the thickness caps may spend beyond the field without them
    /// (sets the floor solved in the component).
    pub thick_budget: f64,
    /// A lower bound of the floor of the thickness caps in the units of the proxy (the
    /// engine's global floor across components), or None. The floor used is the larger of
    /// this and the one solved in the component, so a component never spends more than its
    /// own budget on thickness (a global floor set by a thin neighbour does not move quads
    /// inside a thick component).
    pub thick_floor: Option<f64>,
    /// Lowest floor solved in the component, in h: the thickness cap never asks for quads
    /// smaller than this, which bounds the edge length spread it adds.
    pub thick_floor_min: f64,
}

impl SizingParams {
    /// DESIGN 13 defaults (adaptive 0 as in M1, ring_min 8, grad 0.25, no thickness cap; with
    /// it, a budget of [`THICK_BUDGET`] and the floor solved in the component).
    pub fn new(n_target: f64) -> Self {
        SizingParams {
            n_target,
            adaptive: 0.0,
            ring_min: 8,
            grad: 0.25,
            thickness: false,
            thick_budget: THICK_BUDGET,
            thick_floor: None,
            thick_floor_min: THICK_FLOOR_MIN,
        }
    }
}

/// Tube cap condition: conf above this (DESIGN 6.5).
pub const TUBE_CONF: f64 = 0.5;
/// Tube cap condition: relative anisotropy above this (DESIGN 6.5).
pub const TUBE_ANISO: f64 = 0.6;
/// Clamp of the base g (DESIGN 13).
pub const G_MIN: f64 = 0.25;
pub const G_MAX: f64 = 4.0;
/// Floor of kappa_ref (quad units) so that a flat component does not divide by zero.
pub const KAPPA_REF_MIN: f64 = 1e-3;
/// Default share of N the thickness caps may spend beyond the uncapped field (WP-20).
pub const THICK_BUDGET: f64 = 0.1;
/// Default lowest floor of the thickness caps inside a component, in h (WP-20).
pub const THICK_FLOOR_MIN: f64 = 0.7;

/// What [`sizing_with`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SizingReport {
    /// Final scale s of `h = Grade(min(s g, cap))` (before the final rescale).
    pub scale: f64,
    /// The final rescale factor `sqrt(n/N)` (1 within about 1e-14).
    pub final_rescale: f64,
    /// Area-weighted median of kappa_hat (floored at [`KAPPA_REF_MIN`]).
    pub kappa_ref: f64,
    /// Vertices with a finite tube cap.
    pub tube_verts: u32,
    /// Vertices whose final h is set by their tube cap.
    pub capped_verts: u32,
    /// Vertices lowered by the gradation.
    pub graded_verts: u32,
    /// The caps alone needed more than N quads; the count won.
    pub cap_infeasible: bool,
    /// Uniform factor applied on top of the caps when infeasible (1 otherwise).
    pub cap_relax: f64,
    /// Evaluations of the count during the root search.
    pub evaluations: u32,
    /// Always false since WP-20 (the thickness cap is implemented); kept for the report schema.
    pub thickness_skipped: bool,
    /// Vertices with a finite shape diameter (thickness = true).
    pub thick_verts: u32,
    /// Vertices whose cap (before the gradation) is set by the thickness cap below `s g`.
    pub thick_capped: u32,
    /// The floor of the thickness caps ([`G_MIN`] when the budget is not binding; infinite
    /// when no thickness cap could be afforded).
    pub thick_floor: f64,
    /// Share of N spent by the thickness caps (estimate without gradation, at the final s).
    pub thick_share: f64,
    /// Milliseconds spent on the shape diameter.
    pub thick_ms: f64,
    /// The density was given and used.
    pub density_used: bool,
}

/// Undirected proxy edges in CSR form with their lengths, neighbours sorted by id.
pub(crate) struct Adjacency {
    off: Vec<usize>,
    nbr: Vec<(u32, f64)>,
}

impl Adjacency {
    pub(crate) fn build(px: &ProxyRef) -> Result<Self, AqError> {
        let m = px.mesh;
        let t = px.topo;
        let nv = m.pos.len();
        let nh = t.twin.len();
        let mut deg = aq_core::alloc::try_vec(nv + 1, 0usize).map_err(restamp)?;
        for h in 0..nh as u32 {
            if t.is_canonical(h) {
                deg[t.tail(m, h) as usize] += 1;
                deg[t.head(m, h) as usize] += 1;
            }
        }
        let mut off = aq_core::alloc::try_vec(nv + 1, 0usize).map_err(restamp)?;
        for v in 0..nv {
            off[v + 1] = off[v] + deg[v];
        }
        let mut fill = off.clone();
        let mut nbr = aq_core::alloc::try_vec(off[nv], (0u32, 0.0f64)).map_err(restamp)?;
        for h in 0..nh as u32 {
            if t.is_canonical(h) {
                let a = t.tail(m, h);
                let b = t.head(m, h);
                let l = math::dist(m.pos[a as usize], m.pos[b as usize]);
                nbr[fill[a as usize]] = (b, l);
                fill[a as usize] += 1;
                nbr[fill[b as usize]] = (a, l);
                fill[b as usize] += 1;
            }
        }
        for v in 0..nv {
            nbr[off[v]..off[v + 1]].sort_by_key(|x| x.0);
        }
        Ok(Adjacency { off, nbr })
    }

    pub(crate) fn neighbours(&self, v: usize) -> &[(u32, f64)] {
        &self.nbr[self.off[v]..self.off[v + 1]]
    }
}

/// Persson gradation in place: afterwards `log x[y] <= log x[v] + grad |vy| / x[v]` on every
/// edge (v, y). Dijkstra in increasing x (ties by vertex id); values only decrease. Returns
/// the number of lowered vertices. All values must be positive (infinity allowed).
pub(crate) fn grade(adj: &Adjacency, x: &mut [f64], grad: f64, ctl: &Ctl) -> Result<u32, AqError> {
    if grad.is_infinite() {
        return Ok(0);
    }
    let nv = x.len();
    // Positive f64 bit patterns order like the values.
    let init: Vec<Reverse<(u64, u32)>> = (0..nv).map(|v| Reverse((x[v].to_bits(), v as u32))).collect();
    let mut heap = BinaryHeap::from(init);
    let mut done = aq_core::alloc::try_vec(nv, false).map_err(restamp)?;
    let mut lowered = aq_core::alloc::try_vec(nv, false).map_err(restamp)?;
    let mut pops = 0usize;
    while let Some(Reverse((bits, v))) = heap.pop() {
        pops += 1;
        if pops.is_multiple_of(4096) {
            ctl.check()?;
        }
        let v = v as usize;
        if done[v] || bits != x[v].to_bits() {
            continue;
        }
        done[v] = true;
        let xv = x[v];
        if !xv.is_finite() {
            continue;
        }
        for &(w, l) in adj.neighbours(v) {
            let w = w as usize;
            if done[w] {
                continue;
            }
            let b = xv * (grad * l / xv).exp();
            if b < x[w] {
                x[w] = b;
                lowered[w] = true;
                heap.push(Reverse((b.to_bits(), w as u32)));
            }
        }
    }
    Ok(lowered.iter().filter(|b| **b).count() as u32)
}

/// Mean of three values, exactly `a` when all three agree (infinite when one is).
#[inline]
fn mean3(a: f64, b: f64, c: f64) -> f64 {
    if a == b && a == c { a } else { (a + b + c) / 3.0 }
}

/// Neumaier-compensated sum.
fn csum(it: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in it {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

/// `sum_f A_f / h_f^2` with h_f the mean of the vertex values.
fn count_of(px: &ProxyRef, hv: &[f64]) -> f64 {
    let m = px.mesh;
    csum(m.tris.iter().zip(&px.frames.area).map(|(t, a)| {
        let h = mean3(hv[t[0] as usize], hv[t[1] as usize], hv[t[2] as usize]);
        a / (h * h)
    }))
}

/// Area-weighted (barycentric vertex area) median of kappa_hat.
fn weighted_median(px: &ProxyRef, kh: &[f64]) -> f64 {
    let m = px.mesh;
    let mut w = vec![0.0f64; m.pos.len()];
    for (t, a) in m.tris.iter().zip(&px.frames.area) {
        for &v in t {
            w[v as usize] += a / 3.0;
        }
    }
    let mut pairs: Vec<(f64, f64)> = kh.iter().copied().zip(w).filter(|(_, w)| *w > 0.0).collect();
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    let mut acc = 0.0;
    for (k, wk) in &pairs {
        acc += wk;
        if acc >= 0.5 * total {
            return *k;
        }
    }
    pairs[pairs.len() - 1].0
}

/// Thickness cap of one vertex: `max(raw, floor)` (infinite raw or floor stays infinite).
#[inline]
fn thick_cap(raw: f64, floor: f64) -> f64 {
    if raw.is_finite() && floor.is_finite() { raw.max(floor) } else { f64::INFINITY }
}

/// Barycentric vertex areas.
fn vertex_areas(px: &ProxyRef) -> Vec<f64> {
    let mut w = vec![0.0f64; px.mesh.pos.len()];
    for (t, a) in px.mesh.tris.iter().zip(&px.frames.area) {
        for &v in t {
            w[v as usize] += a / 3.0;
        }
    }
    w
}

/// `sum_v A_v / min(s base_v, tube_v, thick_v)^2` with `thick_v = thick_cap(raw_v, floor)`.
/// `tube` or `raw` may be empty (no such cap). No gradation: this is the estimate used to pick
/// the floor and to plan budgets.
fn count_v(av: &[f64], base: &[f64], tube: &[f64], s: f64, raw: &[f64], floor: f64) -> f64 {
    csum((0..av.len()).map(|v| {
        let mut h = s * base.get(v).copied().unwrap_or(1.0);
        if let Some(&t) = tube.get(v) {
            h = h.min(t);
        }
        if let Some(&r) = raw.get(v) {
            h = h.min(thick_cap(r, floor));
        }
        av[v] / (h * h)
    }))
}

/// s with `count_v(s) = n` (log space); None when even s = infinity leaves at least n quads
/// (the caps alone use the whole budget). Returns the upper end of a bracket, so
/// `count_v(s) <= n`, within 1e-10 in ln s or 1e-12 n in the count.
///
/// M2 integration: Illinois regula falsi on `count / n - 1` against ln s instead of a plain
/// bisection. ln count is close to linear in ln s (slope -2 where nothing is capped), so a
/// solve takes about 5 count evaluations instead of about 35; [`thickness_floor`] runs one
/// solve per floor it tries, which made the floor search about 150 ms per S3 pass on
/// basemesh_female's 60k proxy vertices.
fn solve_scale(av: &[f64], base: &[f64], tube: &[f64], raw: &[f64], floor: f64, n: f64) -> Option<f64> {
    if count_v(av, base, tube, f64::INFINITY, raw, floor) >= n {
        return None;
    }
    let s0 = (count_v(av, base, &[], 1.0, &[], 1.0) / n).sqrt();
    if !(s0.is_finite() && s0 > 0.0) {
        return None;
    }
    let g = |x: f64| count_v(av, base, tube, x.exp(), raw, floor) / n - 1.0;
    let (mut lo, mut hi) = (s0.ln(), s0.ln());
    // Bracket: count(lo) > n >= count(hi). Caps only lower h, so count(s0) >= n.
    let (mut glo, mut ghi) = (0.0, g(hi));
    for _ in 0..200 {
        if ghi <= 0.0 {
            break;
        }
        (lo, glo) = (hi, ghi);
        hi += std::f64::consts::LN_2;
        ghi = g(hi);
    }
    if !(glo > 0.0 && ghi <= 0.0) {
        return Some(hi.exp());
    }
    // (wlo, whi): the Illinois weights (the end values, halved when that end is kept twice).
    let (mut wlo, mut whi) = (glo, ghi);
    let mut side = 0i8;
    for _ in 0..200 {
        if hi - lo < 1e-10 || ghi >= -1e-12 {
            break;
        }
        let x = (lo * whi - hi * wlo) / (whi - wlo);
        let x = if x > lo && x < hi { x } else { 0.5 * (lo + hi) };
        let gx = g(x);
        if gx > 0.0 {
            (lo, wlo) = (x, gx);
            if side == 1 {
                whi *= 0.5;
            }
            side = 1;
        } else {
            (hi, ghi, whi) = (x, gx, gx);
            if side == -1 {
                wlo *= 0.5;
            }
            side = -1;
        }
    }
    Some(hi.exp())
}

/// Thickness budget (WP-20): the smallest floor `f >= G_MIN` for which the thickness caps
/// `max(raw, f)` spend at most `budget * n` quads beyond the field with the tube caps only,
/// both at the scale s that gives n quads ([`count_v`] estimates, without gradation). Returns
/// [`G_MIN`] when that already holds, and infinity (no thickness cap) when the tube caps alone
/// exceed n or no finite floor fits. `av` are vertex areas (any units, with `n` quads of size
/// about `sqrt(sum av / n)`), `base` the base g, `tube` the tube caps (may be empty), `raw` =
/// pi d / ring_min (infinite where there is no diameter). Bisection on ln f to 0.1%.
pub fn thickness_floor(
    av: &[f64],
    base: &[f64],
    tube: &[f64],
    raw: &[f64],
    n: f64,
    budget: f64,
    ctl: &Ctl,
) -> Result<f64, AqError> {
    let nv = av.len();
    if base.len() != nv || (!tube.is_empty() && tube.len() != nv) || raw.len() != nv {
        return Err(bad("thickness_floor: slice lengths differ"));
    }
    if !(n.is_finite() && n > 0.0) {
        return Err(bad(format!("thickness_floor: n {n} must be finite and > 0")));
    }
    if solve_scale(av, base, tube, &[], 1.0, n).is_none() {
        return Ok(f64::INFINITY);
    }
    // The quads spent by the thickness caps at floor f: n minus what the field without them
    // takes at the same s; false when the caps alone use the whole budget.
    let ok = |f: f64| -> bool {
        match solve_scale(av, base, tube, raw, f, n) {
            Some(s) => n - count_v(av, base, tube, s, &[], 1.0) <= budget * n,
            None => false,
        }
    };
    if ok(G_MIN) {
        return Ok(G_MIN);
    }
    // The largest finite raw cap: a floor above it switches every thickness cap off.
    let top = raw.iter().copied().filter(|x| x.is_finite()).fold(G_MIN, f64::max);
    let mut lo = G_MIN.ln(); // fails
    let mut hi = lo;
    let mut found = false;
    for _ in 0..80 {
        ctl.check()?;
        hi += std::f64::consts::LN_2;
        if ok(hi.exp()) {
            found = true;
            break;
        }
        lo = hi;
        if hi.exp() > 2.0 * top {
            break;
        }
    }
    if !found {
        return Ok(f64::INFINITY);
    }
    for _ in 0..40 {
        ctl.check()?;
        let mid = 0.5 * (lo + hi);
        if ok(mid.exp()) {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1e-3 {
            break;
        }
    }
    Ok(hi.exp())
}

/// Per-group quad demand of a global plan (WP-20, used by the engine across components):
/// `sum_v A_v / min(s, max(raw_v, floor))^2` per group, where `group[v]` is the group of
/// vertex v and s solves the total = n (no tube caps, base 1, no gradation). Returns
/// `(demand per group, s)`, or None when the caps alone exceed n.
pub fn thickness_demand(av: &[f64], raw: &[f64], group: &[u32], n_groups: usize, floor: f64, n: f64) -> Option<(Vec<f64>, f64)> {
    let base = vec![1.0; av.len()];
    let s = solve_scale(av, &base, &[], raw, floor, n)?;
    let mut d = vec![0.0f64; n_groups];
    for v in 0..av.len() {
        let h = s.min(thick_cap(raw[v], floor));
        if let Some(x) = d.get_mut(group[v] as usize) {
            *x += av[v] / (h * h);
        }
    }
    Some((d, s))
}

/// S3 sizing (DESIGN 6.5 API): no density (rho = 1).
pub fn sizing(px: &ProxyRef, c: &CurvField, p: &SizingParams, ctl: &Ctl) -> Result<SizeField, AqError> {
    sizing_with(px, c, p, None, ctl).map(|(s, _)| s)
}

/// Monotone family of size fields indexed by one positive parameter.
struct Family<'a> {
    px: &'a ProxyRef<'a>,
    adj: &'a Adjacency,
    base: &'a [f64],
    cap: &'a [f64],
    grad: f64,
    /// true: `Grade(min(s base, cap))`; false: `Grade(s base)`.
    capped: bool,
    evals: u32,
}

impl Family<'_> {
    fn eval(&mut self, s: f64, out: &mut [f64], ctl: &Ctl) -> Result<f64, AqError> {
        self.evals += 1;
        ctl.check()?;
        for v in 0..out.len() {
            let x = s * self.base[v];
            out[v] = if self.capped { x.min(self.cap[v]) } else { x };
        }
        grade(self.adj, out, self.grad, ctl)?;
        Ok(count_of(self.px, out))
    }

    /// Solves `count(s) = n` for s >= lo, where count is non-increasing and count(lo) >= n.
    /// Returns `None` when count stays above n up to lo * 2^64.
    fn solve(&mut self, lo: f64, n: f64, buf: &mut [f64], ctl: &Ctl) -> Result<Option<f64>, AqError> {
        let f = |c: f64| c.ln() - n.ln();
        let c_lo = self.eval(lo, buf, ctl)?;
        let mut fa = f(c_lo);
        if fa <= 1e-15 {
            return Ok(Some(lo));
        }
        // count(s) decreases to count(infinity) (only the caps and what they grade remain): when
        // that limit is not below n, no scale reaches n.
        if self.capped && f(self.eval(f64::INFINITY, buf, ctl)?) >= 0.0 {
            return Ok(None);
        }
        let mut a = lo.ln();
        let mut b = a;
        let mut fb = fa;
        for _ in 0..64 {
            b += std::f64::consts::LN_2;
            fb = f(self.eval(b.exp(), buf, ctl)?);
            if fb <= 0.0 {
                break;
            }
            a = b;
            fa = fb;
        }
        if fb > 0.0 {
            return Ok(None);
        }
        if fb == 0.0 {
            return Ok(Some(b.exp()));
        }
        // Illinois regula falsi on (ln s, ln count - ln n); fa > 0 > fb.
        let (mut best, mut best_f) = (b, fb.abs());
        let mut side = 0i8;
        for _ in 0..200 {
            let c = if fa != fb { (a * fb - b * fa) / (fb - fa) } else { 0.5 * (a + b) };
            let c = if c > a.min(b) && c < a.max(b) { c } else { 0.5 * (a + b) };
            let fc = f(self.eval(c.exp(), buf, ctl)?);
            if fc.abs() < best_f {
                best = c;
                best_f = fc.abs();
            }
            if fc.abs() <= 1e-15 || (b - a).abs() <= 1e-15 * a.abs().max(1.0) {
                break;
            }
            if fc > 0.0 {
                a = c;
                fa = fc;
                if side == 1 {
                    fb *= 0.5;
                }
                side = 1;
            } else {
                b = c;
                fb = fc;
                if side == -1 {
                    fa *= 0.5;
                }
                side = -1;
            }
        }
        Ok(Some(best.exp()))
    }
}

/// [`sizing`] with an optional per-vertex density multiplier rho (already raised to
/// density_strength by aq_prep), returning a [`SizingReport`].
///
/// Errors (stage Curv): `InvalidInput` for mismatched sizes or bad parameters (n_target not
/// finite and positive, adaptive outside [0, 1], grad not positive, a density not finite and
/// positive), `Numerical` when the result is not finite, `Cancelled` / `Timeout`.
pub fn sizing_with(
    px: &ProxyRef,
    c: &CurvField,
    p: &SizingParams,
    density: Option<&[f64]>,
    ctl: &Ctl,
) -> Result<(SizeField, SizingReport), AqError> {
    sizing_diam(px, c, p, density, None, ctl)
}

/// [`sizing_with`] with the shape diameters already computed (`diam`, one per vertex; only
/// read when `p.thickness`); None computes them here.
pub fn sizing_diam(
    px: &ProxyRef,
    c: &CurvField,
    p: &SizingParams,
    density: Option<&[f64]>,
    diam: Option<&[f64]>,
    ctl: &Ctl,
) -> Result<(SizeField, SizingReport), AqError> {
    check_proxy(px)?;
    let m = px.mesh;
    let nv = m.pos.len();
    let nf = m.tris.len();
    c.validate(nf, nv).map_err(|e| bad(format!("sizing: invalid CurvField: {}", e.msg)))?;
    if !(p.n_target.is_finite() && p.n_target > 0.0) {
        return Err(bad(format!("sizing: n_target {} must be finite and > 0", p.n_target)));
    }
    if !(0.0..=1.0).contains(&p.adaptive) {
        return Err(bad(format!("sizing: adaptive {} outside [0, 1]", p.adaptive)));
    }
    if p.grad.is_nan() || p.grad <= 0.0 {
        return Err(bad(format!("sizing: grad {} must be > 0", p.grad)));
    }
    if !(p.thick_floor_min.is_finite() && p.thick_floor_min > 0.0) {
        return Err(bad(format!("sizing: thick_floor_min {} must be finite and > 0", p.thick_floor_min)));
    }
    if !(p.thick_budget.is_finite() && p.thick_budget >= 0.0) {
        return Err(bad(format!("sizing: thick_budget {} must be finite and >= 0", p.thick_budget)));
    }
    if let Some(f) = p.thick_floor
        && (f.is_nan() || f <= 0.0)
    {
        return Err(bad(format!("sizing: thick_floor {f} must be > 0")));
    }
    if let Some(d) = diam
        && d.len() != nv
    {
        return Err(bad(format!("sizing: {} shape diameters for {nv} vertices", d.len())));
    }
    if let Some(d) = density {
        if d.len() != nv {
            return Err(bad(format!("sizing: density has {} values for {nv} vertices", d.len())));
        }
        if let Some(v) = d.iter().position(|x| !(x.is_finite() && *x > 0.0)) {
            return Err(bad(format!("sizing: density[{v}] = {} is not finite and positive", d[v])));
        }
    }
    let area_total = csum(px.frames.area.iter().copied());
    if !(area_total.is_finite() && area_total > 0.0) {
        return Err(bad("sizing: the proxy has no area"));
    }
    ctl.progress(Stage::Curv, 0.0);
    let mut rep = SizingReport {
        thick_floor: G_MIN,
        density_used: density.is_some(),
        cap_relax: 1.0,
        ..Default::default()
    };

    // Base g.
    let alpha = 0.7 * p.adaptive;
    rep.kappa_ref = weighted_median(px, &c.kappa_hat).max(KAPPA_REF_MIN);
    let mut base = aq_core::alloc::try_vec(nv, 1.0f64).map_err(restamp)?;
    if alpha > 0.0 || density.is_some() {
        for v in 0..nv {
            let a = if alpha > 0.0 { (1.0 + c.kappa_hat[v] / rep.kappa_ref).powf(-alpha) } else { 1.0 };
            let r = density.map_or(1.0, |d| d[v].powf(-0.5));
            base[v] = (a * r).clamp(G_MIN, G_MAX);
        }
    }

    // Tube cap per vertex.
    let mut cap = aq_core::alloc::try_vec(nv, f64::INFINITY).map_err(restamp)?;
    if p.ring_min > 0 {
        for f in 0..nf {
            let (k1, k2) = (c.k1[f], c.k2[f]);
            if c.conf[f] > TUBE_CONF && anisotropy(k1, k2) > TUBE_ANISO && k1.abs() > 0.0 {
                let b = TAU / (p.ring_min as f64 * k1.abs());
                for &v in &m.tris[f] {
                    let cv = &mut cap[v as usize];
                    *cv = cv.min(b);
                }
            }
        }
    }
    rep.tube_verts = cap.iter().filter(|x| x.is_finite()).count() as u32;
    let adj = Adjacency::build(px)?;
    ctl.check()?;

    let n = p.n_target;
    // Thickness cap (WP-20): max(pi d / ring_min, floor), with the floor set so that the caps
    // spend at most thick_budget N quads (or the engine's global floor).
    if p.thickness && p.ring_min > 0 {
        let t0 = std::time::Instant::now();
        let owned;
        let d = match diam {
            Some(d) => d,
            None => {
                owned = crate::thick::shape_diameter(px, ctl)?;
                &owned[..]
            }
        };
        rep.thick_ms = t0.elapsed().as_secs_f64() * 1e3;
        let raw: Vec<f64> = d.iter().map(|x| PI * x / p.ring_min as f64).collect();
        rep.thick_verts = raw.iter().filter(|x| x.is_finite()).count() as u32;
        if rep.thick_verts > 0 {
            let av = vertex_areas(px);
            let local = thickness_floor(&av, &base, &cap, &raw, n, p.thick_budget, ctl)?.max(p.thick_floor_min);
            let floor = p.thick_floor.map_or(local, |f| local.max(f));
            rep.thick_floor = floor;
            if floor.is_finite() {
                let s_est = solve_scale(&av, &base, &cap, &raw, floor, n).unwrap_or(1.0);
                let tube = cap.clone();
                for v in 0..nv {
                    let tc = thick_cap(raw[v], floor);
                    if tc < cap[v] {
                        if tc < s_est * base[v] {
                            rep.thick_capped += 1;
                        }
                        cap[v] = tc;
                    }
                }
                let free = count_v(&av, &base, &tube, s_est, &[], 1.0);
                rep.thick_share = ((n - free) / n).max(0.0);
            }
        }
    }
    let count_g = count_of(px, &base);
    let s0 = (count_g / n).sqrt();
    let mut hv = aq_core::alloc::try_vec(nv, 0.0f64).map_err(restamp)?;
    let mut fam = Family { px, adj: &adj, base: &base, cap: &cap, grad: p.grad, capped: true, evals: 0 };
    let s = match fam.solve(s0, n, &mut hv, ctl)? {
        Some(s) => s,
        None => {
            // The caps alone exceed N: scale the capped, graded shape uniformly.
            rep.cap_infeasible = true;
            let s_star = cap
                .iter()
                .zip(&base)
                .filter(|(c, _)| c.is_finite())
                .map(|(c, b)| c / b)
                .fold(s0, f64::max);
            let mut shape = aq_core::alloc::try_vec(nv, 0.0f64).map_err(restamp)?;
            fam.eval(s_star, &mut shape, ctl)?;
            let evals = fam.evals;
            let mut fam2 =
                Family { px, adj: &adj, base: &shape, cap: &cap, grad: p.grad, capped: false, evals };
            let lam = fam2.solve(1.0, n, &mut hv, ctl)?.ok_or_else(|| {
                AqError::new(Stage::Curv, ErrKind::Numerical, "sizing: no scale reaches the target count")
            })?;
            rep.cap_relax = lam;
            fam2.eval(lam, &mut hv, ctl)?;
            rep.evaluations = fam2.evals;
            rep.scale = s_star;
            lam * s_star
        }
    };
    if !rep.cap_infeasible {
        fam.eval(s, &mut hv, ctl)?;
        rep.evaluations = fam.evals;
        rep.scale = s;
    }
    // Statistics (feasible case): vertices set by their cap, vertices lowered by the grading.
    if !rep.cap_infeasible {
        for v in 0..nv {
            let x = s * base[v];
            if cap[v] < x && hv[v] == cap[v] {
                rep.capped_verts += 1;
            } else if hv[v] < x.min(cap[v]) {
                rep.graded_verts += 1;
            }
        }
    }
    // Exact count: rescale by sqrt(n/N) (a factor within about 1e-14 of 1 after the search).
    rep.final_rescale = 1.0;
    for _ in 0..3 {
        let cnt = count_of(px, &hv);
        if (cnt - n).abs() <= 1e-14 * n {
            break;
        }
        let r = (cnt / n).sqrt();
        rep.final_rescale *= r;
        for h in hv.iter_mut() {
            *h *= r;
        }
    }
    let h_face: Vec<f64> = m.tris.iter().map(|t| mean3(hv[t[0] as usize], hv[t[1] as usize], hv[t[2] as usize])).collect();
    if let Some(v) = hv.iter().chain(&h_face).position(|h| !(h.is_finite() && *h > 0.0)) {
        return Err(AqError::new(Stage::Curv, ErrKind::Numerical, format!("sizing: size entry {v} is not finite and positive")));
    }
    let n_pred = csum(h_face.iter().zip(&px.frames.area).map(|(h, a)| a / (h * h)));
    let sf = SizeField { h_vert: hv, h_face, n_pred };
    sf.validate(px.frames, nv).map_err(restamp)?;
    ctl.progress(Stage::Curv, 1.0);
    Ok((sf, rep))
}

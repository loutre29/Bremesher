//! DESIGN 6.6 step 1: constraints, smoothness and alignment weights.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f64::consts::FRAC_PI_2;

use aq_core::alloc::try_vec;
use aq_core::cross::wrap_pi;
use aq_core::math;
use aq_core::tol::clamp_weight;
use aq_core::{AqError, Ctl, CurvField, FeatureKind, NONE, SizeField, Stage, V3};
use aq_linalg::{CTriplets, Jacobi, complex_to_real, pcg_ctl, real_to_complex, try_hermitian_to_real};

use crate::{EIGEN_CONF, FieldParams, ProxyRef, inv};

/// `theta - snap(theta, a)`: the offset of theta from the nearest branch `a + k pi/2`, in
/// [-pi/4, pi/4].
#[inline]
pub(crate) fn branch_offset(theta: f64, a: f64) -> f64 {
    theta - snap(theta, a)
}

/// The branch `a + k pi/2` nearest theta.
#[inline]
pub(crate) fn snap(theta: f64, a: f64) -> f64 {
    let k = ((theta - a) / FRAC_PI_2).round();
    a + k * FRAC_PI_2
}

/// Everything the solvers and the edits need, computed once per call.
pub(crate) struct Setup {
    /// Per face: theta fixed by a constraint edge.
    pub hard: Vec<bool>,
    /// Per hard face: the exact angle (face frame) of its constraint edge.
    pub edge_angle: Vec<f64>,
    /// Per hard face: the segment-smoothed chain tangent (face frame), the Knoppel constraint.
    pub psi_s: Vec<f64>,
    /// Per halfedge: smoothness weight `w_e` (same value on both halves, 0 on the boundary).
    pub w: Vec<f64>,
    /// Per face: curvature alignment weight `lambda_f A_f conf_f` and its target angle phi_f.
    pub a_c: Vec<f64>,
    pub phi: Vec<f64>,
    /// Per face: the curvature confidence clamped to [0, 1] (the eigen path test of step 2).
    pub conf: Vec<f64>,
    /// Per face: the coherent confidence (see [`coherence`]), used by the alignment weight and
    /// the low-confidence cancellation disc.
    pub conf_eff: Vec<f64>,
    /// Per face: h_f^2.
    pub h2: Vec<f64>,
    /// Per face: soft guide weight `20 guide_w / h^2 A_f` and its target angle (0 when no guide).
    pub a_g: Vec<f64>,
    pub psi_g: Vec<f64>,
    /// Per vertex: on a chain or on the boundary.
    pub chain_vert: Vec<bool>,
    /// Per vertex: on a Symmetry chain.
    pub sym_vert: Vec<bool>,
    /// Per vertex: on a non-Symmetry chain or on a boundary edge that is not Symmetry.
    pub other_vert: Vec<bool>,
}

fn is_hard_he(px: &ProxyRef, p: &FieldParams, h: u32) -> bool {
    if px.topo.twin[h as usize] == NONE {
        return true;
    }
    match px.feats.he_kind[h as usize] {
        FeatureKind::Boundary | FeatureKind::Symmetry => true,
        FeatureKind::Hard | FeatureKind::GuideHard => p.hard_mode == 2,
        _ => false,
    }
}

fn is_guide_he(px: &ProxyRef, p: &FieldParams, h: u32) -> bool {
    if px.topo.twin[h as usize] == NONE {
        return false;
    }
    match px.feats.he_kind[h as usize] {
        FeatureKind::Guide => true,
        FeatureKind::Hard | FeatureKind::GuideHard => p.hard_mode == 1,
        _ => false,
    }
}

fn centroid(px: &ProxyRef, f: u32) -> V3 {
    let [a, b, c] = px.mesh.tri_pos(f);
    math::scale(math::add(math::add(a, b), c), 1.0 / 3.0)
}

impl Setup {
    pub fn new(px: &ProxyRef, c: &CurvField, s: &SizeField, p: &FieldParams, ctl: Option<&Ctl>) -> Result<Self, AqError> {
        let m = px.mesh;
        let t = px.topo;
        let fr = px.frames;
        let nf = m.tris.len();
        let nh = 3 * nf;
        let nv = m.pos.len();

        // Chain position of every chain halfedge (and its twin).
        let mut chain_pos = try_vec(nh, (NONE, NONE))?;
        for (ci, ch) in px.feats.chains.iter().enumerate() {
            for (i, &h) in ch.hes.iter().enumerate() {
                chain_pos[h as usize] = (ci as u32, i as u32);
                let tw = t.twin[h as usize];
                if tw != NONE {
                    chain_pos[tw as usize] = (ci as u32, i as u32);
                }
            }
        }
        let smoothed = |f: u32, h: u32| -> f64 {
            let (ci, i) = chain_pos[h as usize];
            let exact = fr.edge_angle(m, h);
            if ci == NONE {
                return exact;
            }
            let ch = &px.feats.chains[ci as usize];
            let n = ch.hes.len();
            let i = i as usize;
            let vec_of = |j: usize| -> V3 {
                let e = ch.hes[j];
                math::sub(m.pos[t.head(m, e) as usize], m.pos[t.tail(m, e) as usize])
            };
            let unit = |j: usize| -> (V3, f64) {
                let v = vec_of(j);
                let l = math::norm(v);
                (if l > 0.0 { math::scale(v, 1.0 / l) } else { [0.0; 3] }, l)
            };
            let is_corner = |j: usize| ch.corners.binary_search(&(j as u32)).is_ok();
            let window = 0.5 * s.h_face[f as usize];
            let half = 0.5 * window;
            let (u0, l0) = unit(i);
            let mut sum = math::scale(u0, l0.min(window));
            // Forward: entering edge jn crosses the vertex tail(hes[jn]).
            let mut start = 0.5 * l0;
            let mut j = i;
            while start < half {
                let mut jn = j + 1;
                if jn == n {
                    if ch.closed {
                        jn = 0;
                    } else {
                        break;
                    }
                }
                if jn == i || is_corner(jn) {
                    break;
                }
                let (u, l) = unit(jn);
                let ov = (start + l).min(half) - start;
                sum = math::add(sum, math::scale(u, ov));
                start += l;
                j = jn;
            }
            // Backward: leaving edge j crosses the vertex tail(hes[j]).
            let mut start = 0.5 * l0;
            let mut j = i;
            while start < half {
                if is_corner(j) {
                    break;
                }
                let jp = if j == 0 {
                    if ch.closed {
                        n - 1
                    } else {
                        break;
                    }
                } else {
                    j - 1
                };
                if jp == i {
                    break;
                }
                let (u, l) = unit(jp);
                let ov = (start + l).min(half) - start;
                sum = math::add(sum, math::scale(u, ov));
                start += l;
                j = jp;
            }
            let a = fr.to_face_2d(f, sum);
            if a[0].hypot(a[1]) <= 1e-12 * l0.max(1e-300) {
                exact
            } else {
                a[1].atan2(a[0])
            }
        };

        // Hard faces and their constraint edges.
        let mut hard = try_vec(nf, false)?;
        let mut edge_angle = try_vec(nf, 0.0f64)?;
        let mut psi_s = try_vec(nf, 0.0f64)?;
        for f in 0..nf as u32 {
            let mut cons: Vec<(u8, u32)> = Vec::new();
            for k in 0..3 {
                let h = 3 * f + k;
                if is_hard_he(px, p, h) {
                    let pri = if t.twin[h as usize] == NONE {
                        0
                    } else if px.feats.he_kind[h as usize] == FeatureKind::Symmetry {
                        1
                    } else {
                        2
                    };
                    cons.push((pri, h));
                }
            }
            cons.sort_unstable();
            let Some(&(_, h0)) = cons.first() else { continue };
            let a0 = fr.edge_angle(m, h0);
            for &(_, h) in &cons[1..] {
                let a = fr.edge_angle(m, h);
                let dev = wrap_pi(4.0 * (a - a0)).abs() / 4.0;
                if dev >= 1e-9 {
                    return Err(inv(format!(
                        "face {f} has two constraint edges that are not orthogonal ({dev:.3e} rad); \
                         feature isolation (DESIGN 6.4) must run before S4"
                    )));
                }
            }
            hard[f as usize] = true;
            edge_angle[f as usize] = a0;
            psi_s[f as usize] = smoothed(f, h0);
        }

        // Smoothness weights.
        let mut w = try_vec(nh, 0.0f64)?;
        let mut sum_w = 0.0;
        let mut cnt = 0usize;
        for h in 0..nh {
            let tw = t.twin[h];
            if tw == NONE || (tw as usize) < h {
                continue;
            }
            let e = math::sub(m.pos[t.head(m, h as u32) as usize], m.pos[t.tail(m, h as u32) as usize]);
            let den = fr.area[h / 3] + fr.area[tw as usize / 3];
            let x = math::norm2(e) / den;
            w[h] = x;
            sum_w += x;
            cnt += 1;
        }
        if cnt > 0 {
            let mean = sum_w / cnt as f64;
            for h in 0..nh {
                let tw = t.twin[h];
                if tw == NONE || (tw as usize) < h {
                    continue;
                }
                let x = clamp_weight(w[h], mean);
                w[h] = x;
                w[tw as usize] = x;
            }
        }

        // Soft guides: faces with a guide edge, spread over the faces within 0.5h (dual
        // Dijkstra over centroids, the target transported with kappa).
        let mut a_g = try_vec(nf, 0.0f64)?;
        let mut psi_g = try_vec(nf, 0.0f64)?;
        if p.guide_w > 0.0 {
            let mut dist = try_vec(nf, f64::INFINITY)?;
            let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
            for f in 0..nf as u32 {
                if hard[f as usize] {
                    continue;
                }
                if let Some(h) = (0..3).map(|k| 3 * f + k).find(|&h| is_guide_he(px, p, h)) {
                    dist[f as usize] = 0.0;
                    psi_g[f as usize] = smoothed(f, h);
                    heap.push(Reverse((0f64.to_bits(), f)));
                }
            }
            let mut pops = 0usize;
            while let Some(Reverse((db, f))) = heap.pop() {
                let d = f64::from_bits(db);
                if d > dist[f as usize] {
                    continue;
                }
                pops += 1;
                if pops.is_multiple_of(1024)
                    && let Some(ctl) = ctl
                {
                    ctl.check()?;
                }
                let cf = centroid(px, f);
                for k in 0..3 {
                    let h = 3 * f + k;
                    let tw = t.twin[h as usize];
                    if tw == NONE {
                        continue;
                    }
                    let g = tw / 3;
                    if hard[g as usize] {
                        continue;
                    }
                    let nd = d + math::dist(cf, centroid(px, g));
                    if nd <= 0.5 * s.h_face[g as usize] && nd < dist[g as usize] {
                        dist[g as usize] = nd;
                        psi_g[g as usize] = psi_g[f as usize] + fr.kappa[h as usize];
                        heap.push(Reverse((nd.to_bits(), g)));
                    }
                }
            }
            for f in 0..nf {
                if dist[f].is_finite() {
                    let hf = s.h_face[f];
                    a_g[f] = p.guide_w * 20.0 / (hf * hf) * fr.area[f];
                }
            }
        }

        // Curvature alignment, with the coherent confidence and target.
        let mut conf = try_vec(nf, 0.0f64)?;
        for f in 0..nf {
            conf[f] = c.conf[f].clamp(0.0, 1.0);
        }
        // The coherence filter only matters on components that take the linear solve.
        let degenerate = degenerate_comps(px, &hard, &a_g, &conf, p);
        let (conf_eff, phi) = if p.align > 0.0 && degenerate.iter().any(|&d| !d) {
            coherence(px, &w, &conf, &c.phi, &s.h_face, ctl)?
        } else {
            (conf.clone(), c.phi.clone())
        };
        let mut a_c = try_vec(nf, 0.0f64)?;
        for f in 0..nf {
            let hf = s.h_face[f];
            a_c[f] = p.align / (hf * hf) * fr.area[f] * conf_eff[f];
        }

        // Vertex classes for the edits.
        let mut chain_vert = try_vec(nv, false)?;
        let mut sym_vert = try_vec(nv, false)?;
        let mut other_vert = try_vec(nv, false)?;
        for h in 0..nh as u32 {
            let bnd = t.twin[h as usize] == NONE;
            let on_chain = px.feats.he_chain[h as usize] != NONE;
            if !bnd && !on_chain {
                continue;
            }
            let sym = px.feats.he_kind[h as usize] == FeatureKind::Symmetry;
            for v in [t.tail(m, h), t.head(m, h)] {
                chain_vert[v as usize] = true;
                if sym {
                    sym_vert[v as usize] = true;
                } else {
                    other_vert[v as usize] = true;
                }
            }
        }
        let h2 = s.h_face.iter().map(|h| h * h).collect();
        Ok(Setup { hard, edge_angle, psi_s, w, a_c, phi, conf, conf_eff, h2, a_g, psi_g, chain_vert, sym_vert, other_vert })
    }

    /// Total alignment weight of face f (curvature plus guide).
    #[inline]
    pub fn a_tot(&self, f: usize) -> f64 {
        self.a_c[f] + self.a_g[f]
    }
}

/// Per component: true when it takes the eigen path of DESIGN 6.6 step 2 (no hard face, no
/// guide face, and no alignment or an area-weighted mean confidence below `EIGEN_CONF`).
pub(crate) fn degenerate_comps(px: &ProxyRef, hard: &[bool], a_g: &[f64], conf: &[f64], p: &FieldParams) -> Vec<bool> {
    let t = px.topo;
    let nc = t.n_comp as usize;
    let mut has_hard = vec![false; nc];
    let mut has_guide = vec![false; nc];
    let mut area = vec![0.0f64; nc];
    let mut area_conf = vec![0.0f64; nc];
    for f in 0..hard.len() {
        let c = t.face_comp[f] as usize;
        has_hard[c] |= hard[f];
        has_guide[c] |= a_g[f] > 0.0;
        area[c] += px.frames.area[f];
        area_conf[c] += px.frames.area[f] * conf[f];
    }
    (0..nc).map(|c| !has_hard[c] && !has_guide[c] && (p.align <= 0.0 || area_conf[c] < EIGEN_CONF * area[c])).collect()
}

/// Smoothing radius of the coherence filter, in h (1.0 and 2.0 were also measured, see
/// CHANGES.md, WP-08 review).
pub(crate) const COHERENCE_RADIUS: f64 = 1.5;
/// Relative residual and iteration cap of the coherence filter's PCG solve.
const COHERENCE_TOL: f64 = 1e-3;
const COHERENCE_MAX_IT: usize = 400;

/// Coherence filter of the curvature target (deviation, CHANGES.md WP-08 review).
///
/// `q_f = conf_f exp(4 i phi_f)` is smoothed by one implicit diffusion step with the
/// 4-connection, `(M + L_t) qs = M q`, where `L_t` has the weights `t_e w_e`,
/// `t_e = (r h_f)(r h_g)` and `r = COHERENCE_RADIUS`. Where the principal directions agree over
/// about one quad, `|qs_f|` stays near `conf_f`; where they are noise (random directions at
/// the sub-quad scale), the average cancels and `|qs_f|` collapses. The coherent confidence is
/// `conf_f * min(1, |qs_f| / conf_f)^2` and the target is `arg(qs_f) / 4`
/// (`phi_f` where `qs_f` vanishes).
pub(crate) fn coherence(
    px: &ProxyRef,
    w: &[f64],
    conf: &[f64],
    phi: &[f64],
    h_face: &[f64],
    ctl: Option<&Ctl>,
) -> Result<(Vec<f64>, Vec<f64>), AqError> {
    let t = px.topo;
    let fr = px.frames;
    let nf = px.mesh.tris.len();
    let mut conf_eff = try_vec(nf, 0.0f64)?;
    let mut phi_eff = phi.to_vec();
    if !conf.iter().any(|&x| x > 0.0) {
        return Ok((conf_eff, phi_eff));
    }
    let r2 = COHERENCE_RADIUS * COHERENCE_RADIUS;
    let mut ct = CTriplets::with_capacity(nf, 4 * nf).map_err(|e| e.at(Stage::Field))?;
    for h in 0..3 * nf {
        let tw = t.twin[h];
        if tw == NONE || (tw as usize) < h {
            continue;
        }
        let (f, g) = (h / 3, tw as usize / 3);
        let we = r2 * h_face[f] * h_face[g] * w[h];
        let (sn, cs) = (4.0 * fr.kappa[h]).sin_cos();
        ct.push(f, f, [we, 0.0]);
        ct.push(g, g, [we, 0.0]);
        ct.push(g, f, [-we * cs, -we * sn]);
    }
    let mut b: Vec<[f64; 2]> = try_vec(nf, [0.0, 0.0])?;
    for f in 0..nf {
        let a = fr.area[f];
        ct.push(f, f, [a, 0.0]);
        let (sn, cs) = (4.0 * phi[f]).sin_cos();
        b[f] = [a * conf[f] * cs, a * conf[f] * sn];
    }
    let a = try_hermitian_to_real(&ct).map_err(|e| e.at(Stage::Field))?;
    drop(ct);
    // A low-pass filter needs no factorization: M + L_t is well conditioned, and Jacobi PCG
    // from the unfiltered q reaches a relative residual of 1e-3 in about 80 matrix products
    // at 82k faces (0.2 s against 0.6 s for a Cholesky factorization; serial, deterministic).
    let rhs = complex_to_real(&b);
    let mut x: Vec<f64> = try_vec(2 * nf, 0.0f64)?;
    for f in 0..nf {
        let (sn, cs) = (4.0 * phi[f]).sin_cos();
        x[2 * f] = conf[f] * cs;
        x[2 * f + 1] = conf[f] * sn;
    }
    let jac = Jacobi::new(&a);
    pcg_ctl(&a, &rhs, &mut x, &|r, z| jac.apply(r, z), COHERENCE_TOL, COHERENCE_MAX_IT, ctl).map_err(|e| e.at(Stage::Field))?;
    let qs = real_to_complex(&x);
    for f in 0..nf {
        let m = qs[f][0].hypot(qs[f][1]);
        if !m.is_finite() {
            return Err(AqError::numerical("coherence filter produced a non-finite value").at(Stage::Field));
        }
        if conf[f] > 0.0 {
            let ratio = (m / conf[f]).min(1.0);
            conf_eff[f] = conf[f] * ratio * ratio;
        }
        if m > 0.0 {
            phi_eff[f] = qs[f][1].atan2(qs[f][0]) / 4.0;
        }
    }
    Ok((conf_eff, phi_eff))
}

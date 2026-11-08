//! DESIGN 6.6 step 2: Knoppel power field `z_f = exp(4 i theta_f)`.
//!
//! Minimizes `sum_e w_e |z_g - exp(4 i kappa_e) z_f|^2 + sum_f a_f |z_f - q_f|^2` over the free
//! faces (hard faces eliminated with `z_f = exp(4 i psi_f)`), as the Hermitian normal equations
//! `H z = b` solved in real form (`aq_linalg::hermitian_to_real`) with Cholesky, followed by
//! at most `ALIGN_NORM_ITERS` normalized iterations on the same factorization. A component
//! with no hard face, no guide and an area-weighted mean confidence below 0.01 (or no
//! alignment at all) takes the smallest generalized eigenvector of `H x = mu M x` by inverse
//! power iteration (8 iterations, shift 1e-8 times the mean diagonal, deterministic start from
//! the seed, mass-normalized, one factorization), followed by at most `EIGEN_NORM_ITERS` (16) normalized
//! iterations on the same factorization.

use aq_core::alloc::try_vec;
use aq_core::{AqError, Ctl, NONE, SplitMix64, Stage};
use aq_linalg::{CTriplets, Chol, Ordering, complex_to_real, real_to_complex, try_hermitian_to_real};

use crate::setup::{Setup, degenerate_comps, snap};
use crate::{FieldParams, ProxyRef};

type C = [f64; 2];

/// Normalized inverse iterations `x <- A^-1 M (x / |x|)` run after the 8 linear ones of the eigen
/// path. The lowest eigenspace of the 4-connection Laplacian is degenerate on round shapes (on
/// a sphere it is the 9-dimensional spin-4, l = 4 space), so the linear iteration returns an
/// arbitrary member, whose 8 cones were measured 0.36 R to 0.96 R apart (closest pair, 60
/// seeds and resolutions). Normalizing per face and projecting again with the same
/// factorization selects the member of most uniform magnitude, a Ginzburg-Landau type step that
/// repels same-sign cones: 0.93 R to 1.21 R with at most 16 iterations and the early stop
/// below (60 runs; 0.99 R after a fixed 24, at twice the cost). See CHANGES.md, WP-08.
pub(crate) const EIGEN_NORM_ITERS: usize = 16;
/// The normalized iterations stop early when the mass-norm change of the (mass-normalized)
/// vector drops below this (0.03 measurably degraded the spread, 0.01 did not).
pub(crate) const EIGEN_NORM_TOL: f64 = 0.01;

#[inline]
fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
#[inline]
fn conj(a: C) -> C {
    [a[0], -a[1]]
}
#[inline]
fn cis(x: f64) -> C {
    let (s, c) = x.sin_cos();
    [c, s]
}
#[inline]
fn cadd(a: &mut C, b: C, s: f64) {
    a[0] += s * b[0];
    a[1] += s * b[1];
}

fn factor(a: &aq_linalg::Csc, ctl: &Ctl) -> Result<Chol, AqError> {
    let sym = Chol::analyze_ctl(a, Ordering::Auto, Some(ctl)).map_err(|e| e.at(Stage::Field))?;
    ctl.check()?;
    let ch = Chol::factor(&sym, a).map_err(|e| e.at(Stage::Field))?;
    ctl.check()?;
    Ok(ch)
}

/// Returns theta per face (hard faces snapped to their constraint edge) and whether any
/// component took the eigen path.
pub(crate) fn knoppel_theta(px: &ProxyRef, st: &Setup, p: &FieldParams, ctl: &Ctl) -> Result<(Vec<f64>, bool), AqError> {
    let t = px.topo;
    let fr = px.frames;
    let nf = px.mesh.tris.len();
    let nh = 3 * nf;
    let nc = t.n_comp as usize;

    let degenerate = degenerate_comps(px, &st.hard, &st.a_g, &st.conf, p);

    let mut z: Vec<C> = try_vec(nf, [0.0, 0.0])?;
    for f in 0..nf {
        if st.hard[f] {
            z[f] = cis(4.0 * st.psi_s[f]);
        }
    }

    // Linear solve over the free faces of the non-degenerate components.
    let mut idx = try_vec(nf, NONE)?;
    let mut n = 0usize;
    for f in 0..nf {
        if !st.hard[f] && !degenerate[t.face_comp[f] as usize] {
            idx[f] = n as u32;
            n += 1;
        }
    }
    if n > 0 {
        let mut ct = CTriplets::with_capacity(n, 4 * n).map_err(|e| e.at(Stage::Field))?;
        let mut b: Vec<C> = try_vec(n, [0.0, 0.0])?;
        for h in 0..nh {
            let tw = t.twin[h];
            if tw == NONE || (tw as usize) < h {
                continue;
            }
            let (f, g) = (h / 3, tw as usize / 3);
            if degenerate[t.face_comp[f] as usize] {
                continue;
            }
            let w = st.w[h];
            let r = cis(4.0 * fr.kappa[h]);
            match (idx[f], idx[g]) {
                (NONE, NONE) => {}
                (fi, NONE) => {
                    ct.push(fi as usize, fi as usize, [w, 0.0]);
                    cadd(&mut b[fi as usize], cmul(conj(r), z[g]), w);
                }
                (NONE, gi) => {
                    ct.push(gi as usize, gi as usize, [w, 0.0]);
                    cadd(&mut b[gi as usize], cmul(r, z[f]), w);
                }
                (fi, gi) => {
                    ct.push(fi as usize, fi as usize, [w, 0.0]);
                    ct.push(gi as usize, gi as usize, [w, 0.0]);
                    ct.push(gi as usize, fi as usize, [-w * r[0], -w * r[1]]);
                }
            }
        }
        for f in 0..nf {
            let fi = idx[f];
            if fi == NONE {
                continue;
            }
            let a = st.a_tot(f);
            ct.push(fi as usize, fi as usize, [a, 0.0]);
            cadd(&mut b[fi as usize], cis(4.0 * st.phi[f]), st.a_c[f]);
            cadd(&mut b[fi as usize], cis(4.0 * st.psi_g[f]), st.a_g[f]);
        }
        let a = try_hermitian_to_real(&ct).map_err(|e| e.at(Stage::Field))?;
        drop(ct);
        let ch = factor(&a, ctl)?;
        let mut rhs = complex_to_real(&b);
        ch.try_solve(&mut rhs).map_err(|e| e.at(Stage::Field))?;
        let mut sol = real_to_complex(&rhs);
        align_norm_iters(st, &fr.area, &idx, n, &b, &ch, &mut sol, ctl)?;
        for f in 0..nf {
            if idx[f] != NONE {
                z[f] = sol[idx[f] as usize];
            }
        }
    }

    // Eigen path per degenerate component.
    let any_deg = degenerate.iter().any(|&d| d);
    if any_deg {
        let mut rng = SplitMix64::new(p.seed);
        for c in 0..nc {
            if !degenerate[c] {
                continue;
            }
            ctl.check()?;
            eigen_component(px, st, c as u32, &mut rng, &mut z, ctl)?;
        }
    }

    // theta = arg(z)/4; tiny |z| takes the transported mean of its neighbours.
    let mut theta = try_vec(nf, 0.0f64)?;
    let mut done = try_vec(nf, false)?;
    let mut mean_abs = 0.0;
    let mut cnt = 0usize;
    for f in 0..nf {
        if !st.hard[f] {
            mean_abs += z[f][0].hypot(z[f][1]);
            cnt += 1;
        }
    }
    let tiny = if cnt > 0 { 1e-9 * mean_abs / cnt as f64 } else { 0.0 };
    for f in 0..nf {
        if st.hard[f] {
            theta[f] = snap(st.psi_s[f], st.edge_angle[f]);
            done[f] = true;
        } else if z[f][0].hypot(z[f][1]) > tiny && z[f][0].is_finite() && z[f][1].is_finite() {
            theta[f] = z[f][1].atan2(z[f][0]) / 4.0;
            done[f] = true;
        }
    }
    let mut pending: Vec<usize> = (0..nf).filter(|&f| !done[f]).collect();
    while !pending.is_empty() {
        let mut next = Vec::new();
        let mut progressed = false;
        for &f in &pending {
            let mut acc = [0.0f64, 0.0];
            let mut any = false;
            for k in 0..3 {
                let h = 3 * f + k;
                let tw = t.twin[h];
                if tw == NONE {
                    continue;
                }
                let g = tw as usize / 3;
                if done[g] {
                    // Angle a in f is a + kappa[h] in g, so theta_g reads theta_g - kappa[h] in f.
                    cadd(&mut acc, cis(4.0 * (theta[g] - fr.kappa[h])), 1.0);
                    any = true;
                }
            }
            if any && acc[0].hypot(acc[1]) > 0.0 {
                theta[f] = acc[1].atan2(acc[0]) / 4.0;
                done[f] = true;
                progressed = true;
            } else {
                next.push(f);
            }
        }
        if !progressed {
            break; // isolated faces keep theta = 0
        }
        pending = next;
    }
    Ok((theta, any_deg))
}

/// Normalized iterations of the aligned solve (deviation, CHANGES.md WP-08 review):
/// `H z_{k+1} = b + mu M z_k / |z_k|` with `mu_f = ALIGN_NORM_MU A_f / h_f^2`, on the
/// factorization of the linear solve. The linear power field shrinks |z| towards 0 wherever the
/// alignment target is weak or incoherent, and every such dip becomes a cone. Feeding back the
/// unit field as an extra target of the weight of a `LOW_CONF` face keeps |z| uniform there, so
/// the field follows its smoothest continuation instead, as on the eigen path.
pub(crate) const ALIGN_NORM_ITERS: usize = 16;
/// Weight of the feedback term, in units of `A_f / h_f^2` (the weight of a face of conf 0.2 at
/// align 1).
pub(crate) const ALIGN_NORM_MU: f64 = 0.2;
/// The iterations stop when the area-weighted RMS change of the unit field drops below this.
pub(crate) const ALIGN_NORM_TOL: f64 = 0.01;

fn align_norm_iters(st: &Setup, area: &[f64], idx: &[u32], n: usize, b: &[C], ch: &Chol, sol: &mut Vec<C>, ctl: &Ctl) -> Result<(), AqError> {
    let nf = idx.len();
    let mut mass = try_vec(n, 0.0f64)?;
    let mut mu = try_vec(n, 0.0f64)?;
    for f in 0..nf {
        if idx[f] != NONE {
            let i = idx[f] as usize;
            mass[i] = area[f];
            mu[i] = ALIGN_NORM_MU * area[f] / st.h2[f];
        }
    }
    let total: f64 = mass.iter().sum();
    if total <= 0.0 {
        return Ok(());
    }
    let unit = |z: &[C]| -> Vec<C> {
        z.iter()
            .map(|v| {
                let a = v[0].hypot(v[1]);
                if a > 0.0 && a.is_finite() { [v[0] / a, v[1] / a] } else { [0.0, 0.0] }
            })
            .collect()
    };
    let mut u = unit(sol);
    for _ in 0..ALIGN_NORM_ITERS {
        let mut rhs: Vec<C> = b.to_vec();
        for i in 0..n {
            cadd(&mut rhs[i], u[i], mu[i]);
        }
        let mut r = complex_to_real(&rhs);
        ch.try_solve(&mut r).map_err(|e| e.at(Stage::Field))?;
        *sol = real_to_complex(&r);
        let un = unit(sol);
        let change: f64 =
            un.iter().zip(&u).zip(&mass).map(|((a, c), m)| m * ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2))).sum();
        u = un;
        ctl.check()?;
        if change <= ALIGN_NORM_TOL * ALIGN_NORM_TOL * total {
            break;
        }
    }
    Ok(())
}

/// Smallest generalized eigenvector of the connection Laplacian on one component.
fn eigen_component(
    px: &ProxyRef,
    st: &Setup,
    comp: u32,
    rng: &mut SplitMix64,
    z: &mut [C],
    ctl: &Ctl,
) -> Result<(), AqError> {
    let t = px.topo;
    let fr = px.frames;
    let nf = px.mesh.tris.len();
    let faces: Vec<usize> = (0..nf).filter(|&f| t.face_comp[f] == comp).collect();
    let n = faces.len();
    if n == 0 {
        return Ok(());
    }
    let mut loc = try_vec(nf, NONE)?;
    for (i, &f) in faces.iter().enumerate() {
        loc[f] = i as u32;
    }
    let mut ct = CTriplets::with_capacity(n, 4 * n).map_err(|e| e.at(Stage::Field))?;
    let mut diag = vec![0.0f64; n];
    for &f in &faces {
        for k in 0..3 {
            let h = 3 * f + k;
            let tw = t.twin[h];
            if tw == NONE || (tw as usize) < h {
                continue;
            }
            let g = tw as usize / 3;
            let (fi, gi) = (loc[f] as usize, loc[g] as usize);
            let w = st.w[h];
            let r = cis(4.0 * fr.kappa[h]);
            ct.push(fi, fi, [w, 0.0]);
            ct.push(gi, gi, [w, 0.0]);
            ct.push(gi, fi, [-w * r[0], -w * r[1]]);
            diag[fi] += w;
            diag[gi] += w;
        }
    }
    let mass: Vec<f64> = faces.iter().map(|&f| fr.area[f]).collect();
    let mean_mass = mass.iter().sum::<f64>() / n as f64;
    let mean_diag = diag.iter().sum::<f64>() / n as f64;
    let mut shift = 1e-8 * mean_diag;
    if shift.is_nan() || shift <= 0.0 {
        shift = 1e-8;
    }
    for i in 0..n {
        ct.push(i, i, [shift * mass[i] / mean_mass, 0.0]);
    }
    let a = try_hermitian_to_real(&ct).map_err(|e| e.at(Stage::Field))?;
    drop(ct);
    let ch = factor(&a, ctl)?;
    let mut x: Vec<C> = (0..n).map(|_| [rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0)]).collect();
    let normalize = |x: &mut [C]| {
        let s: f64 = x.iter().zip(&mass).map(|(v, m)| m * (v[0] * v[0] + v[1] * v[1])).sum();
        if s > 0.0 && s.is_finite() {
            let k = 1.0 / s.sqrt();
            for v in x.iter_mut() {
                v[0] *= k;
                v[1] *= k;
            }
        }
    };
    normalize(&mut x);
    for _ in 0..8 {
        let mut rhs: Vec<f64> = Vec::with_capacity(2 * n);
        for i in 0..n {
            rhs.push(mass[i] * x[i][0]);
            rhs.push(mass[i] * x[i][1]);
        }
        ch.try_solve(&mut rhs).map_err(|e| e.at(Stage::Field))?;
        x = real_to_complex(&rhs);
        normalize(&mut x);
        ctl.check()?;
    }
    // Normalized iterations: pick, inside the (possibly degenerate) lowest eigenspace, the member
    // with the most uniform magnitude (see EIGEN_NORM_ITERS).
    for _ in 0..EIGEN_NORM_ITERS {
        let mut rhs: Vec<f64> = Vec::with_capacity(2 * n);
        for i in 0..n {
            let a = x[i][0].hypot(x[i][1]);
            let (u, v) = if a > 0.0 { (x[i][0] / a, x[i][1] / a) } else { (0.0, 0.0) };
            rhs.push(mass[i] * u);
            rhs.push(mass[i] * v);
        }
        ch.try_solve(&mut rhs).map_err(|e| e.at(Stage::Field))?;
        let mut xn = real_to_complex(&rhs);
        normalize(&mut xn);
        // Change in the mass norm (both vectors have mass norm 1).
        let ch2: f64 = xn
            .iter()
            .zip(&x)
            .zip(&mass)
            .map(|((a, b), m)| m * ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)))
            .sum();
        x = xn;
        ctl.check()?;
        if ch2 <= EIGEN_NORM_TOL * EIGEN_NORM_TOL {
            break;
        }
    }
    for (i, &f) in faces.iter().enumerate() {
        z[f] = x[i];
    }
    Ok(())
}

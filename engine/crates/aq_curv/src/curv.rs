//! S3 curvature at the quad scale (DESIGN 6.5 "Curvature").
//!
//! 1. Per interior edge the signed dihedral angle
//!    `beta_e = atan2((n_f x n_g) . e_hat, n_f . n_g)` with `e_hat` along the canonical
//!    halfedge in f (positive on convex edges). Boundary edges have no dihedral angle.
//! 2. Per vertex the normal-cycle tensor `T0(v) = (1/A_v) sum_{e at v} (1/2) beta_e |e| e_hat e_hat^T`
//!    (ambient 3x3, 6 scalars, A_v barycentric).
//! 3. Implicit diffusion `(M + L_t) T = M T0` of the 6 components with one factorization,
//!    `L_t` = cotan Laplacian with edge weights times `t_e = (c_s hbar_e)^2`. Note that
//!    `M T0(v)` is the plain edge sum, so A_v never divides anything.
//! 4. Per face: mean of the 3 vertex tensors, projected to the face plane, plus the isotropic
//!    return of the normal-normal leak (see below), rotated by 90 degrees in the plane:
//!    `C_f = J T_f J^T`. The eigenvectors of C_f are the principal directions, its eigenvalues
//!    the signed principal curvatures, ordered so that |k1| >= |k2|.
//! 5. `conf_f = smoothstep(h_f |k1 - k2|; conf_lo, conf_hi) * |k1 - k2| / (|k1| + |k2| + 1e-12)`.
//! 6. `kappa_hat(v) = sqrt(k1^2 + k2^2)` of the vertex tensor (same projection and return).
//!
//! **Normal leak return (CHANGES.md, WP-07).** Diffusing an ambient tensor field over a curved
//! surface rotates part of its tangent content into the normal direction: on a sphere of
//! radius R the diffused tensor is `(1/R)(a I - b n n^T)` with a tangent eigenvalue below 1/R
//! (0.8/R for R = 2, t = 1: the l = 2 harmonics decay by `1/(1 + 6t/R^2)`), while the trace
//! (the mean curvature) is preserved exactly. The projection therefore adds `(n^T T n)/2` to
//! both tangent eigenvalues, which gives the trace back. It is exact on spheres, a no-op on
//! cylinders (`T = (1/R) a a^T` with a the axis has no normal part) and second order elsewhere.

use std::f64::consts::FRAC_PI_2;

use aq_core::math;
use aq_core::{AqError, Ctl, CurvField, ErrKind, NONE, Stage, V3};
use aq_linalg::{Chol, Ordering, try_cotan_laplacian, try_lumped_mass};
use rayon::prelude::*;

use crate::{ProxyRef, bad, check_proxy, restamp};

/// Parameters of [`curvature`] (DESIGN 6.5 API).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvParams {
    /// `c_s = flow_smoothness`: the diffusion time per edge is `t_e = (c_s hbar_e)^2`.
    pub scale: f64,
    /// Lower edge of the confidence gate (turn per quad, radians).
    pub conf_lo: f64,
    /// Upper edge of the confidence gate.
    pub conf_hi: f64,
}

impl Default for CurvParams {
    /// DESIGN 13: c_s = 1.0, conf gate (0.02, 0.25).
    fn default() -> Self {
        CurvParams { scale: 1.0, conf_lo: 0.02, conf_hi: 0.25 }
    }
}

/// A symmetric 3x3 tensor stored as (xx, xy, xz, yy, yz, zz).
pub type Sym3 = [f64; 6];

/// Diagnostic output of [`curvature_detail`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CurvDetail {
    /// Diffused vertex tensors T(v).
    pub vert_tensor: Vec<Sym3>,
    /// Per face: relative anisotropy `|k1 - k2| / (|k1| + |k2| + 1e-12)`.
    pub aniso: Vec<f64>,
    /// Number of unknowns of the factorized system (vertices).
    pub n_unknowns: usize,
}

/// `smoothstep(x; a, b) = s^2 (3 - 2s)`, `s = clamp((x - a)/(b - a), 0, 1)`.
pub fn smoothstep(x: f64, a: f64, b: f64) -> f64 {
    let s = ((x - a) / (b - a)).clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}

/// Relative anisotropy `|k1 - k2| / (|k1| + |k2| + 1e-12)`: 1 on cylinders and saddles,
/// 0 on umbilics.
pub fn anisotropy(k1: f64, k2: f64) -> f64 {
    (k1 - k2).abs() / (k1.abs() + k2.abs() + 1e-12)
}

/// DESIGN 6.5 confidence for local size h.
pub fn confidence(k1: f64, k2: f64, h: f64, p: &CurvParams) -> f64 {
    let c = smoothstep(h * (k1 - k2).abs(), p.conf_lo, p.conf_hi) * anisotropy(k1, k2);
    if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 }
}

/// S3 curvature (DESIGN 6.5): k1, k2, phi and conf per face, kappa_hat per vertex.
/// `h_vert` is the local size per vertex in quad units (all 1 in pass 1).
///
/// Errors (stage Curv): `InvalidInput` for mismatched sizes or bad parameters, `Numerical`
/// for a failed factorization or a non-finite result, `Cancelled` / `Timeout` from `ctl`.
pub fn curvature(px: &ProxyRef, h_vert: &[f64], p: &CurvParams, ctl: &Ctl) -> Result<CurvField, AqError> {
    curvature_detail(px, h_vert, p, ctl).map(|(c, _)| c)
}

/// [`curvature`] that also returns the diffused vertex tensors and the per-face anisotropy.
pub fn curvature_detail(
    px: &ProxyRef,
    h_vert: &[f64],
    p: &CurvParams,
    ctl: &Ctl,
) -> Result<(CurvField, CurvDetail), AqError> {
    curvature_impl(px, h_vert, p, true, ctl)
}

fn sym_add(t: &mut Sym3, e: V3, w: f64) {
    t[0] += w * e[0] * e[0];
    t[1] += w * e[0] * e[1];
    t[2] += w * e[0] * e[2];
    t[3] += w * e[1] * e[1];
    t[4] += w * e[1] * e[2];
    t[5] += w * e[2] * e[2];
}

/// `a^T T b`.
fn sym_form(t: &Sym3, a: V3, b: V3) -> f64 {
    let tb = [
        t[0] * b[0] + t[1] * b[1] + t[2] * b[2],
        t[1] * b[0] + t[3] * b[1] + t[4] * b[2],
        t[2] * b[0] + t[4] * b[1] + t[5] * b[2],
    ];
    math::dot(a, tb)
}

/// Eigen-decomposition of the symmetric 2x2 `[[a, b], [b, c]]` as (k1, k2, angle of the k1
/// eigenvector), with |k1| >= |k2|.
fn eig2(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let m = 0.5 * (a + c);
    let d = 0.5 * (a - c);
    let r = d.hypot(b);
    let ang_max = 0.5 * b.atan2(d);
    if m >= 0.0 { (m + r, m - r, ang_max) } else { (m - r, m + r, ang_max + FRAC_PI_2) }
}

/// Signed dihedral angle of the canonical interior halfedge h (DESIGN 6.5 step 1).
pub fn dihedral(px: &ProxyRef, h: u32) -> f64 {
    let t = px.topo;
    let tw = t.twin[h as usize];
    if tw == NONE {
        return 0.0;
    }
    let nf = px.frames.n[(h / 3) as usize];
    let ng = px.frames.n[(tw / 3) as usize];
    let d = math::sub(px.mesh.pos[t.head(px.mesh, h) as usize], px.mesh.pos[t.tail(px.mesh, h) as usize]);
    let len = math::norm(d);
    if len <= 0.0 || !len.is_finite() {
        return 0.0;
    }
    let e = math::scale(d, 1.0 / len);
    math::dot(math::cross(nf, ng), e).atan2(math::dot(nf, ng))
}

pub(crate) fn curvature_impl(
    px: &ProxyRef,
    h_vert: &[f64],
    p: &CurvParams,
    leak_return: bool,
    ctl: &Ctl,
) -> Result<(CurvField, CurvDetail), AqError> {
    check_proxy(px)?;
    let m = px.mesh;
    let t = px.topo;
    let fr = px.frames;
    let nv = m.pos.len();
    let nf = m.tris.len();
    let nh = 3 * nf;
    if h_vert.len() != nv {
        return Err(bad(format!("curvature: h_vert has {} values for {nv} vertices", h_vert.len())));
    }
    if let Some(v) = h_vert.iter().position(|h| !(h.is_finite() && *h > 0.0)) {
        return Err(bad(format!("curvature: h_vert[{v}] = {} is not finite and positive", h_vert[v])));
    }
    if !(p.scale.is_finite() && p.scale > 0.0) {
        return Err(bad(format!("curvature: scale {} must be finite and > 0", p.scale)));
    }
    if !(p.conf_lo.is_finite() && p.conf_hi.is_finite() && p.conf_lo < p.conf_hi) {
        return Err(bad(format!("curvature: conf gate ({}, {}) must be finite and increasing", p.conf_lo, p.conf_hi)));
    }
    ctl.progress(Stage::Curv, 0.0);
    ctl.check()?;

    // Step 1-2: right-hand side M T0 = sum over edges of (1/2) beta |e| e_hat e_hat^T, and the
    // diffusion times on the canonical halfedges.
    let mut rhs: Vec<Sym3> = aq_core::alloc::try_vec(nv, [0.0; 6]).map_err(restamp)?;
    let mut edge_scale = aq_core::alloc::try_vec(nh, 0.0f64).map_err(restamp)?;
    for h in 0..nh as u32 {
        if h % 16384 == 0 {
            ctl.check()?;
        }
        if !t.is_canonical(h) {
            continue;
        }
        let a = t.tail(m, h) as usize;
        let b = t.head(m, h) as usize;
        let hbar = 0.5 * (h_vert[a] + h_vert[b]);
        let ts = p.scale * hbar;
        edge_scale[h as usize] = ts * ts;
        if t.twin[h as usize] == NONE {
            continue;
        }
        let d = math::sub(m.pos[b], m.pos[a]);
        let len = math::norm(d);
        if !(len > 0.0 && len.is_finite()) {
            continue;
        }
        let e = math::scale(d, 1.0 / len);
        let w = 0.5 * dihedral(px, h) * len;
        sym_add(&mut rhs[a], e, w);
        sym_add(&mut rhs[b], e, w);
    }
    ctl.progress(Stage::Curv, 0.1);

    // Step 3: (M + L_t) T = M T0, one factorization, 6 right-hand sides.
    let mut a = try_cotan_laplacian(m, t, fr, Some(&edge_scale)).map_err(restamp)?;
    let mut mass = try_lumped_mass(m, t, fr).map_err(restamp)?;
    let mean_mass = {
        let (s, c) = mass.iter().filter(|x| **x > 0.0 && x.is_finite()).fold((0.0, 0usize), |(s, c), x| (s + x, c + 1));
        if c > 0 { s / c as f64 } else { 1.0 }
    };
    // An isolated vertex (no face, zero mass, empty Laplacian row) gets a unit diagonal: its
    // right-hand side is 0, so its tensor is 0.
    for x in mass.iter_mut() {
        if !(*x > 0.0 && x.is_finite()) {
            *x = mean_mass;
        }
    }
    a.add_diag_vec(&mass).map_err(restamp)?;
    ctl.check()?;
    let sym = Chol::analyze_ctl(&a, Ordering::Auto, Some(ctl)).map_err(restamp)?;
    ctl.check()?;
    let chol = Chol::factor(&sym, &a).map_err(restamp)?;
    ctl.check()?;
    ctl.progress(Stage::Curv, 0.6);
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(6);
    for c in 0..6 {
        let mut col = aq_core::alloc::try_vec(nv, 0.0f64).map_err(restamp)?;
        for (x, r) in col.iter_mut().zip(&rhs) {
            *x = r[c];
        }
        cols.push(col);
    }
    drop(rhs);
    chol.try_solve_multi(&mut cols).map_err(restamp)?;
    ctl.check()?;
    ctl.progress(Stage::Curv, 0.8);
    let mut tv: Vec<Sym3> = aq_core::alloc::try_vec(nv, [0.0; 6]).map_err(restamp)?;
    for (v, tvv) in tv.iter_mut().enumerate() {
        for c in 0..6 {
            tvv[c] = cols[c][v];
        }
        if tvv.iter().any(|x| !x.is_finite()) {
            return Err(AqError::new(Stage::Curv, ErrKind::Numerical, format!("diffused tensor at vertex {v} is not finite")));
        }
    }
    drop(cols);

    // Step 4-5: per face.
    let per_face: Vec<(f64, f64, f64, f64, f64)> = (0..nf)
        .into_par_iter()
        .with_min_len(1024)
        .map(|f| {
            let tri = m.tris[f];
            let mut tf = [0.0; 6];
            for c in 0..6 {
                tf[c] = (tv[tri[0] as usize][c] + tv[tri[1] as usize][c] + tv[tri[2] as usize][c]) / 3.0;
            }
            let (e1, e2, n) = (fr.e1[f], fr.e2[f], fr.n[f]);
            let mut t11 = sym_form(&tf, e1, e1);
            let t12 = sym_form(&tf, e1, e2);
            let mut t22 = sym_form(&tf, e2, e2);
            if leak_return {
                let leak = 0.5 * sym_form(&tf, n, n);
                t11 += leak;
                t22 += leak;
            }
            // C = J T J^T with J the CCW quarter turn: [[t22, -t12], [-t12, t11]].
            let (k1, k2, phi) = eig2(t22, -t12, t11);
            let hf = (h_vert[tri[0] as usize] + h_vert[tri[1] as usize] + h_vert[tri[2] as usize]) / 3.0;
            let conf = confidence(k1, k2, hf, p);
            (k1, k2, phi, conf, anisotropy(k1, k2))
        })
        .collect();
    ctl.check()?;

    // Step 6: kappa_hat per vertex from the vertex tensor, projected on the vertex tangent
    // plane (area-weighted normal) with the same leak return.
    let mut vn: Vec<V3> = aq_core::alloc::try_vec(nv, [0.0; 3]).map_err(restamp)?;
    for f in 0..nf {
        let c = m.face_cross(f as u32);
        for &v in &m.tris[f] {
            vn[v as usize] = math::add(vn[v as usize], c);
        }
    }
    let kappa_hat: Vec<f64> = (0..nv)
        .into_par_iter()
        .with_min_len(1024)
        .map(|v| vertex_kappa_hat(&tv[v], vn[v], leak_return))
        .collect();

    let mut cf = CurvField {
        k1: Vec::with_capacity(nf),
        k2: Vec::with_capacity(nf),
        phi: Vec::with_capacity(nf),
        conf: Vec::with_capacity(nf),
        kappa_hat,
    };
    let mut aniso = Vec::with_capacity(nf);
    for (f, &(k1, k2, phi, conf, an)) in per_face.iter().enumerate() {
        if !(k1.is_finite() && k2.is_finite() && phi.is_finite()) {
            return Err(AqError::new(Stage::Curv, ErrKind::Numerical, format!("curvature of face {f} is not finite")));
        }
        cf.k1.push(k1);
        cf.k2.push(k2);
        cf.phi.push(phi);
        cf.conf.push(conf);
        aniso.push(an);
    }
    cf.validate(nf, nv).map_err(restamp)?;
    ctl.progress(Stage::Curv, 1.0);
    Ok((cf, CurvDetail { vert_tensor: tv, aniso, n_unknowns: nv }))
}

/// sqrt(k1^2 + k2^2) of the vertex tensor T projected on the plane normal to `n` (not
/// necessarily unit), with the leak return. A zero normal gives the Frobenius norm of T.
fn vertex_kappa_hat(t: &Sym3, n: V3, leak_return: bool) -> f64 {
    let full = [[t[0], t[1], t[2]], [t[1], t[3], t[4]], [t[2], t[4], t[5]]];
    let Some(n) = math::normalize(n) else {
        let s: f64 = full.iter().flatten().map(|x| x * x).sum();
        return s.sqrt();
    };
    // Q = P T P with P = I - n n^T.
    let mut pm = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            pm[i][j] = if i == j { 1.0 } else { 0.0 } - n[i] * n[j];
        }
    }
    let mut tp = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            tp[i][j] = (0..3).map(|k| full[i][k] * pm[k][j]).sum();
        }
    }
    let mut fro2 = 0.0;
    let mut tr = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let q: f64 = (0..3).map(|k| pm[i][k] * tp[k][j]).sum();
            fro2 += q * q;
            if i == j {
                tr += q;
            }
        }
    }
    let k2sum = if leak_return {
        let l = sym_form(t, n, n);
        // Eigenvalues (la + l/2, lb + l/2): la^2 + lb^2 + l (la + lb) + l^2 / 2.
        fro2 + l * tr + 0.5 * l * l
    } else {
        fro2
    };
    k2sum.max(0.0).sqrt()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Measures the effect of the normal leak return (CHANGES.md, WP-07): without it, the
    /// diffused tensor on a sphere of radius 2 loses about 20% of its tangent eigenvalues.
    #[test]
    fn leak_return_on_sphere() {
        let s = aq_testkit::Surface::new(aq_testkit::icosphere(3, 2.0).unwrap()).unwrap();
        let px = ProxyRef::new(&s.mesh, &s.topo, &s.frames, &s.feats);
        let st = aq_core::CtlState::new();
        let ctl = st.unbounded();
        let h = vec![1.0; s.n_verts()];
        let p = CurvParams::default();
        let mean = |c: &CurvField| c.k1.iter().sum::<f64>() / c.k1.len() as f64;
        let (raw, _) = curvature_impl(&px, &h, &p, false, &ctl).unwrap();
        let (fixed, _) = curvature_impl(&px, &h, &p, true, &ctl).unwrap();
        println!("sphere R = 2: mean k1 without leak return {:.4}, with {:.4} (exact 0.5)", mean(&raw), mean(&fixed));
        assert!(mean(&raw) < 0.45);
        assert!((mean(&fixed) - 0.5).abs() < 0.015);
    }
}


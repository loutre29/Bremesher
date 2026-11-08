//! C5 to C13: angles, cross fields, matchings, residuals, indices, transitions, combing,
//! Jacobians and edge families.
//!
//! * C6: per face one real `theta[f]` (not restricted to [0, pi/2)). Branch k of face f is
//!   the direction `theta[f] + k pi/2`; the u-axis is branch 0,
//!   `d_f = cos(theta) e1 + sin(theta) e2`, and the v-axis is branch 1, `n x d_f`.
//! * C7: for a canonical interior he, `p[he] = round((theta[g] - theta[f] - kappa[he]) / (pi/2)) mod 4`
//!   (rounding half away from zero), `p[twin] = (4 - p[he]) mod 4`. Branch k of f maps to
//!   branch `(k - p[he]) mod 4` of g. Once S4 produced p, p is the source of truth.
//! * C8: `delta[he] = theta[g] - theta[f] - kappa[he] - p[he] pi/2`, reduced modulo 2pi into
//!   (-pi, pi] ([`delta_edge`]); `delta[twin] = -delta[he]`. Because p is stored mod 4, the
//!   unreduced formula is only defined up to multiples of 2pi; the reduction picks the
//!   representative of p nearest to the actual rotation. Freshly rounded, delta lies in
//!   [-pi/4, pi/4]; after edits it may be larger, and it must stay inside (-pi, pi).
//! * C9: interior `index4(v) = round((2/pi) (K(v) + sum_{H_in(v)} delta))`; boundary
//!   `n_c = round((omega - sum delta)/(pi/2))`, `index4 = 2 - n_c`; residual < 1e-6 rad,
//!   otherwise `Invariant`. Poincare-Hopf: `sum_v index4 = 4 chi` per component.
//! * C10: `R(k)` is the CCW rotation by `k pi/2`, `R(1)(u,v) = (-v, u)`. Across a cut
//!   halfedge from f to g, `phi_g = R(-r) phi_f + t`; twin: `r' = (4 - r) mod 4`,
//!   `t' = -R(r) t` (the inverse).
//! * C11: `J_f` = Jacobian of (u, v) w.r.t. (e1, e2), row 0 grad u, row 1 grad v. A face is
//!   folded when `det J_f <= eps_det = 1e-10 median |det J|`.
//! * C12: combing `theta' = theta + c_f pi/2`, `r = (p + c_g - c_f) mod 4`. A raw branch k
//!   of f is the combed branch `(k - c_f) mod 4` (see CHANGES.md: DESIGN writes `k + c_f`;
//!   both agree mod 2, i.e. on the family).
//! * C13: edge family U (0) when the edge is parallel to the combed u-axis
//!   (`|wrap_pi(edge_angle - theta')| < pi/4` or `> 3pi/4`), else V (1). A U edge lies on a
//!   v-iso line.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use crate::error::{AqError, ensure};
use crate::frames::Frames;
use crate::math;
use crate::mesh::{NONE, TriMesh, V2, V3};
use crate::tol;
use crate::topo::Topology;

/// Maps x to (-pi, pi]. Values already inside are returned unchanged (bitwise).
pub fn wrap_pi(x: f64) -> f64 {
    if x > -PI && x <= PI {
        return x;
    }
    if !x.is_finite() {
        return x;
    }
    let mut y = (x + PI).rem_euclid(TAU) - PI;
    if y <= -PI {
        y += TAU;
    }
    if y > PI {
        y -= TAU;
    }
    y
}

/// C10: CCW rotation by `k pi/2` (k taken mod 4). Exact; returns +0.0 for zero entries.
#[inline]
pub fn rot(k: u8, a: V2) -> V2 {
    match k % 4 {
        0 => [a[0] + 0.0, a[1] + 0.0],
        1 => [-a[1] + 0.0, a[0] + 0.0],
        2 => [-a[0] + 0.0, -a[1] + 0.0],
        _ => [a[1] + 0.0, -a[0] + 0.0],
    }
}

/// `(k - p) mod 4`: the branch of g that branch k of f maps to across a halfedge of
/// matching p (C7). The same rule maps combed branches with r.
#[inline]
pub fn map_branch(k: u8, p: u8) -> u8 {
    (k % 4 + 4 - p % 4) % 4
}

/// C10 chart transition across a halfedge from f to g: `phi_g = R(-r) phi_f + t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Transition {
    pub r: u8,
    pub t: V2,
}

impl Transition {
    pub const IDENTITY: Transition = Transition { r: 0, t: [0.0, 0.0] };

    /// `R(-r) a + t` (maps a point of chart f to chart g). Exact for dyadic inputs.
    #[inline]
    pub fn apply(&self, a: V2) -> V2 {
        let b = rot((4 - self.r % 4) % 4, a);
        [b[0] + self.t[0], b[1] + self.t[1]]
    }

    /// Applies only the rotation part, for vectors.
    #[inline]
    pub fn apply_vec(&self, a: V2) -> V2 {
        rot((4 - self.r % 4) % 4, a)
    }

    /// The transition of the twin halfedge: `r' = (4 - r) mod 4`, `t' = -R(r) t`.
    #[inline]
    pub fn inverse(&self) -> Self {
        let rt = rot(self.r, self.t);
        Transition { r: (4 - self.r % 4) % 4, t: [-rt[0] + 0.0, -rt[1] + 0.0] }
    }

    /// First `self`, then `then`: `then.apply(self.apply(a))`.
    /// `r = r1 + r2`, `t = R(-r2) t1 + t2`.
    #[inline]
    pub fn compose(&self, then: &Self) -> Self {
        let a = rot((4 - then.r % 4) % 4, self.t);
        Transition {
            r: (self.r % 4 + then.r % 4) % 4,
            t: [a[0] + then.t[0] + 0.0, a[1] + then.t[1] + 0.0],
        }
    }

    pub fn is_identity(&self) -> bool {
        self.r.is_multiple_of(4) && self.t[0] == 0.0 && self.t[1] == 0.0
    }

    /// r < 4 and t finite.
    pub fn validate(&self) -> Result<(), AqError> {
        ensure(self.r < 4 && math::is_finite2(self.t), || format!("bad transition {self:?}"))
    }
}

/// C6..C9 cross field.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrossField {
    /// Per face.
    pub theta: Vec<f64>,
    /// Per halfedge, authoritative (0 on the boundary).
    pub p: Vec<u8>,
    /// Per halfedge, C8.
    pub delta: Vec<f64>,
    /// Per vertex, C9.
    pub index4: Vec<i8>,
    /// Per face: theta fixed by a constraint.
    pub hard: Vec<bool>,
}

impl CrossField {
    /// Builds a field from theta: matching rounded (C7), delta (C8) and index4 (C9).
    pub fn from_theta(
        m: &TriMesh,
        t: &Topology,
        fr: &Frames,
        theta: Vec<f64>,
        hard: Vec<bool>,
    ) -> Result<Self, AqError> {
        ensure(theta.len() == m.tris.len() && hard.len() == m.tris.len(), || {
            "CrossField::from_theta: sizes".into()
        })?;
        let (p, delta) = matching_from_theta(m, t, fr, &theta);
        let index4 = index4_all(m, t, fr, &delta)?;
        Ok(CrossField { theta, p, delta, index4, hard })
    }

    /// Cone vertices (index4 != 0) with their index, sorted by vertex.
    pub fn cones(&self) -> Vec<(u32, i8)> {
        self.index4
            .iter()
            .enumerate()
            .filter(|(_, i)| **i != 0)
            .map(|(v, i)| (v as u32, *i))
            .collect()
    }

    /// Checks: sizes, p < 4, p = 0 and delta = 0 on the boundary, `p[twin] = (4-p)%4`,
    /// `delta[twin] = -delta` bitwise, delta consistent with (theta, kappa, p) within 1e-9,
    /// index4 equal to its recomputation from delta, and Poincare-Hopf per component.
    pub fn validate(&self, m: &TriMesh, t: &Topology, fr: &Frames) -> Result<(), AqError> {
        let nf = m.tris.len();
        ensure(
            self.theta.len() == nf
                && self.hard.len() == nf
                && self.p.len() == 3 * nf
                && self.delta.len() == 3 * nf
                && self.index4.len() == m.pos.len(),
            || "CrossField sizes do not match the mesh".into(),
        )?;
        for (f, th) in self.theta.iter().enumerate() {
            ensure(th.is_finite(), || format!("theta[{f}] not finite"))?;
        }
        for h in 0..3 * nf {
            let tw = t.twin[h];
            ensure(self.p[h] < 4, || format!("p[{h}] = {} >= 4", self.p[h]))?;
            if tw == NONE {
                ensure(self.p[h] == 0 && self.delta[h] == 0.0, || {
                    format!("boundary he {h}: p and delta must be 0")
                })?;
                continue;
            }
            let tw = tw as usize;
            ensure(self.p[tw] == (4 - self.p[h]) % 4, || format!("p[twin] != (4-p)%4 at {h}"))?;
            ensure(self.delta[tw].to_bits() == (-self.delta[h]).to_bits(), || {
                format!("delta[twin] != -delta at {h}")
            })?;
            if h < tw {
                let f = h / 3;
                let g = tw / 3;
                let d = delta_edge(self.theta[f], self.theta[g], fr.kappa[h], self.p[h]);
                ensure((d - self.delta[h]).abs() <= 1e-9, || {
                    format!("delta[{h}] = {} inconsistent with theta and p ({d})", self.delta[h])
                })?;
            }
        }
        let idx = index4_all(m, t, fr, &self.delta)?;
        ensure(idx == self.index4, || "index4 differs from its recomputation".into())?;
        poincare_hopf(m, t, &self.index4)
    }
}

/// C6: direction of branch k of face f for the angle theta (3D unit vector).
pub fn branch_dir(fr: &Frames, f: u32, theta: f64, k: u8) -> V3 {
    fr.dir_at(f, theta + (k % 4) as f64 * FRAC_PI_2)
}

/// C8 for one canonical halfedge: `theta_g - theta_f - kappa - p pi/2`, reduced into (-pi, pi].
#[inline]
pub fn delta_edge(theta_f: f64, theta_g: f64, kappa: f64, p: u8) -> f64 {
    wrap_pi(theta_g - theta_f - kappa - (p % 4) as f64 * FRAC_PI_2)
}

/// C7 rounding for one canonical halfedge: `round((theta_g - theta_f - kappa)/(pi/2)) mod 4`.
#[inline]
pub fn matching_edge(theta_f: f64, theta_g: f64, kappa: f64) -> u8 {
    let x = (theta_g - theta_f - kappa) / FRAC_PI_2;
    let r = x.round();
    if !r.is_finite() {
        return 0;
    }
    // r mod 4 without overflow for huge values.
    r.rem_euclid(4.0) as u8 % 4
}

/// C7 + C8: rounded matching and residual for every halfedge (computed on the canonical
/// halfedge, derived for the twin; 0 on the boundary).
pub fn matching_from_theta(
    m: &TriMesh,
    t: &Topology,
    fr: &Frames,
    theta: &[f64],
) -> (Vec<u8>, Vec<f64>) {
    let nh = 3 * m.tris.len();
    let mut p = vec![0u8; nh];
    let mut delta = vec![0.0f64; nh];
    for h in 0..nh {
        let tw = t.twin[h];
        if tw == NONE || (tw as usize) < h {
            continue;
        }
        let tw = tw as usize;
        let (f, g) = (h / 3, tw / 3);
        let ph = matching_edge(theta[f], theta[g], fr.kappa[h]);
        let d = delta_edge(theta[f], theta[g], fr.kappa[h], ph);
        p[h] = ph;
        p[tw] = (4 - ph) % 4;
        delta[h] = d;
        delta[tw] = -d;
    }
    (p, delta)
}

/// C8 for a given (authoritative) matching p.
pub fn delta_from_p(m: &TriMesh, t: &Topology, fr: &Frames, theta: &[f64], p: &[u8]) -> Vec<f64> {
    let nh = 3 * m.tris.len();
    let mut delta = vec![0.0f64; nh];
    for h in 0..nh {
        let tw = t.twin[h];
        if tw == NONE || (tw as usize) < h {
            continue;
        }
        let tw = tw as usize;
        let d = delta_edge(theta[h / 3], theta[tw / 3], fr.kappa[h], p[h]);
        delta[h] = d;
        delta[tw] = -d;
    }
    delta
}

/// C9 for one vertex. An isolated vertex has index 0.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN residuals must fail
pub fn index4_vertex(
    m: &TriMesh,
    t: &Topology,
    fr: &Frames,
    delta: &[f64],
    v: u32,
) -> Result<i8, AqError> {
    if t.vert_he.get(v as usize).copied().unwrap_or(NONE) == NONE {
        return Ok(0);
    }
    let sd: f64 = t.incoming(m, v).map(|h| delta[h as usize]).sum();
    let omega = fr.angle_sum(t, v);
    let (s, bnd) = if t.is_bnd_vert[v as usize] {
        (omega - sd, true)
    } else {
        ((TAU - omega) + sd, false)
    };
    let q = (s / FRAC_PI_2).round();
    let res = s - q * FRAC_PI_2;
    if !(res.abs() < tol::INDEX_RESIDUAL) || q.abs() > 100.0 {
        return Err(AqError::invariant(format!(
            "index residual {res:e} at vertex {v} ({})",
            if bnd { "boundary: field not aligned with the boundary?" } else { "interior" }
        )));
    }
    let q = q as i32;
    Ok(if bnd { (2 - q) as i8 } else { q as i8 })
}

/// C9 for every vertex. A residual >= 1e-6 rad anywhere gives `Invariant`.
pub fn index4_all(
    m: &TriMesh,
    t: &Topology,
    fr: &Frames,
    delta: &[f64],
) -> Result<Vec<i8>, AqError> {
    (0..m.pos.len() as u32).map(|v| index4_vertex(m, t, fr, delta, v)).collect()
}

/// Poincare-Hopf: `sum_v index4(v) == 4 chi` for every component, or `Invariant`.
pub fn poincare_hopf(m: &TriMesh, t: &Topology, index4: &[i8]) -> Result<(), AqError> {
    let mut sum = vec![0i64; t.n_comp as usize];
    for (v, &h) in t.vert_he.iter().enumerate() {
        if h != NONE {
            sum[t.face_comp[(h / 3) as usize] as usize] += index4[v] as i64;
        }
    }
    for (c, s) in sum.iter().enumerate() {
        let chi = t.euler_char(m, c as u32);
        ensure(*s == 4 * chi, || {
            format!("Poincare-Hopf fails on component {c}: sum index4 = {s}, 4 chi = {}", 4 * chi)
        })?;
    }
    Ok(())
}

/// C12: combed angle `theta + c pi/2`.
#[inline]
pub fn comb_theta(theta: f64, c: u8) -> f64 {
    theta + (c % 4) as f64 * FRAC_PI_2
}

/// C12: combed matching `(p + c_g - c_f) mod 4`.
#[inline]
pub fn comb_matching(p: u8, c_f: u8, c_g: u8) -> u8 {
    (p % 4 + c_g % 4 + 4 - c_f % 4) % 4
}

/// C12: the combed branch of raw branch k in a face with comb offset c_f: `(k - c_f) mod 4`.
#[inline]
pub fn raw_to_combed_branch(k: u8, c_f: u8) -> u8 {
    (k % 4 + 4 - c_f % 4) % 4
}

/// C12: inverse of [`raw_to_combed_branch`]: `(j + c_f) mod 4`.
#[inline]
pub fn combed_to_raw_branch(j: u8, c_f: u8) -> u8 {
    (j % 4 + c_f % 4) % 4
}

/// C13: family of halfedge h in its face: 0 (U) if parallel to the combed u-axis, else 1 (V).
pub fn edge_family(fr: &Frames, m: &TriMesh, h: u32, theta_combed: f64) -> u8 {
    let x = wrap_pi(fr.edge_angle(m, h) - theta_combed).abs();
    if x < FRAC_PI_4 || x > 3.0 * FRAC_PI_4 { 0 } else { 1 }
}

/// C11: Jacobian of (u, v) with respect to the frame (e1, e2) of face f, from the per-corner
/// uv. Row 0 is grad u, row 1 is grad v.
pub fn jacobian(fr: &Frames, m: &TriMesh, f: u32, uv: &[V2; 3]) -> [[f64; 2]; 2] {
    let [_, q1, q2] = fr.face_coords(m, f);
    let dq = q1[0] * q2[1] - q1[1] * q2[0];
    let inv = 1.0 / dq;
    let g = |c: usize| -> [f64; 2] {
        let d1 = uv[1][c] - uv[0][c];
        let d2 = uv[2][c] - uv[0][c];
        [(q2[1] * d1 - q1[1] * d2) * inv, (-q2[0] * d1 + q1[0] * d2) * inv]
    };
    [g(0), g(1)]
}

/// Determinant of a 2x2 matrix.
#[inline]
pub fn det2(j: &[[f64; 2]; 2]) -> f64 {
    j[0][0] * j[1][1] - j[0][1] * j[1][0]
}

/// C11: `eps_det = 1e-10 * median(|det J|)` over the given determinants (0 when empty).
pub fn fold_eps(dets: &[f64]) -> f64 {
    let mut a: Vec<f64> = dets.iter().map(|d| d.abs()).filter(|d| !d.is_nan()).collect();
    if a.is_empty() {
        return 0.0;
    }
    let mid = a.len() / 2;
    a.select_nth_unstable_by(mid, |x, y| x.total_cmp(y));
    tol::FOLD_EPS_REL * a[mid]
}

/// C11: per-face fold flags (`det J_f <= eps_det`) for per-corner uv (NaN counts as folded).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn folded_faces(fr: &Frames, m: &TriMesh, uv: &[[V2; 3]]) -> Vec<bool> {
    let dets: Vec<f64> =
        (0..m.tris.len() as u32).map(|f| det2(&jacobian(fr, m, f, &uv[f as usize]))).collect();
    let eps = fold_eps(&dets);
    dets.iter().map(|&d| !(d > eps)).collect()
}

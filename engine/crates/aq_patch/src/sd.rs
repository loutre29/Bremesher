//! Barrier symmetric Dirichlet repair (DESIGN 6.12 step 3).
//!
//! Energy, relative to the per-face targets `T_f` (in the C4 frame of each face):
//! `E = sum_f A_f (|D_f|^2 + |D_f^-1|^2)` with `D_f = J_f T_f^-1`, infinite as soon as a
//! face has `det D_f <= 0` (the barrier). Degenerate faces (zero 3D area) carry no energy but
//! are still kept unfolded by the line search.
//!
//! Each iteration is one local-global step:
//! * local: the rotation `R_f` closest to `D_f` (2D polar decomposition, the rotation of the
//!   SVD), giving the proxy target `R_f T_f`;
//! * global: the area-weighted Laplacian least squares toward the proxy targets with the
//!   fixed vertices held (factored once);
//! * line search on the direction `d = x* - x`: the largest step keeping every triangle
//!   positive comes from the roots of the quadratic `det(t)` per face (Smith and Schaefer
//!   2015), times 0.8 and capped at 1, then halved until the Armijo condition
//!   `E(x + t d) <= E(x) + 1e-4 t dE/dt` holds and every triangle stays positive under the
//!   exact orient2d.
//!
//! The energy therefore decreases strictly at every accepted step, and the map stays
//! injective. An initial map that is folded is replaced by the Floater map with the same
//! fixed vertices (the barrier cannot start from a fold).

use aq_core::exact::orient2d;
use aq_core::{AqError, Ctl, ErrKind, Topology, TriMesh, V2};

use crate::floater::floater_mvc_ctl;
use crate::geom::{M2, det, err, face_geo, frob2, input_err, inv, jac, mul};
use crate::ls::Stiffness;

const ARMIJO: f64 = 1e-4;
const STEP_SHRINK: f64 = 0.8;
const MAX_BACKTRACK: u32 = 40;

/// Trace of one [`sd_barrier_trace`] run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SdTrace {
    /// Accepted local-global steps.
    pub iters: u32,
    /// Energy of the start map, then after each accepted step (strictly decreasing).
    pub energies: Vec<f64>,
    /// True when the given map was folded and the Floater map was used as the start.
    pub restarted: bool,
}

/// True when every triangle is strictly positive in `uv` (exact orient2d).
pub(crate) fn all_positive(m: &TriMesh, uv: &[V2]) -> bool {
    m.tris.iter().all(|t| orient2d(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]) > 0.0)
}

/// Number of triangles that are not strictly positive in `uv` (exact orient2d).
pub fn count_flips(m: &TriMesh, uv: &[V2]) -> usize {
    m.tris.iter().filter(|t| !(orient2d(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]) > 0.0)).count()
}

struct Prep {
    tinv: Vec<Option<M2>>,
}

fn prepare(disk: &TriMesh, targets: &[M2]) -> Result<Prep, AqError> {
    if targets.len() != disk.tris.len() {
        return Err(input_err("sd_barrier: one target per face is required"));
    }
    let mut tinv = Vec::with_capacity(targets.len());
    for (f, t) in targets.iter().enumerate() {
        let d = det(t);
        if !(d.is_finite() && d > 0.0) {
            return Err(input_err(format!("sd_barrier: target of face {f} is not orientation preserving")));
        }
        tinv.push(inv(t));
    }
    Ok(Prep { tinv })
}

fn face_energy(d: &M2) -> f64 {
    let dt = det(d);
    if !(dt > 0.0) {
        return f64::INFINITY;
    }
    let f = frob2(d);
    f + f / (dt * dt)
}

fn energy(disk: &TriMesh, st: &Stiffness, pr: &Prep, uv: &[V2]) -> f64 {
    let mut e = 0.0;
    for (f, t) in disk.tris.iter().enumerate() {
        let (Some(fg), Some(ti)) = (&st.geo[f], &pr.tinv[f]) else { continue };
        let j = jac(&fg.g, [uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]]);
        e += fg.area * face_energy(&mul(&j, ti));
    }
    e
}

/// Symmetric Dirichlet energy of `uv` relative to `targets` (per face, in the C4 face frame).
/// Infinite when a face is folded; degenerate faces are skipped.
pub fn sd_energy(disk: &TriMesh, uv: &[V2], targets: &[M2]) -> f64 {
    let Ok(pr) = prepare(disk, targets) else { return f64::NAN };
    let mut e = 0.0;
    for (f, t) in disk.tris.iter().enumerate() {
        let (Some(fg), Some(ti)) = (face_geo(disk.tri_pos(f as u32)), &pr.tinv[f]) else { continue };
        let j = jac(&fg.g, [uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]]);
        e += fg.area * face_energy(&mul(&j, ti));
    }
    e
}

/// Directional derivative of the energy at `uv` along `d`.
fn energy_slope(disk: &TriMesh, st: &Stiffness, pr: &Prep, uv: &[V2], d: &[V2]) -> f64 {
    let mut s = 0.0;
    for (f, t) in disk.tris.iter().enumerate() {
        let (Some(fg), Some(ti)) = (&st.geo[f], &pr.tinv[f]) else { continue };
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let dm = mul(&jac(&fg.g, [uv[a], uv[b], uv[c]]), ti);
        let dd = mul(&jac(&fg.g, [d[a], d[b], d[c]]), ti);
        let dt = det(&dm);
        let fr = frob2(&dm);
        let inner = dm[0][0] * dd[0][0] + dm[0][1] * dd[0][1] + dm[1][0] * dd[1][0] + dm[1][1] * dd[1][1];
        let ddet = dm[0][0] * dd[1][1] + dm[1][1] * dd[0][0] - dm[0][1] * dd[1][0] - dm[1][0] * dd[0][1];
        s += fg.area * (2.0 * inner * (1.0 + 1.0 / (dt * dt)) - 2.0 * fr * ddet / (dt * dt * dt));
    }
    s
}

/// Smallest positive t where some triangle's signed area `c0 + c1 t + c2 t^2` reaches 0.
fn max_step(disk: &TriMesh, uv: &[V2], d: &[V2]) -> f64 {
    let mut tmax = f64::INFINITY;
    for t in &disk.tris {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let e1 = [uv[b][0] - uv[a][0], uv[b][1] - uv[a][1]];
        let e2 = [uv[c][0] - uv[a][0], uv[c][1] - uv[a][1]];
        let f1 = [d[b][0] - d[a][0], d[b][1] - d[a][1]];
        let f2 = [d[c][0] - d[a][0], d[c][1] - d[a][1]];
        let c0 = e1[0] * e2[1] - e1[1] * e2[0];
        let c1 = e1[0] * f2[1] - e1[1] * f2[0] + f1[0] * e2[1] - f1[1] * e2[0];
        let c2 = f1[0] * f2[1] - f1[1] * f2[0];
        if !(c0 > 0.0) {
            return 0.0;
        }
        let root = smallest_positive_root(c2, c1, c0);
        if root < tmax {
            tmax = root;
        }
    }
    tmax
}

fn smallest_positive_root(a: f64, b: f64, c: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if a.abs() <= 1e-14 * scale {
        return if b < 0.0 { -c / b } else { f64::INFINITY };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return f64::INFINITY;
    }
    let sq = disc.sqrt();
    let q = -0.5 * (b + if b >= 0.0 { sq } else { -sq });
    let mut best = f64::INFINITY;
    for r in [q / a, if q != 0.0 { c / q } else { f64::INFINITY }] {
        if r > 0.0 && r < best {
            best = r;
        }
    }
    best
}

/// Barrier symmetric Dirichlet repair (DESIGN 6.12 API). `targets[f]` is the target Jacobian
/// of face f in its C4 frame (`e1 = normalize(p1 - p0)`, `e2 = n x e1`) with a positive
/// determinant; `fixed` vertices keep their uv. Returns the number of accepted steps.
pub fn sd_barrier(
    disk: &TriMesh,
    t: &Topology,
    uv: &mut [V2],
    targets: &[[[f64; 2]; 2]],
    fixed: &[bool],
    iters: u32,
) -> Result<u32, AqError> {
    sd_barrier_ctl(disk, t, uv, targets, fixed, iters, None).map(|r| r.iters)
}

/// [`sd_barrier`] returning the energy trace.
pub fn sd_barrier_trace(
    disk: &TriMesh,
    t: &Topology,
    uv: &mut [V2],
    targets: &[[[f64; 2]; 2]],
    fixed: &[bool],
    iters: u32,
) -> Result<SdTrace, AqError> {
    sd_barrier_ctl(disk, t, uv, targets, fixed, iters, None)
}

pub(crate) fn sd_barrier_ctl(
    disk: &TriMesh,
    t: &Topology,
    uv: &mut [V2],
    targets: &[M2],
    fixed: &[bool],
    iters: u32,
    ctl: Option<&Ctl>,
) -> Result<SdTrace, AqError> {
    let nv = disk.pos.len();
    if uv.len() != nv || fixed.len() != nv {
        return Err(input_err("sd_barrier: uv and fixed need one entry per vertex"));
    }
    let pr = prepare(disk, targets)?;
    let mut trace = SdTrace::default();
    if !all_positive(disk, uv) {
        let bnd: Vec<(u32, V2)> = (0..nv).filter(|&v| fixed[v]).map(|v| (v as u32, uv[v])).collect();
        let start = floater_mvc_ctl(disk, t, &bnd, ctl)?;
        if !all_positive(disk, &start) {
            return Err(err(ErrKind::Numerical, "sd_barrier: the Floater start map is folded"));
        }
        uv.copy_from_slice(&start);
        trace.restarted = true;
    }
    let st = Stiffness::new(disk, fixed)?;
    let mut x: Vec<V2> = uv.to_vec();
    let mut e0 = energy(disk, &st, &pr, &x);
    trace.energies.push(e0);
    let mut proxy: Vec<M2> = vec![[[0.0; 2]; 2]; disk.tris.len()];
    let scale = x.iter().fold(0.0f64, |m, p| m.max(p[0].abs()).max(p[1].abs())).max(1.0);
    for _ in 0..iters {
        if let Some(c) = ctl {
            c.check()?;
        }
        if !e0.is_finite() {
            break;
        }
        // Local step.
        for (f, tri) in disk.tris.iter().enumerate() {
            let (Some(fg), Some(ti)) = (&st.geo[f], &pr.tinv[f]) else { continue };
            let j = jac(&fg.g, [x[tri[0] as usize], x[tri[1] as usize], x[tri[2] as usize]]);
            let r = crate::geom::closest_rotation(&mul(&j, ti));
            proxy[f] = mul(&r, &targets[f]);
        }
        // Global step.
        let xs = st.solve(disk, &x, &proxy)?;
        let d: Vec<V2> = xs.iter().zip(&x).map(|(a, b)| [a[0] - b[0], a[1] - b[1]]).collect();
        let dmax = d.iter().fold(0.0f64, |m, p| m.max(p[0].abs()).max(p[1].abs()));
        if !(dmax.is_finite()) || dmax <= 1e-13 * scale {
            break;
        }
        let slope = energy_slope(disk, &st, &pr, &x, &d);
        if !(slope < 0.0) {
            break;
        }
        let mut step = (STEP_SHRINK * max_step(disk, &x, &d)).min(1.0);
        let mut accepted = None;
        for _ in 0..MAX_BACKTRACK {
            if !(step > 0.0) {
                break;
            }
            let xt: Vec<V2> = x.iter().zip(&d).map(|(p, q)| [p[0] + step * q[0], p[1] + step * q[1]]).collect();
            let fixed_ok = (0..nv).all(|v| !fixed[v] || xt[v] == x[v]);
            if fixed_ok && all_positive(disk, &xt) {
                let e = energy(disk, &st, &pr, &xt);
                if e.is_finite() && e <= e0 + ARMIJO * step * slope && e < e0 {
                    accepted = Some((xt, e));
                    break;
                }
            }
            step *= 0.5;
        }
        let Some((xt, e)) = accepted else { break };
        x = xt;
        e0 = e;
        trace.energies.push(e);
        trace.iters += 1;
    }
    uv.copy_from_slice(&x);
    Ok(trace)
}

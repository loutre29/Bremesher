//! Floater mean value coordinates (DESIGN 6.12 step 3 and 4).
//!
//! For an interior vertex i and a neighbour j, `w_ij = (tan(a/2) + tan(b/2)) / |x_i - x_j|`
//! where a and b are the angles at i of the two triangles sharing the edge (i, j). The weights
//! are positive, so with the boundary fixed on a convex polygon every interior vertex is a
//! convex combination of its neighbours and the map is bijective (Floater 2003), provided no
//! interior edge joins two boundary vertices of the same straight side (the pre-split of
//! [`crate::presplit`] guarantees it). `tan(a/2)` is floored at a = 1e-3 rad (and capped at
//! pi - 1e-3), edge lengths at 1e-9.
//!
//! The system is not symmetric. It is solved with BiCGSTAB on the row-normalized matrix
//! (unit diagonal, so Jacobi preconditioning is implicit), with Gauss-Seidel sweeps as the
//! fallback when BiCGSTAB stalls (the matrix is an irreducibly diagonally dominant M-matrix,
//! so Gauss-Seidel always converges).

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::math;
use aq_core::{AqError, Ctl, NONE, Topology, TriMesh, V2};

use crate::geom::input_err;

const ANGLE_FLOOR: f64 = 1e-3;
const LEN_FLOOR: f64 = 1e-9;
const REL_TOL: f64 = 1e-14;
const GS_MAX_SWEEPS: usize = 20_000;

/// Row-normalized sparse system `x_i - sum_j l_ij x_j = b_i` over the free unknowns.
struct Rows {
    ptr: Vec<usize>,
    col: Vec<u32>,
    val: Vec<f64>,
}

impl Rows {
    fn mul(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate() {
            let mut s = x[i];
            for k in self.ptr[i]..self.ptr[i + 1] {
                s -= self.val[k] * x[self.col[k] as usize];
            }
            *yi = s;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// BiCGSTAB (identity preconditioner, the diagonal is 1). Returns the final residual norm.
fn bicgstab(a: &Rows, b: &[f64], x: &mut [f64], tol: f64, max_it: usize, ctl: Option<&Ctl>) -> Result<f64, AqError> {
    let n = b.len();
    let mut r = vec![0.0; n];
    a.mul(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let mut rhat = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];
    let (mut rho, mut alpha, mut omega) = (1.0f64, 1.0f64, 1.0f64);
    let mut res = norm(&r);
    for it in 0..max_it {
        if res <= tol {
            break;
        }
        if it % 64 == 63
            && let Some(c) = ctl
        {
            c.check()?;
        }
        let rho_new = dot(&rhat, &r);
        if rho_new == 0.0 || !rho_new.is_finite() {
            // Restart with a fresh shadow residual.
            rhat.copy_from_slice(&r);
            p.fill(0.0);
            v.fill(0.0);
            rho = 1.0;
            alpha = 1.0;
            omega = 1.0;
            continue;
        }
        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        a.mul(&p, &mut v);
        let den = dot(&rhat, &v);
        if den == 0.0 || !den.is_finite() {
            break;
        }
        alpha = rho / den;
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        if norm(&s) <= tol {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            a.mul(x, &mut r);
            for i in 0..n {
                r[i] = b[i] - r[i];
            }
            res = norm(&r);
            break;
        }
        a.mul(&s, &mut t);
        let tt = dot(&t, &t);
        if tt == 0.0 || !tt.is_finite() {
            break;
        }
        omega = dot(&t, &s) / tt;
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }
        res = norm(&r);
        if omega == 0.0 || !res.is_finite() {
            break;
        }
    }
    Ok(res)
}

/// Gauss-Seidel sweeps until the residual is below `tol` (or `max_sweeps`).
fn gauss_seidel(a: &Rows, b: &[f64], x: &mut [f64], tol: f64, max_sweeps: usize, ctl: Option<&Ctl>) -> Result<f64, AqError> {
    let n = b.len();
    let mut r = vec![0.0; n];
    for sweep in 0..max_sweeps {
        for i in 0..n {
            let mut s = b[i];
            for k in a.ptr[i]..a.ptr[i + 1] {
                s += a.val[k] * x[a.col[k] as usize];
            }
            x[i] = s;
        }
        if sweep % 16 == 15 {
            if let Some(c) = ctl {
                c.check()?;
            }
            a.mul(x, &mut r);
            let res = r.iter().zip(b).map(|(y, bb)| (bb - y) * (bb - y)).sum::<f64>().sqrt();
            if res <= tol {
                return Ok(res);
            }
        }
    }
    a.mul(x, &mut r);
    Ok(r.iter().zip(b).map(|(y, bb)| (bb - y) * (bb - y)).sum::<f64>().sqrt())
}

/// Floater mean value map of a disk with the given fixed vertices (DESIGN 6.12 API).
///
/// Every boundary vertex of `disk` must appear in `boundary_uv` (`InvalidInput` otherwise);
/// interior vertices may be fixed too. Vertices used by no face keep their fixed value or get
/// (0, 0). The result has one uv per vertex.
pub fn floater_mvc(disk: &TriMesh, t: &Topology, boundary_uv: &[(u32, V2)]) -> Result<Vec<V2>, AqError> {
    floater_mvc_ctl(disk, t, boundary_uv, None)
}

pub(crate) fn floater_mvc_ctl(
    disk: &TriMesh,
    t: &Topology,
    boundary_uv: &[(u32, V2)],
    ctl: Option<&Ctl>,
) -> Result<Vec<V2>, AqError> {
    let nv = disk.pos.len();
    if t.is_bnd_vert.len() != nv || t.twin.len() != 3 * disk.tris.len() {
        return Err(input_err("floater_mvc: topology does not match the mesh"));
    }
    let mut fixed: Vec<Option<V2>> = try_vec(nv, None)?;
    for &(v, uv) in boundary_uv {
        let slot = fixed.get_mut(v as usize).ok_or_else(|| input_err(format!("floater_mvc: vertex {v} out of range")))?;
        if !math::is_finite2(uv) {
            return Err(input_err(format!("floater_mvc: boundary uv of vertex {v} is not finite")));
        }
        *slot = Some(uv);
    }
    if let Some(v) = (0..nv).find(|&v| t.is_bnd_vert[v] && fixed[v].is_none()) {
        return Err(input_err(format!("floater_mvc: boundary vertex {v} has no uv")));
    }
    // Unknown numbering.
    let mut idx: Vec<u32> = try_vec(nv, NONE)?;
    let mut nfree = 0u32;
    let mut used = try_vec(nv, false)?;
    for tri in &disk.tris {
        for &v in tri {
            used[v as usize] = true;
        }
    }
    for v in 0..nv {
        if fixed[v].is_none() && used[v] {
            idx[v] = nfree;
            nfree += 1;
        }
    }
    let mut out: Vec<V2> = (0..nv).map(|v| fixed[v].unwrap_or([0.0, 0.0])).collect();
    if nfree == 0 {
        return Ok(out);
    }
    // Weights (i, j, w) for free i.
    let tan_lo = (0.5 * ANGLE_FLOOR).tan();
    let tan_hi = (0.5 * (std::f64::consts::PI - ANGLE_FLOOR)).tan();
    let mut ent: Vec<(u32, u32, f64)> = try_with_capacity(6 * disk.tris.len())?;
    for tri in &disk.tris {
        for k in 0..3 {
            let i = tri[k];
            if idx[i as usize] == NONE {
                continue;
            }
            let j = tri[(k + 1) % 3];
            let l = tri[(k + 2) % 3];
            let pi = disk.pos[i as usize];
            let a = math::sub(disk.pos[j as usize], pi);
            let b = math::sub(disk.pos[l as usize], pi);
            let la = math::norm(a);
            let lb = math::norm(b);
            let c = math::norm(math::cross(a, b));
            let den = la * lb + math::dot(a, b);
            let th = if den > 0.0 { c / den } else { tan_hi };
            let th = if th.is_finite() { th.clamp(tan_lo, tan_hi) } else { tan_hi };
            ent.push((i, j, th / la.max(LEN_FLOOR)));
            ent.push((i, l, th / lb.max(LEN_FLOOR)));
        }
    }
    ent.sort_by_key(|x| (x.0, x.1));
    // Merge, then build rows of the free unknowns.
    let nf = nfree as usize;
    let mut ptr = Vec::with_capacity(nf + 1);
    let mut col: Vec<u32> = Vec::new();
    let mut val: Vec<f64> = Vec::new();
    let mut rhs = [vec![0.0; nf], vec![0.0; nf]];
    ptr.push(0);
    let mut k = 0;
    let mut row = 0usize;
    // Rows are emitted in vertex order, which is unknown order.
    while k < ent.len() {
        let vi = ent[k].0;
        let ri = idx[vi as usize] as usize;
        while row < ri {
            ptr.push(col.len());
            row += 1;
        }
        let mut merged: Vec<(u32, f64)> = Vec::new();
        while k < ent.len() && ent[k].0 == vi {
            let j = ent[k].1;
            let mut w = 0.0;
            while k < ent.len() && ent[k].0 == vi && ent[k].1 == j {
                w += ent[k].2;
                k += 1;
            }
            merged.push((j, w));
        }
        let tot: f64 = merged.iter().map(|m| m.1).sum();
        if !(tot.is_finite() && tot > 0.0) {
            return Err(input_err(format!("floater_mvc: vertex {vi} has no valid weight")));
        }
        for (j, w) in merged {
            let l = w / tot;
            match fixed[j as usize] {
                Some(u) => {
                    rhs[0][ri] += l * u[0];
                    rhs[1][ri] += l * u[1];
                }
                None => {
                    col.push(idx[j as usize]);
                    val.push(l);
                }
            }
        }
        ptr.push(col.len());
        row += 1;
    }
    while row < nf {
        ptr.push(col.len());
        row += 1;
    }
    let rows = Rows { ptr, col, val };
    // Start from the mean of the fixed values.
    let mut c0 = [0.0, 0.0];
    let mut cnt = 0.0;
    for u in fixed.iter().flatten() {
        c0[0] += u[0];
        c0[1] += u[1];
        cnt += 1.0;
    }
    if cnt > 0.0 {
        c0 = [c0[0] / cnt, c0[1] / cnt];
    }
    let max_it = 4 * nf + 200;
    for c in 0..2 {
        let b = &rhs[c];
        let bn = norm(b).max(c0[c].abs() * (nf as f64).sqrt()).max(1e-300);
        let tol = REL_TOL * bn;
        let mut x = vec![c0[c]; nf];
        let res = bicgstab(&rows, b, &mut x, tol, max_it, ctl)?;
        if !(res <= tol * 100.0) || x.iter().any(|v| !v.is_finite()) {
            if x.iter().any(|v| !v.is_finite()) {
                x.fill(c0[c]);
            }
            gauss_seidel(&rows, b, &mut x, tol, GS_MAX_SWEEPS, ctl)?;
        }
        for v in 0..nv {
            let i = idx[v];
            if i != NONE {
                out[v][c] = x[i as usize];
            }
        }
    }
    Ok(out)
}

//! Meshing of one patch: interior map (field-aligned least squares, then barrier symmetric
//! Dirichlet, then Floater) and grid sampling (DESIGN 6.12 "Interior map" and "Sampling").

use aq_core::{AqError, Ctl, ErrKind, Layout, V2, V3};

use crate::disk::{Disk, build_disk};
use crate::floater::floater_mvc_ctl;
use crate::geom::{M2, det, err, face_coords, hat_grads, jac};
use crate::ls::Stiffness;
use crate::sample::{Locator, bary};
use crate::sd::{all_positive, sd_barrier_ctl};
use crate::{MapKind, PatchParams};

/// One meshed patch.
pub(crate) struct PatchMap {
    pub kind: MapKind,
    /// Lifted positions per grid point `j * (W + 1) + i` (meaningful for interior points).
    pub pos: Vec<V3>,
    /// Reference surface normal per grid point (the W' face containing it).
    pub normal: Vec<V3>,
    pub n_split: u32,
}

/// Per-face target Jacobians `T_f = diag(W/l0, H/l1) J_f^seam` in the C4 frame of each face.
/// A face whose seamless Jacobian is degenerate or folded gets the isotropic target
/// `sqrt(W H / A) I` (A = disk area), so every target is orientation preserving.
fn targets(lay: &Layout, p: u32, d: &Disk, dims: [u32; 2]) -> Vec<M2> {
    let pa = &lay.tmesh.patches[p as usize];
    let side_len = |k: usize| -> f64 { pa.sides[k].iter().map(|&(a, _)| lay.tmesh.arcs[a as usize].len).sum() };
    let l0 = 0.5 * (side_len(0) + side_len(2));
    let l1 = 0.5 * (side_len(1) + side_len(3));
    let (w, h) = (dims[0] as f64, dims[1] as f64);
    let sx = if l0 > 0.0 && l0.is_finite() { w / l0 } else { 1.0 };
    let sy = if l1 > 0.0 && l1.is_finite() { h / l1 } else { 1.0 };
    let area: f64 = (0..d.mesh.tris.len() as u32).map(|f| d.mesh.face_area(f)).sum();
    let iso = if area > 0.0 { (w * h / area).sqrt() } else { 1.0 };
    let fallback = [[iso, 0.0], [0.0, iso]];
    d.mesh
        .tris
        .iter()
        .enumerate()
        .map(|(f, t)| {
            let Some((q, a)) = face_coords(d.mesh.tri_pos(f as u32)) else { return fallback };
            let g = hat_grads(q, a);
            let j = jac(&g, [d.uv[t[0] as usize], d.uv[t[1] as usize], d.uv[t[2] as usize]]);
            let tt = [[sx * j[0][0], sx * j[0][1]], [sy * j[1][0], sy * j[1][1]]];
            let dt = det(&tt);
            if dt.is_finite() && dt > 0.0 && tt.iter().flatten().all(|x| x.is_finite()) { tt } else { fallback }
        })
        .collect()
}

/// Samples the grid of a patch map; `None` when an interior grid point is not covered.
/// Lifted positions and reference normals per grid point.
type Sampled = (Vec<V3>, Vec<V3>);

fn sample(lay: &Layout, d: &Disk, x: &[V2], dims: [u32; 2], ctl: &Ctl) -> Result<Option<Sampled>, AqError> {
    let (w, h) = (dims[0], dims[1]);
    let loc = Locator::new(&d.mesh, x, w as f64, h as f64);
    let cnt = (w as usize + 1) * (h as usize + 1);
    let mut pos = vec![[0.0; 3]; cnt];
    let mut nrm = vec![[0.0; 3]; cnt];
    let mut missing = Vec::new();
    let mut mean_n = [0.0; 3];
    for f in 0..d.mesh.tris.len() {
        let c = d.mesh.face_cross(f as u32);
        mean_n = [mean_n[0] + c[0], mean_n[1] + c[1], mean_n[2] + c[2]];
    }
    let mean_n = aq_core::math::normalize(mean_n).unwrap_or([0.0, 0.0, 1.0]);
    for j in 0..=h {
        ctl.check()?;
        for i in 0..=w {
            let g = (j * (w + 1) + i) as usize;
            let pt = [i as f64, j as f64];
            let interior = i > 0 && i < w && j > 0 && j < h;
            let Some(f) = loc.locate(&d.mesh, x, pt) else {
                if interior {
                    return Ok(None);
                }
                missing.push(g);
                continue;
            };
            let wf = d.wface[f as usize];
            nrm[g] = lay
                .wmesh
                .face_normal(wf)
                .or_else(|| d.mesh.face_normal(f))
                .unwrap_or(mean_n);
            if interior {
                let b = bary(&d.mesh, x, f, pt);
                let t = d.mesh.tris[f as usize];
                let (p0, p1, p2) = (d.mesh.pos[t[0] as usize], d.mesh.pos[t[1] as usize], d.mesh.pos[t[2] as usize]);
                pos[g] = [
                    b[0] * p0[0] + b[1] * p1[0] + b[2] * p2[0],
                    b[0] * p0[1] + b[1] * p1[1] + b[2] * p2[1],
                    b[0] * p0[2] + b[1] * p1[2] + b[2] * p2[2],
                ];
            }
        }
    }
    for g in missing {
        nrm[g] = mean_n;
    }
    Ok(Some((pos, nrm)))
}

/// Meshes patch p. With `force_floater`, only the Floater map is used (the re-mesh after a
/// failed final check).
#[allow(clippy::too_many_arguments)]
pub(crate) fn map_patch(
    lay: &Layout,
    face_patch: &[u32],
    p: u32,
    n_arc: &[u32],
    dims: [u32; 2],
    prm: &PatchParams,
    force_floater: bool,
    ctl: &Ctl,
) -> Result<PatchMap, AqError> {
    ctl.check()?;
    let d = build_disk(lay, face_patch, p, n_arc, dims, prm.eps_zero)?;
    let nv = d.mesh.pos.len();
    let fixed = &d.is_bnd;
    let x0: Vec<V2> = (0..nv).map(|v| if fixed[v] { d.rect[v] } else { [0.0, 0.0] }).collect();
    let tg = targets(lay, p, &d, dims);
    let done = |kind: MapKind, s: Sampled| PatchMap { kind, pos: s.0, normal: s.1, n_split: d.n_split };

    // A numerical failure of a solver moves on to the next map; cancel, timeout and budget
    // errors propagate.
    let soft = |r: Result<Vec<V2>, AqError>| -> Result<Option<Vec<V2>>, AqError> {
        match r {
            Ok(x) => Ok(Some(x)),
            Err(e) if matches!(e.kind, ErrKind::Cancelled | ErrKind::Timeout | ErrKind::Budget) => Err(e),
            Err(_) => Ok(None),
        }
    };
    if !force_floater {
        // 1. Field-aligned least squares.
        let ls = soft(Stiffness::new(&d.mesh, fixed).and_then(|st| st.solve(&d.mesh, &x0, &tg)))?;
        if let Some(x) = ls
            && x.iter().all(|q| q[0].is_finite() && q[1].is_finite())
            && all_positive(&d.mesh, &x)
            && let Some(s) = sample(lay, &d, &x, dims, ctl)?
        {
            return Ok(done(MapKind::FieldAligned, s));
        }
    }
    // Floater start (always valid for a convex boundary).
    let bnd: Vec<(u32, V2)> = (0..nv).filter(|&v| fixed[v]).map(|v| (v as u32, d.rect[v])).collect();
    let fl = floater_mvc_ctl(&d.mesh, &d.topo, &bnd, Some(ctl))?;
    let fl_ok = all_positive(&d.mesh, &fl);
    if !force_floater && fl_ok {
        // 2. Barrier symmetric Dirichlet from the Floater map.
        let mut x = fl.clone();
        let sd = soft(sd_barrier_ctl(&d.mesh, &d.topo, &mut x, &tg, fixed, prm.sd_iters, Some(ctl)).map(|_| Vec::new()))?;
        if sd.is_some() && all_positive(&d.mesh, &x) && let Some(s) = sample(lay, &d, &x, dims, ctl)? {
            return Ok(done(MapKind::Barrier, s));
        }
    }
    // 3. Floater.
    if fl_ok && let Some(s) = sample(lay, &d, &fl, dims, ctl)? {
        return Ok(done(MapKind::Floater, s));
    }
    Err(err(ErrKind::Numerical, format!("patch {p}: no injective map (the Floater map is folded)")))
}

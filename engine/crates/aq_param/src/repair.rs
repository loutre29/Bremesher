//! Injectivity repair before the snap (DESIGN 6.9 "Injectivity repair", M1 scope).
//!
//! 1. Folded faces (C11, `det J <= eps_det` with eps from the current solution) are grown by 2
//!    vertex rings; the grown set is split into edge-connected regions.
//! 2. A region is repaired when it is a topological disk with one boundary loop, has no cone
//!    among its interior vertices, and unfolds consistently into one chart (BFS through its
//!    interior edges applying the transitions, every interior edge checked). Its boundary
//!    polygon must be simple (exact orient2d segment tests). Floater mean value coordinates
//!    then place the movable interior vertices with everything else fixed; when every det is
//!    positive, up to 10 local-global iterations of the barrier symmetric Dirichlet energy
//!    toward the targets `d'/h` follow (line search bounded by the det roots times 0.8, plus
//!    energy decrease). Otherwise the region grows by 2 more rings, up to `max_grow` times.
//! 3. Regions with an interior cone are left as they are (M1) and counted.
//!
//! Only movable vertices change: interior vertices of the region whose two rep variables are
//! free and used by no other vertex, no penalty row and no translation ([`System::movable`]).
//! Their new chart position is written back as the free values, so every corner outside the
//! fans of the moved vertices is bitwise unchanged. The fixed boundary keeps the uv area of
//! the region, so N_pred does not change.

use std::collections::BTreeSet;

use aq_core::cross::{det2, fold_eps, jacobian};
use aq_core::exact::orient2d;
use aq_core::{AqError, Ctl, CutGraph, Frames, NONE, Topology, Transition, TriMesh, V2};
use aq_linalg::{Chol, Triplets, dense};

use crate::ProxyRef;

use crate::solve::{bary_grads, face_target};
use crate::system::System;

/// Largest number of movable vertices in one region (dense Floater solve).
pub const MAX_REGION_VERTS: usize = 1000;
/// Largest region (faces) before giving up.
pub const MAX_REGION_FACES: usize = 20000;
/// Local-global barrier iterations after Floater.
pub const SD_ITERS: u32 = 10;

/// What the repair did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RepairStats {
    pub folds_before: u32,
    pub folds_after: u32,
    /// Regions that held a fold when processed.
    pub regions: u32,
    pub repaired: u32,
    /// Regions left because of an interior cone.
    pub skipped_cone: u32,
    /// Regions that could not be repaired within the growth limit.
    pub failed: u32,
    /// Growth steps taken.
    pub grows: u32,
    /// Accepted barrier iterations over all regions.
    pub sd_iters: u32,
    /// Faces of every repaired region (sorted).
    pub region_faces: Vec<Vec<u32>>,
    /// Vertices whose rep values changed (sorted).
    pub moved_verts: Vec<u32>,
    /// Accepted single-vertex moves of [`untangle`] (WP-16).
    pub untangled: u32,
}

/// Inputs of the repair shared by every region.
pub struct RepairInput<'a> {
    pub px: &'a ProxyRef<'a>,
    pub cut: &'a CutGraph,
    /// Per vertex (C9).
    pub index4: &'a [i8],
    /// Per face, the combed u-axis angle.
    pub theta_combed: &'a [f64],
    pub h_face: &'a [f64],
    /// Per vertex, its lowest-index face (rep wedge face).
    pub rep_face: &'a [u32],
}

enum Outcome {
    /// New positions (vertex, rep chart value) of the movable vertices.
    Done(Vec<(u32, V2)>, u32),
    Cone,
    Grow,
    Fail,
}

fn face_uv(sys: &System, y: &[f64], f: usize) -> [V2; 3] {
    let mut o = [[0.0; 2]; 3];
    for k in 0..3 {
        for q in 0..2 {
            let (ix, cs) = sys.slot(2 * (3 * f + k) + q);
            o[k][q] = ix
                .iter()
                .zip(cs)
                .map(|(&i, &c)| c as f64 * y[i as usize])
                .sum::<f64>()
                + 0.0;
        }
    }
    o
}

/// Signed uv angle (radians) at corner k of a triangle.
fn corner_angle_uv(uv: &[V2; 3], k: usize) -> f64 {
    let a = uv[k];
    let b = uv[(k + 1) % 3];
    let c = uv[(k + 2) % 3];
    let (u, w) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
    (u[0] * w[1] - u[1] * w[0]).atan2(u[0] * w[0] + u[1] * w[1])
}

/// WP-16 round 3: every interior vertex of the given faces winds as its index says (uv angle
/// sum `2 pi - index4 pi/2`). A fold-free solution can still wind a -1 cone a full turn short
/// (90 instead of 450 degrees), which S8 rejects ("1 axis rays instead of 5").
fn windings_ok(inp: &RepairInput, sys: &System, y: &[f64], faces: &[u32]) -> bool {
    let (m, t) = (inp.px.mesh, inp.px.topo);
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for &f in faces {
        for &v in &m.tris[f as usize] {
            if t.is_bnd_vert[v as usize] || !seen.insert(v) {
                continue;
            }
            let i4 = inp.index4.get(v as usize).copied().unwrap_or(0) as f64;
            let sum: f64 = t
                .fan_out(v)
                .map(|h| corner_angle_uv(&face_uv(sys, y, (h / 3) as usize), (h % 3) as usize))
                .sum();
            let want = std::f64::consts::TAU - i4 * std::f64::consts::FRAC_PI_2;
            if (sum - want).abs() > 0.1 {
                return false;
            }
        }
    }
    true
}

/// Float transition of halfedge h at the current translations.
fn he_trans(cut: &CutGraph, tr: &[[f64; 2]], h: u32) -> Transition {
    if !cut.is_cut[h as usize] {
        return Transition::IDENTITY;
    }
    let (a, side) = cut.he_arc[h as usize];
    if a == NONE || a as usize >= tr.len() {
        return Transition::IDENTITY;
    }
    let ta = Transition {
        r: cut.arcs[a as usize].r % 4,
        t: tr[a as usize],
    };
    if side > 0 { ta } else { ta.inverse() }
}

/// Faces sharing a vertex with `faces`, `rings` times (sorted).
fn grow(m: &TriMesh, t: &Topology, faces: &[u32], rings: u32) -> Vec<u32> {
    let mut set: BTreeSet<u32> = faces.iter().copied().collect();
    for _ in 0..rings {
        let cur: Vec<u32> = set.iter().copied().collect();
        for f in cur {
            for &v in &m.tris[f as usize] {
                for h in t.fan_out(v) {
                    set.insert(h / 3);
                }
            }
        }
    }
    set.into_iter().collect()
}

/// Closed segments ab and cd intersect (exact).
fn seg_intersect(a: V2, b: V2, c: V2, d: V2) -> bool {
    let o1 = orient2d(a, b, c);
    let o2 = orient2d(a, b, d);
    let o3 = orient2d(c, d, a);
    let o4 = orient2d(c, d, b);
    if ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0))
        && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))
    {
        return true;
    }
    let on = |p: V2, q: V2, r: V2| {
        r[0] >= p[0].min(q[0])
            && r[0] <= p[0].max(q[0])
            && r[1] >= p[1].min(q[1])
            && r[1] <= p[1].max(q[1])
    };
    (o1 == 0.0 && on(a, b, c))
        || (o2 == 0.0 && on(a, b, d))
        || (o3 == 0.0 && on(c, d, a))
        || (o4 == 0.0 && on(c, d, b))
}

/// The closed polygon is simple (no repeated point, no touching non-adjacent sides, no
/// folded-back adjacent sides).
fn polygon_simple(p: &[V2]) -> bool {
    let n = p.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        for j in i + 1..n {
            if p[i] == p[j] {
                return false;
            }
        }
    }
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        // Adjacent side (b, c): must not fold back onto ab.
        let c = p[(i + 2) % n];
        if orient2d(a, b, c) == 0.0
            && (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) < 0.0
        {
            return false;
        }
        for j in i + 2..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            if seg_intersect(a, b, p[j], p[(j + 1) % n]) {
                return false;
            }
        }
    }
    true
}

/// 2x2 helpers (row-major [[a, b], [c, d]]).
fn inv2(m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let d = det2(m);
    [[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]]
}

fn mul2(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn frob2(a: &[[f64; 2]; 2]) -> f64 {
    a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1]
}

/// Region-local data of one repair attempt.
struct Region {
    /// Region vertices (sorted) and their chart positions.
    verts: Vec<u32>,
    pos: Vec<V2>,
    /// Per local vertex: index among the movable vertices or NONE.
    mov: Vec<u32>,
    movable: Vec<u32>,
    /// Faces with local vertex indices, gradients, targets (region chart) and areas.
    faces: Vec<u32>,
    fv: Vec<[usize; 3]>,
    gb: Vec<[V2; 3]>,
    tinv: Vec<[[f64; 2]; 2]>,
    tgt: Vec<[[f64; 2]; 2]>,
    area: Vec<f64>,
}

impl Region {
    fn jac(&self, p: &[V2], i: usize) -> [[f64; 2]; 2] {
        let mut j = [[0.0; 2]; 2];
        for k in 0..3 {
            let q = p[self.fv[i][k]];
            for r in 0..2 {
                for c in 0..2 {
                    j[r][c] += q[r] * self.gb[i][k][c];
                }
            }
        }
        j
    }

    fn all_valid(&self, p: &[V2], eps: f64) -> bool {
        (0..self.faces.len()).all(|i| det2(&self.jac(p, i)) > eps)
    }

    /// Symmetric Dirichlet energy relative to the targets; infinite when a det <= eps.
    fn sd_energy(&self, p: &[V2], eps: f64) -> f64 {
        let mut e = 0.0;
        for i in 0..self.faces.len() {
            let j = self.jac(p, i);
            if det2(&j) <= eps {
                return f64::INFINITY;
            }
            let m = mul2(&j, &self.tinv[i]);
            let dm = det2(&m);
            let fm = frob2(&m);
            e += self.area[i] * (fm + fm / (dm * dm));
        }
        e
    }
}

/// One attempt on a region (faces sorted).
#[allow(clippy::too_many_arguments)]
fn try_region(
    inp: &RepairInput,
    sys: &System,
    y: &[f64],
    tr: &[[f64; 2]],
    faces: &[u32],
    in_r: &mut [bool],
    eps: f64,
    ctl: &Ctl,
) -> Result<Outcome, AqError> {
    let (m, t, fr): (&TriMesh, &Topology, &Frames) = (inp.px.mesh, inp.px.topo, inp.px.frames);
    if faces.len() > MAX_REGION_FACES {
        return Ok(Outcome::Fail);
    }
    for &f in faces {
        in_r[f as usize] = true;
    }
    let res = (|| -> Result<Outcome, AqError> {
        let inr = |f: u32| in_r[f as usize];
        // WP-16 round 2: a region with an interior cone is repaired in the rep charts.
        let has_cone = faces.iter().any(|&f| {
            m.tris[f as usize].iter().any(|&v| {
                !t.is_bnd_vert[v as usize]
                    && inp.index4.get(v as usize).copied().unwrap_or(0) != 0
                    && t.fan_out(v).all(|h| inr(h / 3))
            })
        });
        if has_cone {
            return cone_region(inp, sys, y, faces, &inr, eps, ctl);
        }
        // Disk check.
        let mut vs: BTreeSet<u32> = BTreeSet::new();
        let mut n_e = 0i64;
        let mut bnd_out: Vec<(u32, u32)> = Vec::new(); // (tail, halfedge)
        for &f in faces {
            for k in 0..3u32 {
                let h = 3 * f + k;
                vs.insert(m.tris[f as usize][k as usize]);
                let tw = t.twin[h as usize];
                if tw == NONE || !inr(tw / 3) {
                    n_e += 1;
                    bnd_out.push((t.tail(m, h), h));
                } else if h < tw {
                    n_e += 1;
                }
            }
        }
        let verts: Vec<u32> = vs.into_iter().collect();
        if verts.len() as i64 - n_e + faces.len() as i64 != 1 || bnd_out.is_empty() {
            return Ok(Outcome::Grow);
        }
        bnd_out.sort_unstable();
        for w in bnd_out.windows(2) {
            if w[0].0 == w[1].0 {
                return Ok(Outcome::Grow); // pinched boundary vertex
            }
        }
        let mut loop_v: Vec<u32> = Vec::new();
        {
            let h0 = bnd_out[0].1;
            let mut h = h0;
            loop {
                loop_v.push(t.tail(m, h));
                let hd = t.head(m, h);
                let Ok(k) = bnd_out.binary_search_by_key(&hd, |e| e.0) else {
                    return Ok(Outcome::Grow);
                };
                h = bnd_out[k].1;
                if h == h0 {
                    break;
                }
                if loop_v.len() > bnd_out.len() {
                    return Ok(Outcome::Grow);
                }
            }
        }
        if loop_v.len() != bnd_out.len() {
            return Ok(Outcome::Grow); // more than one boundary loop
        }
        let loc = |v: u32| verts.binary_search(&v).unwrap_or(usize::MAX);
        // Interior vertices and cones.
        let mut interior = vec![false; verts.len()];
        for (i, &v) in verts.iter().enumerate() {
            if t.is_bnd_vert[v as usize] {
                continue;
            }
            if t.fan_out(v).all(|h| inr(h / 3)) {
                interior[i] = true;
                if inp.index4.get(v as usize).copied().unwrap_or(0) != 0 {
                    return Ok(Outcome::Cone);
                }
            }
        }
        // Unfold into one chart.
        let fl = |f: u32| faces.binary_search(&f).unwrap_or(usize::MAX);
        let mut chart: Vec<Option<Transition>> = vec![None; faces.len()];
        chart[0] = Some(Transition::IDENTITY);
        let mut queue = vec![0usize];
        let mut qi = 0;
        while qi < queue.len() {
            let i = queue[qi];
            qi += 1;
            let Some(tf) = chart[i] else { continue };
            let f = faces[i];
            for k in 0..3 {
                let h = 3 * f + k;
                let tw = t.twin[h as usize];
                if tw == NONE || !inr(tw / 3) {
                    continue;
                }
                let j = fl(tw / 3);
                let cand = he_trans(inp.cut, tr, h).inverse().compose(&tf);
                match chart[j] {
                    None => {
                        chart[j] = Some(cand);
                        queue.push(j);
                    }
                    Some(c) => {
                        let tol = 1e-8 * (1.0 + c.t[0].abs().max(c.t[1].abs()));
                        if c.r != cand.r
                            || (c.t[0] - cand.t[0]).abs() > tol
                            || (c.t[1] - cand.t[1]).abs() > tol
                        {
                            return Ok(Outcome::Cone);
                        }
                    }
                }
            }
        }
        let charts: Vec<Transition> = match chart.into_iter().collect::<Option<Vec<_>>>() {
            Some(c) => c,
            None => return Ok(Outcome::Grow),
        };
        // Positions in the region chart.
        let mut pos: Vec<Option<V2>> = vec![None; verts.len()];
        for (i, &f) in faces.iter().enumerate() {
            let uv = face_uv(sys, y, f as usize);
            for k in 0..3 {
                let li = loc(m.tris[f as usize][k]);
                let p = charts[i].apply(uv[k]);
                match pos[li] {
                    None => pos[li] = Some(p),
                    Some(q) => {
                        let tol = 1e-7 * (1.0 + q[0].abs().max(q[1].abs()));
                        if (p[0] - q[0]).abs() > tol || (p[1] - q[1]).abs() > tol {
                            return Ok(Outcome::Fail);
                        }
                    }
                }
            }
        }
        let pos: Vec<V2> = pos.into_iter().map(|p| p.unwrap_or([0.0, 0.0])).collect();
        // Movable vertices.
        let mut mov = vec![NONE; verts.len()];
        let mut movable = Vec::new();
        for (i, &v) in verts.iter().enumerate() {
            if interior[i] && sys.movable(v) && (inp.rep_face[v as usize] != NONE) {
                mov[i] = movable.len() as u32;
                movable.push(i as u32);
            }
        }
        if movable.is_empty() {
            return Ok(Outcome::Grow);
        }
        if movable.len() > MAX_REGION_VERTS {
            return Ok(Outcome::Fail);
        }
        // Boundary polygon.
        let poly: Vec<V2> = loop_v.iter().map(|&v| pos[loc(v)]).collect();
        if !polygon_simple(&poly) {
            return Ok(Outcome::Grow);
        }
        // Face data.
        let mut reg = Region {
            verts: verts.clone(),
            pos,
            mov,
            movable,
            faces: faces.to_vec(),
            fv: Vec::with_capacity(faces.len()),
            gb: Vec::with_capacity(faces.len()),
            tinv: Vec::with_capacity(faces.len()),
            tgt: Vec::with_capacity(faces.len()),
            area: Vec::with_capacity(faces.len()),
        };
        for (i, &f) in faces.iter().enumerate() {
            let tri = m.tris[f as usize];
            reg.fv.push([loc(tri[0]), loc(tri[1]), loc(tri[2])]);
            reg.gb.push(bary_grads(fr, m, f));
            let tt = face_target(inp.theta_combed[f as usize], inp.h_face[f as usize]);
            // Rotate the target into the region chart: J_reg = R(-r) J_face.
            let rk = (4 - charts[i].r % 4) % 4;
            let c0 = aq_core::rot(rk, [tt[0], tt[2]]);
            let c1 = aq_core::rot(rk, [tt[1], tt[3]]);
            let tg = [[c0[0], c1[0]], [c0[1], c1[1]]];
            reg.tinv.push(inv2(&tg));
            reg.tgt.push(tg);
            reg.area.push(fr.area[f as usize]);
        }
        let saved = reg.pos.clone();
        // Floater mean value coordinates.
        let nm = reg.movable.len();
        let mut a = vec![0.0f64; nm * nm];
        let mut bu = vec![0.0f64; nm];
        let mut bv = vec![0.0f64; nm];
        for (mi, &li) in reg.movable.iter().enumerate() {
            let v = reg.verts[li as usize];
            for h in t.fan_out(v) {
                let f = h / 3;
                let k = (h % 3) as usize;
                let tri = m.tris[f as usize];
                let alpha = fr.corner_angle[h as usize].max(1e-3);
                let ta = (0.5 * alpha).tan();
                for j in [tri[(k + 1) % 3], tri[(k + 2) % 3]] {
                    let len = aq_core::math::dist(m.pos[v as usize], m.pos[j as usize]).max(1e-9);
                    let w = ta / len;
                    let lj = loc(j);
                    a[mi * nm + mi] += w;
                    if reg.mov[lj] != NONE {
                        a[mi * nm + reg.mov[lj] as usize] -= w;
                    } else {
                        bu[mi] += w * reg.pos[lj][0];
                        bv[mi] += w * reg.pos[lj][1];
                    }
                }
            }
        }
        ctl.check()?;
        if dense::lu_solve(nm, &a, &mut bu).is_err() || dense::lu_solve(nm, &a, &mut bv).is_err() {
            return Ok(Outcome::Fail);
        }
        for (mi, &li) in reg.movable.iter().enumerate() {
            reg.pos[li as usize] = [bu[mi], bv[mi]];
        }
        if !reg.all_valid(&reg.pos, eps) {
            reg.pos = saved;
            // WP-16 round 3: when Floater folds, foldover-free untangling from the current
            // positions (local: only the folded neighbourhood moves much). The unknowns are ordered
            // by lowest outgoing halfedge, so that the floating point work does not depend on the
            // vertex numbering (gauge_invariance_genus2).
            let mut ff_order: Vec<usize> = (0..reg.movable.len()).collect();
            ff_order.sort_by_key(|&mi| {
                t.fan_out(reg.verts[reg.movable[mi] as usize])
                    .min()
                    .unwrap_or(u32::MAX)
            });
            let mut ff_of = vec![0usize; reg.movable.len()];
            for (j, &mi) in ff_order.iter().enumerate() {
                ff_of[mi] = j;
            }
            let ff: Vec<FfFace> = (0..reg.faces.len())
                .map(|i| {
                    let cor = std::array::from_fn(|k| {
                        let li = reg.fv[i][k];
                        match reg.mov[li] {
                            NONE => FfCorner::Fixed(reg.pos[li]),
                            mi => FfCorner::Free(
                                ff_of[mi as usize],
                                [[1.0, 0.0], [0.0, 1.0]],
                                [0.0, 0.0],
                            ),
                        }
                    });
                    FfFace {
                        gb: reg.gb[i],
                        area: reg.area[i],
                        cor,
                    }
                })
                .collect();
            let mut xf: Vec<V2> = ff_order
                .iter()
                .map(|&mi| reg.pos[reg.movable[mi] as usize])
                .collect();
            let ff_ok = foldover_free(&ff, &mut xf, eps, ctl)?;
            if !ff_ok {
                return Ok(Outcome::Grow);
            }
            for (j, &mi) in ff_order.iter().enumerate() {
                reg.pos[reg.movable[mi] as usize] = xf[j];
            }
            if !reg.all_valid(&reg.pos, eps) {
                return Ok(Outcome::Grow);
            }
        }
        // Barrier symmetric Dirichlet, local-global.
        let iters = sd_barrier(&mut reg, eps, SD_ITERS, ctl)?;
        let out = reg
            .movable
            .iter()
            .map(|&li| {
                let v = reg.verts[li as usize];
                let rf = inp.rep_face[v as usize];
                let i = fl(rf);
                let x = if i == usize::MAX {
                    reg.pos[li as usize]
                } else {
                    charts[i].inverse().apply(reg.pos[li as usize])
                };
                (v, x)
            })
            .collect();
        Ok(Outcome::Done(out, iters))
    })();
    for &f in faces {
        in_r[f as usize] = false;
    }
    res
}

/// A corner as an affine function of its vertex's rep value: (M, c) with corner = M x + c.
type CornerLin = ([[f64; 2]; 2], V2);

/// WP-16 round 2 (a part of the M2 cone-aware repair, CHANGES.md): a region with interior
/// cones, which the disk repair above cannot unfold into one chart.
///
/// The unknowns are the rep values x_v of the movable interior vertices themselves. The corner
/// of v in face f is `M_fv x_v + c_fv`, with M_fv a quarter-turn rotation read off the
/// system's corner expressions; so every face works in its own chart and cones (fixed, like
/// every non-movable vertex) need no slit.
/// 1. Floater mean value coordinates, written in v's rep chart:
///    `sum_(f, w) w_vw (x_v - M_fv^T (p_fw - c_fv)) = 0`, where p_fw is the corner of w in f
///    (`M_fw x_w + c_fw` when w is movable). The system is block diagonally dominant.
/// 2. Kernel sweeps: the boundary is not convex and cones twist the neighbourhood, so a movable
///    vertex whose star is flipped or a sliver goes to the centroid of its star's kernel.
/// 3. The region is accepted when every det is above eps.
///
/// Cones and the region boundary do not move, so the signed uv area, and N_pred, do not change.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN dets count as folds
fn cone_region(
    inp: &RepairInput,
    sys: &System,
    y: &[f64],
    faces: &[u32],
    inr: &dyn Fn(u32) -> bool,
    eps: f64,
    ctl: &Ctl,
) -> Result<Outcome, AqError> {
    let (m, t, fr): (&TriMesh, &Topology, &Frames) = (inp.px.mesh, inp.px.topo, inp.px.frames);
    let mut vs: BTreeSet<u32> = BTreeSet::new();
    for &f in faces {
        vs.extend(m.tris[f as usize]);
    }
    // Movable: interior to the region, not a cone, both rep values free and unshared.
    let mut movable: Vec<u32> = vs
        .iter()
        .copied()
        .filter(|&v| {
            !t.is_bnd_vert[v as usize]
                && inp.index4.get(v as usize).copied().unwrap_or(0) == 0
                && sys.movable(v)
                && inp.rep_face.get(v as usize).copied().unwrap_or(NONE) != NONE
                && t.fan_out(v).all(|h| inr(h / 3))
        })
        .collect();
    if movable.is_empty() {
        return Ok(Outcome::Grow);
    }
    if movable.len() > MAX_FF_REGION_VERTS {
        return Ok(Outcome::Fail);
    }
    // Lowest halfedge order: the LU and the sweeps then run in the same order under a vertex
    // permutation (gauge_invariance_genus2).
    movable.sort_by_key(|&v| t.fan_out(v).min().unwrap_or(u32::MAX));
    let nm = movable.len();
    let index_of: std::collections::BTreeMap<u32, usize> =
        movable.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mi_of = |v: u32| index_of.get(&v).copied();
    // M_fv (columns: d corner / d rep value) for a movable v.
    let mmat = |f: u32, k: usize, v: u32| -> Option<[[f64; 2]; 2]> {
        let r = sys.rep_free(v);
        let mut mm = [[0.0; 2]; 2];
        for q in 0..2 {
            let (ix, cs) = sys.slot(2 * (3 * f as usize + k) + q);
            for (col, &rv) in r.iter().enumerate() {
                mm[q][col] = ix
                    .iter()
                    .zip(cs)
                    .filter(|(i, _)| **i == rv)
                    .map(|(_, c)| *c as f64)
                    .sum();
            }
        }
        let d = mm[0][0] * mm[1][1] - mm[0][1] * mm[1][0];
        let orth = (mm[0][0] - mm[1][1]).abs() < 1e-12 && (mm[0][1] + mm[1][0]).abs() < 1e-12;
        ((d - 1.0).abs() < 1e-12 && orth).then_some(mm)
    };
    let mul = |mm: &[[f64; 2]; 2], p: V2| {
        [
            mm[0][0] * p[0] + mm[0][1] * p[1],
            mm[1][0] * p[0] + mm[1][1] * p[1],
        ]
    };
    let mul_t = |mm: &[[f64; 2]; 2], p: V2| {
        [
            mm[0][0] * p[0] + mm[1][0] * p[1],
            mm[0][1] * p[0] + mm[1][1] * p[1],
        ]
    };
    // Per region face: current corners, and for movable corners (M, c).
    let fl = |f: u32| faces.binary_search(&f).unwrap_or(usize::MAX);
    let mut cur: Vec<[V2; 3]> = Vec::with_capacity(faces.len());
    let mut lin: Vec<[Option<CornerLin>; 3]> = Vec::with_capacity(faces.len());
    // The unknown of v is its corner in the face of its lowest outgoing halfedge (a canonical
    // chart: the rep face depends on the vertex numbering, and quarter-turn changes of chart
    // are exact, so a vertex permutation gives the same arithmetic). M0 maps rep to canonical.
    let mut x0: Vec<V2> = Vec::with_capacity(nm);
    let mut m0: Vec<[[f64; 2]; 2]> = Vec::with_capacity(nm);
    for &v in &movable {
        let Some(h0) = t.fan_out(v).min() else {
            return Ok(Outcome::Fail);
        };
        let (f0, k0) = (h0 / 3, (h0 % 3) as usize);
        let Some(mm) = mmat(f0, k0, v) else {
            return Ok(Outcome::Fail);
        };
        x0.push(face_uv(sys, y, f0 as usize)[k0]);
        m0.push(mm);
    }
    for &f in faces {
        let uv = face_uv(sys, y, f as usize);
        let tri = m.tris[f as usize];
        let mut l = [None; 3];
        for k in 0..3 {
            if let Some(mi) = mi_of(tri[k]) {
                let Some(mf) = mmat(f, k, tri[k]) else {
                    return Ok(Outcome::Fail);
                };
                // corner = M_fv M0^T x_can + c.
                let a = m0[mi];
                let mm = [
                    [
                        mf[0][0] * a[0][0] + mf[0][1] * a[0][1],
                        mf[0][0] * a[1][0] + mf[0][1] * a[1][1],
                    ],
                    [
                        mf[1][0] * a[0][0] + mf[1][1] * a[0][1],
                        mf[1][0] * a[1][0] + mf[1][1] * a[1][1],
                    ],
                ];
                let mx = mul(&mm, x0[mi]);
                l[k] = Some((mm, [uv[k][0] - mx[0], uv[k][1] - mx[1]]));
            }
        }
        cur.push(uv);
        lin.push(l);
    }
    // Corner (i, k) at x.
    let corner = |x: &[V2], i: usize, k: usize| -> V2 {
        match lin[i][k] {
            Some((mm, c)) => {
                let mi = mi_of(m.tris[faces[i] as usize][k]).unwrap_or(0);
                let p = mul(&mm, x[mi]);
                [p[0] + c[0], p[1] + c[1]]
            }
            None => cur[i][k],
        }
    };
    // Back to rep values: x_rep = x0_rep + M0^T (x_can - x0_can).
    let back_to_rep = |x: &[V2]| -> Vec<(u32, V2)> {
        movable
            .iter()
            .enumerate()
            .map(|(mi, &v)| {
                let r = sys.rep_free(v);
                let d = mul_t(&m0[mi], [x[mi][0] - x0[mi][0], x[mi][1] - x0[mi][1]]);
                (v, [y[r[0] as usize] + d[0], y[r[1] as usize] + d[1]])
            })
            .collect()
    };
    // Kernel sweeps, each star in its vertex's rep chart. A vertex whose kernel is empty goes
    // to the max-min point of its star (WP-16 round 3), so that a neighbour can finish the job.
    let sweeps = |x: &mut [V2]| -> Option<()> {
        for _ in 0..KERNEL_SWEEPS {
            let mut changed = false;
            for mi in 0..nm {
                let v = movable[mi];
                let p = x[mi];
                let mut edges: Vec<(V2, V2)> = Vec::new();
                let mut bad = false;
                for h in t.fan_out(v) {
                    let i = fl(h / 3);
                    let k = (h % 3) as usize;
                    let (mv, cv) = lin[i][k]?;
                    let to_rep = |q: V2| mul_t(&mv, [q[0] - cv[0], q[1] - cv[1]]);
                    let ea = to_rep(corner(x, i, (k + 1) % 3));
                    let eb = to_rep(corner(x, i, (k + 2) % 3));
                    let l2 = (eb[0] - ea[0]).powi(2) + (eb[1] - ea[1]).powi(2);
                    if !(orient2d(p, ea, eb) > KERNEL_MARGIN * l2) {
                        bad = true;
                    }
                    edges.push((ea, eb));
                }
                if !bad {
                    continue;
                }
                let np = kernel_centroid(&edges, KERNEL_MARGIN)
                    .or_else(|| kernel_centroid(&edges, 0.0))
                    .or_else(|| maxmin_point(&edges, p));
                if let Some(np) = np
                    && np != x[mi]
                {
                    x[mi] = np;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Some(())
    };
    let n_bad = |x: &[V2]| {
        faces
            .iter()
            .enumerate()
            .filter(|&(i, &f)| {
                let uv = [corner(x, i, 0), corner(x, i, 1), corner(x, i, 2)];
                !(det2(&jacobian(fr, m, f, &uv)) > eps)
            })
            .count()
    };
    // Interior cones of the region and their fans (halfedges out of the cone, in fan order),
    // for the winding check (see windings_ok).
    let cones_in: Vec<(u32, f64, Vec<u32>)> = vs
        .iter()
        .copied()
        .filter(|&v| {
            !t.is_bnd_vert[v as usize]
                && inp.index4.get(v as usize).copied().unwrap_or(0) != 0
                && t.fan_out(v).all(|h| inr(h / 3))
        })
        .map(|v| {
            let i4 = inp.index4[v as usize] as f64;
            (
                v,
                std::f64::consts::TAU - i4 * std::f64::consts::FRAC_PI_2,
                t.fan_out(v).collect(),
            )
        })
        .collect();
    let winding_bad = |x: &[V2]| -> Vec<usize> {
        (0..cones_in.len())
            .filter(|&ci| {
                let sum: f64 = cones_in[ci]
                    .2
                    .iter()
                    .map(|&h| {
                        let i = fl(h / 3);
                        let uv = [corner(x, i, 0), corner(x, i, 1), corner(x, i, 2)];
                        corner_angle_uv(&uv, (h % 3) as usize)
                    })
                    .sum();
                (sum - cones_in[ci].1).abs() > 0.1
            })
            .collect()
    };
    // 1. Floater (dense, so only on small regions), then kernel sweeps.
    if nm <= MAX_CONE_REGION_VERTS {
        let n2 = 2 * nm;
        let mut a = vec![0.0f64; n2 * n2];
        let mut rhs = vec![0.0f64; n2];
        for (mi, &v) in movable.iter().enumerate() {
            for h in t.fan_out(v) {
                let f = h / 3;
                let i = fl(f);
                let k = (h % 3) as usize;
                let Some((mv, cv)) = lin[i][k] else {
                    return Ok(Outcome::Fail);
                };
                let tri = m.tris[f as usize];
                let ta = (0.5 * fr.corner_angle[h as usize].max(1e-3)).tan();
                for kk in [(k + 1) % 3, (k + 2) % 3] {
                    let w = tri[kk];
                    let len = aq_core::math::dist(m.pos[v as usize], m.pos[w as usize]).max(1e-9);
                    let wt = ta / len;
                    a[(2 * mi) * n2 + 2 * mi] += wt;
                    a[(2 * mi + 1) * n2 + 2 * mi + 1] += wt;
                    // y_w = M_fv^T (p_fw - c_fv).
                    match lin[i][kk] {
                        Some((mw, cw)) => {
                            let mj = mi_of(w).unwrap_or(0);
                            // M_fv^T M_fw
                            let b = [
                                [
                                    mv[0][0] * mw[0][0] + mv[1][0] * mw[1][0],
                                    mv[0][0] * mw[0][1] + mv[1][0] * mw[1][1],
                                ],
                                [
                                    mv[0][1] * mw[0][0] + mv[1][1] * mw[1][0],
                                    mv[0][1] * mw[0][1] + mv[1][1] * mw[1][1],
                                ],
                            ];
                            for r in 0..2 {
                                for cc in 0..2 {
                                    a[(2 * mi + r) * n2 + 2 * mj + cc] -= wt * b[r][cc];
                                }
                            }
                            let cst = mul_t(&mv, [cw[0] - cv[0], cw[1] - cv[1]]);
                            rhs[2 * mi] += wt * cst[0];
                            rhs[2 * mi + 1] += wt * cst[1];
                        }
                        None => {
                            let p = cur[i][kk];
                            let yw = mul_t(&mv, [p[0] - cv[0], p[1] - cv[1]]);
                            rhs[2 * mi] += wt * yw[0];
                            rhs[2 * mi + 1] += wt * yw[1];
                        }
                    }
                }
            }
        }
        ctl.check()?;
        if dense::lu_solve(n2, &a, &mut rhs).is_err() {
            return Ok(Outcome::Fail);
        }
        let mut x: Vec<V2> = (0..nm).map(|mi| [rhs[2 * mi], rhs[2 * mi + 1]]).collect();
        if sweeps(&mut x).is_none() {
            return Ok(Outcome::Fail);
        }
        if n_bad(&x) == 0 && winding_bad(&x).is_empty() {
            return Ok(Outcome::Done(back_to_rep(&x), 0));
        }
    }
    // 2. WP-16 round 3: kernel sweeps from the current solution (Floater rebuilds the whole
    // region and, with cones inside, can wind a cone fan wrong; the current solution usually
    // has only a few folded faces next to the cone).
    let mut x: Vec<V2> = x0.clone();
    if sweeps(&mut x).is_none() {
        return Ok(Outcome::Fail);
    }
    if n_bad(&x) == 0 && winding_bad(&x).is_empty() {
        return Ok(Outcome::Done(back_to_rep(&x), 0));
    }
    // 3. WP-16 round 3: foldover-free untangling from the current solution. When it leaves a
    // cone wound a full turn short, the cone's movable link vertices are re-placed around it
    // at equal angles summing to the cone's angle, and the untangling runs again.
    {
        let ff: Vec<FfFace> = faces
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let cor = std::array::from_fn(|k| match lin[i][k] {
                    Some((mm, c)) => {
                        FfCorner::Free(mi_of(m.tris[f as usize][k]).unwrap_or(0), mm, c)
                    }
                    None => FfCorner::Fixed(cur[i][k]),
                });
                FfFace {
                    gb: bary_grads(fr, m, f),
                    area: fr.area[f as usize],
                    cor,
                }
            })
            .collect();
        let mut xf = x0.clone();
        for attempt in 0..2 {
            let wrong = winding_bad(&xf);
            if attempt > 0 && wrong.is_empty() {
                break;
            }
            for &ci in &wrong {
                let (_, want, fan) = &cones_in[ci];
                let th = want / fan.len() as f64;
                let (ct, st) = (th.cos(), th.sin());
                // Link radius: the mean 3D spoke length (uv and 3D share the quad unit).
                let cv = cones_in[ci].0;
                let r0 = fan
                    .iter()
                    .map(|&h| aq_core::math::dist(m.pos[cv as usize], m.pos[t.head(m, h) as usize]))
                    .sum::<f64>()
                    / fan.len().max(1) as f64;
                for &h in fan.iter().take(fan.len().saturating_sub(1)) {
                    let i = fl(h / 3);
                    let k = (h % 3) as usize;
                    let (ka, kb) = ((k + 1) % 3, (k + 2) % 3);
                    let Some((mb, cb)) = lin[i][kb] else { continue };
                    let Some(mb_i) = mi_of(m.tris[faces[i] as usize][kb]) else {
                        continue;
                    };
                    let pc = corner(&xf, i, k);
                    let pa = corner(&xf, i, ka);
                    let mut d = [pa[0] - pc[0], pa[1] - pc[1]];
                    let dl = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    if dl > 0.0 && r0 > 0.0 {
                        d = [d[0] * r0 / dl, d[1] * r0 / dl];
                    }
                    let tgt = [pc[0] + ct * d[0] - st * d[1], pc[1] + st * d[0] + ct * d[1]];
                    xf[mb_i] = mul_t(&mb, [tgt[0] - cb[0], tgt[1] - cb[1]]);
                }
            }
            let ok = foldover_free(&ff, &mut xf, eps, ctl)?;
            if ok && n_bad(&xf) == 0 && winding_bad(&xf).is_empty() {
                return Ok(Outcome::Done(back_to_rep(&xf), 0));
            }
        }
    }
    Ok(Outcome::Grow)
}

/// Kernel sweeps of [`cone_region`] after Floater.
const KERNEL_SWEEPS: u32 = 30;
/// Largest number of movable vertices of a cone region for the untangling (WP-16 round 3).
const MAX_FF_REGION_VERTS: usize = 4000;
/// Largest number of movable vertices of a cone region (dense 2n x 2n Floater solve).
const MAX_CONE_REGION_VERTS: usize = 400;
/// A star triangle (p, a, b) is accepted when the height of p over ab is above this fraction
/// of |ab| (the cone chart map back can flip slivers).
const KERNEL_MARGIN: f64 = 0.05;

/// Centroid of the kernel of a star: the points left of every directed edge a -> b (the star
/// triangles (p, a, b) are counterclockwise). None when the kernel is empty or degenerate.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn kernel_centroid(edges: &[(V2, V2)], margin: f64) -> Option<V2> {
    let c = kernel_poly_centroid(edges, margin)?;
    // Strictly inside every half-plane.
    edges
        .iter()
        .all(|&(a, b)| orient2d(c, a, b) > 0.0)
        .then_some(c)
}

/// WP-16 round 3: the max-min point of a star with an empty kernel: the centroid of the
/// smallest relaxed kernel `{p : (b - a) x (p - a) > margin |ab|^2}` with margin < 0 (found by
/// bisection), clipped to the star's bounding box. It raises the worst signed height of the
/// star so that a neighbour can finish the unfolding. None when even margin -64 is empty.
/// It works relative to `o` (the vertex's current position) and returns the centroid of the
/// kernel relaxed a little further (1.25 times the bisected margin, a non-degenerate polygon),
/// so that a translated chart gives the same point up to rounding (gauge_invariance_genus2).
fn maxmin_point(edges: &[(V2, V2)], o: V2) -> Option<V2> {
    let rel: Vec<(V2, V2)> = edges
        .iter()
        .map(|&(a, b)| ([a[0] - o[0], a[1] - o[1]], [b[0] - o[0], b[1] - o[1]]))
        .collect();
    let mut lo = -0.05;
    while kernel_poly_centroid(&rel, lo).is_none() {
        lo *= 2.0;
        if lo < -64.0 {
            return None;
        }
    }
    let mut hi = 0.0;
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if kernel_poly_centroid(&rel, mid).is_some() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let c = kernel_poly_centroid(&rel, 1.25 * lo)?;
    Some([c[0] + o[0], c[1] + o[1]])
}

/// Centroid of the half-plane intersection `{p : (b - a) x (p - a) > margin |ab|^2}` over the
/// star's edges, clipped to their bounding box; None when it is empty or degenerate.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn kernel_poly_centroid(edges: &[(V2, V2)], margin: f64) -> Option<V2> {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for (a, b) in edges {
        for p in [a, b] {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let mut poly: Vec<V2> = vec![
        [lo[0], lo[1]],
        [hi[0], lo[1]],
        [hi[0], hi[1]],
        [lo[0], hi[1]],
    ];
    for &(a, b) in edges {
        let l2 = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2);
        let side =
            |p: V2| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) - margin * l2;
        let mut out = Vec::with_capacity(poly.len() + 1);
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            let (sp, sq) = (side(p), side(q));
            if sp > 0.0 {
                out.push(p);
            }
            if (sp > 0.0) != (sq > 0.0) && sp != sq {
                let s = sp / (sp - sq);
                out.push([p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])]);
            }
        }
        poly = out;
        if poly.len() < 3 {
            return None;
        }
    }
    let n = poly.len();
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        let cr = p[0] * q[1] - q[0] * p[1];
        a2 += cr;
        cx += (p[0] + q[0]) * cr;
        cy += (p[1] + q[1]) * cr;
    }
    if !(a2 > 0.0) {
        return None;
    }
    Some([cx / (3.0 * a2), cy / (3.0 * a2)])
}

/// Local-global barrier symmetric Dirichlet iterations (the positions stay valid).
fn sd_barrier(reg: &mut Region, eps: f64, iters: u32, ctl: &Ctl) -> Result<u32, AqError> {
    let nm = reg.movable.len();
    let nfc = reg.faces.len();
    // Global matrix L_ab = sum_f A_f gB_a . gB_b over movable corners.
    let mut tr = Triplets::new(nm);
    for i in 0..nfc {
        for ka in 0..3 {
            let ma = reg.mov[reg.fv[i][ka]];
            if ma == NONE {
                continue;
            }
            for kb in 0..3 {
                let mb = reg.mov[reg.fv[i][kb]];
                if mb == NONE || mb < ma {
                    continue;
                }
                let g = reg.gb[i][ka][0] * reg.gb[i][kb][0] + reg.gb[i][ka][1] * reg.gb[i][kb][1];
                tr.push(mb as usize, ma as usize, reg.area[i] * g);
            }
        }
    }
    let Ok(lap) = tr.to_csc() else { return Ok(0) };
    let Ok(chol) = Chol::new(&lap) else {
        return Ok(0);
    };
    let mut e_old = reg.sd_energy(&reg.pos, eps);
    let mut done = 0;
    for _ in 0..iters {
        ctl.check()?;
        // Local step: closest rotation per face, target J* = Q T.
        let mut ru = vec![0.0; nm];
        let mut rv = vec![0.0; nm];
        for i in 0..nfc {
            let j = reg.jac(&reg.pos, i);
            let mm = mul2(&j, &reg.tinv[i]);
            let phi = (mm[1][0] - mm[0][1]).atan2(mm[0][0] + mm[1][1]);
            let (s, c) = phi.sin_cos();
            let js = mul2(&[[c, -s], [s, c]], &reg.tgt[i]);
            for k in 0..3 {
                let mk = reg.mov[reg.fv[i][k]];
                if mk == NONE {
                    continue;
                }
                let g = reg.gb[i][k];
                let a = reg.area[i];
                ru[mk as usize] += a * (js[0][0] * g[0] + js[0][1] * g[1]);
                rv[mk as usize] += a * (js[1][0] * g[0] + js[1][1] * g[1]);
                for l in 0..3 {
                    if reg.mov[reg.fv[i][l]] != NONE {
                        continue;
                    }
                    let gl = reg.gb[i][l];
                    let d = a * (g[0] * gl[0] + g[1] * gl[1]);
                    let p = reg.pos[reg.fv[i][l]];
                    ru[mk as usize] -= d * p[0];
                    rv[mk as usize] -= d * p[1];
                }
            }
        }
        if chol.try_solve(&mut ru).is_err() || chol.try_solve(&mut rv).is_err() {
            break;
        }
        let dir: Vec<V2> = reg
            .movable
            .iter()
            .enumerate()
            .map(|(mi, &li)| {
                [
                    ru[mi] - reg.pos[li as usize][0],
                    rv[mi] - reg.pos[li as usize][1],
                ]
            })
            .collect();
        if dir.iter().any(|d| !d[0].is_finite() || !d[1].is_finite()) {
            break;
        }
        // Step bound from the det roots.
        let mut t_max: f64 = 1.0;
        for i in 0..nfc {
            let j = reg.jac(&reg.pos, i);
            let mut kk = [[0.0; 2]; 2];
            for k in 0..3 {
                let mk = reg.mov[reg.fv[i][k]];
                if mk == NONE {
                    continue;
                }
                let d = dir[mk as usize];
                for r in 0..2 {
                    for c in 0..2 {
                        kk[r][c] += d[r] * reg.gb[i][k][c];
                    }
                }
            }
            let a0 = det2(&j);
            let a1 =
                j[0][0] * kk[1][1] + j[1][1] * kk[0][0] - j[0][1] * kk[1][0] - j[1][0] * kk[0][1];
            let a2 = det2(&kk);
            let root = if a2.abs() <= 1e-300 {
                if a1 < 0.0 { -a0 / a1 } else { f64::INFINITY }
            } else {
                let disc = a1 * a1 - 4.0 * a2 * a0;
                if disc < 0.0 {
                    f64::INFINITY
                } else {
                    let sq = disc.sqrt();
                    let r1 = (-a1 - sq) / (2.0 * a2);
                    let r2 = (-a1 + sq) / (2.0 * a2);
                    [r1, r2]
                        .into_iter()
                        .filter(|r| *r > 0.0)
                        .fold(f64::INFINITY, f64::min)
                }
            };
            t_max = t_max.min(0.8 * root);
        }
        if t_max.is_nan() || t_max <= 0.0 {
            break;
        }
        let mut step = t_max;
        let mut accepted = false;
        for _ in 0..30 {
            let mut p = reg.pos.clone();
            for (mi, &li) in reg.movable.iter().enumerate() {
                p[li as usize] = [
                    p[li as usize][0] + step * dir[mi][0],
                    p[li as usize][1] + step * dir[mi][1],
                ];
            }
            let e = reg.sd_energy(&p, eps);
            if e < e_old {
                reg.pos = p;
                e_old = e;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        done += 1;
    }
    Ok(done)
}

/// Repairs the folds of the current solution y in place (see the module documentation).
pub fn repair(
    inp: &RepairInput,
    sys: &System,
    y: &mut [f64],
    max_grow: u32,
    ctl: &Ctl,
) -> Result<RepairStats, AqError> {
    let (m, t, fr) = (inp.px.mesh, inp.px.topo, inp.px.frames);
    let nf = m.tris.len();
    let mut st = RepairStats::default();
    let uv = sys.corners(y);
    let mut dets: Vec<f64> = (0..nf)
        .map(|f| det2(&jacobian(fr, m, f as u32, &uv[f])))
        .collect();
    let eps = fold_eps(&dets);
    let is_fold = |d: f64| d.is_nan() || d <= eps;
    let folds: Vec<u32> = (0..nf as u32)
        .filter(|&f| is_fold(dets[f as usize]))
        .collect();
    st.folds_before = folds.len() as u32;
    if folds.is_empty() {
        return Ok(st);
    }
    let tr: Vec<[f64; 2]> = (0..inp.cut.arcs.len())
        .map(|a| sys.translation(a, y))
        .collect();
    // Grown fold set and its edge-connected regions.
    let grown = grow(m, t, &folds, 2);
    let mut in_g = vec![false; nf];
    for &f in &grown {
        in_g[f as usize] = true;
    }
    let mut comp = vec![NONE; nf];
    let mut regions: Vec<Vec<u32>> = Vec::new();
    for &f0 in &grown {
        if comp[f0 as usize] != NONE {
            continue;
        }
        let id = regions.len() as u32;
        let mut reg = vec![f0];
        comp[f0 as usize] = id;
        let mut qi = 0;
        while qi < reg.len() {
            let f = reg[qi];
            qi += 1;
            for k in 0..3 {
                let tw = t.twin[(3 * f + k) as usize];
                if tw == NONE {
                    continue;
                }
                let g = tw / 3;
                if in_g[g as usize] && comp[g as usize] == NONE {
                    comp[g as usize] = id;
                    reg.push(g);
                }
            }
        }
        reg.sort_unstable();
        regions.push(reg);
    }
    let mut in_r = vec![false; nf];
    let mut moved: BTreeSet<u32> = BTreeSet::new();
    for region in regions {
        ctl.check()?;
        if !region.iter().any(|&f| is_fold(dets[f as usize])) {
            continue;
        }
        st.regions += 1;
        let mut reg = region;
        let mut result = None;
        let mut cone = false;
        for attempt in 0..=max_grow {
            match try_region(inp, sys, y, &tr, &reg, &mut in_r, eps, ctl)? {
                Outcome::Done(p, it) => {
                    result = Some((p, it));
                    break;
                }
                Outcome::Cone => {
                    st.skipped_cone += 1;
                    cone = true;
                    break;
                }
                Outcome::Fail => break,
                Outcome::Grow => {
                    if attempt == max_grow {
                        break;
                    }
                    reg = grow(m, t, &reg, 2);
                    st.grows += 1;
                }
            }
        }
        let Some((newp, it)) = result else {
            if !cone {
                st.failed += 1;
            }
            continue;
        };
        // Write back and verify with the global eps.
        let old: Vec<(u32, [f64; 2])> = newp
            .iter()
            .map(|&(v, _)| {
                let r = sys.rep_free(v);
                (v, [y[r[0] as usize], y[r[1] as usize]])
            })
            .collect();
        for &(v, x) in &newp {
            let r = sys.rep_free(v);
            y[r[0] as usize] = x[0];
            y[r[1] as usize] = x[1];
        }
        let ok = reg.iter().all(|&f| {
            let u = face_uv(sys, y, f as usize);
            !is_fold(det2(&jacobian(fr, m, f, &u)))
        }) && windings_ok(inp, sys, y, &reg);
        if !ok {
            for &(v, x) in &old {
                let r = sys.rep_free(v);
                y[r[0] as usize] = x[0];
                y[r[1] as usize] = x[1];
            }
            st.failed += 1;
            continue;
        }
        for &f in &reg {
            let u = face_uv(sys, y, f as usize);
            dets[f as usize] = det2(&jacobian(fr, m, f, &u));
        }
        st.repaired += 1;
        st.sd_iters += it;
        for &(v, _) in &newp {
            moved.insert(v);
        }
        st.region_faces.push(reg);
    }
    st.folds_after = dets.iter().filter(|&&d| is_fold(d)).count() as u32;
    st.moved_verts = moved.into_iter().collect();
    Ok(st)
}

/// Per-vertex untangling (WP-16 addition, see CHANGES.md). The region repair above skips every
/// region with an interior cone (M1), and S8 cannot start a separatrix next to a folded face.
/// This pass moves single non-cone movable vertices of the remaining folded faces, one at a
/// time, inside the kernel of their star: the star of a non-cone interior vertex unfolds into
/// its rep chart (no holonomy around it), so the move only needs the vertex's own two rep
/// variables. The new position is the minimum of the MIPS energy `sum A |J|^2 / det J` over the
/// star, started from the centroid of the kernel `{p : det J_g(p) >= tau for every star face}`.
/// When the kernel is empty the vertex goes to the point that maximizes the smallest det of
/// its star (bisection on tau), so that alternating neighbours can finish the job; neighbours
/// of folded faces with a valid star make room the same way. Before the sweeps, an interior
/// cone whose fan winds a full turn short (uv angle sum 90 instead of 450 degrees, one fan
/// triangle near -180) gets the opposite edge pushed past it (`fix_cone_winding`). No move
/// may turn a vertex of the star from right to wrong winding, and candidates are visited in
/// halfedge order so that the result does not depend on the vertex numbering. A polish pass
/// then re-places the touched vertices with a near-degenerate star at their MIPS optimum.
/// Cones are never moved (their position is the fixed point of their closure rows), and
/// N_pred is unchanged (the signed uv area of a star is the area of its fixed ring polygon).
/// Returns (unfolding moves, folds left).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN dets count as folds
pub fn untangle(
    inp: &RepairInput,
    sys: &System,
    y: &mut [f64],
    sweeps: u32,
    ctl: &Ctl,
) -> Result<(u32, u32), AqError> {
    let (m, t, fr) = (inp.px.mesh, inp.px.topo, inp.px.frames);
    let nf = m.tris.len();
    let dets_of = |y: &[f64], f: u32| det2(&jacobian(fr, m, f, &face_uv(sys, y, f as usize)));
    let all: Vec<f64> = (0..nf as u32).map(|f| dets_of(y, f)).collect();
    let eps = fold_eps(&all);
    let mut med: Vec<f64> = all
        .iter()
        .copied()
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    med.sort_by(f64::total_cmp);
    let median = med.get(med.len() / 2).copied().unwrap_or(1.0);
    let tr: Vec<[f64; 2]> = (0..inp.cut.arcs.len())
        .map(|a| sys.translation(a, y))
        .collect();
    let mut moved = 0u32;
    // Vertices are visited in the order of their lowest outgoing halfedge, not by id, so that
    // the result does not depend on the vertex numbering (aq_param gauge test).
    let key = |v: u32| -> (u32, u32) { (t.fan_out(v).min().unwrap_or(u32::MAX), v) };
    let mut touched: BTreeSet<(u32, u32)> = BTreeSet::new();
    // Cone fans that wind a full turn short: push the offending edge past the cone first.
    let mut cones: Vec<u32> = (0..m.pos.len() as u32)
        .filter(|&w| {
            inp.index4.get(w as usize).copied().unwrap_or(0) != 0 && !t.is_bnd_vert[w as usize]
        })
        .collect();
    cones.sort_by_key(|&c| key(c));
    // A move must not turn a vertex of the star (v and its 1-ring) from right to wrong winding.
    let star_of = |v: u32| -> Vec<u32> {
        let mut st = vec![v];
        for h in t.fan_out(v) {
            for w in [
                t.head(m, h),
                m.tris[(h / 3) as usize][((h % 3 + 2) % 3) as usize],
            ] {
                if w != NONE && !st.contains(&w) {
                    st.push(w);
                }
            }
        }
        st
    };
    let winds_ok = |y: &[f64], st: &[u32]| -> Vec<bool> {
        st.iter().map(|&w| !wrong_winding(inp, sys, y, w)).collect()
    };
    let regressed = |a: &[bool], b: &[bool]| a.iter().zip(b).any(|(x, z)| *x && !*z);
    let mut pushed = 0u32;
    for &c in &cones {
        let was = wrong_winding(inp, sys, y, c);
        let fixed = was && fix_cone_winding(inp, sys, y, c);
        if fixed {
            pushed += 1;
            touched.insert(key(c));
            for h in t.fan_out(c) {
                let w = t.head(m, h);
                if w != NONE {
                    touched.insert(key(w));
                }
            }
        }
    }
    let _ = pushed;
    for _ in 0..sweeps {
        ctl.check()?;
        let folds: Vec<u32> = (0..nf as u32).filter(|&f| !(dets_of(y, f) > eps)).collect();
        if folds.is_empty() {
            break;
        }
        let mut cand: BTreeSet<(u32, u32)> = BTreeSet::new();
        for &f in &folds {
            for &v in &m.tris[f as usize] {
                cand.insert(key(v));
                // The 1-ring too: a neighbour with a valid star can still make room.
                for h in t.fan_out(v) {
                    let w = t.head(m, h);
                    if w != NONE {
                        cand.insert(key(w));
                    }
                }
            }
        }
        let mut progress = false;
        for (_, v) in cand {
            if t.is_bnd_vert[v as usize]
                || inp.index4.get(v as usize).copied().unwrap_or(0) != 0
                || !sys.movable(v)
                || inp.rep_face.get(v as usize).copied().unwrap_or(NONE) == NONE
            {
                continue;
            }
            let valid = t.fan_out(v).all(|h| dets_of(y, h / 3) > eps);
            if valid
                && !t.fan_out(v).any(|h| {
                    let f = h / 3;
                    m.tris[f as usize]
                        .iter()
                        .any(|&w| w != v && t.fan_out(w).any(|g| !(dets_of(y, g / 3) > eps)))
                })
            {
                continue;
            }
            if let Some((p, full)) = untangle_vertex(inp, sys, y, &tr, v, eps, median, valid)? {
                let r = sys.rep_free(v);
                let old = [y[r[0] as usize], y[r[1] as usize]];
                let min_det = |y: &[f64]| {
                    t.fan_out(v)
                        .map(|h| dets_of(y, h / 3))
                        .fold(f64::INFINITY, f64::min)
                };
                let before = min_det(y);
                let st = star_of(v);
                let ok0 = winds_ok(y, &st);
                y[r[0] as usize] = p[0];
                y[r[1] as usize] = p[1];
                let after = if regressed(&ok0, &winds_ok(y, &st)) {
                    f64::NEG_INFINITY
                } else {
                    min_det(y)
                };
                if full && after > eps && !(before > eps) {
                    moved += 1;
                    progress = true;
                    touched.insert(key(v));
                } else if !full && after > before + 1e-3 * median {
                    progress = true;
                    touched.insert(key(v));
                } else {
                    y[r[0] as usize] = old[0];
                    y[r[1] as usize] = old[1];
                }
            }
        }
        if !progress {
            break;
        }
    }
    // Polish: the alternating max-min moves leave stars with near-zero dets. Re-place the touched vertices and their neighbours
    // at the MIPS optimum of their kernel while that raises the star's smallest det.
    if !touched.is_empty() {
        let mut ring: BTreeSet<(u32, u32)> = BTreeSet::new();
        for &(_, v) in &touched {
            ring.insert(key(v));
            for h in t.fan_out(v) {
                let w = t.head(m, h);
                if w != NONE {
                    ring.insert(key(w));
                }
            }
        }
        let thr = 0.05 * median;
        for _ in 0..3 {
            ctl.check()?;
            let mut any = false;
            for &(_, v) in &ring {
                if t.is_bnd_vert[v as usize]
                    || inp.index4.get(v as usize).copied().unwrap_or(0) != 0
                    || !sys.movable(v)
                    || inp.rep_face.get(v as usize).copied().unwrap_or(NONE) == NONE
                {
                    continue;
                }
                let min_det = |y: &[f64]| {
                    t.fan_out(v)
                        .map(|h| dets_of(y, h / 3))
                        .fold(f64::INFINITY, f64::min)
                };
                let before = min_det(y);
                if !(before > eps) || before >= thr {
                    continue;
                }
                if let Some((p, true)) = untangle_vertex(inp, sys, y, &tr, v, eps, median, false)? {
                    let r = sys.rep_free(v);
                    let old = [y[r[0] as usize], y[r[1] as usize]];
                    let st = star_of(v);
                    let ok0 = winds_ok(y, &st);
                    y[r[0] as usize] = p[0];
                    y[r[1] as usize] = p[1];
                    if min_det(y) > before && !regressed(&ok0, &winds_ok(y, &st)) {
                        any = true;
                    } else {
                        y[r[0] as usize] = old[0];
                        y[r[1] as usize] = old[1];
                    }
                }
            }
            if !any {
                break;
            }
        }
    }
    let left = (0..nf as u32).filter(|&f| !(dets_of(y, f) > eps)).count() as u32;
    Ok((moved, left))
}

/// Signed uv angle sum at w over its fan (chart invariant).
fn angle_sum(t: &Topology, sys: &System, y: &[f64], w: u32) -> f64 {
    t.fan_out(w)
        .map(|h| {
            let f = (h / 3) as usize;
            let k = (h % 3) as usize;
            let uv = face_uv(sys, y, f);
            let a = [uv[(k + 1) % 3][0] - uv[k][0], uv[(k + 1) % 3][1] - uv[k][1]];
            let b = [uv[(k + 2) % 3][0] - uv[k][0], uv[(k + 2) % 3][1] - uv[k][1]];
            (a[0] * b[1] - a[1] * b[0]).atan2(a[0] * b[0] + a[1] * b[1])
        })
        .sum()
}

/// True when the interior vertex w winds to the wrong total angle: its uv angle sum is more
/// than pi away from `2 pi - index4 pi / 2`.
fn wrong_winding(inp: &RepairInput, sys: &System, y: &[f64], w: u32) -> bool {
    let t = inp.px.topo;
    if t.is_bnd_vert[w as usize] {
        return false;
    }
    let i4 = f64::from(inp.index4.get(w as usize).copied().unwrap_or(0));
    let target = std::f64::consts::TAU - i4 * std::f64::consts::FRAC_PI_2;
    (angle_sum(t, sys, y, w) - target).abs() > std::f64::consts::PI
}

/// Sets the corner of movable vertex w in face f (corner k) to `u` by solving for its two rep
/// variables (the corner is an affine function of them, probed numerically).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN dets count as folds
fn set_corner(sys: &System, y: &mut [f64], w: u32, f: usize, k: usize, u: V2) -> bool {
    let rv = sys.rep_free(w);
    if rv.contains(&NONE) {
        return false;
    }
    let base = face_uv(sys, y, f)[k];
    let y0 = [y[rv[0] as usize], y[rv[1] as usize]];
    y[rv[0] as usize] = y0[0] + 1.0;
    let ex = face_uv(sys, y, f)[k];
    y[rv[0] as usize] = y0[0];
    y[rv[1] as usize] = y0[1] + 1.0;
    let ey = face_uv(sys, y, f)[k];
    y[rv[1] as usize] = y0[1];
    let a = [ex[0] - base[0], ex[1] - base[1]];
    let b = [ey[0] - base[0], ey[1] - base[1]];
    let det = a[0] * b[1] - a[1] * b[0];
    if !(det.abs() > 0.5) {
        return false;
    }
    let d = [u[0] - base[0], u[1] - base[1]];
    y[rv[0] as usize] = y0[0] + (d[0] * b[1] - d[1] * b[0]) / det;
    y[rv[1] as usize] = y0[1] + (a[0] * d[1] - a[1] * d[0]) / det;
    true
}

/// WP-16: an interior cone whose fan winds a full turn short has one fan triangle (c, w1, w2)
/// with an angle near -180 degrees at c: the cone sits just outside the opposite edge. The
/// edge is pushed past the cone (w1 and w2 move together by the same uv offset) so that the
/// triangle turns positive and the fan winds right; the sweeps then repair the 2-ring.
/// Returns false when no edge can be pushed (y may be partly written; see fix_cone_winding).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN dets count as folds
fn push_cone_edge(inp: &RepairInput, sys: &System, y: &mut [f64], c: u32) -> bool {
    let (m, t) = (inp.px.mesh, inp.px.topo);
    let mut worst: Option<(f64, u32)> = None;
    for h in t.fan_out(c) {
        let f = (h / 3) as usize;
        let k = (h % 3) as usize;
        let uv = face_uv(sys, y, f);
        let a = [uv[(k + 1) % 3][0] - uv[k][0], uv[(k + 1) % 3][1] - uv[k][1]];
        let b = [uv[(k + 2) % 3][0] - uv[k][0], uv[(k + 2) % 3][1] - uv[k][1]];
        let ang = (a[0] * b[1] - a[1] * b[0]).atan2(a[0] * b[0] + a[1] * b[1]);
        if worst.is_none_or(|(x, _)| ang < x) {
            worst = Some((ang, h));
        }
    }
    let Some((ang, h)) = worst else { return false };
    if !(ang < 0.0) {
        return false;
    }
    let f = (h / 3) as usize;
    let k = (h % 3) as usize;
    let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
    let tri = m.tris[f];
    let (w1, w2) = (tri[k1], tri[k2]);
    for w in [w1, w2] {
        if t.is_bnd_vert[w as usize]
            || !sys.movable(w)
            || inp.index4.get(w as usize).copied().unwrap_or(0) != 0
        {
            return false;
        }
    }
    let uv = face_uv(sys, y, f);
    let (pc, p1, p2) = (uv[k], uv[k1], uv[k2]);
    let e = [p2[0] - p1[0], p2[1] - p1[1]];
    let len = (e[0] * e[0] + e[1] * e[1]).sqrt();
    if !(len > 0.0) {
        return false;
    }
    // Unit normal of the edge pointing toward the cone, and the cone's distance to the line.
    let mut n = [-e[1] / len, e[0] / len];
    let mut dist = (pc[0] - p1[0]) * n[0] + (pc[1] - p1[1]) * n[1];
    if dist < 0.0 {
        n = [-n[0], -n[1]];
        dist = -dist;
    }
    let s = dist + 0.05 * len;
    set_corner(sys, y, w1, f, k1, [p1[0] + s * n[0], p1[1] + s * n[1]])
        && set_corner(sys, y, w2, f, k2, [p2[0] + s * n[0], p2[1] + s * n[1]])
}

/// Up to three edge pushes at cone c until its fan winds right; y is restored on failure.
fn fix_cone_winding(inp: &RepairInput, sys: &System, y: &mut [f64], c: u32) -> bool {
    let saved = y.to_vec();
    for _ in 0..3 {
        if !push_cone_edge(inp, sys, y, c) {
            break;
        }
        if !wrong_winding(inp, sys, y, c) {
            return true;
        }
    }
    y.copy_from_slice(&saved);
    false
}

/// One star triangle (p, a, b) in the rep chart of the moved vertex.
struct StarTri {
    a: V2,
    b: V2,
    gb: [V2; 3],
    area: f64,
}

/// The new rep-chart position of vertex v and whether it unfolds the whole star (false: the
/// star has an empty kernel and the point maximizes its smallest det instead), or None.
/// With `maxmin` the kernel step is skipped and the max-min point is returned directly.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN dets count as folds
fn untangle_vertex(
    inp: &RepairInput,
    sys: &System,
    y: &[f64],
    tr: &[[f64; 2]],
    v: u32,
    eps: f64,
    median: f64,
    maxmin: bool,
) -> Result<Option<(V2, bool)>, AqError> {
    let (m, t, fr) = (inp.px.mesh, inp.px.topo, inp.px.frames);
    let rf = inp.rep_face[v as usize];
    // Charts of the faces of the fan, relative to the rep face (BFS around the fan through
    // the edges incident to v).
    let fan: Vec<u32> = t.fan_out(v).map(|h| h / 3).collect();
    if fan.is_empty() || !fan.contains(&rf) {
        return Ok(None);
    }
    let mut chart: Vec<(u32, Transition)> = vec![(rf, Transition::IDENTITY)];
    let mut qi = 0;
    while qi < chart.len() {
        let (f, tf) = chart[qi];
        qi += 1;
        for k in 0..3u32 {
            let h = 3 * f + k;
            let tw = t.twin[h as usize];
            if tw == NONE {
                continue;
            }
            let g = tw / 3;
            if !fan.contains(&g) || chart.iter().any(|c| c.0 == g) {
                continue;
            }
            let tri = m.tris[f as usize];
            if tri[k as usize] != v && tri[((k + 1) % 3) as usize] != v {
                continue;
            }
            chart.push((g, he_trans(inp.cut, tr, h).inverse().compose(&tf)));
        }
    }
    if chart.len() != fan.len() {
        return Ok(None);
    }
    let mut star: Vec<StarTri> = Vec::with_capacity(chart.len());
    let mut p_cur = [f64::NAN, f64::NAN];
    for &(f, tf) in &chart {
        let tri = m.tris[f as usize];
        let Some(k) = (0..3).find(|&k| tri[k] == v) else {
            return Ok(None);
        };
        let uv = face_uv(sys, y, f as usize);
        let gb0 = crate::solve::bary_grads(fr, m, f);
        let (ka, kb) = ((k + 1) % 3, (k + 2) % 3);
        if f == rf {
            p_cur = uv[k];
        }
        star.push(StarTri {
            a: tf.apply(uv[ka]),
            b: tf.apply(uv[kb]),
            gb: [gb0[k], gb0[ka], gb0[kb]],
            area: fr.area[f as usize],
        });
    }
    let r = sys.rep_free(v);
    if !((p_cur[0] - y[r[0] as usize]).abs() <= 1e-9 * (1.0 + p_cur[0].abs())
        && (p_cur[1] - y[r[1] as usize]).abs() <= 1e-9 * (1.0 + p_cur[1].abs()))
    {
        return Ok(None);
    }
    // Work relative to the current position (translation-invariant rounding).
    let o = p_cur;
    for s in star.iter_mut() {
        s.a = [s.a[0] - o[0], s.a[1] - o[1]];
        s.b = [s.b[0] - o[0], s.b[1] - o[1]];
    }
    let p_cur: V2 = [0.0, 0.0];
    // det J = signed uv area / 3D area.
    let det_at = |s: &StarTri, p: V2| {
        let (u, w) = (
            [s.a[0] - p[0], s.a[1] - p[1]],
            [s.b[0] - p[0], s.b[1] - p[1]],
        );
        0.5 * (u[0] * w[1] - u[1] * w[0]) / s.area
    };
    let energy = |p: V2| -> f64 {
        let mut e = 0.0;
        for s in &star {
            let d = det_at(s, p);
            if !(d > eps) {
                return f64::INFINITY;
            }
            let q = [p, s.a, s.b];
            let mut j = [[0.0; 2]; 2];
            for kk in 0..3 {
                for rr in 0..2 {
                    for cc in 0..2 {
                        j[rr][cc] += q[kk][rr] * s.gb[kk][cc];
                    }
                }
            }
            e += s.area * frob2(&j) / d;
        }
        e
    };
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for s in &star {
        for q in [s.a, s.b] {
            for c in 0..2 {
                lo[c] = lo[c].min(q[c]);
                hi[c] = hi[c].max(q[c]);
            }
        }
    }
    let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12);
    // Centroid of the kernel { p : det >= tau on every star triangle }, None when it is empty.
    let kernel = |tau: f64| -> Option<V2> {
        let mut poly: Vec<V2> = vec![
            [lo[0] - ext, lo[1] - ext],
            [hi[0] + ext, lo[1] - ext],
            [hi[0] + ext, hi[1] + ext],
            [lo[0] - ext, hi[1] + ext],
        ];
        for s in &star {
            // (a - p) x (b - p) >= 2 tau A  <=>  a x b + p x (a - b) - 2 tau A >= 0.
            let c = s.a[0] * s.b[1] - s.a[1] * s.b[0] - 2.0 * tau * s.area;
            let d = [s.a[0] - s.b[0], s.a[1] - s.b[1]];
            let val = |p: V2| c + p[0] * d[1] - p[1] * d[0];
            let mut out: Vec<V2> = Vec::with_capacity(poly.len() + 2);
            for i in 0..poly.len() {
                let (p0, p1) = (poly[i], poly[(i + 1) % poly.len()]);
                let (v0, v1) = (val(p0), val(p1));
                if v0 >= 0.0 {
                    out.push(p0);
                }
                if (v0 >= 0.0) != (v1 >= 0.0) {
                    let tt = v0 / (v0 - v1);
                    out.push([p0[0] + tt * (p1[0] - p0[0]), p0[1] + tt * (p1[1] - p0[1])]);
                }
            }
            poly = out;
            if poly.len() < 3 {
                return None;
            }
        }
        // Shoelace centroid relative to the vertex mean (no cancellation on a thin kernel);
        // the mean itself when the area underflows (both lie inside the convex kernel).
        let k = poly.len() as f64;
        let o = poly
            .iter()
            .fold([0.0, 0.0], |s, q| [s[0] + q[0] / k, s[1] + q[1] / k]);
        let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
        for i in 0..poly.len() {
            let (p0, p1) = (poly[i], poly[(i + 1) % poly.len()]);
            let (p0, p1) = ([p0[0] - o[0], p0[1] - o[1]], [p1[0] - o[0], p1[1] - o[1]]);
            let cr = p0[0] * p1[1] - p1[0] * p0[1];
            a2 += cr;
            cx += (p0[0] + p1[0]) * cr;
            cy += (p0[1] + p1[1]) * cr;
        }
        if !(a2.abs() > 0.0) {
            return Some(o);
        }
        Some([o[0] + cx / (3.0 * a2), o[1] + cy / (3.0 * a2)])
    };
    // Kernel for det >= tau, tau relaxed toward eps when it is empty.
    for frac in [0.05, 0.01, 1e-3] {
        if maxmin {
            break;
        }
        let tau = (frac * median).max(2.0 * eps);
        let Some(p0) = kernel(tau) else { continue };
        let mut p = p0;
        let mut e = energy(p);
        if !e.is_finite() {
            continue;
        }
        // Descent with backtracking (numerical gradient, 2 unknowns).
        let hstep = 1e-6 * ext;
        for _ in 0..20 {
            let gx = (energy([p[0] + hstep, p[1]]) - energy([p[0] - hstep, p[1]])) / (2.0 * hstep);
            let gy = (energy([p[0], p[1] + hstep]) - energy([p[0], p[1] - hstep])) / (2.0 * hstep);
            let gn = (gx * gx + gy * gy).sqrt();
            if !(gn.is_finite() && gn > 0.0) {
                break;
            }
            let mut s = 0.25 * ext;
            let mut improved = false;
            for _ in 0..30 {
                let q = [p[0] - s * gx / gn, p[1] - s * gy / gn];
                let eq = energy(q);
                if eq < e {
                    p = q;
                    e = eq;
                    improved = true;
                    break;
                }
                s *= 0.5;
            }
            if !improved {
                break;
            }
        }
        return Ok(Some(([p[0] + o[0], p[1] + o[1]], true)));
    }
    // WP-16: empty kernel. Move to the point that maximizes the smallest det of the star (a
    // 3-variable LP solved by bisection on tau), so that a neighbour can finish the job.
    let cur = star
        .iter()
        .map(|s| det_at(s, p_cur))
        .fold(f64::INFINITY, f64::min);
    if !cur.is_finite() {
        return Ok(None);
    }
    let (mut t_lo, mut t_hi) = (
        cur,
        if maxmin {
            2.0 * median
        } else {
            (1e-3 * median).max(2.0 * eps)
        },
    );
    if !(t_hi > t_lo) {
        return Ok(None);
    }
    let mut best = None;
    for _ in 0..40 {
        let tm = 0.5 * (t_lo + t_hi);
        match kernel(tm) {
            Some(q) => {
                best = Some(q);
                t_lo = tm;
            }
            None => t_hi = tm,
        }
    }
    Ok(best.map(|q| ([q[0] + o[0], q[1] + o[1]], false)))
}

/// A corner of a face in [`foldover_free`]: fixed, or `M x[mi] + c` for movable vertex mi.
#[derive(Clone, Copy)]
enum FfCorner {
    Fixed(V2),
    Free(usize, [[f64; 2]; 2], V2),
}

/// A face of [`foldover_free`]: barycentric gradients in the 3D face frame, 3D area, corners.
struct FfFace {
    gb: [[f64; 2]; 3],
    area: f64,
    cor: [FfCorner; 3],
}

/// Area term weight of the untangling energy.
const FF_LAMBDA: f64 = 1.0;
/// Outer rounds (epsilon updates) and L-BFGS iterations per round of [`foldover_free`].
const FF_ROUNDS: usize = 30;
/// Rounds after which a fold-free result is accepted without waiting for convergence.
const FF_MIN_ROUNDS: usize = 4;
const FF_ITERS: usize = 50;
/// L-BFGS history length.
const FF_HIST: usize = 8;

fn ff_jac(fc: &FfFace, x: &[V2]) -> [[f64; 2]; 2] {
    let mut j = [[0.0; 2]; 2];
    for k in 0..3 {
        let q = match fc.cor[k] {
            FfCorner::Fixed(p) => p,
            FfCorner::Free(mi, mm, c) => [
                mm[0][0] * x[mi][0] + mm[0][1] * x[mi][1] + c[0],
                mm[1][0] * x[mi][0] + mm[1][1] * x[mi][1] + c[1],
            ],
        };
        for r in 0..2 {
            for cc in 0..2 {
                j[r][cc] += q[r] * fc.gb[k][cc];
            }
        }
    }
    j
}

fn ff_det(fc: &FfFace, x: &[V2]) -> f64 {
    let j = ff_jac(fc, x);
    j[0][0] * j[1][1] - j[0][1] * j[1][0]
}

/// Energy (and gradient) of the regularized untangling energy of Garanzha et al. 2021
/// ("Foldover-free maps in 50 lines of code"): `sum A (|J|^2 + lambda (det^2 + 1)) / chi(det)`
/// with `chi(d) = (d + sqrt(eps^2 + d^2)) / 2`. It is finite for any input, folds included.
fn ff_energy(faces: &[FfFace], x: &[V2], eps: f64, mut grad: Option<&mut [V2]>) -> f64 {
    let mut e = 0.0;
    if let Some(g) = grad.as_deref_mut() {
        g.iter_mut().for_each(|v| *v = [0.0, 0.0]);
    }
    for fc in faces {
        let j = ff_jac(fc, x);
        let d = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let s = (eps * eps + d * d).sqrt();
        let chi = 0.5 * (d + s);
        let dchi = 0.5 * (1.0 + d / s);
        let fro = j[0][0] * j[0][0] + j[0][1] * j[0][1] + j[1][0] * j[1][0] + j[1][1] * j[1][1];
        let num = fro + FF_LAMBDA * (d * d + 1.0);
        e += fc.area * num / chi;
        if let Some(g) = grad.as_deref_mut() {
            // dE/dJ = A 2 J / chi + A (2 lambda d / chi - num chi' / chi^2) cof(J).
            let cof = [[j[1][1], -j[1][0]], [-j[0][1], j[0][0]]];
            let a1 = 2.0 * fc.area / chi;
            let a2 = fc.area * (2.0 * FF_LAMBDA * d / chi - num * dchi / (chi * chi));
            let mut dj = [[0.0; 2]; 2];
            for r in 0..2 {
                for cc in 0..2 {
                    dj[r][cc] = a1 * j[r][cc] + a2 * cof[r][cc];
                }
            }
            for k in 0..3 {
                if let FfCorner::Free(mi, mm, _) = fc.cor[k] {
                    // dE/dq = dJ gb_k, dE/dx = M^T dE/dq.
                    let dq = [
                        dj[0][0] * fc.gb[k][0] + dj[0][1] * fc.gb[k][1],
                        dj[1][0] * fc.gb[k][0] + dj[1][1] * fc.gb[k][1],
                    ];
                    g[mi][0] += mm[0][0] * dq[0] + mm[1][0] * dq[1];
                    g[mi][1] += mm[0][1] * dq[0] + mm[1][1] * dq[1];
                }
            }
        }
    }
    e
}

fn ff_min_det(faces: &[FfFace], x: &[V2]) -> f64 {
    faces
        .iter()
        .map(|fc| ff_det(fc, x))
        .fold(f64::INFINITY, f64::min)
}

fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// WP-16 round 3: untangles the movable vertices x of a region with a fixed boundary by
/// minimizing [`ff_energy`] with L-BFGS under the epsilon schedule of Garanzha et al. Returns
/// true when every det ends above `det_eps`. Sequential with fixed caps, so deterministic.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn foldover_free(faces: &[FfFace], x: &mut [V2], det_eps: f64, ctl: &Ctl) -> Result<bool, AqError> {
    let n = x.len();
    if n == 0 || faces.is_empty() {
        return Ok(false);
    }
    let mut md = ff_min_det(faces, x);
    if !md.is_finite() {
        return Ok(false);
    }
    let mean_det = faces.iter().map(|fc| ff_det(fc, x).abs()).sum::<f64>() / faces.len() as f64;
    if !(mean_det > 0.0) {
        return Ok(false);
    }
    let mut eps = if md > 0.0 {
        1e-3 * mean_det
    } else {
        (1e-6 * mean_det * mean_det + 0.04 * md * md).sqrt()
    };
    let mut grad = vec![[0.0; 2]; n];
    for round in 0..FF_ROUNDS {
        ctl.check()?;
        let e_prev = ff_energy(faces, x, eps, None);
        let mut s_hist: Vec<Vec<f64>> = Vec::new();
        let mut y_hist: Vec<Vec<f64>> = Vec::new();
        let mut e = ff_energy(faces, x, eps, Some(&mut grad));
        let mut gflat: Vec<f64> = grad.iter().flat_map(|g| [g[0], g[1]]).collect();
        for _ in 0..FF_ITERS {
            // Two-loop recursion.
            let mut q = gflat.clone();
            let k = s_hist.len();
            let mut alpha = vec![0.0; k];
            for i in (0..k).rev() {
                let rho = 1.0 / dotv(&y_hist[i], &s_hist[i]);
                alpha[i] = rho * dotv(&s_hist[i], &q);
                for (qq, yy) in q.iter_mut().zip(&y_hist[i]) {
                    *qq -= alpha[i] * yy;
                }
            }
            if k > 0 {
                let gamma =
                    dotv(&s_hist[k - 1], &y_hist[k - 1]) / dotv(&y_hist[k - 1], &y_hist[k - 1]);
                q.iter_mut().for_each(|v| *v *= gamma);
            } else {
                let gn = dotv(&q, &q).sqrt().max(1e-300);
                let sc = 1e-2 / gn;
                q.iter_mut().for_each(|v| *v *= sc);
            }
            for i in 0..k {
                let rho = 1.0 / dotv(&y_hist[i], &s_hist[i]);
                let beta = rho * dotv(&y_hist[i], &q);
                for (qq, ss) in q.iter_mut().zip(&s_hist[i]) {
                    *qq += (alpha[i] - beta) * ss;
                }
            }
            let dir: Vec<f64> = q.iter().map(|v| -v).collect();
            let slope = dotv(&dir, &gflat);
            if !(slope < 0.0) {
                break;
            }
            // Backtracking line search (Armijo).
            let x_old: Vec<V2> = x.to_vec();
            let mut step = 1.0;
            let mut ok = false;
            for _ in 0..40 {
                for i in 0..n {
                    x[i] = [
                        x_old[i][0] + step * dir[2 * i],
                        x_old[i][1] + step * dir[2 * i + 1],
                    ];
                }
                let e_new = ff_energy(faces, x, eps, None);
                if e_new.is_finite() && e_new <= e + 1e-4 * step * slope {
                    ok = true;
                    break;
                }
                step *= 0.5;
            }
            if !ok {
                x.copy_from_slice(&x_old);
                break;
            }
            let g_old = std::mem::take(&mut gflat);
            let e_old = e;
            e = ff_energy(faces, x, eps, Some(&mut grad));
            gflat = grad.iter().flat_map(|g| [g[0], g[1]]).collect();
            let s: Vec<f64> = dir.iter().map(|d| step * d).collect();
            let yv: Vec<f64> = gflat.iter().zip(&g_old).map(|(a, b)| a - b).collect();
            if dotv(&s, &yv) > 1e-300 {
                if s_hist.len() == FF_HIST {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
                s_hist.push(s);
                y_hist.push(yv);
            }
            if (e_old - e).abs() <= 1e-12 * e.abs() {
                break;
            }
        }
        md = ff_min_det(faces, x);
        if md > det_eps && (round >= FF_MIN_ROUNDS || (e_prev - e).abs() <= 1e-5 * e.abs()) {
            break;
        }
        // Epsilon update (Garanzha et al.).
        let sigma = (1.0 - e / e_prev).max(0.1);
        let chi_e = 0.5 * (md + (eps * eps + md * md).sqrt());
        let mu = (1.0 - sigma) * chi_e;
        eps = if md < mu {
            2.0 * (mu * (mu - md)).max(0.0).sqrt()
        } else {
            1e-12 * mean_det
        };
        if !(eps > 0.0) {
            eps = 1e-12 * mean_det;
        }
    }
    Ok(ff_min_det(faces, x) > det_eps)
}

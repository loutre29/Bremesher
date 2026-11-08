//! Exact seamless parametrizations (DESIGN 6.2 "Analytic contracts", C10, C14).
//!
//! Every param is built the same way: the per-corner uv is constructed exactly (dyadic, 2^-24),
//! then [`param_from_corners`] derives everything else from it:
//! * the transition of every interior halfedge from its two endpoints (`R(-r) a_f + t = a_g`,
//!   exact), `is_cut` = non-identity transitions;
//! * the cross field: per face the angle of the rotation part of the Jacobian (grad u, C11),
//!   snapped to boundary and chain edges, matching rounded (C7); the rounded matching must equal
//!   the transition rotations (`p == r`, comb offsets 0, C12);
//! * the cut graph (arcs between nodes, C12 r), `he_family` on chains (C13), cones, fold flags.
//!
//! The result is checked with `SeamlessParam::validate`, `CrossField::validate` and
//! `CutGraph::validate`. Closed iso-lines can be traced into exact [`LoopCand`]s.

use std::f64::consts::TAU;

use aq_core::cross::{det2, edge_family, folded_faces, jacobian, rot};
use aq_core::math;
use aq_core::param::FAMILY_NONE;
use aq_core::tol::{DYADIC_QUANTUM, snap_dyadic};
use aq_core::{
    AqError, CrossField, CutGraph, FeatureKind, LoopCand, NONE, ParamReport, SeamlessParam, SizeField,
    SplitMix64, SurfacePoint, Topology, Transition, TriMesh, V2, V3,
};

use crate::cut::cut_graph_from_flags;
use crate::field::{snap_to_constraints, uniform_size};
use crate::meshes::{CUBE_FACES, cube_lattice, extreme_axes};
use crate::surface::{CORNER_DEG, SYM_CORNER_DEG, Surface};
use crate::util::{bad, ensure, inv, snap_to};

/// Mesh options of the analytic params.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamOpts {
    /// Mesh vertices per uv unit along each direction (the proxy runs at about h/3).
    pub res: u32,
    /// Vertex jitter as a fraction of the mesh spacing, in [0, 0.25] (larger values can fold
    /// triangles). 0 gives the grid-aligned variant whose iso values hit vertices exactly (SoS
    /// cases).
    pub jitter: f64,
    pub seed: u64,
}

impl Default for ParamOpts {
    fn default() -> Self {
        ParamOpts { res: 3, jitter: 0.25, seed: 1 }
    }
}

impl ParamOpts {
    /// Grid-aligned variant (no jitter).
    pub fn aligned(res: u32) -> Self {
        ParamOpts { res, jitter: 0.0, seed: 1 }
    }
}

/// An analytic param with every upstream contract it is consistent with.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamCase {
    pub surface: Surface,
    /// Field derived from the param (theta = rotation part of J), matching p == r.
    pub field: CrossField,
    /// Cut = the halfedges with non-identity transitions; comb offsets all 0.
    pub cut: CutGraph,
    pub param: SeamlessParam,
    /// Uniform h = 1 (quad units).
    pub size: SizeField,
    /// Exact closed iso-line loops; `param.loops_exact` lists all of them.
    pub loops: Vec<LoopCand>,
    /// Per vertex: uv in its rep wedge (the wedge of its lowest-index face, DESIGN 6.9).
    pub rep_uv: Vec<V2>,
}

impl ParamCase {
    /// Re-runs every validator.
    pub fn validate(&self) -> Result<(), AqError> {
        let s = &self.surface;
        self.field.validate(&s.mesh, &s.topo, &s.frames)?;
        self.cut.validate(&s.mesh, &s.topo, Some(&self.field))?;
        self.param.validate(&s.mesh, &s.topo)?;
        self.size.validate(&s.frames, s.n_verts())?;
        for l in &self.loops {
            l.validate(&s.mesh, &s.topo)?;
        }
        Ok(())
    }

    /// Number of folded faces (C11).
    pub fn n_folded(&self) -> usize {
        self.param.folded.iter().filter(|&&f| f).count()
    }

    /// The lowest face whose uv triangle strictly contains `p` (in that face's chart).
    pub fn face_at(&self, p: V2) -> Option<u32> {
        (0..self.param.uv.len() as u32).find(|&f| {
            let q = self.param.uv[f as usize];
            let o = |a: V2, b: V2| aq_core::exact::orient2d_sign(a, b, p);
            o(q[0], q[1]) > 0 && o(q[1], q[2]) > 0 && o(q[2], q[0]) > 0
        })
    }

    /// Traces the closed iso-line through `start` (a uv point in the chart of its face, see
    /// [`ParamCase::face_at`]) along `axis` (0: u varies, v = start[1] is constant; 1: v varies)
    /// in the `forward` (+) or backward direction, appends it to `loops` and `loops_exact`, and
    /// returns its index. The constant coordinate is snapped to 2^-24; it must not equal a
    /// corner coordinate on the way (pick a generic value).
    pub fn add_iso_loop(&mut self, start: V2, axis: u8, forward: bool, ring: bool) -> Result<u32, AqError> {
        let f0 = self.face_at(start).ok_or_else(|| bad(format!("no face contains uv {start:?}")))?;
        let l = trace_iso_loop(&self.surface, &self.param, f0, axis, forward, start, ring)?;
        l.validate(&self.surface.mesh, &self.surface.topo)?;
        let id = self.loops.len() as u32;
        self.loops.push(l);
        self.param.loops_exact.push(id);
        self.param.report.loops_exact += 1;
        Ok(id)
    }
}

/// Derives the transition of every halfedge from the per-corner uv (exact); errors when the two
/// endpoints of an edge are not related by a rotation by a multiple of pi/2 plus a translation.
fn derive_transitions(m: &TriMesh, t: &Topology, uv: &[[V2; 3]]) -> Result<Vec<Transition>, AqError> {
    let nh = 3 * m.tris.len();
    let mut tr = vec![Transition::IDENTITY; nh];
    for h in 0..nh {
        let tw = t.twin[h];
        if tw == NONE || (tw as usize) < h {
            continue;
        }
        let (f, k) = (h / 3, h % 3);
        let (g, kg) = (tw as usize / 3, tw as usize % 3);
        let (a_f, b_f) = (uv[f][k], uv[f][(k + 1) % 3]);
        let (b_g, a_g) = (uv[g][kg], uv[g][(kg + 1) % 3]);
        let d_f = [b_f[0] - a_f[0], b_f[1] - a_f[1]];
        let d_g = [b_g[0] - a_g[0], b_g[1] - a_g[1]];
        let r = (0..4u8)
            .find(|&r| rot((4 - r) % 4, d_f) == [d_g[0] + 0.0, d_g[1] + 0.0])
            .ok_or_else(|| inv(format!("edge {h}: endpoints not related by a quarter rotation")))?;
        let ra = rot((4 - r) % 4, a_f);
        let tt = Transition { r, t: [a_g[0] - ra[0] + 0.0, a_g[1] - ra[1] + 0.0] };
        ensure(tt.apply(b_f) == b_g && tt.apply(a_f) == a_g, || format!("edge {h}: transition not exact"))?;
        tr[h] = tt;
        tr[tw as usize] = tt.inverse();
    }
    Ok(tr)
}

/// Builds a [`ParamCase`] from a mesh, its feature edges and exact per-corner uv (see the
/// module doc). No loops; add them with [`ParamCase::add_iso_loop`].
pub fn param_from_corners(
    mesh: TriMesh,
    feature_edges: &[([u32; 2], FeatureKind)],
    uv: Vec<[V2; 3]>,
) -> Result<ParamCase, AqError> {
    let s = Surface::with_features(mesh, feature_edges, CORNER_DEG, SYM_CORNER_DEG)?;
    let (m, t, fr) = (&s.mesh, &s.topo, &s.frames);
    let nf = m.tris.len();
    let nh = 3 * nf;
    ensure(uv.len() == nf, || "param_from_corners: uv size".into())?;
    let trans = derive_transitions(m, t, &uv)?;
    let is_cut: Vec<bool> = (0..nh).map(|h| !trans[h].is_identity()).collect();
    let mut theta: Vec<f64> = (0..nf as u32)
        .map(|f| {
            let j = jacobian(fr, m, f, &uv[f as usize]);
            (j[0][1] - j[1][0]).atan2(j[0][0] + j[1][1])
        })
        .collect();
    let hard = snap_to_constraints(&s, &mut theta)?;
    let field = CrossField::from_theta(m, t, fr, theta, hard)?;
    for h in 0..nh {
        if t.twin[h] != NONE {
            ensure(field.p[h] == trans[h].r, || {
                format!("he {h}: rounded matching {} differs from the transition rotation {}", field.p[h], trans[h].r)
            })?;
        }
    }
    let cut = cut_graph_from_flags(&s, &field, is_cut.clone(), vec![0; nf])?;
    let mut he_family = vec![FAMILY_NONE; nh];
    for h in 0..nh as u32 {
        if s.feats.he_chain[h as usize] != NONE {
            he_family[h as usize] = edge_family(fr, m, h, cut.theta_combed[(h / 3) as usize]);
        }
    }
    let folded = folded_faces(fr, m, &uv);
    let n_pred: f64 = (0..nf as u32).map(|f| fr.area[f as usize] * det2(&jacobian(fr, m, f, &uv[f as usize]))).sum();
    let n_fold = folded.iter().filter(|&&x| x).count() as u32;
    let report = ParamReport {
        n_pred,
        scale: 1.0,
        flips_initial: n_fold,
        flips_after_stiff: n_fold,
        folds_after_repair: n_fold,
        ..Default::default()
    };
    let rep_uv: Vec<V2> = (0..m.pos.len() as u32)
        .map(|v| {
            t.fan_out(v).min_by_key(|&h| h / 3).map(|h| uv[(h / 3) as usize][(h % 3) as usize]).unwrap_or([0.0, 0.0])
        })
        .collect();
    let param = SeamlessParam {
        uv,
        trans,
        is_cut,
        cones: field.cones(),
        he_family,
        loops_exact: Vec::new(),
        folded,
        report,
    };
    param.validate(m, t)?;
    let size = uniform_size(&s, 1.0)?;
    let case = ParamCase { surface: s, field, cut, param, size, loops: Vec::new(), rep_uv };
    case.validate()?;
    Ok(case)
}

/// Traces a closed iso-line (see [`ParamCase::add_iso_loop`]).
fn trace_iso_loop(
    s: &Surface,
    prm: &SeamlessParam,
    f0: u32,
    axis: u8,
    forward: bool,
    start: V2,
    ring: bool,
) -> Result<LoopCand, AqError> {
    let (m, t) = (&s.mesh, &s.topo);
    let a0 = (axis % 2) as usize;
    let c0 = snap_dyadic(start[1 - a0]);
    let s0: f64 = if forward { 1.0 } else { -1.0 };
    // Crossings of the line coordinate[q] = c with face f: (local edge k, param s, along x).
    let crossings = |f: u32, a: usize, c: f64| -> Result<Vec<(usize, f64, f64)>, AqError> {
        let q = 1 - a;
        let uvf = prm.uv[f as usize];
        let mut out = Vec::new();
        for k in 0..3 {
            let (p, r) = (uvf[k], uvf[(k + 1) % 3]);
            ensure(p[q] != c, || format!("iso value {c} hits a vertex of face {f}"))?;
            if (p[q] - c) * (r[q] - c) < 0.0 {
                let sk = (c - p[q]) / (r[q] - p[q]);
                out.push((k, sk, p[a] + sk * (r[a] - p[a])));
            }
        }
        ensure(out.len() == 2, || format!("iso line crosses face {f} {} times", out.len()))?;
        Ok(out)
    };
    let point3 = |f: u32, k: usize, sk: f64| -> V3 {
        let tri = m.tris[f as usize];
        math::lerp(m.pos[tri[k] as usize], m.pos[tri[(k + 1) % 3] as usize], sk)
    };
    let cr = crossings(f0, a0, c0)?;
    let (exit0, entry0) = if s0 * cr[0].2 >= s0 * cr[1].2 { (cr[0], cr[1]) } else { (cr[1], cr[0]) };
    // Seed: midpoint of the chord in the start face.
    let mid_x = 0.5 * (exit0.2 + entry0.2);
    let mut pmid = [0.0; 2];
    pmid[a0] = mid_x;
    pmid[1 - a0] = c0;
    let q = prm.uv[f0 as usize];
    let den = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
    let l1 = ((pmid[0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (pmid[1] - q[0][1])) / den;
    let l2 = ((q[1][0] - q[0][0]) * (pmid[1] - q[0][1]) - (pmid[0] - q[0][0]) * (q[1][1] - q[0][1])) / den;
    let bary = [1.0 - l1 - l2, l1, l2];
    let seed = SurfacePoint { face: f0, bary };
    let seed3 = seed.position(m);
    let mut hes = Vec::new();
    let mut pts = vec![seed3];
    let (mut f, mut a, mut sg, mut c) = (f0, a0, s0, c0);
    let mut exit = exit0;
    let max_steps = 4 * m.tris.len() + 8;
    for _ in 0..max_steps {
        let h = 3 * f + exit.0 as u32;
        pts.push(point3(f, exit.0, exit.1));
        hes.push(h);
        let tw = t.twin[h as usize];
        ensure(tw != NONE, || format!("iso line reaches the boundary at he {h}"))?;
        let tr = prm.trans[h as usize];
        let mut p = [0.0; 2];
        p[a] = exit.2;
        p[1 - a] = c;
        let mut dv = [0.0; 2];
        dv[a] = sg;
        let p2 = tr.apply(p);
        let d2 = tr.apply_vec(dv);
        let a2 = if d2[0] != 0.0 { 0 } else { 1 };
        let (g, sg2, c2) = (tw / 3, d2[a2], p2[1 - a2]);
        if g == f0 && a2 == a0 && sg2 == s0 && c2 == c0 {
            pts.push(seed3);
            let length: f64 = pts.windows(2).map(|w| math::dist(w[0], w[1])).sum();
            let branch = match (a0, forward) {
                (0, true) => 0,
                (1, true) => 1,
                (0, false) => 2,
                _ => 3,
            };
            return Ok(LoopCand { hes, start_face: f0, seed, branch, length, ring });
        }
        let kin = (tw % 3) as usize;
        let cr = crossings(g, a2, c2)?;
        let nxt = if cr[0].0 == kin { cr[1] } else if cr[1].0 == kin { cr[0] } else {
            return Err(inv(format!("iso line enters face {g} through a non-crossed edge")));
        };
        f = g;
        a = a2;
        sg = sg2;
        c = c2;
        exit = nxt;
    }
    Err(inv("iso line does not close"))
}

fn check_opts(o: &ParamOpts) -> Result<(), AqError> {
    if o.res == 0 || !(0.0..=0.25).contains(&o.jitter) {
        return Err(bad("ParamOpts: res >= 1 and jitter in [0, 0.25]"));
    }
    Ok(())
}

/// Dyadic jitter of up to `j` (absolute).
fn djit(rng: &mut SplitMix64, j: f64) -> f64 {
    snap_dyadic(j * rng.range_f64(-1.0, 1.0))
}

fn push_cell(tris: &mut Vec<[u32; 3]>, q: [u32; 4], alt: bool) {
    let [a, b, c, d] = q;
    if alt {
        tris.push([a, b, d]);
        tris.push([b, c, d]);
    } else {
        tris.push([a, b, c]);
        tris.push([a, c, d]);
    }
}

/// [`plane_param_with`] with a square of side 8 (uv units) and [`ParamOpts::default`].
pub fn plane_param(alpha: f64, s: f64) -> Result<ParamCase, AqError> {
    plane_param_with(alpha, s, 8.0, ParamOpts::default())
}

/// Planar square disk with an affine param, the grid rotated by `alpha` relative to the mesh
/// x axis: the mesh is `[0, side]^2` in uv (side snapped to 2^-24) at `res` vertices per unit,
/// and `x = R(alpha) uv / s` in z = 0 (so `s` is uv units per world unit, h = 1/s). No cut.
/// The boundary is one Boundary chain with 4 corners, iso exact (U on the bottom and top
/// sides, V on the left and right).
pub fn plane_param_with(alpha: f64, s: f64, side: f64, o: ParamOpts) -> Result<ParamCase, AqError> {
    check_opts(&o)?;
    if !(s.is_finite() && s > 0.0 && side.is_finite() && side > 0.0) {
        return Err(bad("plane_param: s and side must be > 0"));
    }
    let side = snap_dyadic(side);
    let n = ((side * o.res as f64).round() as u32).max(1);
    let sp = side / n as f64;
    let mut rng = SplitMix64::new(o.seed);
    let mut uvv = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (ju, jv) = (djit(&mut rng, o.jitter * sp), djit(&mut rng, o.jitter * sp));
            let u = if i == 0 { 0.0 } else if i == n { side } else { snap_dyadic(i as f64 * sp) + ju };
            let v = if j == 0 { 0.0 } else if j == n { side } else { snap_dyadic(j as f64 * sp) + jv };
            uvv.push([u, v]);
        }
    }
    let pos: Vec<V3> = uvv
        .iter()
        .map(|p| {
            let (sn, cs) = alpha.sin_cos();
            [(cs * p[0] - sn * p[1]) / s, (sn * p[0] + cs * p[1]) / s, 0.0]
        })
        .collect();
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let mut tris = Vec::new();
    for j in 0..n {
        for i in 0..n {
            push_cell(&mut tris, [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], (i + j) % 2 == 1);
        }
    }
    let uv: Vec<[V2; 3]> = tris.iter().map(|t| [uvv[t[0] as usize], uvv[t[1] as usize], uvv[t[2] as usize]]).collect();
    param_from_corners(TriMesh { pos, tris }, &[], uv)
}

/// Generic constant offset of the loop iso values (keeps them off the vertex coordinates).
const LOOP_OFFSET: f64 = 0.5 + 3.0 / 128.0;

/// [`cylinder_param_with`] with length `max(2, n_around / 2)` and [`ParamOpts::default`].
pub fn cylinder_param(n_around: u32) -> Result<ParamCase, AqError> {
    cylinder_param_with(n_around, (n_around / 2).max(2) as f64, ParamOpts::default())
}

/// Open cylinder along +z of circumference `n_around` (radius `n_around / 2pi`) and `length`
/// (quad units), u = arc length around (period `n_around`), v = z. The cut is the seam u = 0
/// (transition `t = (-n_around, 0)` from the last strip to the first, r = 0). Both rims are
/// closed Boundary chains (family U, iso exact). One exact ring loop per unit of length.
pub fn cylinder_param_with(n_around: u32, length: f64, o: ParamOpts) -> Result<ParamCase, AqError> {
    check_opts(&o)?;
    if n_around < 2 || !(length.is_finite() && length > 0.0) {
        return Err(bad("cylinder_param: n_around >= 2 and length > 0"));
    }
    let na = n_around as f64;
    let radius = na / TAU;
    let nu = n_around * o.res;
    let nv = ((length * o.res as f64).round() as u32).max(1);
    let length = snap_dyadic(length);
    let (du, dv) = (1.0 / o.res as f64, length / nv as f64);
    let mut rng = SplitMix64::new(o.seed);
    let mut uvv = Vec::new();
    for j in 0..=nv {
        for i in 0..nu {
            let (ju, jv) = (djit(&mut rng, o.jitter * du), djit(&mut rng, o.jitter * dv));
            let u = snap_dyadic(i as f64 * du) + ju;
            let v = if j == 0 { 0.0 } else if j == nv { length } else { snap_dyadic(j as f64 * dv) + jv };
            uvv.push([u, v]);
        }
    }
    let pos: Vec<V3> = uvv
        .iter()
        .map(|p| {
            let a = TAU * p[0] / na;
            [radius * a.cos(), radius * a.sin(), p[1]]
        })
        .collect();
    let id = |i: u32, j: u32| j * nu + (i % nu);
    let mut tris = Vec::new();
    let mut uv = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let mut q = Vec::new();
            push_cell(&mut q, [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], (i + j) % 2 == 1);
            for t in q {
                let c = t.map(|v| {
                    let mut p = uvv[v as usize];
                    if i == nu - 1 && v % nu == 0 {
                        p[0] += na;
                    }
                    p
                });
                tris.push(t);
                uv.push(c);
            }
        }
    }
    let mut case = param_from_corners(TriMesh { pos, tris }, &[], uv)?;
    let mut k = 0.0;
    while k + LOOP_OFFSET < length {
        case.add_iso_loop([0.5 * du + 1.0 / 1024.0, k + LOOP_OFFSET], 0, true, true)?;
        k += 1.0;
    }
    Ok(case)
}

/// [`torus_param_with`] with tube radius `m / 2pi`, major radius `max(n / 2pi, 2 m / 2pi)` and
/// [`ParamOpts::default`].
pub fn torus_param(n: u32, m: u32) -> Result<ParamCase, AqError> {
    let r = m as f64 / TAU;
    let big_r = (n as f64 / TAU).max(2.0 * r);
    torus_param_with(n, m, big_r, r, ParamOpts::default())
}

/// Torus about z (major radius `big_r`, tube radius `r`) with u = n phi / 2pi around z (period
/// n) and v = m psi / 2pi around the tube (period m). The cut is the two seam circles u = 0
/// and v = 0 meeting at one node. Exact loops: every longitude v = k + 0.523 (branch 0, not
/// rings) and every meridian u = k + 0.523 (branch 1, rings when shorter than 1.5 * 2pi r).
pub fn torus_param_with(n: u32, m: u32, big_r: f64, r: f64, o: ParamOpts) -> Result<ParamCase, AqError> {
    check_opts(&o)?;
    if n < 2 || m < 2 || !(r > 0.0 && big_r > r) {
        return Err(bad("torus_param: n, m >= 2 and big_r > r > 0"));
    }
    let (nf_, mf) = (n as f64, m as f64);
    let (nu, nv) = (n * o.res, m * o.res);
    let d = 1.0 / o.res as f64;
    let mut rng = SplitMix64::new(o.seed);
    let mut uvv = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let u = snap_dyadic(i as f64 * d) + djit(&mut rng, o.jitter * d);
            let v = snap_dyadic(j as f64 * d) + djit(&mut rng, o.jitter * d);
            uvv.push([u, v]);
        }
    }
    let pos: Vec<V3> = uvv
        .iter()
        .map(|p| {
            let (a, b) = (TAU * p[0] / nf_, TAU * p[1] / mf);
            let rr = big_r + r * b.cos();
            [rr * a.cos(), rr * a.sin(), r * b.sin()]
        })
        .collect();
    let id = |i: u32, j: u32| (i % nu) * nv + (j % nv);
    let mut tris = Vec::new();
    let mut uv = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let mut q = Vec::new();
            push_cell(&mut q, [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], (i + j) % 2 == 1);
            for t in q {
                let c = t.map(|v| {
                    let mut p = uvv[v as usize];
                    let (vi, vj) = (v / nv, v % nv);
                    if i == nu - 1 && vi == 0 {
                        p[0] += nf_;
                    }
                    if j == nv - 1 && vj == 0 {
                        p[1] += mf;
                    }
                    p
                });
                tris.push(t);
                uv.push(c);
            }
        }
    }
    let mut case = param_from_corners(TriMesh { pos, tris }, &[], uv)?;
    let ring_len = 1.5 * TAU * r;
    for k in 0..m {
        let id = case.add_iso_loop([0.5 * d + 1.0 / 1024.0, k as f64 + LOOP_OFFSET], 0, true, false)?;
        let l = &mut case.loops[id as usize];
        l.ring = l.length < ring_len;
    }
    for k in 0..n {
        let id = case.add_iso_loop([k as f64 + LOOP_OFFSET, 0.5 * d + 1.0 / 1024.0], 1, true, true)?;
        let l = &mut case.loops[id as usize];
        l.ring = l.length < ring_len;
    }
    Ok(case)
}

/// Options of the cube-sphere params.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubeSphereOpts {
    /// Mesh squares per uv unit along each cube edge (the mesh is (n res)^2 per cube face).
    pub res: u32,
    /// Jitter as a fraction of the mesh spacing, in [0, 0.25]; face-interior vertices move in
    /// the face, cube-edge vertices along their edge, corners stay.
    pub jitter: f64,
    pub seed: u64,
    /// Sphere radius; `None` gives `n sqrt(6 / 4pi)` (area about 6 n^2, about one quad per
    /// uv unit square).
    pub radius: Option<f64>,
}

impl Default for CubeSphereOpts {
    fn default() -> Self {
        CubeSphereOpts { res: 1, jitter: 0.0, seed: 1, radius: None }
    }
}

/// [`cube_sphere_param_with`] with [`CubeSphereOpts::default`]: a cube with n x n squares per
/// face (as DESIGN 6.2 states), uv the cube-face coordinates in [0, n].
pub fn cube_sphere_param(n: u32) -> Result<ParamCase, AqError> {
    cube_sphere_param_with(n, CubeSphereOpts::default())
}

/// Cube subdivided `n res` x `n res` per face, projected to the sphere. uv is the cube-face
/// coordinates (each face [0, n]^2) laid out in a net: a spanning tree of 5 glued cube edges
/// from face +x; the other 7 cube edges are the cut, with quarter-rotation plus integer
/// translation transitions. The 8 corners are cones of index4 +1. Exact seamless.
pub fn cube_sphere_param_with(n: u32, o: CubeSphereOpts) -> Result<ParamCase, AqError> {
    if n == 0 || o.res == 0 || !(0.0..=0.25).contains(&o.jitter) {
        return Err(bad("cube_sphere_param: n, res >= 1 and jitter in [0, 0.25]"));
    }
    let l = (n * o.res) as i64;
    let nn = n as f64;
    let (keys, tris, tri_face) = cube_lattice(l);
    let mut rng = SplitMix64::new(o.seed);
    let d = 1.0 / o.res as f64;
    let q: Vec<V3> = keys
        .iter()
        .map(|k| {
            let ext = extreme_axes(k, l);
            let mut p = [0.0; 3];
            for a in 0..3 {
                let jit = djit(&mut rng, o.jitter * d);
                p[a] = if ext[a] { if k[a] == 0 { 0.0 } else { nn } } else { snap_dyadic(k[a] as f64 * d) + jit };
            }
            p
        })
        .collect();
    let radius = o.radius.unwrap_or(nn * (6.0 / (2.0 * TAU)).sqrt());
    let ctr = [0.5 * nn; 3];
    let pos: Vec<V3> = q
        .iter()
        .map(|p| math::scale(math::normalize(math::sub(*p, ctr)).unwrap_or([0.0, 0.0, 1.0]), radius))
        .collect();
    // Net placement: per cube face (rotation k, translation T): uv = rot(k, (Q[ua], Q[va])) + T.
    let local = |fi: usize, p: V3| -> V2 {
        let (_, _, ua, va) = CUBE_FACES[fi];
        [p[ua], p[va]]
    };
    let mut place: [Option<(u8, V2)>; 6] = [None; 6];
    place[0] = Some((0, [0.0, 0.0]));
    let mut queue = vec![0usize];
    let mut qi = 0;
    while qi < queue.len() {
        let fa = queue[qi];
        qi += 1;
        let Some((ka, ta)) = place[fa] else { continue };
        for gb in 0..6 {
            let (na_f, max_f, _, _) = CUBE_FACES[fa];
            let (na_g, max_g, _, _) = CUBE_FACES[gb];
            if place[gb].is_some() || na_f == na_g {
                continue;
            }
            let w = 3 - na_f - na_g;
            let mut c1 = [0.0; 3];
            c1[na_f] = if max_f { nn } else { 0.0 };
            c1[na_g] = if max_g { nn } else { 0.0 };
            let mut c2 = c1;
            c2[w] = nn;
            let uvf = |p: V3| {
                let r = rot(ka, local(fa, p));
                [r[0] + ta[0], r[1] + ta[1]]
            };
            let (u1, u2) = (uvf(c1), uvf(c2));
            let (g1, g2) = (local(gb, c1), local(gb, c2));
            let dg = [g2[0] - g1[0], g2[1] - g1[1]];
            let du = [u2[0] - u1[0], u2[1] - u1[1]];
            let kg = (0..4u8).find(|&k| rot(k, dg) == du).ok_or_else(|| inv("cube net placement failed"))?;
            let rg = rot(kg, g1);
            place[gb] = Some((kg, [u1[0] - rg[0] + 0.0, u1[1] - rg[1] + 0.0]));
            queue.push(gb);
        }
    }
    let mut uv = Vec::with_capacity(tris.len());
    for (t, &fi) in tris.iter().zip(&tri_face) {
        let Some((k, tt)) = place[fi as usize] else { return Err(inv("cube face not placed")) };
        uv.push(t.map(|v| {
            let r = rot(k, local(fi as usize, q[v as usize]));
            [r[0] + tt[0] + 0.0, r[1] + tt[1] + 0.0]
        }));
    }
    param_from_corners(TriMesh { pos, tris }, &[], uv)
}

/// Per corner of v: the composite transition from v's rep wedge chart to that corner's face
/// chart (walking the fan with `trans`), plus, for an interior vertex, the closure transition
/// (rep chart around the full fan back to itself).
fn fan_composites(s: &Surface, trans: &[Transition], v: u32) -> (Vec<(u32, Transition)>, Option<Transition>) {
    let t = &s.topo;
    let fan: Vec<u32> = t.fan_out(v).collect();
    let n = fan.len();
    if n == 0 {
        return (Vec::new(), None);
    }
    let i0 = (0..n).min_by_key(|&i| fan[i] / 3).unwrap_or(0);
    let mut comp = vec![Transition::IDENTITY; n];
    let bnd = t.is_bnd_vert[v as usize];
    let mut closure = None;
    if bnd {
        for i in i0..n - 1 {
            comp[i + 1] = comp[i].compose(&trans[Topology::prev(fan[i]) as usize]);
        }
        for i in (1..=i0).rev() {
            comp[i - 1] = comp[i].compose(&trans[fan[i] as usize]);
        }
    } else {
        for k in 1..=n {
            let i = (i0 + k - 1) % n;
            let nx = (i0 + k) % n;
            let c = comp[i].compose(&trans[Topology::prev(fan[i]) as usize]);
            if nx == i0 {
                closure = Some(c);
            } else {
                comp[nx] = c;
            }
        }
    }
    (fan.iter().zip(comp).map(|(&h, c)| (h, c)).collect(), closure)
}

/// Solves `x = R(-r) x + tau` for a closure with r != 0 (exact for dyadic tau: only halves).
fn closure_fixed_point(c: &Transition) -> Option<V2> {
    let tau = c.t;
    match c.r % 4 {
        1 => Some([0.5 * (tau[0] + tau[1]) + 0.0, 0.5 * (tau[1] - tau[0]) + 0.0]),
        2 => Some([0.5 * tau[0] + 0.0, 0.5 * tau[1] + 0.0]),
        3 => Some([0.5 * (tau[0] - tau[1]) + 0.0, 0.5 * (tau[0] + tau[1]) + 0.0]),
        _ => None,
    }
}

/// [`perturbed_cube_sphere_param_with`] with [`CubeSphereOpts::default`].
pub fn perturbed_cube_sphere_param(n: u32, amp: f64, seed: u64) -> Result<ParamCase, AqError> {
    perturbed_cube_sphere_param_with(n, amp, seed, CubeSphereOpts::default())
}

/// The cube-sphere param with its cones moved off the integers while staying exactly seamless:
/// every cut arc gets a random translation increment (|dt| <= `amp` per coordinate, multiples of
/// 2^-20), the cone positions follow from their closure `x = R x + tau` (aq_param's closure
/// rows), the uv of the faces within 0.3 n of a cone moves rigidly with it, and every other rep
/// vertex value comes from a tiny private solver: Dirichlet energy of the displacement over all
/// face edges with the transitions of 6.9, Jacobi-preconditioned CG. The free values are
/// snapped to 2^-24 and every corner is recomputed from its exact expression.
pub fn perturbed_cube_sphere_param_with(n: u32, amp: f64, seed: u64, o: CubeSphereOpts) -> Result<ParamCase, AqError> {
    if !(amp.is_finite() && amp >= 0.0) {
        return Err(bad("perturbed_cube_sphere_param: amp >= 0"));
    }
    let base = cube_sphere_param_with(n, o)?;
    let s = &base.surface;
    let (m, t) = (&s.mesh, &s.topo);
    let nv = m.pos.len();
    let nh = 3 * m.tris.len();
    let mut rng = SplitMix64::new(seed);
    const DT_Q: f64 = 1.0 / 1_048_576.0; // 2^-20, so the halved cone positions stay on 2^-24
    let mut new_tr = base.param.trans.clone();
    let mut d_tr = vec![Transition::IDENTITY; nh];
    for arc in &base.cut.arcs {
        let dt = [snap_to(amp * rng.range_f64(-1.0, 1.0), DT_Q), snap_to(amp * rng.range_f64(-1.0, 1.0), DT_Q)];
        for &h in &arc.hes {
            let tw = t.twin[h as usize] as usize;
            let b = base.param.trans[h as usize];
            let nt = Transition { r: b.r, t: [b.t[0] + dt[0] + 0.0, b.t[1] + dt[1] + 0.0] };
            new_tr[h as usize] = nt;
            new_tr[tw] = nt.inverse();
            let dd = Transition { r: b.r, t: dt };
            d_tr[h as usize] = dd;
            d_tr[tw] = dd.inverse();
        }
    }
    // Displacement composites per vertex.
    let comps: Vec<(Vec<(u32, Transition)>, Option<Transition>)> =
        (0..nv as u32).map(|v| fan_composites(s, &d_tr, v)).collect();
    let mut corner_comp = vec![Transition::IDENTITY; nh];
    for (cs, _) in &comps {
        for &(c, tr) in cs {
            corner_comp[c as usize] = tr;
        }
    }
    let mut delta: Vec<Option<V2>> = vec![None; nv];
    let mut cones = Vec::new();
    for v in 0..nv {
        if let Some(cl) = comps[v].1 {
            if cl.r % 4 != 0 {
                delta[v] = closure_fixed_point(&cl);
                cones.push(v as u32);
            } else {
                ensure(cl.t[0].abs() < 1e-12 && cl.t[1].abs() < 1e-12, || {
                    format!("vertex {v}: displacement does not close (non-cone cut node)")
                })?;
            }
        }
    }
    // Rigid neighbourhoods of the cones.
    let rho = 0.3 * n as f64;
    let mut face_delta: Vec<Option<V2>> = vec![None; m.tris.len()];
    for &v in &cones {
        let dv = delta[v as usize].unwrap_or([0.0, 0.0]);
        for &(c, tr) in &comps[v as usize].0 {
            let f0 = c / 3;
            let dfc = tr.apply(dv);
            let uc = base.param.uv[f0 as usize][(c % 3) as usize];
            let mut stack = vec![f0];
            while let Some(f) = stack.pop() {
                if let Some(old) = face_delta[f as usize] {
                    ensure((old[0] - dfc[0]).abs() < 1e-9 && (old[1] - dfc[1]).abs() < 1e-9, || {
                        format!("cone neighbourhoods overlap at face {f}")
                    })?;
                    continue;
                }
                face_delta[f as usize] = Some(dfc);
                for k in 0..3 {
                    let h = (3 * f + k) as usize;
                    let tw = t.twin[h];
                    if tw == NONE || base.param.is_cut[h] {
                        continue;
                    }
                    let g = tw / 3;
                    if face_delta[g as usize].is_some() {
                        continue;
                    }
                    let near = base.param.uv[g as usize].iter().all(|p| (p[0] - uc[0]).hypot(p[1] - uc[1]) <= rho);
                    if near {
                        stack.push(g);
                    }
                }
            }
        }
    }
    for f in 0..m.tris.len() {
        let Some(df) = face_delta[f] else { continue };
        for k in 0..3 {
            let c = 3 * f + k;
            let w = m.tris[f][k] as usize;
            let dw = corner_comp[c].inverse().apply(df);
            match delta[w] {
                Some(old) => ensure((old[0] - dw[0]).abs() < 1e-9 && (old[1] - dw[1]).abs() < 1e-9, || {
                    format!("inconsistent rigid displacement at vertex {w}")
                })?,
                None => delta[w] = Some(dw),
            }
        }
    }
    let fixed: Vec<bool> = delta.iter().map(|d| d.is_some()).collect();
    let x0: Vec<V2> = delta.iter().map(|d| d.unwrap_or([0.0, 0.0])).collect();
    let x = solve_displacement(s, &corner_comp, &fixed, x0)?;
    // Exact rebuild.
    let comps_new: Vec<(Vec<(u32, Transition)>, Option<Transition>)> =
        (0..nv as u32).map(|v| fan_composites(s, &new_tr, v)).collect();
    let mut rep = vec![[0.0; 2]; nv];
    for v in 0..nv {
        rep[v] = match comps_new[v].1 {
            Some(cl) if cl.r % 4 != 0 => closure_fixed_point(&cl).ok_or_else(|| inv("closure"))?,
            _ => [snap_dyadic(base.rep_uv[v][0] + x[v][0]), snap_dyadic(base.rep_uv[v][1] + x[v][1])],
        };
        if let Some(cl) = comps_new[v].1 {
            ensure(cl.apply(rep[v]) == rep[v], || format!("vertex {v}: rebuilt fan does not close exactly"))?;
        }
    }
    let mut uv = vec![[[0.0; 2]; 3]; m.tris.len()];
    for v in 0..nv {
        for &(c, tr) in &comps_new[v].0 {
            uv[(c / 3) as usize][(c % 3) as usize] = tr.apply(rep[v]);
        }
    }
    for c in uv.iter().flatten().flatten() {
        ensure(c.abs() < 1e7 && (c / DYADIC_QUANTUM).fract() == 0.0, || "rebuilt uv not dyadic".into())?;
    }
    param_from_corners(m.clone(), &[], uv)
}

/// Minimizes `sum_f sum_edges |d_ci - d_cj|^2` with `d_c = comp_c(x_v)` over the free vertices
/// (Jacobi PCG, matrix free).
fn solve_displacement(s: &Surface, comp: &[Transition], fixed: &[bool], mut x: Vec<V2>) -> Result<Vec<V2>, AqError> {
    let m = &s.mesh;
    let nv = m.pos.len();
    let rotm = |c: usize, a: V2| comp[c].apply_vec(a);
    let rot_t = |c: usize, a: V2| rot(comp[c].r % 4, a);
    // Gradient of the energy at y (with translations when `affine`).
    let grad = |y: &[V2], affine: bool| -> Vec<V2> {
        let mut g = vec![[0.0; 2]; nv];
        for f in 0..m.tris.len() {
            for k in 0..3 {
                let (ci, cj) = (3 * f + k, 3 * f + (k + 1) % 3);
                let (vi, vj) = (m.tris[f][k] as usize, m.tris[f][(k + 1) % 3] as usize);
                let (mut di, mut dj) = (rotm(ci, y[vi]), rotm(cj, y[vj]));
                if affine {
                    di = [di[0] + comp[ci].t[0], di[1] + comp[ci].t[1]];
                    dj = [dj[0] + comp[cj].t[0], dj[1] + comp[cj].t[1]];
                }
                let e = [di[0] - dj[0], di[1] - dj[1]];
                let (gi, gj) = (rot_t(ci, e), rot_t(cj, e));
                g[vi] = [g[vi][0] + gi[0], g[vi][1] + gi[1]];
                g[vj] = [g[vj][0] - gj[0], g[vj][1] - gj[1]];
            }
        }
        for v in 0..nv {
            if fixed[v] {
                g[v] = [0.0, 0.0];
            }
        }
        g
    };
    let mut diag = vec![0.0f64; nv];
    for tri in &m.tris {
        for k in 0..3 {
            diag[tri[k] as usize] += 2.0;
        }
    }
    let dot = |a: &[V2], b: &[V2]| -> f64 { a.iter().zip(b).map(|(p, q)| p[0] * q[0] + p[1] * q[1]).sum() };
    let g0 = grad(&x, true);
    let mut r: Vec<V2> = g0.iter().map(|g| [-g[0], -g[1]]).collect();
    let pre = |r: &[V2]| -> Vec<V2> {
        r.iter().zip(&diag).map(|(a, &d)| if d > 0.0 { [a[0] / d, a[1] / d] } else { [0.0, 0.0] }).collect()
    };
    let mut z = pre(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let r0 = dot(&r, &r).sqrt();
    if r0 == 0.0 {
        return Ok(x);
    }
    for _ in 0..20_000 {
        let hp = grad(&p, false);
        let php = dot(&p, &hp);
        if php <= 0.0 {
            break;
        }
        let alpha = rz / php;
        for v in 0..nv {
            x[v] = [x[v][0] + alpha * p[v][0], x[v][1] + alpha * p[v][1]];
            r[v] = [r[v][0] - alpha * hp[v][0], r[v][1] - alpha * hp[v][1]];
        }
        if dot(&r, &r).sqrt() <= 1e-13 * r0 {
            return Ok(x);
        }
        z = pre(&r);
        let rz2 = dot(&r, &z);
        let beta = rz2 / rz;
        rz = rz2;
        for v in 0..nv {
            p[v] = [z[v][0] + beta * p[v][0], z[v][1] + beta * p[v][1]];
        }
    }
    let res = dot(&r, &r).sqrt();
    ensure(res <= 1e-8 * r0, || format!("displacement solver did not converge ({res:e})"))?;
    Ok(x)
}

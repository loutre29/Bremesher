//! S2 proxy remesh (DESIGN 6.4 "Proxy remesh"): Botsch-Kobbelt 2004 isotropic remeshing
//! with exact feature preservation.
//!
//! Chains: every source chain (after `hard_mode`) becomes an arc-length parametrized polyline.
//! A non-corner chain vertex stores (chain, s) and its position is always `polyline(s)`, so it
//! stays exactly on its source edges; corners (chain ends, kind changes and turning-angle
//! corners measured over 0.5 h on the source) are frozen. Splits of chain edges insert the
//! vertex at an interpolated arc length (the midpoint, or the foot of a cap apex in the sliver
//! pass); collapses only run along a chain; flips never touch a chain edge.
//!
//! Each iteration ends with a sliver pass (flips, collapses and split-collapses aimed at the
//! faces below 2 degrees), and the first iteration of a coarse source alternates single split
//! rounds with a pre-collapse (see `iterate` and CHANGES.md, WP-03 review follow-up).

use std::collections::BTreeMap;

use rayon::prelude::*;

use aq_core::math;
use aq_core::{AqError, Ctl, ErrKind, FeatureKind, NONE, Projector, Stage, Topology, TriMesh, V3};

use crate::chains::{CornerRule, apply_hard_mode, build_feature_set};
use crate::dynmesh::{DynMesh, NewVert, min_angle, next, prev, try_copy};
use crate::{ComponentMesh, MIN_PROXY_TRIS, Proxy, ProxyParams};

const SPLIT_HI: f64 = 4.0 / 3.0;
const COLLAPSE_LO: f64 = 4.0 / 5.0;
const LAMBDA: f64 = 0.3;
const MAX_NORMAL_TURN_DEG: f64 = 60.0;
/// First-iteration split cap, as a multiple of the expected proxy face count.
const SPLIT_CAP: f64 = 1.5;
/// Split rounds per split pass.
const SPLIT_ROUNDS: usize = 12;
const MIN_ANGLE_DEG: f64 = 10.0;
const FLIP_NORMAL_DEG: f64 = 30.0;
/// Faces with a minimum angle below this are repaired by the sliver pass (DESIGN 7.3 allows
/// at most 0.1% below 1 degree after S2; the margin absorbs the later isolation edits).
const SLIVER_DEG: f64 = 2.0;
/// Length slack of the sliver pass: a repair may create an edge up to this times the target.
const SLIVER_LONG: f64 = 5.0 / 3.0;
/// Pre-collapse of the first iteration: a new edge longer than 4/3 len is accepted when it is
/// at most this factor longer than the edge it replaces.
const PRE_GROWTH: f64 = 1.02;

fn perr(kind: ErrKind, msg: impl Into<String>) -> AqError {
    AqError::new(Stage::Proxy, kind, msg)
}

/// One source chain as an arc-length parametrized polyline.
struct Poly {
    pts: Vec<V3>,
    cum: Vec<f64>,
    /// Source halfedge of each segment.
    src_he: Vec<u32>,
    /// Source vertex at each point.
    verts: Vec<u32>,
    /// Closed chain without corners: s is periodic.
    wrap: bool,
    len: f64,
}

struct Chains {
    polys: Vec<Poly>,
    /// (source vertex, chain) -> arc lengths of that frozen vertex on the chain.
    frozen_s: BTreeMap<(u32, u32), Vec<f64>>,
}

impl Chains {
    /// Point at arc length s on chain c: (position, source face, bary).
    fn at(&self, c: u32, s: f64) -> (V3, u32, V3) {
        let p = &self.polys[c as usize];
        let n = p.src_he.len();
        let s = if p.wrap && p.len > 0.0 { s.rem_euclid(p.len) } else { s.clamp(0.0, p.len) };
        let i = p.cum.partition_point(|&x| x <= s).saturating_sub(1).min(n - 1);
        let d = p.cum[i + 1] - p.cum[i];
        let t = if d > 0.0 { ((s - p.cum[i]) / d).clamp(0.0, 1.0) } else { 0.0 };
        let pos = math::lerp(p.pts[i], p.pts[i + 1], t);
        let he = p.src_he[i];
        let mut bary = [0.0; 3];
        bary[(he % 3) as usize] = 1.0 - t;
        bary[((he % 3 + 1) % 3) as usize] = t;
        (pos, he / 3, bary)
    }

    /// s shifted by whole periods to be nearest to r (periodic chains only).
    fn near(&self, c: u32, s: f64, r: f64) -> f64 {
        let p = &self.polys[c as usize];
        if p.wrap && p.len > 0.0 && r.is_finite() {
            s + ((r - s) / p.len).round() * p.len
        } else {
            s
        }
    }
}

/// Arc length of vertex v on chain c, nearest to r.
fn s_of(dm: &DynMesh, ch: &Chains, v: u32, c: u32, r: f64) -> Option<f64> {
    if dm.frozen[v as usize] {
        let cands = ch.frozen_s.get(&(dm.vorig[v as usize], c))?;
        let mut best: Option<f64> = None;
        for &s in cands {
            let s = ch.near(c, s, r);
            if best.is_none_or(|b| (s - r).abs() < (b - r).abs()) {
                best = Some(s);
            }
        }
        best
    } else {
        let (cc, s) = dm.vchain[v as usize];
        if cc != c {
            return None;
        }
        Some(ch.near(c, s, r))
    }
}

/// Arc lengths of the two ends of chain edge (a, b) on chain c.
fn edge_s(dm: &DynMesh, ch: &Chains, a: u32, b: u32, c: u32) -> Option<(f64, f64)> {
    if !dm.frozen[a as usize] {
        let sa = s_of(dm, ch, a, c, f64::NAN)?;
        Some((sa, s_of(dm, ch, b, c, sa)?))
    } else if !dm.frozen[b as usize] {
        let sb = s_of(dm, ch, b, c, f64::NAN)?;
        Some((s_of(dm, ch, a, c, sb)?, sb))
    } else {
        let ca = ch.frozen_s.get(&(dm.vorig[a as usize], c))?;
        let cb = ch.frozen_s.get(&(dm.vorig[b as usize], c))?;
        let mut best: Option<(f64, f64)> = None;
        for &x in ca {
            for &y in cb {
                let y = ch.near(c, y, x);
                if (x - y).abs() > 0.0 && best.is_none_or(|(p, q)| (x - y).abs() < (p - q).abs()) {
                    best = Some((x, y));
                }
            }
        }
        best
    }
}

struct Ctx<'a, 'p> {
    proj: &'a Projector<'p>,
    ch: Chains,
    same_src: bool,
    /// Face count cap of the split rounds.
    cap: usize,
}

/// Scratch buffers.
#[derive(Default)]
struct Scratch {
    fan: Vec<u32>,
    fan2: Vec<u32>,
    na: Vec<u32>,
    nb: Vec<u32>,
    nx: Vec<u32>,
    faces: Vec<u32>,
    /// Collapse round in which a vertex's neighborhood last changed (per vertex).
    stamp: Vec<u32>,
}

/// S2: builds the isotropic proxy of `src` with target edge length `len(x)` (quad units,
/// h/3 in the pipeline), projecting onto `proj` (built over `src.mesh`; `src_face` indexes
/// `proj.mesh`). See DESIGN 6.4. The budget (`p.max_tris`, at least [`MIN_PROXY_TRIS`])
/// scales every length by one factor. After every iteration the mesh is checked (manifold,
/// chi per component, chains intact, frames valid); a failed check reverts to the previous
/// iteration and stops, and a failure of the first iteration is `ErrKind::Topology`.
///
/// The output chains carry corners classified with a window of 0.5 h = 1.5 len; isolation
/// and the final classification are separate calls ([`crate::isolate_features`],
/// [`crate::classify_chains`]).
pub fn build_proxy(
    src: &ComponentMesh,
    proj: &Projector,
    len: &dyn Fn(V3) -> f64,
    p: &ProxyParams,
    ctl: &Ctl,
) -> Result<Proxy, AqError> {
    ctl.progress(Stage::Proxy, 0.0);
    let m = &src.mesh;
    let t = &src.topo;
    let nf = m.tris.len();
    if nf == 0 || t.twin.len() != 3 * nf || src.feats.he_kind.len() != 3 * nf {
        return Err(perr(ErrKind::InvalidInput, "build_proxy: empty or inconsistent component"));
    }
    // Length scale from the budget.
    let mut expected = 0.0;
    for f in 0..nf as u32 {
        let [a, b, c] = m.tri_pos(f);
        let cen = math::scale(math::add(math::add(a, b), c), 1.0 / 3.0);
        let l = len(cen);
        if !(l.is_finite() && l > 0.0) {
            return Err(perr(ErrKind::InvalidInput, format!("build_proxy: len = {l} at face {f}")));
        }
        expected += m.face_area(f) / (0.25 * 3f64.sqrt() * l * l);
    }
    let mut scale = 1.0;
    if p.max_tris > 0 && expected > p.max_tris as f64 {
        scale = (expected / p.max_tris as f64).sqrt();
    }
    let min_t = MIN_PROXY_TRIS as f64;
    if expected / (scale * scale) < min_t && expected > 0.0 {
        scale = (expected / min_t).sqrt();
    }
    let lenf = |x: V3| len(x) * scale;

    // Source chains after hard_mode, with corners over a 0.5 h window.
    let mut kinds = try_copy(&src.feats.he_kind).map_err(|e| e.at(Stage::Proxy))?;
    apply_hard_mode(&mut kinds, p.hard_mode);
    let win = |v: u32| 1.5 * lenf(m.pos[v as usize]);
    let rule = CornerRule { corner_deg: p.corner_deg, sym_corner_deg: p.sym_corner_deg, window: &win, fixed: None };
    let sfs = build_feature_set(m, t, kinds, &rule)?;
    let mut polys = Vec::with_capacity(sfs.chains.len());
    let mut frozen_s: BTreeMap<(u32, u32), Vec<f64>> = BTreeMap::new();
    for (ci, c) in sfs.chains.iter().enumerate() {
        let mut pts = Vec::with_capacity(c.hes.len() + 1);
        let mut verts = Vec::with_capacity(c.hes.len() + 1);
        for &h in &c.hes {
            let v = t.tail(m, h);
            pts.push(m.pos[v as usize]);
            verts.push(v);
        }
        if let Some(&h) = c.hes.last() {
            let v = t.head(m, h);
            pts.push(m.pos[v as usize]);
            verts.push(v);
        }
        let mut cum = vec![0.0; pts.len()];
        for i in 1..pts.len() {
            cum[i] = cum[i - 1] + math::dist(pts[i - 1], pts[i]);
        }
        let l = *cum.last().unwrap_or(&0.0);
        for (i, &v) in verts.iter().enumerate() {
            if sfs.vert_corner[v as usize] {
                frozen_s.entry((v, ci as u32)).or_default().push(cum[i]);
            }
        }
        polys.push(Poly { pts, cum, src_he: c.hes.clone(), verts, wrap: c.closed && c.corners.is_empty(), len: l });
    }
    let ch = Chains { polys, frozen_s };

    // Initial dynamic mesh = the source.
    let same_src = std::ptr::eq(proj.mesh, m) || (proj.mesh.tris == m.tris && proj.mesh.pos == m.pos);
    let pe = |e: AqError| e.at(Stage::Proxy);
    let mut dm = DynMesh::from_mesh(m, t, try_copy(&sfs.he_kind).map_err(pe)?, try_copy(&sfs.he_chain).map_err(pe)?)
        .map_err(pe)?;
    for v in 0..m.pos.len() {
        dm.frozen[v] = sfs.vert_corner[v];
        dm.lv[v] = lenf(m.pos[v]);
        let h = t.vert_he[v];
        if h != NONE {
            dm.vsrc_f[v] = h / 3;
            dm.vsrc_b[v][(h % 3) as usize] = 1.0;
        }
    }
    for (ci, c) in sfs.chains.iter().enumerate() {
        let cum = &ch.polys[ci].cum;
        for (i, &h) in c.hes.iter().enumerate() {
            let v = t.tail(m, h) as usize;
            if !dm.frozen[v] {
                dm.vchain[v] = (ci as u32, cum[i]);
            }
        }
    }
    if !same_src {
        let mut res: Vec<(u32, V3)> = aq_core::alloc::try_with_capacity(m.pos.len()).map_err(pe)?;
        (0..m.pos.len())
            .into_par_iter()
            .map(|v| {
                let (sp, _, _) = proj.bvh.closest(m.pos[v]);
                (sp.face, sp.bary)
            })
            .collect_into_vec(&mut res);
        for (v, (f, b)) in res.into_iter().enumerate() {
            dm.vsrc_f[v] = f;
            dm.vsrc_b[v] = b;
        }
    }
    let cap = ((SPLIT_CAP * expected / (scale * scale)) as usize).max(2 * nf).max(4 * MIN_PROXY_TRIS as usize);
    let ctx = Ctx { proj, ch, same_src, cap };
    let mut chi_ref: Vec<i64> = (0..t.n_comp).map(|c| t.euler_char(m, c)).collect();
    chi_ref.sort_unstable();
    let n_comp = t.n_comp;

    let mut sc = Scratch::default();
    for it in 0..p.iters {
        // The last mesh that passed the check (or the source before the first iteration).
        let snapshot = dm.try_clone().map_err(pe)?;
        ctl.check().map_err(pe)?;
        let res = iterate(&mut dm, &ctx, &lenf, it, &mut sc, ctl);
        if let Err(e) = res
            && matches!(e.kind, ErrKind::Cancelled | ErrKind::Timeout | ErrKind::Budget)
        {
            return Err(e.at(Stage::Proxy));
        }
        dm.compact().map_err(pe)?;
        ctl.check().map_err(pe)?;
        match check(&dm, &ctx.ch, n_comp, &chi_ref).map_err(pe)? {
            Ok(()) => {}
            Err(msg) => {
                if it == 0 {
                    return Err(perr(ErrKind::Topology, format!("proxy remesh failed in iteration 1: {msg}")));
                }
                dm = snapshot;
                break;
            }
        }
        ctl.check().map_err(pe)?;
        ctl.progress(Stage::Proxy, (it + 1) as f32 / p.iters.max(1) as f32);
    }
    dm.compact().map_err(pe)?;
    finish(dm, &ctx, &lenf, p)
}

/// One Botsch-Kobbelt iteration (`it` counts from 0). When the source has edges longer than
/// 4/3 len, the first iteration alternates a pre-collapse (`Mode::Pre`: its length rule only
/// refuses edges that grow) with one bisection round, so a coarse anisotropic source is
/// decimated across its short edges while the long ones are split instead of being refined
/// to several times the expected count first. The split rounds of the first iteration stop at
/// `ctx.cap` faces. Every iteration ends with the sliver pass.
fn iterate(dm: &mut DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, it: u32, sc: &mut Scratch, ctl: &Ctl) -> Result<(), AqError> {
    let cap = if it == 0 { ctx.cap } else { usize::MAX };
    if it == 0 && has_long_edge(dm)? {
        // Coarse source: one bisection round at a time, each preceded by a pre-collapse,
        // until nothing is split or the split cap is reached.
        for _ in 0..SPLIT_ROUNDS {
            collapse_pass(dm, ctx, Mode::Pre, sc, ctl)?;
            ctl.check()?;
            if split_pass(dm, ctx, lenf, cap, 1, ctl)? == 0 || dm.live_faces() >= cap {
                break;
            }
        }
    }
    split_pass(dm, ctx, lenf, cap, SPLIT_ROUNDS, ctl)?;
    ctl.check()?;
    collapse_pass(dm, ctx, Mode::Normal, sc, ctl)?;
    ctl.check()?;
    flip_pass(dm, sc, ctl)?;
    ctl.check()?;
    smooth_pass(dm, ctx)?;
    ctl.check()?;
    update_lengths(dm, lenf);
    sliver_pass(dm, ctx, lenf, sc, ctl)?;
    Ok(())
}

/// Refreshes the per-vertex target length after vertices moved.
fn update_lengths(dm: &mut DynMesh, lenf: &dyn Fn(V3) -> f64) {
    for v in 0..dm.pos.len() {
        if !dm.vdead[v] {
            dm.lv[v] = lenf(dm.pos[v]);
        }
    }
}

fn edge_target(dm: &DynMesh, a: u32, b: u32) -> f64 {
    0.5 * (dm.lv[a as usize] + dm.lv[b as usize])
}

/// Canonical live halfedges (one per edge) with their endpoints.
fn edges(dm: &DynMesh) -> Result<Vec<(u32, u32, u32)>, AqError> {
    let mut out = aq_core::alloc::try_with_capacity(dm.tris.len() * 3 / 2 + 8)?;
    for f in 0..dm.tris.len() as u32 {
        if !dm.alive(f) {
            continue;
        }
        for k in 0..3 {
            let h = 3 * f + k;
            let tw = dm.twin[h as usize];
            if tw == NONE || h < tw {
                out.push((h, dm.tail(h), dm.head(h)));
            }
        }
    }
    Ok(out)
}

/// New-vertex data for the midpoint of edge (a, b) (halfedge h from a to b).
fn midpoint(dm: &DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, h: u32, a: u32, b: u32) -> Option<NewVert> {
    point_on_edge(dm, ctx, lenf, h, a, b, 0.5)
}

/// New-vertex data for the point at parameter t of edge (a, b) (halfedge h from a to b, or
/// its twin): on the chain polyline at the interpolated arc length for a chain edge, else the
/// projected interpolated point.
fn point_on_edge(dm: &DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, h: u32, a: u32, b: u32, t: f64) -> Option<NewVert> {
    let c = dm.cid[h as usize];
    if dm.is_chain(h) && c != NONE && (c as usize) < ctx.ch.polys.len() {
        let (sa, sb) = edge_s(dm, &ctx.ch, a, b, c)?;
        let s = sa + t * (sb - sa);
        let s = {
            let pl = &ctx.ch.polys[c as usize];
            if pl.wrap && pl.len > 0.0 { s.rem_euclid(pl.len) } else { s }
        };
        let (pos, f, bary) = ctx.ch.at(c, s);
        let (src_f, src_b) = if ctx.same_src { (f, bary) } else { (dm.vsrc_f[a as usize], dm.vsrc_b[a as usize]) };
        return Some(NewVert { pos, chain: (c, s), src_f, src_b, lv: lenf(pos) });
    }
    let pm = math::lerp(dm.pos[a as usize], dm.pos[b as usize], t);
    let mut n = dm.face_cross(h / 3);
    let tw = dm.twin[h as usize];
    if tw != NONE {
        n = math::add(n, dm.face_cross(tw / 3));
    }
    let hint = dm.vsrc_f[a as usize];
    let (pos, sf, sb) = match ctx.proj.project(pm, hint, n) {
        Some((sp, q)) => (q, sp.face, sp.bary),
        None => (pm, hint, dm.vsrc_b[a as usize]),
    };
    Some(NewVert { pos, chain: (NONE, 0.0), src_f: sf, src_b: sb, lv: lenf(pos) })
}

/// Splits the edges longer than 4/3 len, longest first, in up to `rounds` rounds until none
/// is left, and returns the number of splits. The rounds stop once the face count would
/// exceed `cap`: in the first iteration that is 1.5 times the expected count, because on a
/// coarse or strongly anisotropic source repeated bisection of long thin faces overshoots by
/// a large factor before any collapse runs; the later iterations (uncapped) split what is
/// left after the collapses, flips and smoothing.
fn split_pass(dm: &mut DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, cap: usize, rounds: usize, ctl: &Ctl) -> Result<usize, AqError> {
    let mut total = 0usize;
    for _round in 0..rounds {
        let live = dm.live_faces();
        if live >= cap {
            break;
        }
        let all = edges(dm)?;
        let mut cand: Vec<(f64, u32, u32)> = aq_core::alloc::try_with_capacity(all.len())?;
        cand.extend(all.into_iter().filter_map(|(_, a, b)| {
            let r = math::dist(dm.pos[a as usize], dm.pos[b as usize]) / edge_target(dm, a, b);
            (r > SPLIT_HI).then_some((r, a, b))
        }));
        if cand.is_empty() {
            break;
        }
        cand.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        cand.truncate((cap - live).div_ceil(2).max(1));
        dm.try_reserve(cand.len(), 2 * cand.len())?;
        let mut done = 0usize;
        for (i, &(_, a, b)) in cand.iter().enumerate() {
            if i % 4096 == 4095 {
                ctl.check().map_err(|e| e.at(Stage::Proxy))?;
            }
            if dm.vdead[a as usize] || dm.vdead[b as usize] {
                continue;
            }
            let mut h = dm.find_he(a, b);
            let (a, b) = if h == NONE {
                h = dm.find_he(b, a);
                if h == NONE {
                    continue;
                }
                (b, a)
            } else {
                (a, b)
            };
            let r = math::dist(dm.pos[a as usize], dm.pos[b as usize]) / edge_target(dm, a, b);
            if r <= SPLIT_HI {
                continue;
            }
            let Some(nv) = midpoint(dm, ctx, lenf, h, a, b) else { continue };
            if dm.split_edge(a, b, nv).is_ok() {
                done += 1;
            }
        }
        total += done;
        if done == 0 {
            break;
        }
    }
    Ok(total)
}

/// True when some edge is longer than 4/3 len (a source coarser than the target somewhere).
fn has_long_edge(dm: &DynMesh) -> Result<bool, AqError> {
    Ok(edges(dm)?
        .into_iter()
        .any(|(_, a, b)| math::dist(dm.pos[a as usize], dm.pos[b as usize]) > SPLIT_HI * edge_target(dm, a, b)))
}

#[derive(Clone, Copy, PartialEq)]
enum Reject {
    TooLong,
    Other,
}

/// Rule set of a collapse.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// DESIGN 6.4 rules (with the CHANGES.md reading of the 10 degree rule).
    Normal,
    /// First-iteration pre-collapse on a coarse source (between single split rounds): as
    /// `Normal`, except that a new edge longer than 4/3 len is accepted when it is at most
    /// `PRE_GROWTH` times the longest edge it replaces (the long edges are not split yet), and
    /// a new face below 10 degrees is refused only when it is also worse than the worst face
    /// of the old neighborhood (the faces of a strip of long thin triangles are all below 10
    /// degrees and get fatter as the short edges merge).
    Pre,
    /// Sliver repair: the new edges may reach `SLIVER_LONG` len, every new face must have a
    /// larger minimum angle than the smallest one of the old neighborhood (the collapse
    /// raises the local minimum), and an old face below `SLIVER_DEG` is checked against the
    /// area-weighted normal of the neighborhood instead of its own unstable normal.
    Sliver,
}

/// Validity and score (minimum new angle) of collapsing a into b. With `pnew`, b also moves
/// to `pnew` (both ends must then be free vertices: no chain, no corner).
fn try_collapse(
    dm: &DynMesh,
    a: u32,
    b: u32,
    pnew: Option<V3>,
    mode: Mode,
    sc: &mut Scratch,
) -> Result<(f64, Vec<(u32, u32, FeatureKind, u32)>), Reject> {
    if dm.frozen[a as usize] {
        return Err(Reject::Other);
    }
    if pnew.is_some() && (dm.frozen[b as usize] || dm.vchain[b as usize].0 != NONE || dm.chain_degree(b, &mut sc.fan2) > 0) {
        return Err(Reject::Other);
    }
    let h_ab = dm.find_he(a, b);
    let h_ba = dm.find_he(b, a);
    if h_ab == NONE && h_ba == NONE {
        return Err(Reject::Other);
    }
    let e_chain = (h_ab != NONE && dm.is_chain(h_ab)) || (h_ba != NONE && dm.is_chain(h_ba));
    let e_cid = if h_ab != NONE { dm.cid[h_ab as usize] } else { dm.cid[h_ba as usize] };
    // Chain rules.
    let a_bnd = dm.fan(a, &mut sc.fan);
    let mut a_cdeg = 0;
    let mut a_cids_ok = true;
    for &h in sc.fan.iter() {
        if dm.is_chain(h) {
            a_cdeg += 1;
            a_cids_ok &= dm.cid[h as usize] == e_cid;
        }
    }
    if a_bnd && let Some(&last) = sc.fan.last() {
        let p = prev(last);
        if dm.is_chain(p) {
            a_cdeg += 1;
            a_cids_ok &= dm.cid[p as usize] == e_cid;
        }
    }
    if a_cdeg > 0 && (!e_chain || a_cdeg != 2 || !a_cids_ok) {
        return Err(Reject::Other);
    }
    let mut merged: Vec<(u32, u32, FeatureKind, u32)> = Vec::new();
    let mut opp: [u32; 2] = [NONE, NONE];
    if h_ab != NONE {
        let c = dm.tail(prev(h_ab));
        let (ac, bc) = (prev(h_ab), next(h_ab));
        if dm.is_chain(ac) && dm.is_chain(bc) {
            return Err(Reject::Other);
        }
        if dm.is_chain(ac) {
            merged.push((b, c, dm.kind[ac as usize], dm.cid[ac as usize]));
        }
        opp[0] = c;
    }
    if h_ba != NONE {
        let d = dm.tail(prev(h_ba));
        let (ad, bd) = (next(h_ba), prev(h_ba));
        if dm.is_chain(ad) && dm.is_chain(bd) {
            return Err(Reject::Other);
        }
        if dm.is_chain(ad) {
            merged.push((b, d, dm.kind[ad as usize], dm.cid[ad as usize]));
        }
        opp[1] = d;
    }
    let b_bnd = dm.is_bnd_vert(b);
    if h_ab != NONE && h_ba != NONE && a_bnd && b_bnd {
        return Err(Reject::Other);
    }
    // Link condition.
    dm.neighbors(a, &mut sc.fan2, &mut sc.na);
    dm.neighbors(b, &mut sc.fan2, &mut sc.nb);
    let mut common = 0usize;
    {
        let (mut i, mut j) = (0, 0);
        while i < sc.na.len() && j < sc.nb.len() {
            match sc.na[i].cmp(&sc.nb[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    let x = sc.na[i];
                    if x != opp[0] && x != opp[1] {
                        return Err(Reject::Other);
                    }
                    common += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
    }
    let n_opp = opp.iter().filter(|&&x| x != NONE).count();
    if opp[0] != NONE && opp[0] == opp[1] {
        return Err(Reject::Other);
    }
    if common != n_opp {
        return Err(Reject::Other);
    }
    // Valences.
    for &x in opp.iter().filter(|&&x| x != NONE) {
        let bx = dm.neighbors(x, &mut sc.fan2, &mut sc.nx);
        let vx = sc.nx.len();
        if (bx && vx < 3) || (!bx && vx < 4) {
            return Err(Reject::Other);
        }
    }
    let vb_after = sc.na.len() + sc.nb.len() - common - 2;
    if vb_after < 3 {
        return Err(Reject::Other);
    }
    // Geometry: faces of a (and of b when it moves) other than f and g.
    let pb = pnew.unwrap_or(dm.pos[b as usize]);
    let f_ab = if h_ab != NONE { h_ab / 3 } else { NONE };
    let f_ba = if h_ba != NONE { h_ba / 3 } else { NONE };
    let cos_turn = MAX_NORMAL_TURN_DEG.to_radians().cos();
    let min_ang = MIN_ANGLE_DEG.to_radians();
    let mut score = f64::INFINITY;
    let mut long = false;
    let lb = dm.lv[b as usize];
    // Length rule (the most frequent rejection, reported after the other rules).
    let moved_nb: &[u32] = if pnew.is_some() { &sc.nb } else { &[] };
    let lim = if mode == Mode::Sliver { SLIVER_LONG } else { SPLIT_HI };
    for &n in sc.na.iter().chain(moved_nb.iter()) {
        if n == b || n == a {
            continue;
        }
        let d = math::dist(pb, dm.pos[n as usize]);
        if d <= lim * edge_target(dm, b, n) {
            continue;
        }
        if mode == Mode::Pre {
            // The longest of the edges a-n and b-n that the new edge replaces.
            let mut d_old: f64 = 0.0;
            if sc.na.binary_search(&n).is_ok() {
                d_old = d_old.max(math::dist(dm.pos[a as usize], dm.pos[n as usize]));
            }
            if sc.nb.binary_search(&n).is_ok() {
                d_old = d_old.max(math::dist(dm.pos[b as usize], dm.pos[n as usize]));
            }
            if d <= PRE_GROWTH * d_old {
                continue;
            }
        }
        long = true;
        break;
    }
    let faces = &mut sc.faces;
    faces.clear();
    faces.extend(sc.fan.iter().map(|&h| h / 3));
    if pnew.is_some() {
        dm.fan(b, &mut sc.fan2);
        faces.extend(sc.fan2.iter().map(|&h| h / 3));
    }
    let sliver = mode == Mode::Sliver;
    let thr = SLIVER_DEG.to_radians();
    // Sliver and Pre modes: smallest old angle and area-weighted normal of the neighborhood.
    let mut region_min = f64::INFINITY;
    let mut n_ref = [0.0; 3];
    if mode != Mode::Normal {
        for &f in faces.iter() {
            let tr = dm.tris[f as usize];
            let po = tr.map(|v| dm.pos[v as usize]);
            region_min = region_min.min(min_angle(po[0], po[1], po[2]));
            n_ref = math::add(n_ref, dm.face_cross(f));
        }
    }
    for &f in faces.iter() {
        if f == f_ab || f == f_ba {
            continue;
        }
        let tr = dm.tris[f as usize];
        let p_old = tr.map(|v| dm.pos[v as usize]);
        let p_new = tr.map(|v| if v == a || v == b { pb } else { dm.pos[v as usize] });
        let n_old = math::cross(math::sub(p_old[1], p_old[0]), math::sub(p_old[2], p_old[0]));
        let n_new = math::cross(math::sub(p_new[1], p_new[0]), math::sub(p_new[2], p_new[0]));
        if !(0.5 * math::norm(n_new) > 1e-10 * lb * lb) {
            return Err(Reject::Other);
        }
        let old_min = min_angle(p_old[0], p_old[1], p_old[2]);
        let n_cmp = if sliver && old_min < thr { n_ref } else { n_old };
        if !(math::dot(n_cmp, n_new) >= cos_turn * math::norm(n_cmp) * math::norm(n_new)) {
            return Err(Reject::Other);
        }
        let new_min = min_angle(p_new[0], p_new[1], p_new[2]);
        if sliver {
            if !(new_min > region_min) {
                return Err(Reject::Other);
            }
        } else if new_min < min_ang && new_min < if mode == Mode::Pre { region_min } else { old_min } {
            return Err(Reject::Other);
        }
        score = score.min(new_min);
    }
    if long {
        return Err(Reject::TooLong);
    }
    Ok((score, merged))
}

/// Collapses the edges shorter than 4/5 len, shortest first, toward the endpoint with the
/// better new minimum angle. Later rounds only retry the edges next to a change of the
/// previous round (a failed collapse stays failed while its neighborhood is unchanged).
fn collapse_pass(dm: &mut DynMesh, ctx: &Ctx, mode: Mode, sc: &mut Scratch, ctl: &Ctl) -> Result<(), AqError> {
    sc.stamp.clear();
    sc.stamp.resize(dm.pos.len(), 0);
    for round in 1..=4u32 {
        let stamp = &sc.stamp;
        let all = edges(dm)?;
        let mut cand: Vec<(f64, u32, u32)> = aq_core::alloc::try_with_capacity(all.len())?;
        cand.extend(all.into_iter().filter_map(|(_, a, b)| {
            if round > 1 && stamp[a as usize] != round - 1 && stamp[b as usize] != round - 1 {
                return None;
            }
            let r = math::dist(dm.pos[a as usize], dm.pos[b as usize]) / edge_target(dm, a, b);
            (r < COLLAPSE_LO).then_some((r, a.min(b), a.max(b)))
        }));
        if cand.is_empty() {
            break;
        }
        cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        let mut done = 0usize;
        for (i, &(_, a, b)) in cand.iter().enumerate() {
            if i % 4096 == 4095 {
                ctl.check().map_err(|e| e.at(Stage::Proxy))?;
            }
            if dm.vdead[a as usize] || dm.vdead[b as usize] {
                continue;
            }
            if dm.find_he(a, b) == NONE && dm.find_he(b, a) == NONE {
                continue;
            }
            let r = math::dist(dm.pos[a as usize], dm.pos[b as usize]) / edge_target(dm, a, b);
            if r >= COLLAPSE_LO {
                continue;
            }
            let x = try_collapse(dm, a, b, None, mode, sc);
            let y = try_collapse(dm, b, a, None, mode, sc);
            let long = x.as_ref().err() == Some(&Reject::TooLong) && y.as_ref().err() == Some(&Reject::TooLong);
            let mut pick = match (x, y) {
                (Ok(x), Ok(y)) => {
                    if y.0 > x.0 {
                        Some((b, a, y.1, None))
                    } else {
                        Some((a, b, x.1, None))
                    }
                }
                (Ok(x), Err(_)) => Some((a, b, x.1, None)),
                (Err(_), Ok(y)) => Some((b, a, y.1, None)),
                (Err(_), Err(_)) => None,
            };
            if pick.is_none() && long {
                // Both endpoints only fail on a too long new edge: try the projected midpoint
                // when both ends are free vertices.
                if mid_lengths_ok(dm, a, b, sc)
                    && let Some(mp) = free_midpoint(dm, ctx, a, b, sc)
                    && let Ok(z) = try_collapse(dm, a, b, Some(mp.0), mode, sc)
                {
                    pick = Some((a, b, z.1, Some(mp)));
                }
            }
            if let Some((rem, keep, merged, mp)) = pick
                && dm.collapse(rem, keep, &merged, &mut sc.fan).is_ok()
            {
                if let Some((p, f, bary)) = mp {
                    dm.pos[keep as usize] = p;
                    dm.vsrc_f[keep as usize] = f;
                    dm.vsrc_b[keep as usize] = bary;
                }
                done += 1;
                dm.neighbors(keep, &mut sc.fan2, &mut sc.nx);
                sc.stamp[keep as usize] = round;
                for &n in sc.nx.iter() {
                    sc.stamp[n as usize] = round;
                }
            }
        }
        if done == 0 {
            break;
        }
    }
    Ok(())
}

/// Length rule of a midpoint collapse of (a, b), checked before the projection with the plain
/// midpoint and a 5% slack.
fn mid_lengths_ok(dm: &DynMesh, a: u32, b: u32, sc: &mut Scratch) -> bool {
    let pm = math::scale(math::add(dm.pos[a as usize], dm.pos[b as usize]), 0.5);
    for v in [a, b] {
        dm.neighbors(v, &mut sc.fan2, &mut sc.nx);
        for &n in sc.nx.iter() {
            if n != a && n != b && math::dist(pm, dm.pos[n as usize]) > 1.05 * SPLIT_HI * edge_target(dm, b, n) {
                return false;
            }
        }
    }
    true
}

/// Projected midpoint of edge (a, b) when both ends are free vertices.
fn free_midpoint(dm: &DynMesh, ctx: &Ctx, a: u32, b: u32, sc: &mut Scratch) -> Option<(V3, u32, V3)> {
    for v in [a, b] {
        if dm.frozen[v as usize] || dm.vchain[v as usize].0 != NONE || dm.chain_degree(v, &mut sc.fan2) > 0 {
            return None;
        }
    }
    let pm = math::scale(math::add(dm.pos[a as usize], dm.pos[b as usize]), 0.5);
    let mut n = [0.0; 3];
    for v in [a, b] {
        dm.fan(v, &mut sc.fan2);
        for &h in sc.fan2.iter() {
            n = math::add(n, dm.face_cross(h / 3));
        }
    }
    let (sp, q) = ctx.proj.project(pm, dm.vsrc_f[a as usize], n)?;
    Some((q, sp.face, sp.bary))
}

/// Valence flips (DESIGN 6.4 step 3). Valences and boundary flags are computed once and
/// updated per flip (a flip never changes the boundary).
fn flip_pass(dm: &mut DynMesh, sc: &mut Scratch, ctl: &Ctl) -> Result<(), AqError> {
    let max_dev = FLIP_NORMAL_DEG.to_radians();
    let nf = dm.tris.len() as u32;
    let nv = dm.pos.len();
    let mut valence: Vec<i64> = aq_core::alloc::try_vec(nv, 0)?;
    let mut bndv: Vec<bool> = aq_core::alloc::try_vec(nv, false)?;
    for v in 0..nv {
        if !dm.vdead[v] && dm.vhe[v] != NONE {
            bndv[v] = dm.neighbors(v as u32, &mut sc.fan2, &mut sc.na);
            valence[v] = sc.na.len() as i64;
        }
        if v % 16384 == 16383 {
            ctl.check()?;
        }
    }
    let mut count = 0usize;
    for f in 0..nf {
        for k in 0..3 {
            count += 1;
            if count.is_multiple_of(8192) {
                ctl.check().map_err(|e| e.at(Stage::Proxy))?;
            }
            if !dm.alive(f) {
                continue;
            }
            let h = 3 * f + k;
            let t = dm.twin[h as usize];
            if t == NONE || h > t || dm.is_chain(h) {
                continue;
            }
            let (a, b) = (dm.tail(h), dm.head(h));
            let c = dm.tail(prev(h));
            let d = dm.tail(prev(t));
            if c == d || dm.find_he(c, d) != NONE || dm.find_he(d, c) != NONE {
                continue;
            }
            let mut val = [0i64; 4];
            let mut tgt = [6i64; 4];
            for (i, &v) in [a, b, c, d].iter().enumerate() {
                val[i] = valence[v as usize];
                if bndv[v as usize] {
                    tgt[i] = 4;
                }
            }
            if val[0] <= 3 || val[1] <= 3 {
                continue;
            }
            let dev = |x: [i64; 4]| -> i64 { (0..4).map(|i| (x[i] - tgt[i]).pow(2)).sum() };
            let before = dev(val);
            let after = dev([val[0] - 1, val[1] - 1, val[2] + 1, val[3] + 1]);
            if after >= before {
                continue;
            }
            let (pa, pb, pc, pd) = (dm.pos[a as usize], dm.pos[b as usize], dm.pos[c as usize], dm.pos[d as usize]);
            let n_old1 = dm.face_cross(f);
            let n_old2 = dm.face_cross(t / 3);
            let n1 = math::cross(math::sub(pa, pc), math::sub(pd, pc));
            let n2 = math::cross(math::sub(pd, pc), math::sub(pb, pc));
            let navg = math::add(
                math::scale(n_old1, 1.0 / math::norm(n_old1).max(1e-300)),
                math::scale(n_old2, 1.0 / math::norm(n_old2).max(1e-300)),
            );
            if math::dot(n1, navg) <= 0.0 || math::dot(n2, navg) <= 0.0 {
                continue;
            }
            let lt = dm.lv[a as usize];
            if 0.5 * math::norm(n1) <= 1e-10 * lt * lt || 0.5 * math::norm(n2) <= 1e-10 * lt * lt {
                continue;
            }
            // The 30 degree rule refuses a flip that creates a deviation above 30 degrees;
            // a pair that already deviates more may flip when the deviation does not grow
            // (CHANGES.md, WP-03).
            let ang_new = math::angle_between(n1, n2);
            let ang_old = math::angle_between(n_old1, n_old2);
            if ang_new > max_dev && ang_new > ang_old {
                continue;
            }
            if dm.flip(h).is_ok() {
                valence[a as usize] -= 1;
                valence[b as usize] -= 1;
                valence[c as usize] += 1;
                valence[d as usize] += 1;
            }
        }
    }
    Ok(())
}

/// Minimum angle of a live face.
fn face_min_angle(dm: &DynMesh, f: u32) -> f64 {
    let t = dm.tris[f as usize];
    min_angle(dm.pos[t[0] as usize], dm.pos[t[1] as usize], dm.pos[t[2] as usize])
}

/// Sliver pass: repairs the faces whose minimum angle is below `SLIVER_DEG`, smallest first,
/// in up to 6 rounds. Per face, in order:
/// 1. flip the edge opposite the largest angle when that raises the minimum angle of the two
///    faces (not on a chain or the boundary; valence is not considered);
/// 2. collapse the shortest edge (either direction, `Mode::Sliver` rules);
/// 3. for a cap (largest angle above 120 degrees) whose apex is a free vertex: split the
///    opposite edge at the foot of the apex (on the chain polyline for a chain edge) and
///    collapse the apex into the new vertex; when that collapse is refused the split is undone.
///
/// Faces that remain are forced by the constraints: the strip between two chains closer than
/// the edge length, or a source fold that the chains follow (CHANGES.md, WP-03).
fn sliver_pass(dm: &mut DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, sc: &mut Scratch, ctl: &Ctl) -> Result<(), AqError> {
    let thr = SLIVER_DEG.to_radians();
    for _round in 0..6 {
        let mut bad: Vec<(f64, u32)> = Vec::new();
        for f in 0..dm.tris.len() as u32 {
            if dm.alive(f) {
                let a = face_min_angle(dm, f);
                if a < thr {
                    aq_core::alloc::try_reserve(&mut bad, 1)?;
                    bad.push((a, f));
                }
            }
        }
        if bad.is_empty() {
            break;
        }
        bad.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        dm.try_reserve(bad.len(), 2 * bad.len())?;
        let mut done = 0usize;
        for (i, &(_, f)) in bad.iter().enumerate() {
            if i % 1024 == 1023 {
                ctl.check()?;
            }
            if !dm.alive(f) || face_min_angle(dm, f) >= thr {
                continue;
            }
            if fix_sliver(dm, ctx, lenf, f, sc) {
                done += 1;
            }
        }
        if done == 0 {
            break;
        }
    }
    Ok(())
}

/// One sliver repair (see [`sliver_pass`]). Returns true when the mesh changed.
fn fix_sliver(dm: &mut DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, f: u32, sc: &mut Scratch) -> bool {
    let tr = dm.tris[f as usize];
    let p = tr.map(|v| dm.pos[v as usize]);
    let ang: [f64; 3] = std::array::from_fn(|k| {
        math::angle_between(math::sub(p[(k + 1) % 3], p[k]), math::sub(p[(k + 2) % 3], p[k]))
    });
    let km = (0..3).fold(0, |m, k| if ang[k] > ang[m] { k } else { m });
    // 1. Flip the edge opposite the largest angle.
    let h_long = 3 * f + ((km + 1) % 3) as u32;
    if sliver_flip(dm, h_long, sc) {
        return true;
    }
    // 2. Collapse the shortest edge.
    let len2: [f64; 3] = std::array::from_fn(|k| math::dist2(p[k], p[(k + 1) % 3]));
    let js = (0..3).fold(0, |m, k| if len2[k] < len2[m] { k } else { m });
    let (a, b) = (tr[js], tr[(js + 1) % 3]);
    let x = try_collapse(dm, a, b, None, Mode::Sliver, sc);
    let y = try_collapse(dm, b, a, None, Mode::Sliver, sc);
    let pick = match (x, y) {
        (Ok(x), Ok(y)) => Some(if y.0 > x.0 { (b, a, y.1) } else { (a, b, x.1) }),
        (Ok(x), Err(_)) => Some((a, b, x.1)),
        (Err(_), Ok(y)) => Some((b, a, y.1)),
        (Err(_), Err(_)) => None,
    };
    if let Some((rem, keep, merged)) = pick
        && dm.collapse(rem, keep, &merged, &mut sc.fan).is_ok()
    {
        return true;
    }
    // 3. Cap with a free apex: split the opposite edge at the foot and collapse the apex.
    if ang[km] < 120f64.to_radians() {
        return false;
    }
    let apex = tr[km];
    let (u, v) = (tr[(km + 1) % 3], tr[(km + 2) % 3]);
    if dm.frozen[apex as usize] || dm.vchain[apex as usize].0 != NONE || dm.chain_degree(apex, &mut sc.fan2) > 0 {
        return false;
    }
    let (pu, pv) = (p[(km + 1) % 3], p[(km + 2) % 3]);
    let uv = math::sub(pv, pu);
    let t = math::dot(math::sub(p[km], pu), uv) / math::dot(uv, uv).max(1e-300);
    if !(0.05..=0.95).contains(&t) || (dm.is_chain(h_long) && dm.cid[h_long as usize] == NONE) {
        return false;
    }
    let Some(nv) = point_on_edge(dm, ctx, lenf, h_long, u, v, t) else { return false };
    let Ok(m) = dm.split_edge(u, v, nv) else { return false };
    if let Ok((_, merged)) = try_collapse(dm, apex, m, None, Mode::Sliver, sc)
        && dm.collapse(apex, m, &merged, &mut sc.fan).is_ok()
    {
        return true;
    }
    // Undo the split: collapse m back into u (the edge u-v and its labels come back).
    let hum = dm.find_he(u, m);
    let hmu = dm.find_he(m, u);
    let he = if hum != NONE { hum } else { hmu };
    let merged: Vec<(u32, u32, FeatureKind, u32)> =
        if he != NONE && dm.is_chain(he) { vec![(u, v, dm.kind[he as usize], dm.cid[he as usize])] } else { Vec::new() };
    if dm.collapse(m, u, &merged, &mut sc.fan).is_err() {
        // The split stays; the new vertex is a regular vertex on the edge.
        return true;
    }
    false
}

/// Angle-improving flip of the edge of halfedge h for the sliver pass: refused on a chain or
/// boundary edge, when c-d is already an edge, when a or b would drop below valence 3, when
/// a new face folds against the area-weighted normal of the two faces and their four
/// neighbors, or (as in the valence flips) when the normal deviation grows above 30 degrees.
fn sliver_flip(dm: &mut DynMesh, h: u32, sc: &mut Scratch) -> bool {
    let t = dm.twin[h as usize];
    if t == NONE || dm.is_chain(h) {
        return false;
    }
    let (a, b) = (dm.tail(h), dm.head(h));
    let c = dm.tail(prev(h));
    let d = dm.tail(prev(t));
    if c == d || dm.find_he(c, d) != NONE || dm.find_he(d, c) != NONE {
        return false;
    }
    for v in [a, b] {
        dm.neighbors(v, &mut sc.fan2, &mut sc.na);
        if sc.na.len() <= 3 {
            return false;
        }
    }
    let (pa, pb, pc, pd) = (dm.pos[a as usize], dm.pos[b as usize], dm.pos[c as usize], dm.pos[d as usize]);
    let old_min = min_angle(pa, pb, pc).min(min_angle(pb, pa, pd));
    let n1 = math::cross(math::sub(pa, pc), math::sub(pd, pc));
    let n2 = math::cross(math::sub(pd, pc), math::sub(pb, pc));
    let new_min = min_angle(pc, pa, pd).min(min_angle(pc, pd, pb));
    if !(new_min > old_min) {
        return false;
    }
    let lt = dm.lv[a as usize];
    if 0.5 * math::norm(n1) <= 1e-10 * lt * lt || 0.5 * math::norm(n2) <= 1e-10 * lt * lt {
        return false;
    }
    let mut n_ref = math::add(dm.face_cross(h / 3), dm.face_cross(t / 3));
    for g in [next(h), prev(h), next(t), prev(t)] {
        let o = dm.twin[g as usize];
        if o != NONE {
            n_ref = math::add(n_ref, dm.face_cross(o / 3));
        }
    }
    if !(math::dot(n1, n_ref) > 0.0 && math::dot(n2, n_ref) > 0.0) {
        return false;
    }
    let ang_new = math::angle_between(n1, n2);
    let ang_old = math::angle_between(dm.face_cross(h / 3), dm.face_cross(t / 3));
    if ang_new > FLIP_NORMAL_DEG.to_radians() && ang_new > ang_old {
        return false;
    }
    dm.flip(h).is_ok()
}

/// Tangential smoothing (lambda 0.3, area-weighted centroid) and projection, in parallel
/// (every vertex reads the old positions only). Chain vertices move along their polyline.
fn smooth_pass(dm: &mut DynMesh, ctx: &Ctx) -> Result<(), AqError> {
    let nv = dm.pos.len();
    let dmr: &DynMesh = dm;
    let mut res: Vec<Option<(V3, f64, u32, V3)>> = aq_core::alloc::try_with_capacity(nv)?;
    (0..nv as u32)
        .into_par_iter()
        .map_init(Vec::new, |fan, v| {
            let vi = v as usize;
            if dmr.vdead[vi] || dmr.frozen[vi] || dmr.vhe[vi] == NONE {
                return None;
            }
            let (c, s) = dmr.vchain[vi];
            if c != NONE {
                // 1D smoothing along the chain.
                let bnd = dmr.fan(v, fan);
                let mut nb: Vec<u32> = Vec::with_capacity(2);
                for &h in fan.iter() {
                    if dmr.is_chain(h) && dmr.cid[h as usize] == c {
                        nb.push(dmr.head(h));
                    }
                }
                if bnd && let Some(&last) = fan.last() {
                    let p = prev(last);
                    if dmr.is_chain(p) && dmr.cid[p as usize] == c {
                        nb.push(dmr.tail(p));
                    }
                }
                if nb.len() != 2 {
                    return None;
                }
                let s1 = s_of(dmr, &ctx.ch, nb[0], c, s)?;
                let s2 = s_of(dmr, &ctx.ch, nb[1], c, s)?;
                let tgt = 0.5 * (s1 + s2);
                let mut ns = s + LAMBDA * (tgt - s);
                let pl = &ctx.ch.polys[c as usize];
                if pl.wrap && pl.len > 0.0 {
                    ns = ns.rem_euclid(pl.len);
                }
                let (pos, f, bary) = ctx.ch.at(c, ns);
                let (f, bary) = if ctx.same_src { (f, bary) } else { (dmr.vsrc_f[vi], dmr.vsrc_b[vi]) };
                return Some((pos, ns, f, bary));
            }
            dmr.fan(v, fan);
            let p = dmr.pos[vi];
            let mut wsum = 0.0;
            let mut cen = [0.0; 3];
            let mut n = [0.0; 3];
            for &h in fan.iter() {
                let f = h / 3;
                let cr = dmr.face_cross(f);
                let a = 0.5 * math::norm(cr);
                let tr = dmr.tris[f as usize];
                let fc = math::scale(
                    math::add(math::add(dmr.pos[tr[0] as usize], dmr.pos[tr[1] as usize]), dmr.pos[tr[2] as usize]),
                    1.0 / 3.0,
                );
                cen = math::add(cen, math::scale(fc, a));
                wsum += a;
                n = math::add(n, cr);
            }
            if !(wsum > 0.0) {
                return None;
            }
            let nn = math::normalize(n)?;
            let cen = math::scale(cen, 1.0 / wsum);
            let d = math::sub(cen, p);
            let dt = math::sub(d, math::scale(nn, math::dot(nn, d)));
            let q = math::add(p, math::scale(dt, LAMBDA));
            let (sp, qq) = ctx.proj.project(q, dmr.vsrc_f[vi], nn)?;
            Some((qq, 0.0, sp.face, sp.bary))
        })
        .collect_into_vec(&mut res);
    for (v, r) in res.into_iter().enumerate() {
        if let Some((pos, s, f, b)) = r {
            dm.pos[v] = pos;
            if dm.vchain[v].0 != NONE {
                dm.vchain[v].1 = s;
            }
            dm.vsrc_f[v] = f;
            dm.vsrc_b[v] = b;
        }
    }
    Ok(())
}

/// Post-iteration check: manifold, components and chi, frames, chains intact. The outer error
/// is an allocation failure (`ErrKind::Budget`), the inner one a failed check.
fn check(dm: &DynMesh, ch: &Chains, n_comp: u32, chi_ref: &[i64]) -> Result<Result<(), String>, AqError> {
    let tm = dm.to_trimesh()?;
    let topo = match Topology::build(&tm) {
        Ok(t) => t,
        Err(e) if e.kind == ErrKind::Budget => return Err(e),
        Err(e) => return Ok(Err(format!("not manifold: {}", e.msg))),
    };
    if topo.twin != dm.twin {
        return Ok(Err("twin mismatch".into()));
    }
    if topo.n_comp != n_comp {
        return Ok(Err(format!("{} components instead of {n_comp}", topo.n_comp)));
    }
    let mut chi: Vec<i64> = (0..topo.n_comp).map(|c| topo.euler_char(&tm, c)).collect();
    chi.sort_unstable();
    if chi != chi_ref {
        return Ok(Err(format!("chi {chi:?} instead of {chi_ref:?}")));
    }
    for f in 0..tm.tris.len() as u32 {
        if !(tm.face_area(f) > 0.0) {
            return Ok(Err(format!("face {f} has zero area")));
        }
    }
    Ok(check_chains(dm, ch))
}

fn check_chains(dm: &DynMesh, ch: &Chains) -> Result<(), String> {
    let nc = ch.polys.len();
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nc];
    for f in 0..dm.tris.len() as u32 {
        if !dm.alive(f) {
            continue;
        }
        for k in 0..3 {
            let h = 3 * f + k;
            let tw = dm.twin[h as usize];
            let c = dm.cid[h as usize];
            if c != NONE && (tw == NONE || h < tw) {
                if c as usize >= nc {
                    return Err(format!("chain id {c} out of range"));
                }
                buckets[c as usize].push((dm.tail(h), dm.head(h)));
            }
        }
    }
    let mut orig: BTreeMap<u32, u32> = BTreeMap::new();
    for v in 0..dm.pos.len() {
        if dm.frozen[v] && !dm.vdead[v] {
            orig.insert(dm.vorig[v], v as u32);
        }
    }
    for (c, edges) in buckets.iter().enumerate() {
        if edges.is_empty() {
            return Err(format!("chain {c} vanished"));
        }
        let pl = &ch.polys[c];
        let mut adj: Vec<(u32, usize)> = Vec::with_capacity(2 * edges.len());
        for (i, &(a, b)) in edges.iter().enumerate() {
            adj.push((a, i));
            adj.push((b, i));
        }
        adj.sort_unstable();
        let (start, end) = if pl.wrap {
            (edges[0].0, edges[0].0)
        } else {
            let (Some(&s), Some(&e)) = (pl.verts.first(), pl.verts.last()) else {
                return Err(format!("chain {c} empty polyline"));
            };
            match (orig.get(&s), orig.get(&e)) {
                (Some(&a), Some(&b)) => (a, b),
                _ => return Err(format!("chain {c} lost an end vertex")),
            }
        };
        let mut used = vec![false; edges.len()];
        let mut cur = start;
        let mut n_used = 0;
        loop {
            let lo = adj.partition_point(|x| x.0 < cur);
            let mut nxt = None;
            for &(v, i) in &adj[lo..] {
                if v != cur {
                    break;
                }
                if !used[i] {
                    nxt = Some(i);
                    break;
                }
            }
            let Some(i) = nxt else { break };
            used[i] = true;
            n_used += 1;
            let (a, b) = edges[i];
            cur = if a == cur { b } else { a };
        }
        if n_used != edges.len() || cur != end {
            return Err(format!("chain {c} is broken ({n_used} of {} edges walked)", edges.len()));
        }
    }
    Ok(())
}

/// Builds the output proxy.
fn finish(dm: DynMesh, ctx: &Ctx, lenf: &dyn Fn(V3) -> f64, p: &ProxyParams) -> Result<Proxy, AqError> {
    let mesh: TriMesh = dm.to_trimesh().map_err(|e| e.at(Stage::Proxy))?;
    let topo = Topology::build(&mesh).map_err(|e| e.at(Stage::Proxy))?;
    let frames = aq_core::Frames::new(&mesh, &topo).map_err(|e| e.at(Stage::Proxy))?;
    let win = |v: u32| 1.5 * lenf(mesh.pos[v as usize]);
    let rule = CornerRule { corner_deg: p.corner_deg, sym_corner_deg: p.sym_corner_deg, window: &win, fixed: None };
    let nv = mesh.pos.len();
    let DynMesh { kind, vsrc_f: mut src_face, vsrc_b: mut src_bary, frozen, vchain, .. } = dm;
    let feats = build_feature_set(&mesh, &topo, kind, &rule)?;
    if !ctx.same_src {
        // Chain vertices were tracked on the component; re-seat them on the projector mesh.
        let fix: Vec<(usize, u32, V3)> = (0..nv)
            .into_par_iter()
            .filter(|&v| frozen[v] || vchain[v].0 != NONE)
            .map(|v| {
                let (sp, _, _) = ctx.proj.bvh.closest(mesh.pos[v]);
                (v, sp.face, sp.bary)
            })
            .collect();
        for (v, f, b) in fix {
            src_face[v] = f;
            src_bary[v] = b;
        }
    }
    let density = aq_core::alloc::try_vec(nv, 1.0).map_err(|e| e.at(Stage::Proxy))?;
    Ok(Proxy { mesh, topo, frames, feats, src_face, src_bary, density })
}

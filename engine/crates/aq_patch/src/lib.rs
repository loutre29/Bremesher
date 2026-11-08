//! # aq_patch: S10 patch maps, sampling and assembly (DESIGN 6.12, WP-13)
//!
//! Input: the S8 layout (W' with its T-mesh) and the S9 combinatorial quad mesh
//! ([`QComb`], `QuantResult::comb`). Output: the quad mesh of the component.
//!
//! * **Arc samples** ([`arc_samples`]): sample j of arc a lies at parametric length
//!   `s = j len_a / n_a` along `arc_verts`, linear in s on the W' edge containing it. Each
//!   arc is computed once, so both adjacent patches use bitwise identical samples. Vertex ids
//!   come from `QComb.arc_ids`; node positions are the averages over their union-find classes
//!   (the nodes sharing one QComb id).
//! * **Boundary map** (per patch, rectangle [0, W] x [0, H]): side k's arcs are laid out
//!   consecutively; a W' boundary vertex at parameter s of arc a maps to
//!   `offset_a + n_a s / len_a`. Zero arcs occupy `[k, k + m eps]` carved from the next
//!   positive arc of the side (at the end of a side: from the previous one). Zero-height and
//!   zero-width patches are not meshed: QComb glued their sides.
//! * **Interior map**: pre-split ([`presplit`]), field-aligned least squares
//!   `min sum A_f |J - T_f|^2` with `T_f = diag(W/l0, H/l1) J_f^seam`; if a triangle is
//!   flipped (exact orient2d), the barrier symmetric Dirichlet repair ([`sd_barrier`]) from
//!   the Floater map ([`floater_mvc`]); if that fails, the Floater map itself.
//! * **Sampling**: interior grid points are located in the mapped triangulation (bucket grid,
//!   exact orient2d, lowest face id on ties) and lifted barycentrically.
//! * **Assembly**: quads from `QComb.quads` (CCW), vflag per DESIGN 8.4, then the final
//!   check: every quad corner Jacobian positive against the proxy normal, manifold, oriented,
//!   Euler characteristic equal to `TMesh.chi`. A failed patch is re-meshed with plain Floater;
//!   a component that still fails gives `Err(Invariant)`.
//!
//! Deviation from DESIGN 6.12 (CHANGES.md, WP-13): `mesh_patches` takes `&QComb` instead of
//! `&QuantResult`, which lives in aq_quantize (aq_patch depends on aq_core and aq_linalg
//! only). The arc lengths are `arc_ids[a].len() - 1`, so nothing else of `QuantResult` is
//! needed.
//!
//! Errors carry `Stage::Patch`. Patches are meshed in parallel (rayon); each patch is
//! sequential, so the result is bit-identical for any thread count.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// `!(x > 0.0)` is used on purpose: NaN must count as a failed test.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod disk;
mod floater;
mod geom;
mod ls;
mod patch;
mod sample;
mod sd;

use aq_core::alloc::try_vec;
use aq_core::math;
use aq_core::{AqError, ArcKind, Ctl, ErrKind, FeatureKind, Layout, NONE, NodeKind, QComb, QuadMesh, Stage, V3};
use rayon::prelude::*;

pub use disk::{PreSplit, presplit};
pub use floater::floater_mvc;
pub use geom::M2;
pub use sd::{SdTrace, count_flips, sd_barrier, sd_barrier_trace, sd_energy};

use geom::{input_err, inv_err};

/// S10 parameters (DESIGN 6.12, 13).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchParams {
    /// Barrier symmetric Dirichlet iterations (20).
    pub sd_iters: u32,
    /// Width of a zero arc in the boundary map (1e-3).
    pub eps_zero: f64,
}

impl Default for PatchParams {
    fn default() -> Self {
        PatchParams { sd_iters: 20, eps_zero: 1e-3 }
    }
}

/// Which interior map a patch ended with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapKind {
    FieldAligned,
    Barrier,
    Floater,
}

/// S10 report (DESIGN 6.12; `zero_patches` and `split_edges` added).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatchReport {
    /// Patches whose field-aligned least squares map was valid.
    pub field_aligned: u32,
    /// Patches repaired by the barrier symmetric Dirichlet map.
    pub barrier: u32,
    /// Patches that ended with the Floater map (including the re-meshed ones).
    pub floater: u32,
    /// Patches re-meshed with plain Floater after a failed final check.
    pub remeshed: u32,
    /// Glued zero-height or zero-width patches (not meshed).
    pub zero_patches: u32,
    /// Interior edges split by the pre-split, over all patches.
    pub split_edges: u32,
    /// Quads left with a non-positive corner against the proxy normal after the final check
    /// and re-mesh (WP-16: reported, not an error).
    pub bad_quads: u32,
}

/// Samples of one arc (DESIGN 6.12 "Arc samples").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArcSamples {
    /// n_a + 1 positions from `ends[0]` to `ends[1]` (one for a zero arc).
    pub pos: Vec<V3>,
    /// Per sample: the W' edge (vertex a, vertex b) and the parameter t on it
    /// (position = a + t (b - a); t = 0 or 1 at the ends).
    pub seg: Vec<(u32, u32, f64)>,
}

/// The grid of one patch as used by the assembly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatchGrid {
    pub dims: [u32; 2],
    /// QComb ids, `j * (W + 1) + i` (as `QComb.patch_ids`).
    pub ids: Vec<u32>,
    /// Final position of every grid point (boundary points are arc samples or nodes).
    pub pos: Vec<V3>,
    /// Reference proxy normal per grid point (empty for a zero patch).
    pub normal: Vec<V3>,
    /// None for a glued zero patch.
    pub kind: Option<MapKind>,
}

/// Everything [`mesh_patches_full`] produces.
#[derive(Clone, Debug, Default)]
pub struct PatchOutput {
    pub mesh: QuadMesh,
    pub report: PatchReport,
    pub grids: Vec<PatchGrid>,
    pub samples: Vec<ArcSamples>,
}

/// S10 (DESIGN 6.12 API, with `&QComb` in place of `&QuantResult`, see the crate docs).
/// vflag kinds (bits 20..23) are derived from the arc kinds (Feature gives Hard).
pub fn mesh_patches(lay: &Layout, q: &QComb, p: &PatchParams, ctl: &Ctl) -> Result<(QuadMesh, PatchReport), AqError> {
    mesh_patches_full(lay, q, &[], p, ctl).map(|o| (o.mesh, o.report))
}

/// [`mesh_patches`] with the FeatureKind of every chain (indexed by chain id, as in
/// `FeatureSet.chains`), used for vflag bits 20..23.
pub fn mesh_patches_with(
    lay: &Layout,
    q: &QComb,
    chain_kind: &[FeatureKind],
    p: &PatchParams,
    ctl: &Ctl,
) -> Result<(QuadMesh, PatchReport), AqError> {
    mesh_patches_full(lay, q, chain_kind, p, ctl).map(|o| (o.mesh, o.report))
}

fn n_of(q: &QComb, a: usize) -> u32 {
    q.arc_ids[a].len().saturating_sub(1) as u32
}

/// Arc samples of every arc (DESIGN 6.12): sample j at `s = j len_a / n_a` along the arc's W'
/// chain, with `len_a` the chain's total parametric length `arc_s.last()` (equal to
/// `TArc.len` within 1e-9, Layout::validate) so the last sample is exactly the end node.
pub fn arc_samples(lay: &Layout, q: &QComb) -> Result<Vec<ArcSamples>, AqError> {
    let na = lay.tmesh.arcs.len();
    if q.arc_ids.len() != na || lay.arc_verts.len() != na || lay.arc_s.len() != na {
        return Err(input_err("arc_samples: arc counts differ"));
    }
    let pos = &lay.wmesh.pos;
    let mut out = Vec::with_capacity(na);
    for a in 0..na {
        let vs = &lay.arc_verts[a];
        let s = &lay.arc_s[a];
        if vs.len() < 2 || s.len() != vs.len() || vs.iter().any(|&v| v as usize >= pos.len()) {
            return Err(input_err(format!("arc {a}: bad W' vertex chain")));
        }
        let n = n_of(q, a);
        let last = vs.len() - 1;
        let stot = s[last];
        let mut smp = ArcSamples { pos: Vec::with_capacity(n as usize + 1), seg: Vec::with_capacity(n as usize + 1) };
        smp.pos.push(pos[vs[0] as usize]);
        smp.seg.push((vs[0], vs[1], 0.0));
        for j in 1..n {
            let sj = j as f64 * stot / n as f64;
            // Largest i < last with s[i] <= sj.
            let i = s[..last].partition_point(|&x| x <= sj).saturating_sub(1);
            let den = s[i + 1] - s[i];
            let t = if den > 0.0 { ((sj - s[i]) / den).clamp(0.0, 1.0) } else { 0.0 };
            let (va, vb) = (vs[i], vs[i + 1]);
            smp.pos.push(math::lerp(pos[va as usize], pos[vb as usize], t));
            smp.seg.push((va, vb, t));
        }
        if n > 0 {
            smp.pos.push(pos[vs[last] as usize]);
            smp.seg.push((vs[last - 1], vs[last], 1.0));
        }
        out.push(smp);
    }
    Ok(out)
}

fn feature_of_arc(k: ArcKind) -> FeatureKind {
    match k {
        ArcKind::Feature => FeatureKind::Hard,
        ArcKind::Boundary => FeatureKind::Boundary,
        ArcKind::Symmetry => FeatureKind::Symmetry,
        ArcKind::Guide => FeatureKind::Guide,
        _ => FeatureKind::None,
    }
}

const VF_CHAIN_MASK: u32 = (1 << 20) - 1;
const VF_CORNER: u32 = 1 << 24;
const VF_CONE: u32 = 1 << 25;
const VF_BOUNDARY: u32 = 1 << 26;

/// Checks the sizes of the layout and the QComb against each other; returns n per arc.
fn check_inputs(lay: &Layout, q: &QComb) -> Result<Vec<u32>, AqError> {
    let t = &lay.tmesh;
    let nf = lay.wmesh.tris.len();
    if lay.patch_faces.len() != t.patches.len()
        || lay.patch_chart_face.len() != t.patches.len()
        || lay.node_vert.len() != t.nodes.len()
        || lay.arc_verts.len() != t.arcs.len()
        || lay.arc_s.len() != t.arcs.len()
        || lay.wuv.len() != nf
        || lay.wtrans.len() != 3 * nf
        || lay.wtopo.twin.len() != 3 * nf
    {
        return Err(input_err("layout sizes are inconsistent"));
    }
    if lay.node_vert.iter().any(|&v| v as usize >= lay.wmesh.pos.len()) {
        return Err(input_err("node vertex out of range"));
    }
    q.validate(t).map_err(|e| AqError::new(Stage::Patch, ErrKind::InvalidInput, format!("QComb: {}", e.msg)))?;
    let n: Vec<u32> = (0..t.arcs.len()).map(|a| n_of(q, a)).collect();
    for (p, pa) in t.patches.iter().enumerate() {
        let mut s = [0u64; 4];
        for (k, side) in pa.sides.iter().enumerate() {
            s[k] = side.iter().map(|&(a, _)| n[a as usize] as u64).sum();
        }
        let d = q.patch_dims[p];
        if s[0] != s[2] || s[1] != s[3] || s[0] != d[0] as u64 || s[1] != d[1] as u64 {
            return Err(input_err(format!("patch {p}: dims {d:?} do not match the arc lengths {s:?}")));
        }
        if d[0] > 0 && d[1] > 0 && q.patch_ids[p].len() != (d[0] as usize + 1) * (d[1] as usize + 1) {
            return Err(input_err(format!("patch {p}: grid size")));
        }
    }
    Ok(n)
}

/// Grid cell (i, j) of each quad: quad k of patch p (in `QComb.quads` order) is cell
/// (k mod W, k div W) when its ids match (the aq_quantize order); otherwise it is searched.
fn quad_cells(q: &QComb) -> Result<Vec<(u32, u32)>, AqError> {
    let mut seen = vec![0u32; q.patch_dims.len()];
    let mut out = Vec::with_capacity(q.quads.len());
    for (qi, quad) in q.quads.iter().enumerate() {
        let p = q.quad_patch[qi] as usize;
        let [w, h] = q.patch_dims[p];
        let ids = &q.patch_ids[p];
        let g = |i: u32, j: u32| ids[(j * (w + 1) + i) as usize];
        let cell_ok =
            |i: u32, j: u32| -> bool { i < w && j < h && [g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)] == *quad };
        let k = seen[p];
        seen[p] += 1;
        let (i, j) = if w > 0 { (k % w, k / w) } else { (0, 0) };
        if cell_ok(i, j) {
            out.push((i, j));
            continue;
        }
        let mut found = None;
        'search: for jj in 0..h {
            for ii in 0..w {
                if cell_ok(ii, jj) {
                    found = Some((ii, jj));
                    break 'search;
                }
            }
        }
        out.push(found.ok_or_else(|| input_err(format!("quad {qi} is not a cell of patch {p}")))?);
    }
    Ok(out)
}

/// Quads of patch p with a non-positive corner against the reference normal.
fn bad_quads_of(q: &QComb, cells: &[(u32, u32)], pos: &[V3], maps: &[Option<patch::PatchMap>], p: u32) -> u32 {
    let Some(m) = &maps[p as usize] else { return 0 };
    let w = q.patch_dims[p as usize][0];
    let mut n = 0;
    for (qi, quad) in q.quads.iter().enumerate() {
        if q.quad_patch[qi] != p {
            continue;
        }
        let (i, j) = cells[qi];
        let grid = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
        let ok = (0..4).all(|c| {
            let v = pos[quad[c] as usize];
            let a = math::sub(pos[quad[(c + 1) % 4] as usize], v);
            let b = math::sub(pos[quad[(c + 3) % 4] as usize], v);
            let (gi, gj) = grid[c];
            math::dot(math::cross(a, b), m.normal[(gj * (w + 1) + gi) as usize]) > 0.0
        });
        if !ok {
            n += 1;
        }
    }
    n
}

/// Patches with a quad corner whose Jacobian is not positive against the reference normal:
/// at corner v of quad (v, a, ..., b), `((a - v) x (b - v)) . n > 0`.
fn bad_patches(q: &QComb, cells: &[(u32, u32)], pos: &[V3], maps: &[Option<patch::PatchMap>]) -> Vec<u32> {
    let mut bad: Vec<u32> = Vec::new();
    for (qi, quad) in q.quads.iter().enumerate() {
        let p = q.quad_patch[qi];
        let Some(m) = &maps[p as usize] else { continue };
        let w = q.patch_dims[p as usize][0];
        let (i, j) = cells[qi];
        let grid = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
        let ok = (0..4).all(|c| {
            let v = pos[quad[c] as usize];
            let a = math::sub(pos[quad[(c + 1) % 4] as usize], v);
            let b = math::sub(pos[quad[(c + 3) % 4] as usize], v);
            let (gi, gj) = grid[c];
            let n = m.normal[(gj * (w + 1) + gi) as usize];
            math::dot(math::cross(a, b), n) > 0.0
        });
        if !ok && !bad.contains(&p) {
            bad.push(p);
        }
    }
    bad.sort_unstable();
    bad
}

/// Manifold check of the assembled mesh: directed edges used once, every vertex used, one
/// fan per vertex.
fn check_manifold(m: &QuadMesh) -> Result<(), AqError> {
    m.validate().map_err(|e| inv_err(format!("quad mesh: {}", e.msg)))?;
    let nv = m.pos.len();
    // (v, prev, next) per corner.
    let mut corners: Vec<(u32, u32, u32)> = Vec::with_capacity(m.face_verts.len());
    for f in 0..m.n_faces() {
        let fv = m.face(f);
        let k = fv.len();
        for i in 0..k {
            corners.push((fv[i], fv[(i + k - 1) % k], fv[(i + 1) % k]));
        }
    }
    corners.sort_unstable();
    let mut start = 0;
    let mut used = vec![false; nv];
    while start < corners.len() {
        let v = corners[start].0;
        let mut end = start;
        while end < corners.len() && corners[end].0 == v {
            end += 1;
        }
        used[v as usize] = true;
        let cs = &corners[start..end];
        // Successor of a corner: the corner of the next face CCW, whose prev is this next.
        let succ = |c: usize| -> Option<usize> { cs.iter().position(|x| x.1 == cs[c].2) };
        let pred = |c: usize| -> Option<usize> { cs.iter().position(|x| x.2 == cs[c].1) };
        let mut seen = vec![false; cs.len()];
        let mut comps = 0;
        for c0 in 0..cs.len() {
            if seen[c0] {
                continue;
            }
            comps += 1;
            seen[c0] = true;
            let mut c = c0;
            while let Some(nx) = succ(c) {
                if seen[nx] {
                    break;
                }
                seen[nx] = true;
                c = nx;
            }
            let mut c = c0;
            while let Some(nx) = pred(c) {
                if seen[nx] {
                    break;
                }
                seen[nx] = true;
                c = nx;
            }
        }
        if comps != 1 {
            return Err(inv_err(format!("quad mesh vertex {v} is non-manifold ({comps} fans)")));
        }
        start = end;
    }
    if let Some(v) = used.iter().position(|&u| !u) {
        return Err(inv_err(format!("quad mesh vertex {v} is used by no quad")));
    }
    Ok(())
}

/// Sorted undirected edges of a polygon mesh, with multiplicity.
fn undirected_edges(m: &QuadMesh) -> Vec<(u32, u32)> {
    let mut e: Vec<(u32, u32)> = Vec::with_capacity(m.face_verts.len());
    for f in 0..m.n_faces() {
        let fv = m.face(f);
        for i in 0..fv.len() {
            let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.sort_unstable();
    e
}

/// Euler characteristic V - E + F of a polygon mesh.
fn euler(m: &QuadMesh) -> i64 {
    let mut e = undirected_edges(m);
    e.dedup();
    m.pos.len() as i64 - e.len() as i64 + m.n_faces() as i64
}

/// Full S10 returning every intermediate result (grids and arc samples), for the engine's
/// invariant checks (DESIGN 7.3) and the tests.
pub fn mesh_patches_full(
    lay: &Layout,
    q: &QComb,
    chain_kind: &[FeatureKind],
    prm: &PatchParams,
    ctl: &Ctl,
) -> Result<PatchOutput, AqError> {
    run(lay, q, chain_kind, prm, ctl).map_err(|e| e.at(Stage::Patch))
}

fn run(lay: &Layout, q: &QComb, chain_kind: &[FeatureKind], prm: &PatchParams, ctl: &Ctl) -> Result<PatchOutput, AqError> {
    ctl.progress(Stage::Patch, 0.0);
    if !(prm.eps_zero > 0.0 && prm.eps_zero < 0.25) {
        return Err(input_err("eps_zero must be in (0, 0.25)"));
    }
    let n = check_inputs(lay, q)?;
    let t = &lay.tmesh;
    let np = t.patches.len();
    let nf = lay.wmesh.tris.len();
    let mut face_patch = try_vec(nf, NONE)?;
    for (p, fs) in lay.patch_faces.iter().enumerate() {
        for &f in fs {
            let slot = face_patch.get_mut(f as usize).ok_or_else(|| input_err("patch face out of range"))?;
            if *slot != NONE {
                return Err(input_err(format!("W' face {f} is in two patches")));
            }
            *slot = p as u32;
        }
    }
    let samples = arc_samples(lay, q)?;
    ctl.check()?;

    // Patch maps, in parallel; the first error in patch order wins (deterministic).
    let results: Vec<Result<Option<patch::PatchMap>, AqError>> = (0..np as u32)
        .into_par_iter()
        .map(|p| {
            let d = q.patch_dims[p as usize];
            if d[0] == 0 || d[1] == 0 {
                return Ok(None);
            }
            patch::map_patch(lay, &face_patch, p, &n, d, prm, false, ctl).map(Some)
        })
        .collect();
    let mut maps = Vec::with_capacity(np);
    for r in results {
        maps.push(r?);
    }
    ctl.progress(Stage::Patch, 0.8);

    // Vertex positions: node classes, then arc samples, then patch interiors.
    let nvq = q.n_verts as usize;
    let mut node_acc = try_vec(nvq, ([0.0f64; 3], 0u32))?;
    let mut smp_acc = try_vec(nvq, ([0.0f64; 3], 0u32))?;
    for (ni, &v) in lay.node_vert.iter().enumerate() {
        let id = q.node_id[ni] as usize;
        let e = &mut node_acc[id];
        e.0 = math::add(e.0, lay.wmesh.pos[v as usize]);
        e.1 += 1;
    }
    for (a, smp) in samples.iter().enumerate() {
        for j in 1..n[a] as usize {
            let id = q.arc_ids[a][j] as usize;
            let e = &mut smp_acc[id];
            e.0 = math::add(e.0, smp.pos[j]);
            e.1 += 1;
        }
    }
    let mut pos: Vec<V3> = try_vec(nvq, [0.0; 3])?;
    let mut have = try_vec(nvq, false)?;
    for id in 0..nvq {
        let (s, c) = if node_acc[id].1 > 0 { node_acc[id] } else { smp_acc[id] };
        if c == 1 {
            pos[id] = s;
            have[id] = true;
        } else if c > 1 {
            pos[id] = math::scale(s, 1.0 / c as f64);
            have[id] = true;
        }
    }
    let set_interior = |pos: &mut [V3], have: &mut [bool], p: usize, m: &patch::PatchMap, first: bool| -> Result<(), AqError> {
        let [w, h] = q.patch_dims[p];
        for j in 1..h {
            for i in 1..w {
                let g = (j * (w + 1) + i) as usize;
                let id = q.patch_ids[p][g] as usize;
                if first && have[id] {
                    return Err(input_err(format!("patch {p}: interior grid id {id} is shared")));
                }
                pos[id] = m.pos[g];
                have[id] = true;
            }
        }
        Ok(())
    };
    for (p, m) in maps.iter().enumerate() {
        if let Some(m) = m {
            set_interior(&mut pos, &mut have, p, m, true)?;
        }
    }
    if let Some(id) = have.iter().position(|&x| !x) {
        return Err(inv_err(format!("QComb vertex {id} has no position")));
    }

    // Final geometric check; failed patches are re-meshed with plain Floater.
    let cells = quad_cells(q)?;
    let mut report = PatchReport::default();
    // WP-16 (CHANGES.md): a patch whose re-mesh does not remove every non-positive corner
    // keeps the map with fewer bad quads, and the count is reported instead of failing the
    // component: S11 relaxation repairs most of them and the engine checks the rest.
    let bad = bad_patches(q, &cells, &pos, &maps);
    for &p in &bad {
        ctl.check()?;
        let pu = p as usize;
        if maps[pu].as_ref().map(|m| m.kind) == Some(MapKind::Floater) {
            continue;
        }
        let before = bad_quads_of(q, &cells, &pos, &maps, p);
        let m = match patch::map_patch(lay, &face_patch, p, &n, q.patch_dims[pu], prm, true, ctl) {
            Ok(m) => m,
            Err(e) if matches!(e.kind, ErrKind::Cancelled | ErrKind::Timeout | ErrKind::Budget) => return Err(e),
            Err(_) => continue,
        };
        let mut pos2 = pos.clone();
        set_interior(&mut pos2, &mut have, pu, &m, false)?;
        let old = maps[pu].replace(m);
        if bad_quads_of(q, &cells, &pos2, &maps, p) < before {
            pos = pos2;
            report.remeshed += 1;
        } else {
            maps[pu] = old;
        }
    }
    if !bad.is_empty() {
        report.bad_quads = (0..q.patch_dims.len() as u32).map(|p| bad_quads_of(q, &cells, &pos, &maps, p)).sum();
    }
    for m in &maps {
        match m {
            None => report.zero_patches += 1,
            Some(m) => {
                report.split_edges += m.n_split;
                match m.kind {
                    MapKind::FieldAligned => report.field_aligned += 1,
                    MapKind::Barrier => report.barrier += 1,
                    MapKind::Floater => report.floater += 1,
                }
            }
        }
    }

    // Assembly.
    let mut mesh = QuadMesh::new();
    mesh.pos = pos;
    mesh.face_verts = aq_core::alloc::try_with_capacity(4 * q.quads.len())?;
    for quad in &q.quads {
        mesh.push_face(quad, 0);
    }
    mesh.vflag = vflags(lay, q, &n, chain_kind, &mesh)?;
    check_manifold(&mesh)?;
    let chi = euler(&mesh);
    if chi != t.chi as i64 {
        return Err(inv_err(format!("quad mesh Euler characteristic {chi} != chi {}", t.chi)));
    }

    let grids = (0..np)
        .map(|p| {
            let ids = q.patch_ids[p].clone();
            let gpos = ids.iter().map(|&id| mesh.pos[id as usize]).collect();
            let (normal, kind) = match &maps[p] {
                Some(m) => (m.normal.clone(), Some(m.kind)),
                None => (Vec::new(), None),
            };
            PatchGrid { dims: q.patch_dims[p], ids, pos: gpos, normal, kind }
        })
        .collect();
    ctl.progress(Stage::Patch, 1.0);
    Ok(PatchOutput { mesh, report, grids, samples })
}

/// vflag per DESIGN 8.4: chain id + 1 (a node's own chain first, else the lowest chain of
/// the arcs through the vertex), the chain's FeatureKind, corner (FeatureCorner or
/// BoundaryCorner node), cone (Cone node), boundary (on a boundary edge of the quad mesh).
/// Chain ids are the layout's; the engine renumbers them globally.
fn vflags(lay: &Layout, q: &QComb, n: &[u32], chain_kind: &[FeatureKind], mesh: &QuadMesh) -> Result<Vec<u32>, AqError> {
    let t = &lay.tmesh;
    let nv = mesh.pos.len();
    let mut node_chain = try_vec(nv, NONE)?;
    let mut arc_chain = try_vec(nv, NONE)?;
    let mut flag = try_vec(nv, 0u32)?;
    for (ni, node) in t.nodes.iter().enumerate() {
        let id = q.node_id[ni] as usize;
        if node.chain != NONE {
            node_chain[id] = node_chain[id].min(node.chain);
        }
        match node.kind {
            NodeKind::FeatureCorner | NodeKind::BoundaryCorner => flag[id] |= VF_CORNER,
            NodeKind::Cone => flag[id] |= VF_CONE,
            _ => {}
        }
    }
    // Kind of each chain from its lowest arc.
    let mut kind_of: std::collections::BTreeMap<u32, FeatureKind> = std::collections::BTreeMap::new();
    for (a, arc) in t.arcs.iter().enumerate() {
        if arc.chain == NONE {
            continue;
        }
        kind_of.entry(arc.chain).or_insert(feature_of_arc(arc.kind));
        for j in 0..=n[a] as usize {
            let id = q.arc_ids[a][j] as usize;
            arc_chain[id] = arc_chain[id].min(arc.chain);
        }
    }
    for v in 0..nv {
        let c = if node_chain[v] != NONE { node_chain[v] } else { arc_chain[v] };
        if c != NONE {
            let k = chain_kind
                .get(c as usize)
                .copied()
                .unwrap_or_else(|| kind_of.get(&c).copied().unwrap_or(FeatureKind::None));
            flag[v] |= c.saturating_add(1).min(VF_CHAIN_MASK) | ((k as u32 & 0xF) << 20);
        }
    }
    let e = undirected_edges(mesh);
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j] == e[i] {
            j += 1;
        }
        if j - i == 1 {
            flag[e[i].0 as usize] |= VF_BOUNDARY;
            flag[e[i].1 as usize] |= VF_BOUNDARY;
        }
        i = j;
    }
    Ok(flag)
}

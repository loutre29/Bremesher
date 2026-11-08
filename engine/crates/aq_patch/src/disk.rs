//! Per-patch disk extraction, the boundary map onto the rectangle [0, W] x [0, H], and the
//! pre-split (DESIGN 6.12 "Boundary map per patch" and "Interior map" step 1).

use std::collections::{BTreeMap, BTreeSet};

use aq_core::math;
use aq_core::{AqError, Layout, NONE, Topology, Transition, TriMesh, V2};

use crate::geom::{input_err, inv_err};

/// A patch as a local triangle disk.
pub(crate) struct Disk {
    pub mesh: TriMesh,
    pub topo: Topology,
    /// Seamless uv in the patch chart, per local vertex.
    pub uv: Vec<V2>,
    /// Rectangle position of boundary vertices (unused for interior ones).
    pub rect: Vec<V2>,
    pub is_bnd: Vec<bool>,
    /// Local face -> W' face.
    pub wface: Vec<u32>,
    /// Edges split by the pre-split.
    pub n_split: u32,
}

/// Interval [t0, t1] of every arc of one side (side parameter, 0 at the side's start).
///
/// A positive arc of n segments occupies n units. A run of m consecutive zero arcs occupies
/// `[k, k + m eps]`, carved from the start of the next positive arc on the side; a run at the
/// end of a side (no positive arc follows) is carved from the end of the previous positive arc
/// instead (`[k - m eps, k]`). eps is `eps_zero`, reduced to `0.25 / (zeros + 1)` when a side
/// carries very many zero arcs, so every positive arc keeps a positive interval.
pub(crate) fn side_intervals(ns: &[u32], eps_zero: f64) -> Result<Vec<(f64, f64)>, AqError> {
    let zeros = ns.iter().filter(|&&n| n == 0).count();
    let eps = eps_zero.min(0.25 / (zeros as f64 + 1.0));
    let mut out = vec![(0.0, 0.0); ns.len()];
    let mut pos = 0.0f64;
    let mut last_pos: Option<usize> = None;
    let mut i = 0;
    while i < ns.len() {
        if ns[i] > 0 {
            out[i] = (pos, pos + ns[i] as f64);
            pos += ns[i] as f64;
            last_pos = Some(i);
            i += 1;
            continue;
        }
        let mut k = i;
        while k < ns.len() && ns[k] == 0 {
            k += 1;
        }
        let m = (k - i) as f64;
        if k < ns.len() {
            for (q, o) in out[i..k].iter_mut().enumerate() {
                *o = (pos + q as f64 * eps, pos + (q + 1) as f64 * eps);
            }
            out[k] = (pos + m * eps, pos + ns[k] as f64);
            pos += ns[k] as f64;
            last_pos = Some(k);
            i = k + 1;
        } else {
            let Some(lp) = last_pos else {
                return Err(inv_err("side without a positive arc"));
            };
            out[lp].1 = pos - m * eps;
            for (q, o) in out[i..k].iter_mut().enumerate() {
                let qq = q as f64;
                *o = (pos - (m - qq) * eps, pos - (m - qq - 1.0) * eps);
            }
            i = k;
        }
    }
    Ok(out)
}

/// Rectangle point of side k at side parameter t.
#[inline]
pub(crate) fn side_point(k: usize, t: f64, w: f64, h: f64) -> V2 {
    match k {
        0 => [t, 0.0],
        1 => [w, t],
        2 => [w - t, h],
        _ => [0.0, h - t],
    }
}

/// Builds the disk of patch p with its boundary map and pre-split.
pub(crate) fn build_disk(
    lay: &Layout,
    face_patch: &[u32],
    p: u32,
    n_arc: &[u32],
    dims: [u32; 2],
    eps_zero: f64,
) -> Result<Disk, AqError> {
    let pu = p as usize;
    let faces = &lay.patch_faces[pu];
    if faces.is_empty() {
        return Err(inv_err(format!("patch {p} has no face")));
    }
    let wm = &lay.wmesh;
    let wt = &lay.wtopo;
    let nf = faces.len();
    // W' face -> local face.
    let mut wf_sorted: Vec<(u32, u32)> = faces.iter().enumerate().map(|(i, &f)| (f, i as u32)).collect();
    wf_sorted.sort_unstable();
    let lface = |w: u32| -> Option<u32> { wf_sorted.binary_search_by(|x| x.0.cmp(&w)).ok().map(|i| wf_sorted[i].1) };
    // Arc edges of this patch's sides. An arc can bound the patch on both of its sides (for
    // example a ring arc that is side 0 and side 2 of a patch wrapping a tube), so the local
    // disk is cut open along every arc edge, never glued across one (WP-16 fix, CHANGES.md).
    let pa = &lay.tmesh.patches[pu];
    let mut arc_edge: BTreeSet<(u32, u32)> = BTreeSet::new();
    for side in pa.sides.iter() {
        for &(a, _) in side {
            let vs = lay.arc_verts.get(a as usize).ok_or_else(|| input_err(format!("arc {a} out of range")))?;
            for w in vs.windows(2) {
                arc_edge.insert((w[0].min(w[1]), w[0].max(w[1])));
            }
        }
    }
    let is_arc_edge = |wf: u32, k: u32| -> bool {
        let t = wm.tris[wf as usize];
        let (a, b) = (t[k as usize], t[((k + 1) % 3) as usize]);
        arc_edge.contains(&(a.min(b), a.max(b)))
    };
    // Local vertices: classes of patch corners glued across the non-arc interior edges.
    let mut uf: Vec<u32> = (0..3 * nf as u32).collect();
    fn find(uf: &mut [u32], mut x: u32) -> u32 {
        while uf[x as usize] != x {
            let g = uf[uf[x as usize] as usize];
            uf[x as usize] = g;
            x = g;
        }
        x
    }
    for (lf, &wf) in faces.iter().enumerate() {
        for k in 0..3u32 {
            let he = 3 * wf + k;
            let tw = wt.twin[he as usize];
            if tw == NONE || is_arc_edge(wf, k) {
                continue;
            }
            let Some(lg) = lface(tw / 3) else { continue };
            let c_tail = 3 * lf as u32 + k;
            let c_head = 3 * lf as u32 + (k + 1) % 3;
            let g_tail = 3 * lg + Topology::next(tw) % 3;
            let g_head = 3 * lg + tw % 3;
            for (x, y) in [(c_tail, g_tail), (c_head, g_head)] {
                let (rx, ry) = (find(&mut uf, x), find(&mut uf, y));
                if rx != ry {
                    uf[rx.max(ry) as usize] = rx.min(ry);
                }
            }
        }
    }
    let mut class_id: Vec<u32> = vec![NONE; 3 * nf];
    let mut pos: Vec<aq_core::V3> = Vec::new();
    let mut corner_local: Vec<u32> = vec![NONE; 3 * nf];
    for c in 0..3 * nf as u32 {
        let r = find(&mut uf, c) as usize;
        if class_id[r] == NONE {
            class_id[r] = pos.len() as u32;
            let wf = faces[(c / 3) as usize] as usize;
            pos.push(wm.pos[wm.tris[wf][(c % 3) as usize] as usize]);
        }
        corner_local[c as usize] = class_id[r];
    }
    let tris: Vec<[u32; 3]> =
        (0..nf).map(|lf| [corner_local[3 * lf], corner_local[3 * lf + 1], corner_local[3 * lf + 2]]).collect();
    let mesh = TriMesh { pos, tris };
    let topo = Topology::build(&mesh).map_err(|e| inv_err(format!("patch {p} is not a manifold disk: {}", e.msg)))?;
    if topo.n_comp != 1 || topo.boundary_loops.len() != 1 || topo.euler_char(&mesh, 0) != 1 {
        return Err(inv_err(format!(
            "patch {p} is not a disk ({} components, {} boundary loops)",
            topo.n_comp,
            topo.boundary_loops.len()
        )));
    }
    // Local vertex of W' vertex a on the patch boundary, leaving along the arc step a -> b
    // (the patch lies on the left of its CCW boundary walk).
    let step_local = |a: u32, b: u32| -> Option<u32> {
        let h = wt.find_he(wm, a, b)?;
        let lf = lface(h / 3)?;
        Some(corner_local[(3 * lf + h % 3) as usize])
    };
    // Boundary map.
    let nv = mesh.pos.len();
    let (w, h) = (dims[0] as f64, dims[1] as f64);
    let mut rect = vec![[0.0, 0.0]; nv];
    let mut is_bnd = vec![false; nv];
    let mut walk: Vec<u32> = Vec::new();
    let mut side_start = [0usize; 4];
    for (k, side) in pa.sides.iter().enumerate() {
        side_start[k] = walk.len();
        let ns: Vec<u32> = side.iter().map(|&(a, _)| n_arc[a as usize]).collect();
        let iv = side_intervals(&ns, eps_zero)?;
        for (&(a, rev), &(t0, t1)) in side.iter().zip(&iv) {
            let verts = &lay.arc_verts[a as usize];
            let s = &lay.arc_s[a as usize];
            let len = verts.len();
            if len < 2 || s.len() != len {
                return Err(input_err(format!("arc {a} has a bad vertex chain")));
            }
            let stot = s[len - 1];
            for i in 0..len - 1 {
                let idx = if rev { len - 1 - i } else { i };
                let nxt = if rev { idx - 1 } else { idx + 1 };
                let sp = if rev { stot - s[idx] } else { s[idx] };
                let frac = if i == 0 {
                    0.0
                } else if stot > 0.0 {
                    (sp / stot).clamp(0.0, 1.0)
                } else {
                    i as f64 / (len - 1) as f64
                };
                let tpos = if i == 0 { t0 } else { t0 + (t1 - t0) * frac };
                let lv = step_local(verts[idx], verts[nxt]).ok_or_else(|| {
                    inv_err(format!("patch {p}: vertex {} of arc {a} is not in the patch", verts[idx]))
                })?;
                if is_bnd[lv as usize] {
                    return Err(inv_err(format!("patch {p}: boundary vertex {} visited twice", verts[idx])));
                }
                is_bnd[lv as usize] = true;
                rect[lv as usize] = side_point(k, tpos, w, h);
                walk.push(lv);
            }
        }
    }
    let loop_len = topo.boundary_loops[0].len();
    if walk.len() != loop_len {
        return Err(inv_err(format!(
            "patch {p}: the sides cover {} boundary vertices, the disk boundary has {loop_len}",
            walk.len()
        )));
    }
    for i in 0..walk.len() {
        let a = walk[i];
        let b = walk[(i + 1) % walk.len()];
        let ok = topo.find_he(&mesh, a, b).map(|hh| topo.twin[hh as usize] == NONE).unwrap_or(false);
        if !ok {
            return Err(inv_err(format!("patch {p}: the sides do not follow the disk boundary CCW")));
        }
    }
    // Patch chart: unfold through the transitions from the chart face, across the non-arc
    // edges inside the patch.
    let chart = lay.patch_chart_face[pu];
    let start = lface(chart).ok_or_else(|| inv_err(format!("patch {p} chart face not in the patch")))?;
    let mut chart_map: Vec<Option<Transition>> = vec![None; nf];
    chart_map[start as usize] = Some(Transition::IDENTITY);
    let mut queue = vec![start];
    let mut qi = 0;
    while qi < queue.len() {
        let lf = queue[qi];
        qi += 1;
        let Some(mf) = chart_map[lf as usize] else { continue };
        let wf = faces[lf as usize];
        for k in 0..3 {
            let he = 3 * wf + k;
            let tw = wt.twin[he as usize];
            if tw == NONE || is_arc_edge(wf, k) {
                continue;
            }
            let g = tw / 3;
            if face_patch.get(g as usize).copied() != Some(p) {
                continue;
            }
            let Some(lg) = lface(g) else { continue };
            if chart_map[lg as usize].is_some() {
                continue;
            }
            let tr = lay.wtrans[he as usize];
            chart_map[lg as usize] = Some(tr.inverse().compose(&mf));
            queue.push(lg);
        }
    }
    let mut uv: Vec<Option<V2>> = vec![None; nv];
    for lf in 0..nf {
        let m = chart_map[lf].ok_or_else(|| inv_err(format!("patch {p} faces are not connected")))?;
        let wf = faces[lf] as usize;
        for k in 0..3 {
            let v = mesh.tris[lf][k] as usize;
            if uv[v].is_none() {
                uv[v] = Some(m.apply(lay.wuv[wf][k]));
            }
        }
    }
    let mut uv: Vec<V2> = uv.into_iter().map(|x| x.unwrap_or([0.0, 0.0])).collect();
    // DESIGN 6.10: side0 runs +u in the patch chart. Enforce it with the quarter rotation
    // that turns side 0's chord closest to +u (the identity for a layout that meets the
    // contract), so a chart face taken from a rotated chart cannot rotate the targets.
    if !walk.is_empty() {
        let s = walk[side_start[0]] as usize;
        let e = walk[side_start[1] % walk.len()] as usize;
        let d = [uv[e][0] - uv[s][0], uv[e][1] - uv[s][1]];
        // R(k) d for k = 0..3 has x component d0, -d1, -d0, d1.
        let cand = [d[0], -d[1], -d[0], d[1]];
        let mut k = 0u8;
        for (i, &c) in cand.iter().enumerate() {
            if c > cand[k as usize] {
                k = i as u8;
            }
        }
        if k != 0 {
            for x in uv.iter_mut() {
                *x = aq_core::rot(k, *x);
            }
        }
    }

    // Pre-split.
    let ps = split_edges(&mesh, &topo, &uv, &is_bnd)?;
    let n_new = ps.mesh.pos.len() - nv;
    rect.extend(std::iter::repeat_n([0.0, 0.0], n_new));
    is_bnd.extend(std::iter::repeat_n(false, n_new));
    let wface = ps.parent.iter().map(|&f| faces[f as usize]).collect();
    let topo = if ps.n_split > 0 { Topology::build(&ps.mesh)? } else { topo };
    Ok(Disk { mesh: ps.mesh, topo, uv: ps.uv, rect, is_bnd, wface, n_split: ps.n_split })
}

/// Result of [`presplit`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreSplit {
    pub mesh: TriMesh,
    /// uv per vertex (midpoints for new vertices).
    pub uv: Vec<V2>,
    /// Face -> input face.
    pub parent: Vec<u32>,
    /// Number of split edges (= new vertices, appended after the input vertices).
    pub n_split: u32,
}

/// Pre-split (DESIGN 6.12 interior map step 1): every interior edge whose two endpoints are
/// flagged in `on_boundary` is split at its midpoint (3D position and uv), together with its
/// two triangles. This splits the interior edges of every triangle whose 3 vertices lie on
/// the boundary (in particular those whose 3 vertices lie on one side), and also every other
/// interior edge joining two boundary vertices, which a Floater map with a weakly convex
/// boundary needs (no "dividing edge" with both ends on one straight side). New vertices are
/// interior. Deterministic: edges are split in increasing (lower vertex, higher vertex) order.
pub fn presplit(mesh: &TriMesh, t: &Topology, uv: &[V2], on_boundary: &[bool]) -> Result<PreSplit, AqError> {
    if uv.len() != mesh.pos.len() || on_boundary.len() != mesh.pos.len() || t.twin.len() != 3 * mesh.tris.len() {
        return Err(input_err("presplit: size mismatch"));
    }
    split_edges(mesh, t, uv, on_boundary)
}

fn split_edges(mesh: &TriMesh, t: &Topology, uv: &[V2], flag: &[bool]) -> Result<PreSplit, AqError> {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for h in 0..t.twin.len() as u32 {
        let tw = t.twin[h as usize];
        if tw == NONE || h > tw {
            continue;
        }
        let a = t.tail(mesh, h);
        let b = t.head(mesh, h);
        if flag[a as usize] && flag[b as usize] {
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    let mut out = PreSplit {
        mesh: mesh.clone(),
        uv: uv.to_vec(),
        parent: (0..mesh.tris.len() as u32).collect(),
        n_split: 0,
    };
    if edges.is_empty() {
        return Ok(out);
    }
    let mut dir: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for (f, tr) in mesh.tris.iter().enumerate() {
        for k in 0..3 {
            dir.insert((tr[k], tr[(k + 1) % 3]), f as u32);
        }
    }
    let mut tris = std::mem::take(&mut out.mesh.tris);
    for (a, b) in edges {
        let (Some(&f1), Some(&f2)) = (dir.get(&(a, b)), dir.get(&(b, a))) else {
            return Err(inv_err("presplit: edge lost"));
        };
        let rot_to = |tr: [u32; 3], x: u32| -> [u32; 3] {
            if tr[0] == x {
                tr
            } else if tr[1] == x {
                [tr[1], tr[2], tr[0]]
            } else {
                [tr[2], tr[0], tr[1]]
            }
        };
        let t1 = rot_to(tris[f1 as usize], a); // [a, b, c]
        let t2 = rot_to(tris[f2 as usize], b); // [b, a, d]
        let c = t1[2];
        let d = t2[2];
        let m = out.mesh.pos.len() as u32;
        out.mesh.pos.push(math::lerp(mesh_pos(&out.mesh, a), mesh_pos(&out.mesh, b), 0.5));
        let (ua, ub) = (out.uv[a as usize], out.uv[b as usize]);
        out.uv.push([0.5 * (ua[0] + ub[0]), 0.5 * (ua[1] + ub[1])]);
        let f3 = tris.len() as u32;
        let f4 = f3 + 1;
        tris[f1 as usize] = [a, m, c];
        tris.push([m, b, c]);
        tris[f2 as usize] = [b, m, d];
        tris.push([m, a, d]);
        let p1 = out.parent[f1 as usize];
        let p2 = out.parent[f2 as usize];
        out.parent.push(p1);
        out.parent.push(p2);
        dir.remove(&(a, b));
        dir.remove(&(b, a));
        dir.insert((a, m), f1);
        dir.insert((m, c), f1);
        dir.insert((m, b), f3);
        dir.insert((b, c), f3);
        dir.insert((c, m), f3);
        dir.insert((b, m), f2);
        dir.insert((m, d), f2);
        dir.insert((m, a), f4);
        dir.insert((a, d), f4);
        dir.insert((d, m), f4);
        out.n_split += 1;
    }
    out.mesh.tris = tris;
    Ok(out)
}

#[inline]
fn mesh_pos(m: &TriMesh, v: u32) -> aq_core::V3 {
    m.pos[v as usize]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::side_intervals;

    #[test]
    fn zero_arc_intervals() {
        let e = 1e-3;
        // Positive arcs only.
        assert_eq!(side_intervals(&[2, 3], e).unwrap(), vec![(0.0, 2.0), (2.0, 5.0)]);
        // Leading run of 2 zero arcs: carved from the start of the next positive arc.
        let iv = side_intervals(&[0, 0, 3], e).unwrap();
        assert_eq!(iv[0], (0.0, e));
        assert_eq!(iv[1], (e, 2.0 * e));
        assert_eq!(iv[2], (2.0 * e, 3.0));
        // A zero arc in the middle.
        let iv = side_intervals(&[2, 0, 1], e).unwrap();
        assert_eq!(iv, vec![(0.0, 2.0), (2.0, 2.0 + e), (2.0 + e, 3.0)]);
        // Trailing zero arcs: carved from the end of the previous positive arc.
        let iv = side_intervals(&[0, 2, 0, 0], e).unwrap();
        assert_eq!(iv[0], (0.0, e));
        assert_eq!(iv[1], (e, 2.0 - 2.0 * e));
        assert_eq!(iv[2], (2.0 - 2.0 * e, 2.0 - e));
        assert_eq!(iv[3], (2.0 - e, 2.0));
        // Intervals are consecutive and non-decreasing (injective boundary map).
        for ns in [vec![0u32, 1, 0, 2, 0], vec![1, 0], vec![0, 1]] {
            let iv = side_intervals(&ns, e).unwrap();
            for w in iv.windows(2) {
                assert_eq!(w[0].1, w[1].0);
            }
            for &(a, b) in &iv {
                assert!(b > a);
            }
            let tot: u32 = ns.iter().sum();
            assert_eq!(iv[iv.len() - 1].1, tot as f64);
        }
        // A side without a positive arc is an error.
        assert!(side_intervals(&[0, 0], e).is_err());
        // Very many zero arcs shrink eps.
        let ns: Vec<u32> = std::iter::repeat_n(0, 999).chain([1]).collect();
        let iv = side_intervals(&ns, e).unwrap();
        assert!(iv[999].1 - iv[999].0 > 0.5);
    }
}

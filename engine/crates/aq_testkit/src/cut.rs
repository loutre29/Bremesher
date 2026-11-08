//! Analytic cut graphs (DESIGN 6.2 `analytic_cut`, following the S6 algorithm of 6.8) and the
//! arc builder shared with the params.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use aq_core::cross::{comb_matching, comb_theta};
use aq_core::math;
use aq_core::{AqError, CrossField, CutArc, CutGraph, NONE};

use crate::surface::Surface;
use crate::util::{centroid, ensure};

/// Builds a [`CutGraph`] from cut flags (per halfedge, symmetric, false on the boundary) and
/// comb offsets: `theta' = theta + c pi/2`, `r = (p + c_g - c_f) mod 4` (C12, which must be 0
/// off the cut), nodes (cut degree other than 2, cones on the cut, cut/boundary junctions,
/// feature corners on the cut; a node-free cut cycle gets its lowest vertex) and arcs (maximal
/// node-to-node paths, side A = the halfedges walked forward). Validated against `cf`.
pub fn cut_graph_from_flags(
    s: &Surface,
    cf: &CrossField,
    is_cut: Vec<bool>,
    comb_offset: Vec<u8>,
) -> Result<CutGraph, AqError> {
    let (m, t) = (&s.mesh, &s.topo);
    let nh = 3 * m.tris.len();
    let nv = m.pos.len();
    ensure(is_cut.len() == nh && comb_offset.len() == m.tris.len(), || "cut flags sizes".into())?;
    let mut r = vec![0u8; nh];
    for h in 0..nh {
        let tw = t.twin[h];
        if tw == NONE {
            ensure(!is_cut[h], || format!("boundary halfedge {h} flagged as cut"))?;
            continue;
        }
        ensure(is_cut[tw as usize] == is_cut[h], || format!("cut flag asymmetric at {h}"))?;
        r[h] = comb_matching(cf.p[h], comb_offset[h / 3], comb_offset[tw as usize / 3]);
        ensure(is_cut[h] || r[h] == 0, || format!("combed matching {} != 0 off the cut at he {h}", r[h]))?;
    }
    let theta_combed: Vec<f64> =
        cf.theta.iter().zip(&comb_offset).map(|(&th, &c)| comb_theta(th, c)).collect();
    let mut deg = vec![0u32; nv];
    for h in 0..nh as u32 {
        if is_cut[h as usize] && t.is_canonical(h) {
            deg[t.tail(m, h) as usize] += 1;
            deg[t.head(m, h) as usize] += 1;
        }
    }
    let mut node: Vec<bool> = (0..nv)
        .map(|v| {
            deg[v] > 0
                && (deg[v] != 2 || cf.index4[v] != 0 || t.is_bnd_vert[v] || s.feats.vert_corner[v])
        })
        .collect();
    let edge_id = |h: u32| -> usize {
        let tw = t.twin[h as usize];
        if tw == NONE { h as usize } else { h.min(tw) as usize }
    };
    let mut visited = vec![false; nh];
    let mut arcs: Vec<CutArc> = Vec::new();
    let mut he_arc = vec![(NONE, 0i8); nh];
    let mut starts: Vec<u32> = (0..nv as u32).filter(|&v| node[v as usize]).collect();
    let mut si = 0usize;
    let mut next_unvisited = 0usize;
    loop {
        let v = if si < starts.len() {
            si += 1;
            starts[si - 1]
        } else {
            // A remaining cut cycle without node: its lowest-id halfedge's tail becomes a node.
            while next_unvisited < nh && !(is_cut[next_unvisited] && !visited[edge_id(next_unvisited as u32)]) {
                next_unvisited += 1;
            }
            if next_unvisited >= nh {
                break;
            }
            let v = t.tail(m, next_unvisited as u32);
            node[v as usize] = true;
            starts.push(v);
            si += 1;
            v
        };
        let outs: Vec<u32> = t.fan_out(v).collect();
        for h0 in outs {
            if !is_cut[h0 as usize] || visited[edge_id(h0)] {
                continue;
            }
            visited[edge_id(h0)] = true;
            let mut hes = vec![h0];
            let mut cur = t.head(m, h0);
            let mut guard = 0usize;
            while !node[cur as usize] {
                guard += 1;
                ensure(guard <= nh, || "cut arc walk does not terminate".into())?;
                let Some(h) = t.fan_out(cur).find(|&h| is_cut[h as usize] && !visited[edge_id(h)]) else {
                    return Err(AqError::invariant(format!("cut arc dead end at vertex {cur}")));
                };
                visited[edge_id(h)] = true;
                hes.push(h);
                cur = t.head(m, h);
            }
            let id = arcs.len() as u32;
            for &h in &hes {
                he_arc[h as usize] = (id, 1);
                he_arc[t.twin[h as usize] as usize] = (id, -1);
            }
            arcs.push(CutArc { r: r[h0 as usize], nodes: [v, cur], hes });
        }
    }
    let mut node_verts: Vec<u32> = starts;
    node_verts.sort_unstable();
    node_verts.dedup();
    let cg = CutGraph { is_cut, he_arc, arcs, node_verts, theta_combed, r, comb_offset };
    cg.validate(m, t, Some(cf))?;
    Ok(cg)
}

/// Euler characteristic of the cut mesh of component `comp` (vertices duplicated per wedge).
pub fn cut_mesh_euler(s: &Surface, is_cut: &[bool], comp: u32) -> i64 {
    let (m, t) = (&s.mesh, &s.topo);
    let mut f = 0i64;
    let mut e = 0i64;
    for fi in 0..m.tris.len() {
        if t.face_comp[fi] != comp {
            continue;
        }
        f += 1;
        for k in 0..3 {
            let h = (3 * fi + k) as u32;
            if t.twin[h as usize] == NONE || is_cut[h as usize] {
                e += 1; // a boundary or cut halfedge is its own edge in the cut mesh
            } else if t.is_canonical(h) {
                e += 1;
            }
        }
    }
    let mut v = 0i64;
    for vi in 0..m.pos.len() as u32 {
        let h0 = t.vert_he[vi as usize];
        if h0 == NONE || t.face_comp[(h0 / 3) as usize] != comp {
            continue;
        }
        let c = t.fan_out(vi).filter(|&h| is_cut[h as usize]).count() as i64;
        v += if t.is_bnd_vert[vi as usize] { c + 1 } else { c.max(1) };
    }
    v - e + f
}

/// Analytic cut graph for a field on `s` (DESIGN 6.8 steps 1 to 6): Dijkstra dual tree from the
/// largest face of each component (ties to the lowest id; no rng), cut = interior non-tree
/// edges, pruned at cut-degree-1 vertices that are not cones, boundary vertices or feature
/// corners; disk check `V - E + F = 1` per component; combing along the tree. Validated.
pub fn analytic_cut(s: &Surface, cf: &CrossField) -> Result<CutGraph, AqError> {
    let (m, t) = (&s.mesh, &s.topo);
    let nf = m.tris.len();
    let nh = 3 * nf;
    let nv = m.pos.len();
    cf.validate(m, t, &s.frames)?;
    let cents: Vec<_> = (0..nf as u32).map(|f| centroid(m, f)).collect();
    let mut dist = vec![f64::INFINITY; nf];
    let mut parent_he = vec![NONE; nf];
    let mut done = vec![false; nf];
    let mut order: Vec<u32> = Vec::with_capacity(nf);
    let mut roots = vec![NONE; t.n_comp as usize];
    for f in 0..nf {
        let c = t.face_comp[f] as usize;
        let r = roots[c];
        if r == NONE || s.frames.area[f] > s.frames.area[r as usize] {
            roots[c] = f as u32;
        }
    }
    for &root in &roots {
        if root == NONE {
            continue;
        }
        let mut heap = BinaryHeap::new();
        dist[root as usize] = 0.0;
        heap.push(Reverse((0u64, root)));
        while let Some(Reverse((_, f))) = heap.pop() {
            if done[f as usize] {
                continue;
            }
            done[f as usize] = true;
            order.push(f);
            for k in 0..3 {
                let h = 3 * f + k;
                let tw = t.twin[h as usize];
                if tw == NONE {
                    continue;
                }
                let g = tw / 3;
                if done[g as usize] {
                    continue;
                }
                let nd = dist[f as usize] + math::dist(cents[f as usize], cents[g as usize]);
                if nd < dist[g as usize] {
                    dist[g as usize] = nd;
                    parent_he[g as usize] = h;
                    heap.push(Reverse((nd.to_bits(), g)));
                }
            }
        }
    }
    let mut is_cut = vec![false; nh];
    for h in 0..nh {
        is_cut[h] = t.twin[h] != NONE;
    }
    for &ph in &parent_he {
        if ph != NONE {
            is_cut[ph as usize] = false;
            is_cut[t.twin[ph as usize] as usize] = false;
        }
    }
    // Prune.
    let mut deg = vec![0u32; nv];
    for h in 0..nh as u32 {
        if is_cut[h as usize] && t.is_canonical(h) {
            deg[t.tail(m, h) as usize] += 1;
            deg[t.head(m, h) as usize] += 1;
        }
    }
    let keep = |v: usize| cf.index4[v] != 0 || t.is_bnd_vert[v] || s.feats.vert_corner[v];
    let mut stack: Vec<u32> = (0..nv as u32).filter(|&v| deg[v as usize] == 1 && !keep(v as usize)).collect();
    stack.reverse();
    while let Some(v) = stack.pop() {
        if deg[v as usize] != 1 || keep(v as usize) {
            continue;
        }
        let Some(h) = t.fan_out(v).find(|&h| is_cut[h as usize]) else { continue };
        let tw = t.twin[h as usize];
        is_cut[h as usize] = false;
        is_cut[tw as usize] = false;
        deg[v as usize] -= 1;
        let w = t.head(m, h);
        deg[w as usize] -= 1;
        if deg[w as usize] == 1 && !keep(w as usize) {
            stack.push(w);
        }
    }
    for c in 0..t.n_comp {
        let e = cut_mesh_euler(s, &is_cut, c);
        ensure(e == 1, || format!("cut mesh of component {c} is not a disk (V - E + F = {e})"))?;
    }
    let mut comb = vec![0u8; nf];
    for &f in &order {
        let ph = parent_he[f as usize];
        if ph != NONE {
            let pf = (ph / 3) as usize;
            comb[f as usize] = (comb[pf] + 4 - cf.p[ph as usize] % 4) % 4;
        }
    }
    cut_graph_from_flags(s, cf, is_cut, comb)
}

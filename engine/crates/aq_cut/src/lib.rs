//! # aq_cut: S6 cut graph and combing (DESIGN 6.8, WP-09)
//!
//! [`cut_and_comb`] cuts every component of the proxy to a topological disk and combs the
//! cross field (C12):
//!
//! 1. **Dual spanning tree:** Dijkstra over the faces of each component from its root (the
//!    face of largest area, exact ties broken by the seed rng, C15), with the dual edge length
//!    (centroid to edge midpoint to centroid) as the cost. Interior edges only. Heap ties go
//!    to the lowest face id, so the tree is deterministic.
//! 2. **Initial cut:** every interior edge not crossed by the tree.
//! 3. **Prune:** a cut edge whose endpoint has cut degree 1 is removed, unless that endpoint
//!    is a cone (`index4 != 0`), a boundary vertex or a feature corner. Repeated until stable.
//! 4. **Check:** the cut mesh (faces glued along non-cut edges, vertices duplicated per wedge)
//!    has `V - E + F = 1` per component, otherwise `Invariant`.
//! 5. **Combing:** `c_root = 0` and `c_g = (c_f - p[he]) mod 4` across the tree edges (in tree
//!    order, parents first), so the combed matching `r = (p + c_g - c_f) mod 4` is 0 on every
//!    tree edge; `theta' = theta + c pi/2` (aq_core `comb_theta`, `comb_matching`).
//! 6. **Nodes and arcs:** nodes are the cut vertices of cut degree other than 2, every cone on
//!    the cut and every cut vertex on the boundary (a remaining node-free cut cycle gets its
//!    lowest vertex as a node). Arcs are the maximal node-to-node paths; side A is the walked
//!    halfedges (their faces lie on the left). r must be constant along each arc
//!    (`Invariant` otherwise).
//!
//! The result is checked with [`CutGraph::validate`] against the field.
//!
//! The functions take a [`ProxyRef`] (borrowed mesh, topology, frames and features), because
//! the aq_prep `Proxy` type is not visible from this crate (DESIGN 3.2: aq_cut depends on
//! aq_core only). See CHANGES.md, WP-09.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use aq_core::cross::{comb_matching, comb_theta};
use aq_core::math;
use aq_core::{
    AqError, CrossField, CutArc, CutGraph, FeatureSet, Frames, NONE, SplitMix64, Stage, Topology,
    TriMesh, V3,
};

/// Borrowed view of a proxy: everything S5 and S6 read from an aq_prep `Proxy`.
#[derive(Clone, Copy, Debug)]
pub struct ProxyRef<'a> {
    pub mesh: &'a TriMesh,
    pub topo: &'a Topology,
    pub frames: &'a Frames,
    pub feats: &'a FeatureSet,
}

impl<'a> ProxyRef<'a> {
    pub fn new(mesh: &'a TriMesh, topo: &'a Topology, frames: &'a Frames, feats: &'a FeatureSet) -> Self {
        ProxyRef { mesh, topo, frames, feats }
    }
}

fn inv(msg: impl Into<String>) -> AqError {
    AqError::invariant(msg).at(Stage::Cut)
}

fn bad(msg: impl Into<String>) -> AqError {
    AqError::invalid_input(msg).at(Stage::Cut)
}

fn centroid(m: &TriMesh, f: u32) -> V3 {
    let [a, b, c] = m.tri_pos(f);
    math::scale(math::add(math::add(a, b), c), 1.0 / 3.0)
}

/// Checks the sizes of the inputs against the mesh.
fn check_inputs(px: &ProxyRef, f: &CrossField) -> Result<(), AqError> {
    let nf = px.mesh.tris.len();
    let nh = 3 * nf;
    let nv = px.mesh.pos.len();
    if px.topo.twin.len() != nh
        || px.topo.face_comp.len() != nf
        || px.topo.is_bnd_vert.len() != nv
        || px.topo.vert_he.len() != nv
    {
        return Err(bad("aq_cut: topology does not match the mesh"));
    }
    if px.frames.area.len() != nf {
        return Err(bad("aq_cut: frames do not match the mesh"));
    }
    if px.feats.vert_corner.len() != nv {
        return Err(bad("aq_cut: feature set does not match the mesh"));
    }
    if f.theta.len() != nf || f.p.len() != nh || f.index4.len() != nv {
        return Err(bad("aq_cut: cross field does not match the mesh"));
    }
    Ok(())
}

/// Euler characteristic of the cut mesh, per component (vertices duplicated per wedge: an
/// interior vertex with c > 0 cut edges has c wedges, a boundary vertex c + 1; a cut or
/// boundary halfedge is its own edge). Returns an empty Vec when `is_cut` or the topology does
/// not match the mesh (`is_cut.len() != 3 * n_faces`).
pub fn cut_mesh_euler(px: &ProxyRef, is_cut: &[bool]) -> Vec<i64> {
    let (m, t) = (px.mesh, px.topo);
    let (nf, nv) = (m.tris.len(), m.pos.len());
    if is_cut.len() != 3 * nf
        || t.twin.len() != 3 * nf
        || t.face_comp.len() != nf
        || t.vert_he.len() != nv
        || t.is_bnd_vert.len() != nv
    {
        return Vec::new();
    }
    let nc = t.n_comp as usize;
    let mut chi = vec![0i64; nc];
    for fi in 0..m.tris.len() {
        let c = t.face_comp[fi] as usize;
        if c >= nc {
            continue;
        }
        chi[c] += 1;
        for k in 0..3 {
            let h = (3 * fi + k) as u32;
            if t.twin[h as usize] == NONE || is_cut[h as usize] || t.is_canonical(h) {
                chi[c] -= 1;
            }
        }
    }
    for v in 0..m.pos.len() as u32 {
        let h0 = t.vert_he[v as usize];
        if h0 == NONE {
            continue;
        }
        let c = t.face_comp[(h0 / 3) as usize] as usize;
        if c >= nc {
            continue;
        }
        let n = t.fan_out(v).filter(|&h| is_cut[h as usize]).count() as i64;
        chi[c] += if t.is_bnd_vert[v as usize] { n + 1 } else { n.max(1) };
    }
    chi
}

/// Cut degree per vertex (number of cut edges at the vertex).
fn cut_degree(px: &ProxyRef, is_cut: &[bool]) -> Vec<u32> {
    let (m, t) = (px.mesh, px.topo);
    let mut deg = vec![0u32; m.pos.len()];
    for h in 0..is_cut.len() as u32 {
        if is_cut[h as usize] && t.is_canonical(h) {
            deg[t.tail(m, h) as usize] += 1;
            deg[t.head(m, h) as usize] += 1;
        }
    }
    deg
}

/// Dual spanning forest: per component Dijkstra from the root. Returns the parent halfedge per
/// face (the halfedge of the parent face whose twin lies in the child, NONE at a root) and the
/// faces in tree order (parents first).
fn dual_tree(px: &ProxyRef, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let (m, t, fr) = (px.mesh, px.topo, px.frames);
    let nf = m.tris.len();
    let nc = t.n_comp as usize;
    // Roots: largest area per component, exact ties broken by the rng (C15).
    let mut best = vec![f64::NEG_INFINITY; nc];
    for f in 0..nf {
        let c = t.face_comp[f] as usize;
        if c < nc && fr.area[f] > best[c] {
            best[c] = fr.area[f];
        }
    }
    let mut ties: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for f in 0..nf {
        let c = t.face_comp[f] as usize;
        if c < nc && fr.area[f] == best[c] {
            ties[c].push(f as u32);
        }
    }
    let mut rng = SplitMix64::new(seed);
    let roots: Vec<u32> = ties
        .iter()
        .map(|v| match v.len() {
            0 => NONE,
            1 => v[0],
            n => v[rng.below(n as u64) as usize],
        })
        .collect();
    let cents: Vec<V3> = (0..nf as u32).map(|f| centroid(m, f)).collect();
    let mut dist = vec![f64::INFINITY; nf];
    let mut parent = vec![NONE; nf];
    let mut done = vec![false; nf];
    let mut order = Vec::with_capacity(nf);
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for &root in &roots {
        if root == NONE {
            continue;
        }
        dist[root as usize] = 0.0;
        heap.push(Reverse((0.0f64.to_bits(), root)));
        while let Some(Reverse((dbits, f))) = heap.pop() {
            if done[f as usize] || dbits != dist[f as usize].to_bits() {
                continue;
            }
            done[f as usize] = true;
            order.push(f);
            let tri = m.tris[f as usize];
            for k in 0..3u32 {
                let h = 3 * f + k;
                let tw = t.twin[h as usize];
                if tw == NONE {
                    continue;
                }
                let g = tw / 3;
                if done[g as usize] {
                    continue;
                }
                let (a, b) = (tri[k as usize], tri[((k + 1) % 3) as usize]);
                let mid = math::lerp(m.pos[a as usize], m.pos[b as usize], 0.5);
                let w = math::dist(cents[f as usize], mid) + math::dist(mid, cents[g as usize]);
                let nd = dist[f as usize] + w;
                if nd < dist[g as usize] {
                    dist[g as usize] = nd;
                    parent[g as usize] = h;
                    heap.push(Reverse((nd.to_bits(), g)));
                }
            }
        }
    }
    (parent, order)
}

/// S6: cut graph and combing (DESIGN 6.8). `f` must be a valid field of the proxy (S4 output).
///
/// Errors (`Stage::Cut`): `InvalidInput` for mismatched sizes; `Invariant` for a closed
/// genus-0 component without cones (impossible by Poincare-Hopf, a convention bug), for a cut
/// mesh that is not a disk, or for a combed rotation that varies along an arc.
pub fn cut_and_comb(px: &ProxyRef, f: &CrossField, seed: u64) -> Result<CutGraph, AqError> {
    check_inputs(px, f)?;
    let (m, t) = (px.mesh, px.topo);
    let nf = m.tris.len();
    let nh = 3 * nf;
    let nv = m.pos.len();
    let nc = t.n_comp as usize;

    // Genus-0 closed components need cones (Poincare-Hopf: sum index4 = 8).
    let mut has_cone = vec![false; nc];
    let mut has_bnd = vec![false; nc];
    for v in 0..nv {
        let h0 = t.vert_he[v];
        if h0 == NONE {
            continue;
        }
        let c = t.face_comp[(h0 / 3) as usize] as usize;
        if c >= nc {
            continue;
        }
        has_cone[c] |= f.index4[v] != 0;
        has_bnd[c] |= t.is_bnd_vert[v];
    }
    for c in 0..nc {
        if !has_cone[c] && !has_bnd[c] && t.euler_char(m, c as u32) == 2 {
            return Err(inv(format!(
                "component {c} is a closed genus-0 surface without cones (Poincare-Hopf violated)"
            )));
        }
    }

    // 1-2. Dual tree and initial cut.
    let (parent, order) = dual_tree(px, seed);
    let mut is_cut: Vec<bool> = (0..nh).map(|h| t.twin[h] != NONE).collect();
    for &ph in &parent {
        if ph != NONE {
            is_cut[ph as usize] = false;
            is_cut[t.twin[ph as usize] as usize] = false;
        }
    }

    // 3. Prune dangling cut edges.
    let mut deg = cut_degree(px, &is_cut);
    let keep = |v: usize| f.index4[v] != 0 || t.is_bnd_vert[v] || px.feats.vert_corner[v];
    let mut stack: Vec<u32> =
        (0..nv as u32).rev().filter(|&v| deg[v as usize] == 1 && !keep(v as usize)).collect();
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

    // 4. Disk check.
    for (c, &e) in cut_mesh_euler(px, &is_cut).iter().enumerate() {
        if e != 1 {
            return Err(inv(format!("cut mesh of component {c} is not a disk (V - E + F = {e})")));
        }
    }

    // 5. Combing along the tree, parents first.
    let mut comb = vec![0u8; nf];
    for &g in &order {
        let ph = parent[g as usize];
        if ph != NONE {
            let pf = (ph / 3) as usize;
            comb[g as usize] = (comb[pf] + 4 - f.p[ph as usize] % 4) % 4;
        }
    }
    let theta_combed: Vec<f64> = f.theta.iter().zip(&comb).map(|(&th, &c)| comb_theta(th, c)).collect();
    let mut r = vec![0u8; nh];
    for h in 0..nh {
        let tw = t.twin[h];
        if tw == NONE {
            continue;
        }
        r[h] = comb_matching(f.p[h], comb[h / 3], comb[tw as usize / 3]);
        if !is_cut[h] && r[h] != 0 {
            return Err(inv(format!("combed matching {} != 0 off the cut at halfedge {h}", r[h])));
        }
    }

    // 6. Nodes and arcs.
    let (arcs, he_arc, node_verts) = build_arcs(px, f, &is_cut, &deg, &r)?;
    let cg = CutGraph { is_cut, he_arc, arcs, node_verts, theta_combed, r, comb_offset: comb };
    cg.validate(m, t, Some(f)).map_err(|e| e.at(Stage::Cut))?;
    Ok(cg)
}

/// Canonical edge id of a halfedge.
fn edge_id(t: &Topology, h: u32) -> usize {
    let tw = t.twin[h as usize];
    if tw == NONE { h as usize } else { h.min(tw) as usize }
}

type Arcs = (Vec<CutArc>, Vec<(u32, i8)>, Vec<u32>);

fn build_arcs(px: &ProxyRef, f: &CrossField, is_cut: &[bool], deg: &[u32], r: &[u8]) -> Result<Arcs, AqError> {
    let (m, t) = (px.mesh, px.topo);
    let nh = is_cut.len();
    let nv = m.pos.len();
    let mut node: Vec<bool> =
        (0..nv).map(|v| deg[v] > 0 && (deg[v] != 2 || f.index4[v] != 0 || t.is_bnd_vert[v])).collect();
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
            // A cut cycle without any node: its lowest-id halfedge's tail becomes a node.
            while next_unvisited < nh
                && !(is_cut[next_unvisited] && !visited[edge_id(t, next_unvisited as u32)])
            {
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
            if !is_cut[h0 as usize] || visited[edge_id(t, h0)] {
                continue;
            }
            visited[edge_id(t, h0)] = true;
            let mut hes = vec![h0];
            let mut cur = t.head(m, h0);
            let mut guard = 0usize;
            while !node[cur as usize] {
                guard += 1;
                if guard > nh {
                    return Err(inv("cut arc walk does not terminate"));
                }
                let Some(h) = t.fan_out(cur).find(|&h| is_cut[h as usize] && !visited[edge_id(t, h)]) else {
                    return Err(inv(format!("cut arc dead end at vertex {cur}")));
                };
                visited[edge_id(t, h)] = true;
                hes.push(h);
                cur = t.head(m, h);
            }
            let ra = r[h0 as usize];
            if let Some(&h) = hes.iter().find(|&&h| r[h as usize] != ra) {
                return Err(inv(format!("combed rotation not constant along a cut arc (halfedge {h})")));
            }
            let id = arcs.len() as u32;
            for &h in &hes {
                he_arc[h as usize] = (id, 1);
                he_arc[t.twin[h as usize] as usize] = (id, -1);
            }
            arcs.push(CutArc { hes, nodes: [v, cur], r: ra });
        }
    }
    starts.sort_unstable();
    starts.dedup();
    Ok((arcs, he_arc, starts))
}

fn uf_find(uf: &mut [u32], mut x: u32) -> u32 {
    while uf[x as usize] != x {
        let p = uf[uf[x as usize] as usize];
        uf[x as usize] = p;
        x = p;
    }
    x
}

/// Dual cycles generating the first homology of each component (diagnostics and tests).
///
/// The cut graph with all boundary vertices of a component merged into one super-vertex is
/// spanned by a forest (union-find over the cut edges in increasing edge id); every cut edge
/// closing a cycle gives one generator: cross it from side `face(h)` to `face(twin h)`, then
/// follow the faces glued along non-cut edges (a BFS forest) back to the start face. There are
/// `2g + b - 1` of them per component of genus g with b > 0 boundary loops, `2g` for a closed
/// one. Each cycle is the list of halfedges crossed in order (as `LoopCand::hes`) and starts and
/// ends in the face of its first halfedge.
pub fn homology_generators(px: &ProxyRef, cut: &CutGraph) -> Vec<Vec<u32>> {
    let (m, t) = (px.mesh, px.topo);
    let nf = m.tris.len();
    let nh = 3 * nf;
    let nv = m.pos.len();
    if cut.is_cut.len() != nh || t.twin.len() != nh {
        return Vec::new();
    }
    let mut uf: Vec<u32> = (0..nv as u32).collect();
    let mut bnd_rep = vec![NONE; t.n_comp as usize];
    for v in 0..nv as u32 {
        let h0 = t.vert_he[v as usize];
        if h0 == NONE || !t.is_bnd_vert[v as usize] {
            continue;
        }
        let c = t.face_comp[(h0 / 3) as usize] as usize;
        if c >= bnd_rep.len() {
            continue;
        }
        if bnd_rep[c] == NONE {
            bnd_rep[c] = v;
        } else {
            let (a, b) = (uf_find(&mut uf, v), uf_find(&mut uf, bnd_rep[c]));
            if a != b {
                uf[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let mut closing: Vec<u32> = Vec::new();
    for h in 0..nh as u32 {
        if !cut.is_cut[h as usize] || !t.is_canonical(h) {
            continue;
        }
        let (a, b) = (uf_find(&mut uf, t.tail(m, h)), uf_find(&mut uf, t.head(m, h)));
        if a == b {
            closing.push(h);
        } else {
            uf[a.max(b) as usize] = a.min(b);
        }
    }
    // BFS forest over the non-cut interior edges. parent[g] is the halfedge of g crossing to
    // its parent face.
    let mut parent = vec![NONE; nf];
    let mut depth = vec![u32::MAX; nf];
    let mut queue: Vec<u32> = Vec::with_capacity(nf);
    for s in 0..nf as u32 {
        if depth[s as usize] != u32::MAX {
            continue;
        }
        depth[s as usize] = 0;
        queue.clear();
        queue.push(s);
        let mut qi = 0;
        while qi < queue.len() {
            let fc = queue[qi];
            qi += 1;
            for k in 0..3 {
                let h = 3 * fc + k;
                let tw = t.twin[h as usize];
                if tw == NONE || cut.is_cut[h as usize] {
                    continue;
                }
                let g = tw / 3;
                if depth[g as usize] == u32::MAX {
                    depth[g as usize] = depth[fc as usize] + 1;
                    parent[g as usize] = tw;
                    queue.push(g);
                }
            }
        }
    }
    let mut out = Vec::with_capacity(closing.len());
    for h in closing {
        let tw = t.twin[h as usize];
        // Cross h from a = face(h) into b = face(twin h), then walk b -> lca -> a.
        let (mut x, mut y) = (tw / 3, h / 3);
        let mut up: Vec<u32> = Vec::new();
        let mut down: Vec<u32> = Vec::new();
        let mut guard = 0usize;
        while x != y && guard <= 2 * nf {
            guard += 1;
            if depth[x as usize] >= depth[y as usize] {
                let ph = parent[x as usize];
                if ph == NONE {
                    break;
                }
                up.push(ph);
                x = t.twin[ph as usize] / 3;
            } else {
                let ph = parent[y as usize];
                if ph == NONE {
                    break;
                }
                down.push(t.twin[ph as usize]);
                y = t.twin[ph as usize] / 3;
            }
        }
        if x != y {
            continue;
        }
        let mut cyc = Vec::with_capacity(1 + up.len() + down.len());
        cyc.push(h);
        cyc.extend(up);
        cyc.extend(down.into_iter().rev());
        out.push(cyc);
    }
    out
}

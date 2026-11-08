//! T-mesh extraction and patch validation (DESIGN 6.10 "T-mesh extraction", "Validation").

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_PI_2;

use aq_core::{
    AqError, ArcKind, FeatureKind, Layout, LayoutHealth, NONE, NodeKind, SeamlessParam, Stage, TArc, TMesh, TNode,
    TPatch, Topology, math, rot,
};

use crate::ctx::Ctx;
use crate::curves::{chain_corners_ends, chain_verts};
use crate::embed::Embedded;
use crate::geom::{DIR_ALL, dir_code};
use crate::{CurveKind, Emitter, LayoutIssue, ProxyView, Seed};

/// A proxy vertex that must become a T-mesh node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeHint {
    pub vert: u32,
    pub kind: NodeKind,
    pub index4: i8,
    /// Feature chain id or NONE.
    pub chain: u32,
}

/// Lowest constraint chain id at each proxy vertex.
fn vert_chain_ids(cx: &Ctx) -> Vec<u32> {
    let mut vc = vec![NONE; cx.m.pos.len()];
    for (h, &c) in cx.constraint_he.iter().enumerate() {
        let ch = cx.feats.he_chain[h];
        if !c || ch == NONE {
            continue;
        }
        let tri = cx.m.tris[h / 3];
        for v in [tri[h % 3], tri[(h % 3 + 1) % 3]] {
            let x = &mut vc[v as usize];
            *x = (*x).min(ch);
        }
    }
    vc
}

pub(crate) fn node_hints_ctx(cx: &Ctx, ems: &[Emitter]) -> Vec<NodeHint> {
    let is_cv = chain_verts(cx);
    let (corners, ends) = chain_corners_ends(cx);
    let vc = vert_chain_ids(cx);
    let mut verts: BTreeSet<u32> = cx.cone_index.keys().copied().collect();
    verts.extend(corners.iter().copied());
    verts.extend(ends.iter().copied());
    verts.extend(ems.iter().filter(|e| e.kind == CurveKind::SeedTrace).map(|e| e.vert));
    let mut out = Vec::new();
    for v in verts {
        let vi = v as usize;
        if vi >= cx.m.pos.len() {
            continue;
        }
        let bnd = cx.t.is_bnd_vert[vi];
        let cone = cx.cone_index.contains_key(&v);
        let kind = if corners.contains(&v) || (cone && is_cv[vi]) {
            if bnd { NodeKind::BoundaryCorner } else { NodeKind::FeatureCorner }
        } else if cone {
            NodeKind::Cone
        } else if ends.contains(&v) {
            NodeKind::ChainEnd
        } else {
            NodeKind::Seed
        };
        let index4 = cx.cone_index.get(&v).copied().unwrap_or(0).clamp(-4, 4);
        out.push(NodeHint { vert: v, kind, index4, chain: vc[vi] });
    }
    out
}

/// Public entry: the node hints of a proxy, param and emitter list.
pub fn node_hints(px: &ProxyView, prm: &SeamlessParam, ems: &[Emitter]) -> Result<Vec<NodeHint>, AqError> {
    let cx = Ctx::new(px, prm).map_err(|e| e.at(Stage::Layout))?;
    Ok(node_hints_ctx(&cx, ems))
}

fn fatal(msg: String) -> LayoutIssue {
    LayoutIssue::Fatal(AqError::invariant(msg).at(Stage::Layout))
}

/// testkit `arc_defaults` (DESIGN 6.11 table), allow_collapse off.
fn arc_defaults(kind: ArcKind, len: f64, wrap: bool) -> (f64, u32) {
    if wrap {
        return (8.0, 0);
    }
    match kind {
        ArcKind::Separatrix | ArcKind::SeedTrace => (1.0, 1),
        ArcKind::Feature | ArcKind::Boundary | ArcKind::Guide | ArcKind::Loop => (4.0, if len >= 0.5 { 1 } else { 0 }),
        ArcKind::Symmetry => (8.0, 1),
    }
}

struct Uf {
    p: Vec<u32>,
}
impl Uf {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.p[x as usize] != x {
            let g = self.p[self.p[x as usize] as usize];
            self.p[x as usize] = g;
            x = g;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.p[hi as usize] = lo;
        }
    }
}

pub(crate) fn extract_ctx(cx: &Ctx, e: &Embedded, hints: &[NodeHint]) -> Result<Layout, LayoutIssue> {
    let m = &e.wmesh;
    let t = &e.wtopo;
    let nf = m.tris.len();
    let nv = m.pos.len();
    let nh = 3 * nf;
    if e.he_label.len() != nh || e.wuv.len() != nf || e.parent_face.len() != nf {
        return Err(fatal("embedded sizes".into()));
    }
    let tail = |h: u32| m.tris[(h / 3) as usize][(h % 3) as usize];
    let head = |h: u32| m.tris[(h / 3) as usize][((h % 3 + 1) % 3) as usize];
    // Labels on both halves.
    let mut elab = vec![NONE; nh];
    for h in 0..nh {
        let tw = t.twin[h];
        let l = e.he_label[h];
        let l2 = if tw == NONE { NONE } else { e.he_label[tw as usize] };
        elab[h] = if l != NONE { l } else { l2 };
        if tw == NONE && elab[h] == NONE {
            return Err(fatal(format!("W' boundary halfedge {h} is not on a constraint chain")));
        }
    }
    let ncurves = e.curve_kind.len();
    if elab.iter().any(|&l| l != NONE && l as usize >= ncurves) {
        return Err(fatal("curve label out of range".into()));
    }
    let is_trace = |l: u32| l != NONE && !e.curve_kind[l as usize].is_constraint();
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for h in 0..nh as u32 {
        if elab[h as usize] != NONE && t.is_canonical(h) {
            inc[tail(h) as usize].push(h);
            inc[head(h) as usize].push(h);
        }
    }
    let mut vert_chain = vec![NONE; nv];
    for h in 0..nh as u32 {
        let l = elab[h as usize];
        if l != NONE && e.curve_kind[l as usize] == CurveKind::Chain {
            let ch = e.curve_chain[l as usize];
            for v in [tail(h), head(h)] {
                let x = &mut vert_chain[v as usize];
                *x = (*x).min(ch);
            }
        }
    }
    let hint: BTreeMap<u32, NodeHint> = hints.iter().map(|h| (h.vert, *h)).collect();
    let mut node_of = vec![NONE; nv];
    let mut nodes: Vec<TNode> = Vec::new();
    let mut node_vert: Vec<u32> = Vec::new();
    let add_node = |v: u32, nodes: &mut Vec<TNode>, node_vert: &mut Vec<u32>, node_of: &mut Vec<u32>| {
        let (kind, index4, chain) = match hint.get(&v) {
            Some(h) => (h.kind, h.index4, if h.chain != NONE { h.chain } else { vert_chain[v as usize] }),
            None => (NodeKind::Regular, 0, vert_chain[v as usize]),
        };
        node_of[v as usize] = nodes.len() as u32;
        nodes.push(TNode { kind, index4, chain });
        node_vert.push(v);
    };
    for v in 0..nv as u32 {
        let d = inc[v as usize].len();
        if d == 0 {
            continue;
        }
        let is_node = hint.contains_key(&v) || d != 2 || {
            let (l1, l2) = (elab[inc[v as usize][0] as usize], elab[inc[v as usize][1] as usize]);
            l1 != l2
                && !(is_trace(l1)
                    && is_trace(l2)
                    && e.curve_ends[l1 as usize].contains(&v)
                    && e.curve_ends[l2 as usize].contains(&v))
        };
        if is_node {
            add_node(v, &mut nodes, &mut node_vert, &mut node_of);
        }
    }
    // Arcs.
    let mut used = vec![false; nh];
    let mut arcs_v: Vec<Vec<u32>> = Vec::new();
    let mut arcs_h: Vec<Vec<u32>> = Vec::new();
    let walk = |v0: u32, h0: u32, used: &mut Vec<bool>, node_of: &Vec<u32>| -> Result<(Vec<u32>, Vec<u32>), LayoutIssue> {
        let mut vs = vec![v0];
        let mut hs = Vec::new();
        let (mut cur, mut h) = (v0, h0);
        loop {
            used[h as usize] = true;
            hs.push(h);
            let w = if tail(h) == cur { head(h) } else { tail(h) };
            vs.push(w);
            if node_of[w as usize] != NONE {
                break;
            }
            let Some(&g) = inc[w as usize].iter().find(|&&g| g != h && !used[g as usize]) else {
                return Err(fatal(format!("arc walk stuck at W' vertex {w}")));
            };
            cur = w;
            h = g;
            if hs.len() > nh {
                return Err(fatal("arc walk does not end".into()));
            }
        }
        Ok((vs, hs))
    };
    for ni in 0..node_vert.len() {
        let v = node_vert[ni];
        for &h in &inc[v as usize] {
            if used[h as usize] {
                continue;
            }
            let (vs, hs) = walk(v, h, &mut used, &node_of)?;
            arcs_v.push(vs);
            arcs_h.push(hs);
        }
    }
    for h0 in 0..nh as u32 {
        if elab[h0 as usize] == NONE || !t.is_canonical(h0) || used[h0 as usize] {
            continue;
        }
        // A node-free cycle: Regular node at its lowest vertex.
        let mut vmin = tail(h0);
        let (mut cur, mut h) = (tail(h0), h0);
        let mut steps = 0usize;
        loop {
            let w = if tail(h) == cur { head(h) } else { tail(h) };
            vmin = vmin.min(w);
            if w == tail(h0) {
                break;
            }
            let Some(&g) = inc[w as usize].iter().find(|&&g| g != h) else {
                return Err(fatal("open node-free curve".into()));
            };
            cur = w;
            h = g;
            steps += 1;
            if steps > nh {
                return Err(fatal("node-free cycle does not close".into()));
            }
        }
        add_node(vmin, &mut nodes, &mut node_vert, &mut node_of);
        let Some(&hs0) = inc[vmin as usize].iter().find(|&&g| !used[g as usize]) else {
            return Err(fatal("cycle node without free edge".into()));
        };
        let (vs, hs) = walk(vmin, hs0, &mut used, &node_of)?;
        arcs_v.push(vs);
        arcs_h.push(hs);
    }
    let nn = nodes.len();
    let na = arcs_v.len();
    let mut edge_arc: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut tarcs: Vec<TArc> = Vec::with_capacity(na);
    let mut arc_s: Vec<Vec<f64>> = Vec::with_capacity(na);
    let mut arc_label: Vec<u32> = Vec::with_capacity(na);
    for a in 0..na {
        let mut s = vec![0.0];
        let mut acc = 0.0;
        let mut lab = NONE;
        let mut any_sep = false;
        for (i, &h) in arcs_h[a].iter().enumerate() {
            edge_arc.insert(h, (a as u32, i as u32));
            let f = (h / 3) as usize;
            let j = (h % 3) as usize;
            let d = math::sub2(e.wuv[f][(j + 1) % 3], e.wuv[f][j]);
            acc += d[0].abs().max(d[1].abs());
            s.push(acc);
            let l = elab[h as usize];
            if lab == NONE {
                lab = l;
            }
            if l != NONE && e.curve_kind[l as usize] == CurveKind::Separatrix {
                any_sep = true;
            }
        }
        let kind = match e.curve_kind.get(lab as usize).copied() {
            Some(CurveKind::Chain) => match cx.feats.chains.get(e.curve_chain[lab as usize] as usize).map(|c| c.kind) {
                Some(FeatureKind::Boundary) => ArcKind::Boundary,
                Some(FeatureKind::Symmetry) => ArcKind::Symmetry,
                Some(FeatureKind::Guide) => ArcKind::Guide,
                _ => ArcKind::Feature,
            },
            Some(CurveKind::Loop) => ArcKind::Loop,
            Some(CurveKind::SeedTrace) if !any_sep => ArcKind::SeedTrace,
            _ => ArcKind::Separatrix,
        };
        let chain = if e.curve_kind.get(lab as usize) == Some(&CurveKind::Chain) { e.curve_chain[lab as usize] } else { NONE };
        let (Some(&v0), Some(&v1)) = (arcs_v[a].first(), arcs_v[a].last()) else {
            return Err(fatal("empty arc".into()));
        };
        tarcs.push(TArc {
            ends: [node_of[v0 as usize], node_of[v1 as usize]],
            len: acc,
            kind,
            lb: 0,
            weight: 0.0,
            wrap: false,
            chain,
        });
        arc_s.push(s);
        arc_label.push(lab);
    }
    // Wrap flags.
    let trace_labels_at = |v: u32| -> BTreeSet<u32> {
        inc[v as usize].iter().map(|&h| elab[h as usize]).filter(|&l| is_trace(l)).collect()
    };
    for a in 0..na {
        let ta = &tarcs[a];
        let both_t = nodes[ta.ends[0] as usize].kind == NodeKind::Regular && nodes[ta.ends[1] as usize].kind == NodeKind::Regular;
        if ta.len < 1.5 && both_t && ta.ends[0] != ta.ends[1] {
            let s0 = trace_labels_at(node_vert[ta.ends[0] as usize]);
            let s1 = trace_labels_at(node_vert[ta.ends[1] as usize]);
            let own = arc_label[a];
            if s0.intersection(&s1).any(|&l| l != own) {
                tarcs[a].wrap = true;
            }
        }
        let (w, lb) = arc_defaults(tarcs[a].kind, tarcs[a].len, tarcs[a].wrap);
        tarcs[a].weight = w;
        tarcs[a].lb = lb;
    }
    // Patches.
    let mut pid = vec![NONE; nf];
    let mut patch_faces: Vec<Vec<u32>> = Vec::new();
    for f0 in 0..nf {
        if pid[f0] != NONE {
            continue;
        }
        let p = patch_faces.len() as u32;
        let mut st = vec![f0 as u32];
        pid[f0] = p;
        let mut fs = Vec::new();
        while let Some(f) = st.pop() {
            fs.push(f);
            for j in 0..3 {
                let h = 3 * f as usize + j;
                let tw = t.twin[h];
                if tw == NONE || elab[h] != NONE {
                    continue;
                }
                let g = (tw / 3) as usize;
                if pid[g] == NONE {
                    pid[g] = p;
                    st.push(g as u32);
                }
            }
        }
        fs.sort_unstable();
        patch_faces.push(fs);
    }
    let mut uf = Uf { p: (0..nh as u32).collect() };
    for h in 0..nh as u32 {
        let tw = t.twin[h as usize];
        if tw != NONE && elab[h as usize] == NONE {
            uf.union(h, Topology::next(tw));
            uf.union(Topology::next(h), tw);
        }
    }
    let is_bnd_he = |h: u32| t.twin[h as usize] == NONE || elab[h as usize] != NONE;
    let mut class_angle = vec![0.0f64; nh];
    let mut class_bnd = vec![false; nh];
    for c in 0..nh as u32 {
        let f = (c / 3) as usize;
        let j = (c % 3) as usize;
        let u = &e.wuv[f];
        let e1 = math::sub2(u[(j + 1) % 3], u[j]);
        let e2 = math::sub2(u[(j + 2) % 3], u[j]);
        let ang = math::cross2(e1, e2).atan2(math::dot2(e1, e2));
        let r = uf.find(c);
        class_angle[r as usize] += ang;
        if is_bnd_he(c) {
            class_bnd[r as usize] = true;
        }
        if is_bnd_he(Topology::prev(c)) {
            class_bnd[r as usize] = true;
        }
    }
    let np = patch_faces.len();
    let mut tpatches: Vec<TPatch> = Vec::with_capacity(np);
    let mut side_lens: Vec<[f64; 4]> = Vec::with_capacity(np);
    let mut seeds: Vec<Seed> = Vec::new();
    let mut any_invalid = false;
    for p in 0..np {
        let fs = &patch_faces[p];
        let mut classes: BTreeSet<u32> = BTreeSet::new();
        let mut n_int_he = 0usize;
        let mut bhes: Vec<u32> = Vec::new();
        for &f in fs {
            for j in 0..3u32 {
                let h = 3 * f + j;
                classes.insert(uf.find(h));
                if is_bnd_he(h) {
                    bhes.push(h);
                } else {
                    n_int_he += 1;
                }
            }
        }
        let chi = classes.len() as i64 - (n_int_he / 2 + bhes.len()) as i64 + fs.len() as i64;
        let mut out_of: BTreeMap<u32, u32> = BTreeMap::new();
        let mut pinched = false;
        for &h in &bhes {
            if out_of.insert(uf.find(h), h).is_some() {
                pinched = true;
            }
        }
        let mut loops: Vec<Vec<u32>> = Vec::new();
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        if !pinched {
            for &h0 in &bhes {
                if seen.contains(&h0) {
                    continue;
                }
                let mut lp = Vec::new();
                let mut h = h0;
                loop {
                    if !seen.insert(h) {
                        break;
                    }
                    lp.push(h);
                    match out_of.get(&uf.find(Topology::next(h))) {
                        Some(&g) => h = g,
                        None => {
                            pinched = true;
                            break;
                        }
                    }
                }
                if h != h0 {
                    pinched = true;
                }
                loops.push(lp);
            }
        }
        let cone_inside = classes.iter().any(|&c| {
            let v = tail(c);
            !class_bnd[c as usize] && v < e.n_proxy_verts && cx.cone_index.contains_key(&v)
        });
        let count_of = |c: u32| class_angle[c as usize] / FRAC_PI_2;
        let mut valid = !pinched && chi == 1 && loops.len() == 1 && !cone_inside;
        let mut corners_at: Vec<usize> = Vec::new();
        if valid {
            let lp = &loops[0];
            for (i, &h) in lp.iter().enumerate() {
                let x = count_of(uf.find(h));
                let n = x.round();
                if (x - n).abs() > 0.02 {
                    return Err(LayoutIssue::Fatal(
                        AqError::numerical(format!("patch {p}: uv angle sum {x} quarter turns is not an integer")).at(Stage::Layout),
                    ));
                }
                if n == 1.0 {
                    if node_of[tail(h) as usize] == NONE {
                        return Err(fatal(format!("patch {p}: corner at W' vertex {} is not a node", tail(h))));
                    }
                    corners_at.push(i);
                } else if n != 2.0 {
                    valid = false;
                }
            }
            if corners_at.len() != 4 {
                valid = false;
            }
        }
        if !valid {
            any_invalid = true;
            if let Some(s) = seed_for_patch(cx, e, &bhes, &classes, &mut uf, &class_bnd, &count_of) {
                if !seeds.contains(&s) {
                    seeds.push(s);
                }
            } else {
                return Err(fatal(format!("invalid patch {p} offers no seed vertex")));
            }
            continue;
        }
        if any_invalid {
            continue;
        }
        // Patch chart rotations.
        let mut krot: BTreeMap<u32, u8> = BTreeMap::new();
        let f0 = fs[0];
        krot.insert(f0, 0);
        let mut st = vec![f0];
        while let Some(f) = st.pop() {
            let kf = krot.get(&f).copied().unwrap_or(0);
            for j in 0..3u32 {
                let h = 3 * f + j;
                if is_bnd_he(h) {
                    continue;
                }
                let g = t.twin[h as usize] / 3;
                if let std::collections::btree_map::Entry::Vacant(v) = krot.entry(g) {
                    v.insert((kf + e.wtrans[h as usize].r) % 4);
                    st.push(g);
                }
            }
        }
        let lp = &loops[0];
        let code_of = |h: u32| {
            let f = h / 3;
            let j = (h % 3) as usize;
            let u = &e.wuv[f as usize];
            let d = math::sub2(u[(j + 1) % 3], u[j]);
            dir_code(rot(krot.get(&f).copied().unwrap_or(0), d))
        };
        let Some(start) = corners_at.iter().position(|&i| code_of(lp[i]) == 0) else {
            return Err(fatal(format!("patch {p}: no side runs +u in the patch chart")));
        };
        let mut sides: [Vec<(u32, bool)>; 4] = Default::default();
        let mut lens = [0.0; 4];
        for (k, side) in sides.iter_mut().enumerate() {
            let i0 = corners_at[(start + k) % 4];
            let i1 = corners_at[(start + k + 1) % 4];
            let mut i = i0;
            loop {
                let h = lp[i];
                if code_of(h) != k as u8 {
                    return Err(fatal(format!("patch {p}: side {k} is not axis-consistent")));
                }
                let c = t.twin[h as usize];
                let ch = if c == NONE || h < c { h } else { c };
                let Some(&(a, pos)) = edge_arc.get(&ch) else {
                    return Err(fatal(format!("patch {p}: boundary edge without arc")));
                };
                let rev = tail(h) != arcs_v[a as usize][pos as usize];
                if node_of[tail(h) as usize] != NONE || side.is_empty() {
                    side.push((a, rev));
                    lens[k] += tarcs[a as usize].len;
                }
                i = (i + 1) % lp.len();
                if i == i1 {
                    break;
                }
            }
        }
        tpatches.push(TPatch { sides });
        side_lens.push(lens);
    }
    if any_invalid {
        return Err(LayoutIssue::Validation(seeds));
    }
    // Rings: arcs of loops and closed chains, in walking order.
    let mut by_curve: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for a in 0..na {
        let l = arc_label[a];
        if l == NONE {
            continue;
        }
        let k = e.curve_kind[l as usize];
        if k == CurveKind::Loop || (k == CurveKind::Chain && e.curve_closed[l as usize]) {
            by_curve.entry(l).or_default().push(a as u32);
        }
    }
    let mut rings: Vec<Vec<(u32, bool)>> = Vec::new();
    for list in by_curve.values() {
        let mut usedr = vec![false; list.len()];
        let mut ring: Vec<(u32, bool)> = vec![(list[0], false)];
        usedr[0] = true;
        let start = tarcs[list[0] as usize].ends[0];
        let mut cur = tarcs[list[0] as usize].ends[1];
        while cur != start || ring.len() < list.len() {
            let mut found = None;
            for (i, &a) in list.iter().enumerate() {
                if usedr[i] {
                    continue;
                }
                let en = tarcs[a as usize].ends;
                if en[0] == cur {
                    found = Some((i, a, false, en[1]));
                    break;
                }
                if en[1] == cur {
                    found = Some((i, a, true, en[0]));
                    break;
                }
            }
            let Some((i, a, r, nx)) = found else { break };
            usedr[i] = true;
            ring.push((a, r));
            cur = nx;
        }
        if cur == start && usedr.iter().all(|&u| u) {
            rings.push(ring);
        }
    }
    let mut chi = 0i64;
    for c in 0..t.n_comp {
        chi += t.euler_char(m, c);
    }
    let tmesh = TMesh { nodes, arcs: tarcs, patches: tpatches, rings, chi: chi as i32 };
    tmesh.validate().map_err(|e| LayoutIssue::Fatal(e.at(Stage::Layout)))?;
    let health = LayoutHealth::from_sides(&side_lens);
    let patch_chart_face: Vec<u32> = patch_faces.iter().map(|fs| fs[0]).collect();
    let layout = Layout {
        wmesh: e.wmesh.clone(),
        wtopo: e.wtopo.clone(),
        wuv: e.wuv.clone(),
        wtrans: e.wtrans.clone(),
        parent_face: e.parent_face.clone(),
        tmesh,
        node_vert,
        arc_verts: arcs_v,
        arc_s,
        patch_faces,
        patch_chart_face,
        vert_chain,
        health,
    };
    let _ = nn;
    layout.validate(cx.n_faces()).map_err(|e| LayoutIssue::Fatal(e.at(Stage::Layout)))?;
    Ok(layout)
}

/// Seed for an invalid patch: rule (a) the lowest proxy vertex on the patch boundary whose
/// patch sector spans at least 2 quarter turns, emitting the middle interior ray; rule (b)
/// the lowest interior proxy vertex, emitting every free axis.
fn seed_for_patch(
    cx: &Ctx,
    e: &Embedded,
    bhes: &[u32],
    classes: &BTreeSet<u32>,
    uf: &mut Uf,
    class_bnd: &[bool],
    count_of: &dyn Fn(u32) -> f64,
) -> Option<Seed> {
    let m = &e.wmesh;
    let tail = |h: u32| m.tris[(h / 3) as usize][(h % 3) as usize];
    let mut best: Option<(u32, u32, Seed)> = None;
    for &h in bhes {
        let v = tail(h);
        if v >= e.n_proxy_verts {
            continue;
        }
        let n = count_of(uf.find(h)).round();
        if n < 2.0 {
            continue;
        }
        let f = (h / 3) as usize;
        let j = (h % 3) as usize;
        let d = math::sub2(e.wuv[f][(j + 1) % 3], e.wuv[f][j]);
        let d0 = dir_code(d);
        let off = ((n as i64) / 2).rem_euclid(4) as u8;
        let s = Seed { vert: v, face: e.parent_face[f], dir: (d0 + off) % 4 };
        if best.is_none_or(|b| (v, h) < (b.0, b.1)) {
            best = Some((v, h, s));
        }
    }
    if let Some(b) = best {
        return Some(b.2);
    }
    let mut bestb: Option<(u32, u32)> = None;
    for &c in classes {
        let v = tail(c);
        if class_bnd[c as usize] || v >= e.n_proxy_verts || cx.cone_index.contains_key(&v) {
            continue;
        }
        if bestb.is_none_or(|b| (v, c) < b) {
            bestb = Some((v, c));
        }
    }
    bestb.map(|(v, c)| Seed { vert: v, face: e.parent_face[(c / 3) as usize], dir: DIR_ALL })
}

/// DESIGN 6.10 `extract`: the layout of an embedding, or the seeds a re-run needs.
pub fn extract(e: &Embedded, px: &ProxyView, prm: &SeamlessParam, hints: &[NodeHint]) -> Result<Layout, LayoutIssue> {
    let cx = Ctx::new(px, prm).map_err(|e| LayoutIssue::Fatal(e.at(Stage::Layout)))?;
    extract_ctx(&cx, e, hints)
}

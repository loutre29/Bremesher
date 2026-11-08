//! QComb construction and the validity checks of DESIGN 6.11 step 4.
//!
//! Every T-mesh node, every arc sample `0..=n_a` and every patch grid point `(i, j)` is an
//! element of one union-find. Arc samples are glued to their end nodes and to the boundary
//! rows and columns of the patches they bound, so a zero arc merges its end nodes and a
//! zero-height (zero-width) patch glues side 0 to side 2 (side 1 to side 3) index by index.
//! Every union caused by a collapse is recorded with its cause (the zero arc, or the glued
//! patch), so a violation can be traced back to the zero arcs that produced it.

use aq_core::alloc::try_vec;
use aq_core::{AqError, NONE, NodeKind, QComb, TMesh};

/// A validity violation of a quantized T-mesh (DESIGN 6.11 step 4).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Violation {
    /// (i) Non-manifold QComb: the QComb vertices involved (an edge used twice in the same
    /// direction or by more than 2 quads, a quad repeating a vertex, a doublet, a vertex whose
    /// quads do not form one fan, a vertex in no quad).
    NonManifold(Vec<u32>),
    /// (ii) QComb vertex with an interior valence <= 2 or >= 6 (valence 6 is allowed at a
    /// lone cone of index4 -2, WP-23), or merged node set whose combined index4 is outside
    /// {-1, 0, 1}.
    BadValence(u32),
    /// (iii) QComb vertex where a corner node is merged with a cone, or with a corner of a
    /// different chain.
    CornerMerge(u32),
    /// (iv) Glued zero patch (patch id) identifying two different chains or boundary loops.
    Pinch(u32),
    /// (v) Ring (index into `TMesh.rings`) whose quantized length is below
    /// `max(3, ring_min / 2)`.
    ShortRing(u32),
    /// (vi) The QComb Euler characteristic (differs from `TMesh.chi`).
    Euler(i64),
    /// The lengths do not fit the T-mesh: patch id whose opposite sides differ, or NONE for a
    /// length vector of the wrong size or a QComb above the size budget (addition, CHANGES.md).
    Unbalanced(u32),
}

/// Result of [`check_comb`]: the QComb (even when invalid) and the violations, each with the
/// sorted zero arcs whose collapse caused it.
pub(crate) struct CombCheck {
    pub comb: QComb,
    pub violations: Vec<Violation>,
    pub causes: Vec<Vec<u32>>,
}

/// Largest number of union-find elements a QComb may use.
const MAX_ELEMS: u64 = 1 << 31;

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        let gp = p[p[x as usize] as usize];
        p[x as usize] = gp;
        x = gp;
    }
    x
}

fn union(p: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(p, a), find(p, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        p[hi as usize] = lo;
    }
}

fn unbalanced(id: u32) -> CombCheck {
    CombCheck { comb: QComb::default(), violations: vec![Violation::Unbalanced(id)], causes: vec![Vec::new()] }
}

/// Builds the QComb of `t` with arc lengths `n` and checks it.
pub(crate) fn check_comb(t: &TMesh, n: &[u32], ring_min: u32) -> Result<CombCheck, AqError> {
    let na = t.arcs.len();
    let nn = t.nodes.len();
    let np = t.patches.len();
    if n.len() != na {
        return Ok(unbalanced(NONE));
    }
    // Patch dimensions.
    let mut dims: Vec<[u32; 2]> = Vec::with_capacity(np);
    let mut bad = Vec::new();
    for (p, pa) in t.patches.iter().enumerate() {
        let mut s = [0u64; 4];
        for (k, side) in pa.sides.iter().enumerate() {
            for &(a, _) in side {
                s[k] += *n.get(a as usize).unwrap_or(&0) as u64;
            }
        }
        if s[0] != s[2] || s[1] != s[3] || s[0] > u32::MAX as u64 || s[1] > u32::MAX as u64 {
            bad.push(Violation::Unbalanced(p as u32));
        }
        dims.push([s[0] as u32, s[1] as u32]);
    }
    if !bad.is_empty() {
        let causes = vec![Vec::new(); bad.len()];
        return Ok(CombCheck { comb: QComb::default(), violations: bad, causes });
    }
    // Element layout: nodes, arc samples, patch grids.
    let mut total = nn as u64;
    let mut arc_base = Vec::with_capacity(na);
    for &x in n {
        arc_base.push(total);
        total += x as u64 + 1;
    }
    let mut patch_base = Vec::with_capacity(np);
    for d in &dims {
        patch_base.push(total);
        total += (d[0] as u64 + 1) * (d[1] as u64 + 1);
    }
    if total >= MAX_ELEMS {
        return Ok(unbalanced(NONE));
    }
    let total = total as usize;
    let mut par: Vec<u32> = try_vec(total, 0u32)?;
    for (i, x) in par.iter_mut().enumerate() {
        *x = i as u32;
    }
    // (elem, elem, cause): cause < na is a zero arc, na + p the glue of zero patch p.
    let mut links: Vec<(u32, u32)> = Vec::new();
    for (a, arc) in t.arcs.iter().enumerate() {
        let b = arc_base[a] as u32;
        union(&mut par, arc.ends[0], b);
        union(&mut par, b + n[a], arc.ends[1]);
        if n[a] == 0 {
            links.push((b, a as u32));
        }
    }
    for (p, pa) in t.patches.iter().enumerate() {
        let [w, h] = dims[p];
        let base = patch_base[p] as u32;
        let gid = |i: u32, j: u32| base + j * (w + 1) + i;
        for (k, side) in pa.sides.iter().enumerate() {
            let glue = (k == 2 && h == 0) || (k == 3 && w == 0);
            let mut pos = 0u32;
            for &(a, rev) in side {
                let na_ = n[a as usize];
                let ab = arc_base[a as usize] as u32;
                for s in 0..=na_ {
                    let tpos = pos + s;
                    let (i, j) = match k {
                        0 => (tpos, 0),
                        1 => (w, tpos),
                        2 => (w - tpos, h),
                        _ => (0, h - tpos),
                    };
                    let g = gid(i, j);
                    let e = ab + if rev { na_ - s } else { s };
                    union(&mut par, g, e);
                    if glue {
                        links.push((g, (na + p) as u32));
                    }
                }
                pos += na_;
            }
        }
    }
    // Class ids in order of first element.
    let mut id_of_root: Vec<u32> = try_vec(total, NONE)?;
    let mut elem_id: Vec<u32> = try_vec(total, 0u32)?;
    let mut nv = 0u32;
    for e in 0..total {
        let r = find(&mut par, e as u32) as usize;
        if id_of_root[r] == NONE {
            id_of_root[r] = nv;
            nv += 1;
        }
        elem_id[e] = id_of_root[r];
    }
    let node_id: Vec<u32> = (0..nn).map(|i| elem_id[i]).collect();
    let arc_ids: Vec<Vec<u32>> =
        (0..na).map(|a| (0..=n[a]).map(|s| elem_id[arc_base[a] as usize + s as usize]).collect()).collect();
    let mut patch_ids = Vec::with_capacity(np);
    let mut quads: Vec<[u32; 4]> = Vec::new();
    let mut quad_patch = Vec::new();
    for p in 0..np {
        let [w, h] = dims[p];
        let base = patch_base[p] as usize;
        let cnt = (w as usize + 1) * (h as usize + 1);
        let ids: Vec<u32> = elem_id[base..base + cnt].to_vec();
        let g = |i: u32, j: u32| ids[(j * (w + 1) + i) as usize];
        for j in 0..h {
            for i in 0..w {
                quads.push([g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)]);
                quad_patch.push(p as u32);
            }
        }
        patch_ids.push(ids);
    }
    let comb = QComb { n_verts: nv, node_id, arc_ids, patch_dims: dims.clone(), patch_ids, quads, quad_patch };

    // Causes per class.
    let mut class_cause: Vec<(u32, u32)> = links.iter().map(|&(e, c)| (elem_id[e as usize], c)).collect();
    class_cause.sort_unstable();
    class_cause.dedup();
    let expand = |c: u32, out: &mut Vec<u32>| {
        if (c as usize) < na {
            out.push(c);
            return;
        }
        let p = c as usize - na;
        let [w, h] = dims[p];
        let pa = &t.patches[p];
        let push_side = |k: usize, out: &mut Vec<u32>| {
            for &(a, _) in &pa.sides[k] {
                if n[a as usize] == 0 {
                    out.push(a);
                }
            }
        };
        if h == 0 {
            push_side(1, out);
            push_side(3, out);
        }
        if w == 0 {
            push_side(0, out);
            push_side(2, out);
        }
    };
    let causes_of_class = |v: u32, out: &mut Vec<u32>| {
        let lo = class_cause.partition_point(|&(c, _)| c < v);
        for &(c, cause) in &class_cause[lo..] {
            if c != v {
                break;
            }
            expand(cause, out);
        }
    };
    let finish = |mut v: Vec<u32>| {
        v.sort_unstable();
        v.dedup();
        v
    };

    let mut viol: Vec<Violation> = Vec::new();
    let mut causes: Vec<Vec<u32>> = Vec::new();
    let nvu = nv as usize;
    let nqd = comb.quads.len();

    // (i) Manifoldness.
    let mut nonman: Vec<Vec<u32>> = Vec::new();
    // (lo, hi, quad, k)
    let mut edges: Vec<(u32, u32, u32, u8)> = Vec::with_capacity(4 * nqd);
    for (qi, q) in comb.quads.iter().enumerate() {
        let mut rep = false;
        for k in 0..4 {
            if q[..k].contains(&q[k]) {
                rep = true;
            }
        }
        if rep {
            let mut vs = q.to_vec();
            vs.sort_unstable();
            vs.dedup();
            nonman.push(vs);
        }
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            if a != b {
                edges.push((a.min(b), a.max(b), qi as u32, k as u8));
            }
        }
    }
    edges.sort_unstable();
    let mut corner_par: Vec<u32> = (0..4 * nqd as u32).collect();
    let mut valence = vec![0u32; nvu];
    let mut on_bnd = vec![false; nvu];
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut n_edges = 0i64;
    let mut i = 0;
    while i < edges.len() {
        let mut j = i;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        let (lo, hi) = (edges[i].0, edges[i].1);
        n_edges += 1;
        valence[lo as usize] += 1;
        valence[hi as usize] += 1;
        let uses = &edges[i..j];
        let dir = |u: &(u32, u32, u32, u8)| comb.quads[u.2 as usize][u.3 as usize] == lo;
        match uses.len() {
            1 => {
                on_bnd[lo as usize] = true;
                on_bnd[hi as usize] = true;
            }
            2 if dir(&uses[0]) != dir(&uses[1]) => {
                let (q1, k1) = (uses[0].2, uses[0].3 as u32);
                let (q2, k2) = (uses[1].2, uses[1].3 as u32);
                // Corner of q1 at the tail of its edge with the corner of q2 at the head of its
                // edge (the same vertex), and the other way around.
                union(&mut corner_par, 4 * q1 + k1, 4 * q2 + (k2 + 1) % 4);
                union(&mut corner_par, 4 * q1 + (k1 + 1) % 4, 4 * q2 + k2);
                pairs.push((q1.min(q2), q1.max(q2)));
            }
            _ => {
                nonman.push(vec![lo, hi]);
                on_bnd[lo as usize] = true;
                on_bnd[hi as usize] = true;
            }
        }
        i = j;
    }
    pairs.sort_unstable();
    for w in pairs.windows(2) {
        if w[0] == w[1] {
            let (q1, q2) = (comb.quads[w[0].0 as usize], comb.quads[w[0].1 as usize]);
            let mut vs: Vec<u32> = q1.iter().copied().filter(|v| q2.contains(v)).collect();
            vs.sort_unstable();
            vs.dedup();
            nonman.push(vs);
        }
    }
    // Fans: the corners around each vertex must form one component.
    let mut first_root = vec![NONE; nvu];
    let mut multi = vec![false; nvu];
    for qi in 0..nqd {
        for k in 0..4 {
            let v = comb.quads[qi][k] as usize;
            let r = find(&mut corner_par, (4 * qi + k) as u32);
            if first_root[v] == NONE {
                first_root[v] = r;
            } else if first_root[v] != r {
                multi[v] = true;
            }
        }
    }
    for v in 0..nvu {
        if first_root[v] == NONE || multi[v] {
            nonman.push(vec![v as u32]);
        }
    }
    nonman.sort();
    nonman.dedup();
    for vs in nonman {
        let mut c = Vec::new();
        for &v in &vs {
            causes_of_class(v, &mut c);
        }
        causes.push(finish(c));
        viol.push(Violation::NonManifold(vs));
    }

    // Per-class node data.
    let mut cnt = vec![0u32; nvu];
    let mut idx = vec![0i32; nvu];
    let mut cone = vec![false; nvu];
    let mut corner_chain: Vec<(u32, u32)> = Vec::new(); // (vertex, chain)
    for (i, nd) in t.nodes.iter().enumerate() {
        let v = comb.node_id[i] as usize;
        cnt[v] += 1;
        idx[v] += nd.index4 as i32;
        match nd.kind {
            NodeKind::Cone => cone[v] = true,
            NodeKind::FeatureCorner | NodeKind::BoundaryCorner => corner_chain.push((v as u32, nd.chain)),
            _ => {}
        }
    }
    corner_chain.sort_unstable();
    corner_chain.dedup();

    // (ii) Valence and combined index.
    for v in 0..nvu {
        // WP-23: a lone -2 cone (two -1 cones merged by S4) is an interior valence 6 pole.
        let six_ok = valence[v] == 6 && cnt[v] == 1 && cone[v] && idx[v] == -2;
        let interior_bad = !on_bnd[v] && first_root[v] != NONE && (valence[v] <= 2 || (valence[v] >= 6 && !six_ok));
        let index_bad = cnt[v] >= 2 && !(-1..=1).contains(&idx[v]);
        if interior_bad || index_bad {
            let mut c = Vec::new();
            causes_of_class(v as u32, &mut c);
            causes.push(finish(c));
            viol.push(Violation::BadValence(v as u32));
        }
    }
    // (iii) Corner merges.
    let mut i = 0;
    while i < corner_chain.len() {
        let v = corner_chain[i].0;
        let mut j = i;
        while j < corner_chain.len() && corner_chain[j].0 == v {
            j += 1;
        }
        if cone[v as usize] || j - i >= 2 {
            let mut c = Vec::new();
            causes_of_class(v, &mut c);
            causes.push(finish(c));
            viol.push(Violation::CornerMerge(v));
        }
        i = j;
    }
    // (iv) Pinches by glued zero patches.
    for (p, pa) in t.patches.iter().enumerate() {
        let [w, h] = dims[p];
        let chains = |k: usize| -> Vec<u32> {
            let mut c: Vec<u32> =
                pa.sides[k].iter().map(|&(a, _)| t.arcs[a as usize].chain).filter(|&c| c != NONE).collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let differ = |x: &[u32], y: &[u32]| x.iter().any(|a| y.iter().any(|b| a != b));
        let pinch = (h == 0 && differ(&chains(0), &chains(2))) || (w == 0 && differ(&chains(1), &chains(3)));
        if pinch {
            let mut c = Vec::new();
            expand((na + p) as u32, &mut c);
            causes.push(finish(c));
            viol.push(Violation::Pinch(p as u32));
        }
    }
    // (v) Rings.
    let ring_floor = 3u64.max(ring_min as u64 / 2);
    for (r, ring) in t.rings.iter().enumerate() {
        let sum: u64 = ring.iter().map(|&(a, _)| n[a as usize] as u64).sum();
        if sum < ring_floor {
            let c: Vec<u32> = ring.iter().map(|&(a, _)| a).filter(|&a| n[a as usize] == 0).collect();
            causes.push(finish(c));
            viol.push(Violation::ShortRing(r as u32));
        }
    }
    // (vi) Euler characteristic.
    let euler = nv as i64 - n_edges + nqd as i64;
    if euler != t.chi as i64 {
        let mut c = Vec::new();
        for &(_, cause) in &class_cause {
            expand(cause, &mut c);
        }
        causes.push(finish(c));
        viol.push(Violation::Euler(euler));
    }
    Ok(CombCheck { comb, violations: viol, causes })
}

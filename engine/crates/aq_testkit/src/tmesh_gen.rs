//! T-mesh generators (DESIGN 6.2 "TMesh generators"), mesh-free inputs for aq_quantize and
//! aq_patch. Every result passes `TMesh::validate`. Arc weights and lower bounds follow the
//! DESIGN 6.11 table with `allow_collapse = false` ([`arc_defaults`]); the quantizer may
//! recompute them.

use std::collections::BTreeMap;

use aq_core::{AqError, ArcKind, NONE, NodeKind, SplitMix64, TArc, TMesh, TNode, TPatch};

use crate::util::{bad, ensure};

/// (weight, lb) of an arc per the DESIGN 6.11 table.
pub fn arc_defaults(kind: ArcKind, len: f64, wrap: bool, allow_collapse: bool) -> (f64, u32) {
    if wrap {
        return (8.0, 0);
    }
    match kind {
        ArcKind::Separatrix | ArcKind::SeedTrace => (1.0, if allow_collapse { 0 } else { 1 }),
        ArcKind::Feature | ArcKind::Boundary | ArcKind::Guide | ArcKind::Loop => {
            (4.0, if len >= 0.5 { 1 } else { 0 })
        }
        ArcKind::Symmetry => (8.0, 1),
    }
}

fn arc(ends: [u32; 2], len: f64, kind: ArcKind, wrap: bool, chain: u32) -> TArc {
    let (weight, lb) = arc_defaults(kind, len, wrap, false);
    TArc { ends, len, kind, lb, weight, wrap, chain }
}

fn finish(t: TMesh) -> Result<TMesh, AqError> {
    t.validate()?;
    Ok(t)
}

/// Builds the T-mesh of an axis-aligned rectangle dissection of a rectangular domain. `rects`
/// are `[x0, y0, x1, y1]` (x0 < x1, y0 < y1) tiling the domain without overlap; `extra` adds
/// degree-2 nodes on rectangle sides. Nodes: every corner (exact coordinates); arcs: maximal
/// segments between consecutive nodes on each side, oriented left to right / bottom to top;
/// patch sides: bottom (+u), right (+v), top (-u), left (-v). The 4 domain corners are
/// BoundaryCorner nodes of index4 +1; domain sides are Boundary arcs (chain 0) and form the
/// only ring; interior arcs are Separatrix. chi = 1.
pub fn tmesh_from_rects(rects: &[[f64; 4]], extra: &[[f64; 2]]) -> Result<TMesh, AqError> {
    ensure(!rects.is_empty(), || "tmesh_from_rects: no rectangle".into())?;
    for r in rects {
        if !(r.iter().all(|x| x.is_finite()) && r[0] < r[2] && r[1] < r[3]) {
            return Err(bad(format!("tmesh_from_rects: bad rectangle {r:?}")));
        }
    }
    let norm = |x: f64| x + 0.0;
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for r in rects {
        for p in [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]] {
            pts.push([norm(p[0]), norm(p[1])]);
        }
    }
    pts.extend(extra.iter().map(|p| [norm(p[0]), norm(p[1])]));
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for r in rects {
        xmin = xmin.min(r[0]);
        ymin = ymin.min(r[1]);
        xmax = xmax.max(r[2]);
        ymax = ymax.max(r[3]);
    }
    let on_bnd = |p: [f64; 2]| p[0] == xmin || p[0] == xmax || p[1] == ymin || p[1] == ymax;
    let nodes: Vec<TNode> = pts
        .iter()
        .map(|&p| {
            let corner = (p[0] == xmin || p[0] == xmax) && (p[1] == ymin || p[1] == ymax);
            if corner {
                TNode { kind: NodeKind::BoundaryCorner, index4: 1, chain: 0 }
            } else {
                TNode { kind: NodeKind::Regular, index4: 0, chain: if on_bnd(p) { 0 } else { NONE } }
            }
        })
        .collect();
    // Nodes on a segment, sorted along it. `horizontal`: y = c, x in [lo, hi].
    let on_segment = |horizontal: bool, c: f64, lo: f64, hi: f64| -> Vec<u32> {
        let mut v: Vec<u32> = (0..pts.len() as u32)
            .filter(|&i| {
                let p = pts[i as usize];
                if horizontal { p[1] == c && p[0] >= lo && p[0] <= hi } else { p[0] == c && p[1] >= lo && p[1] <= hi }
            })
            .collect();
        let k = if horizontal { 0 } else { 1 };
        v.sort_by(|&a, &b| pts[a as usize][k].total_cmp(&pts[b as usize][k]));
        v
    };
    let mut arcs: Vec<TArc> = Vec::new();
    let mut arc_id: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut get_arc = |a: u32, b: u32, horizontal: bool, arcs: &mut Vec<TArc>| -> u32 {
        *arc_id.entry((a, b)).or_insert_with(|| {
            let (pa, pb) = (pts[a as usize], pts[b as usize]);
            let len = (pb[0] - pa[0]).abs() + (pb[1] - pa[1]).abs();
            let bnd = if horizontal { pa[1] == ymin || pa[1] == ymax } else { pa[0] == xmin || pa[0] == xmax };
            let kind = if bnd { ArcKind::Boundary } else { ArcKind::Separatrix };
            arcs.push(arc([a, b], len, kind, false, if bnd { 0 } else { NONE }));
            (arcs.len() - 1) as u32
        })
    };
    let mut patches = Vec::with_capacity(rects.len());
    for r in rects {
        let mut sides: [Vec<(u32, bool)>; 4] = Default::default();
        let segs = [
            (true, r[1], r[0], r[2], false),
            (false, r[2], r[1], r[3], false),
            (true, r[3], r[0], r[2], true),
            (false, r[0], r[1], r[3], true),
        ];
        for (k, &(hz, c, lo, hi, rev)) in segs.iter().enumerate() {
            let ns = on_segment(hz, c, lo, hi);
            ensure(ns.len() >= 2, || format!("rectangle {r:?}: side {k} lacks its corners"))?;
            let mut list: Vec<(u32, bool)> = ns.windows(2).map(|w| (get_arc(w[0], w[1], hz, &mut arcs), rev)).collect();
            if rev {
                list.reverse();
            }
            sides[k] = list;
        }
        patches.push(TPatch { sides });
    }
    // Boundary ring, CCW from (xmin, ymin).
    let mut ring = Vec::new();
    for (hz, c, lo, hi, rev) in [
        (true, ymin, xmin, xmax, false),
        (false, xmax, ymin, ymax, false),
        (true, ymax, xmin, xmax, true),
        (false, xmin, ymin, ymax, true),
    ] {
        let ns = on_segment(hz, c, lo, hi);
        let mut list: Vec<(u32, bool)> = Vec::new();
        for w in ns.windows(2) {
            let id = arc_id.get(&(w[0], w[1])).copied().ok_or_else(|| bad("domain boundary not covered by rectangles"))?;
            list.push((id, rev));
        }
        if rev {
            list.reverse();
        }
        ring.extend(list);
    }
    let chi = nodes.len() as i64 - arcs.len() as i64 + patches.len() as i64;
    ensure(chi == 1, || format!("rectangle dissection has chi {chi} (overlap or gap)"))?;
    finish(TMesh { nodes, arcs, patches, rings: vec![ring], chi: 1 })
}

/// One rectangular patch with side lengths (l0, l1, l0, l1).
pub fn tmesh_rect(l0: f64, l1: f64) -> Result<TMesh, AqError> {
    tmesh_from_rects(&[[0.0, 0.0, l0, l1]], &[])
}

/// [`tmesh_grid_sized`] with unit cells.
pub fn tmesh_grid(rows: u32, cols: u32, t_offsets: &[f64]) -> Result<TMesh, AqError> {
    tmesh_grid_sized(rows, cols, 1.0, 1.0, t_offsets)
}

/// A rows x cols grid of patches over `[0, cols w] x [0, rows h]`. In patch row r the interior
/// vertical separators sit at `c w + t_offsets[r]` (missing offsets are 0), so unequal offsets
/// of neighbouring rows create T-junctions joined by short arcs.
pub fn tmesh_grid_sized(rows: u32, cols: u32, w: f64, h: f64, t_offsets: &[f64]) -> Result<TMesh, AqError> {
    if rows == 0 || cols == 0 || !(w > 0.0 && h > 0.0) {
        return Err(bad("tmesh_grid: rows, cols >= 1 and positive cell sizes"));
    }
    let mut rects = Vec::new();
    for r in 0..rows {
        let off = t_offsets.get(r as usize).copied().unwrap_or(0.0);
        let x = |c: u32| -> f64 {
            if c == 0 { 0.0 } else if c == cols { cols as f64 * w } else { c as f64 * w + off }
        };
        for c in 0..cols {
            let (x0, x1) = (x(c), x(c + 1));
            if x0.partial_cmp(&x1) != Some(std::cmp::Ordering::Less) {
                return Err(bad(format!("tmesh_grid: offset {off} folds row {r}")));
            }
            rects.push([x0, r as f64 * h, x1, (r + 1) as f64 * h]);
        }
    }
    tmesh_from_rects(&rects, &[])
}

/// Box T-mesh: 8 Cone nodes (index4 +1), 12 Separatrix arcs (lengths a, b, c along x, y, z), 6
/// patches, chi = 2.
pub fn tmesh_box(a: f64, b: f64, c: f64) -> Result<TMesh, AqError> {
    if !(a > 0.0 && b > 0.0 && c > 0.0) {
        return Err(bad("tmesh_box: positive lengths"));
    }
    let id = |x: u32, y: u32, z: u32| x + 2 * y + 4 * z;
    let faces: [[(u32, u32, u32); 4]; 6] = [
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    ];
    let nodes = vec![TNode { kind: NodeKind::Cone, index4: 1, chain: NONE }; 8];
    let mut arcs: Vec<TArc> = Vec::new();
    let mut ids: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut patches = Vec::new();
    for f in &faces {
        let mut sides: [Vec<(u32, bool)>; 4] = Default::default();
        for k in 0..4 {
            let (p, q) = (f[k], f[(k + 1) % 4]);
            let (u, v) = (id(p.0, p.1, p.2), id(q.0, q.1, q.2));
            let key = (u.min(v), u.max(v));
            let len = if p.0 != q.0 { a } else if p.1 != q.1 { b } else { c };
            let ai = *ids.entry(key).or_insert_with(|| {
                arcs.push(arc([key.0, key.1], len, ArcKind::Separatrix, false, NONE));
                (arcs.len() - 1) as u32
            });
            sides[k] = vec![(ai, u != key.0)];
        }
        patches.push(TPatch { sides });
    }
    finish(TMesh { nodes, arcs, patches, rings: Vec::new(), chi: 2 })
}

/// [`tmesh_box`] with unit lengths: the cube (its QComb has 8 valence-3 vertices).
pub fn tmesh_cube() -> Result<TMesh, AqError> {
    tmesh_box(1.0, 1.0, 1.0)
}

/// Helix layout on an open cylinder of circumference `circ` and `length` (DESIGN 6.11 helix
/// test): a seed trace T runs axially from the bottom rim to the top rim, and a separatrix S
/// spirals `turns` times around, each turn ending `pitch` higher on T. T is split into wrap
/// arcs of length `pitch` (flagged `wrap` when pitch < 1.5, weight 8, target 0) between the
/// consecutive crossings. Patches: the bottom region, `turns - 1` helical bands, the top region.
/// Both rims are Boundary rings. chi = 0.
/// Arc ids: 0 bottom rim, 1 top rim, 2 T below the helix, 3 T above it, `4 + k` the k-th wrap
/// arc, `4 + turns + k` the k-th turn of S.
pub fn tmesh_helix(circ: f64, length: f64, pitch: f64, turns: u32) -> Result<TMesh, AqError> {
    let k = turns as usize;
    let below = 0.5 * (length - turns as f64 * pitch);
    if turns < 2 || !(circ > 0.0 && pitch > 0.0 && below > 0.0) {
        return Err(bad("tmesh_helix: turns >= 2, positive sizes, turns * pitch < length"));
    }
    let wrap = pitch < 1.5;
    let h = |i: usize| (1 + i) as u32; // h_0 .. h_K
    let (b, top) = (0u32, (k + 2) as u32);
    let mut nodes = vec![TNode { kind: NodeKind::Regular, index4: 0, chain: NONE }; k + 3];
    nodes[0].chain = 0;
    nodes[k + 2].chain = 1;
    nodes[1].kind = NodeKind::Seed;
    let mut arcs = vec![
        arc([b, b], circ, ArcKind::Boundary, false, 0),
        arc([top, top], circ, ArcKind::Boundary, false, 1),
        arc([b, h(0)], below, ArcKind::SeedTrace, false, NONE),
        arc([h(k), top], length - below - turns as f64 * pitch, ArcKind::SeedTrace, false, NONE),
    ];
    for i in 0..k {
        arcs.push(arc([h(i), h(i + 1)], pitch, ArcKind::SeedTrace, wrap, NONE));
    }
    for i in 0..k {
        arcs.push(arc([h(i), h(i + 1)], circ, ArcKind::Separatrix, false, NONE));
    }
    let tw = |i: usize| (4 + i) as u32;
    let turn = |i: usize| (4 + k + i) as u32;
    let mut patches = vec![TPatch {
        sides: [vec![(0, false)], vec![(2, false), (tw(0), false)], vec![(turn(0), true)], vec![(2, true)]],
    }];
    for i in 0..k - 1 {
        patches.push(TPatch {
            sides: [vec![(turn(i), false)], vec![(tw(i + 1), false)], vec![(turn(i + 1), true)], vec![(tw(i), true)]],
        });
    }
    patches.push(TPatch {
        sides: [vec![(turn(k - 1), false)], vec![(3, false)], vec![(1, true)], vec![(3, true), (tw(k - 1), true)]],
    });
    finish(TMesh { nodes, arcs, patches, rings: vec![vec![(0, false)], vec![(1, false)]], chi: 0 })
}

/// Random guillotine dissection of a rectangle into `n_patches` patches with integer corner
/// coordinates (so every arc length is a positive integer and the instance is feasible with
/// n >= 1), then every arc length scaled by a random factor in [0.85, 1.15].
pub fn tmesh_random_dissection(n_patches: u32, seed: u64) -> Result<TMesh, AqError> {
    if n_patches == 0 {
        return Err(bad("tmesh_random_dissection: n_patches >= 1"));
    }
    let mut rng = SplitMix64::new(seed);
    let side = 4.0 * (n_patches as f64).sqrt().ceil();
    let mut rects: Vec<[f64; 4]> = vec![[0.0, 0.0, side, side]];
    let mut guard = 0u32;
    while rects.len() < n_patches as usize {
        guard += 1;
        if guard > 100 * n_patches + 100 {
            return Err(AqError::internal("tmesh_random_dissection: cannot split further"));
        }
        let i = rng.below(rects.len() as u64) as usize;
        let r = rects[i];
        let (w, h) = (r[2] - r[0], r[3] - r[1]);
        let vertical = if w >= 2.0 && h >= 2.0 { rng.next_f64() < w / (w + h) } else { w >= 2.0 };
        let span = if vertical { w } else { h };
        if span < 2.0 {
            continue;
        }
        let cut = 1.0 + rng.below(span as u64 - 1) as f64;
        if vertical {
            rects[i] = [r[0], r[1], r[0] + cut, r[3]];
            rects.push([r[0] + cut, r[1], r[2], r[3]]);
        } else {
            rects[i] = [r[0], r[1], r[2], r[1] + cut];
            rects.push([r[0], r[1] + cut, r[2], r[3]]);
        }
    }
    let mut t = tmesh_from_rects(&rects, &[])?;
    for a in t.arcs.iter_mut() {
        a.len *= rng.range_f64(0.85, 1.15);
        let (w, lb) = arc_defaults(a.kind, a.len, a.wrap, false);
        a.weight = w;
        a.lb = lb;
    }
    finish(t)
}

/// Abstract bi-directed constraint instance with exactly `n_arcs` arcs (>= 2): random patches
/// whose sides are split into random slots, slots glued in random pairs (opposite traversal,
/// any sides, so (+,+), (-,-) and (+,-) edges all occur), unglued slots are Boundary arcs.
/// The gluing is planted on an integer solution (each patch an integer W x H rectangle, glued
/// slots of equal length), so the instance is always feasible with n >= lb. Arc lengths are
/// the planted values plus noise in [-0.45, 0.45]. The surface it describes may be exotic;
/// chi is its V - E + F.
pub fn tmesh_random_bidirected(n_arcs: u32, seed: u64) -> Result<TMesh, AqError> {
    if n_arcs < 2 {
        return Err(bad("tmesh_random_bidirected: n_arcs >= 2"));
    }
    let na = n_arcs as usize;
    let mut rng = SplitMix64::new(seed);
    // slots[p][k] = planted values of the slots of side k of patch p.
    let mut plan: Option<(Vec<[Vec<u32>; 4]>, Vec<(usize, usize)>)> = None;
    for attempt in 0..65 {
        let symmetric = attempt == 64;
        let (np, s_total) = if symmetric {
            (1usize, (na + na % 2).max(4))
        } else {
            let pmax = (na / 2).max(1);
            let np = 1 + rng.below(pmax as u64) as usize;
            let s_min = (4 * np).max(na);
            let s_max = 2 * na;
            if s_min > s_max {
                continue;
            }
            (np, s_min + rng.below((s_max - s_min + 1) as u64) as usize)
        };
        let pairs = s_total - na;
        let mut m = vec![[1usize; 4]; np];
        if symmetric {
            let half = s_total / 2;
            let a = 1 + rng.below((half - 1) as u64) as usize;
            m[0] = [a, half - a, a, half - a];
        } else {
            for _ in 0..s_total - 4 * np {
                let p = rng.below(np as u64) as usize;
                let k = rng.below(4) as usize;
                m[p][k] += 1;
            }
        }
        let mut sides: Vec<[Vec<u32>; 4]> = Vec::with_capacity(np);
        for mp in &m {
            let wd = mp[0].max(mp[2]) + if symmetric { 0 } else { rng.below(2) as usize };
            let ht = mp[1].max(mp[3]) + if symmetric { 0 } else { rng.below(2) as usize };
            let mut s: [Vec<u32>; 4] = Default::default();
            for k in 0..4 {
                let total = if k % 2 == 0 { wd } else { ht };
                let mut v = vec![1u32; mp[k]];
                for _ in 0..total - mp[k] {
                    let i = rng.below(mp[k] as u64) as usize;
                    v[i] += 1;
                }
                s[k] = v;
            }
            sides.push(s);
        }
        // Flat slot list (patch, side, index) and greedy random pairing among equal values.
        let mut flat: Vec<(usize, usize, usize)> = Vec::new();
        for (p, s) in sides.iter().enumerate() {
            for k in 0..4 {
                for i in 0..s[k].len() {
                    flat.push((p, k, i));
                }
            }
        }
        let mut order: Vec<usize> = (0..flat.len()).collect();
        rng.shuffle(&mut order);
        let val = |x: usize| {
            let (p, k, i) = flat[x];
            sides[p][k][i]
        };
        let mut used = vec![false; flat.len()];
        let mut glued = Vec::new();
        for oi in 0..order.len() {
            if glued.len() == pairs {
                break;
            }
            let a = order[oi];
            if used[a] {
                continue;
            }
            if let Some(&b) = order[oi + 1..].iter().find(|&&b| !used[b] && val(b) == val(a)) {
                used[a] = true;
                used[b] = true;
                glued.push((a, b));
            }
        }
        if glued.len() == pairs {
            let flat_pairs = glued;
            // Re-express pairs as slot positions in the patch boundary cycles below.
            plan = Some((sides, flat_pairs));
            break;
        }
    }
    let Some((sides, pairs)) = plan else { return Err(AqError::internal("bidirected generator failed")) };
    // Boundary cycle points per patch: slot j of patch p runs from point off[p] + j to the next.
    let mut slot_of: Vec<(usize, usize, usize)> = Vec::new(); // flat index -> (p, k, i)
    let mut cyc_index: Vec<usize> = Vec::new(); // flat index -> position in its patch cycle
    let mut off = Vec::new();
    let mut total_pts = 0usize;
    for (p, s) in sides.iter().enumerate() {
        off.push(total_pts);
        let mut j = 0;
        for k in 0..4 {
            for i in 0..s[k].len() {
                slot_of.push((p, k, i));
                cyc_index.push(j);
                j += 1;
            }
        }
        total_pts += j;
    }
    let cyc_len = |p: usize| sides[p].iter().map(|v| v.len()).sum::<usize>();
    let ends = |x: usize| -> (usize, usize) {
        let p = slot_of[x].0;
        let j = cyc_index[x];
        (off[p] + j, off[p] + (j + 1) % cyc_len(p))
    };
    let mut uf: Vec<usize> = (0..total_pts).collect();
    let find = |uf: &mut Vec<usize>, mut x: usize| -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    };
    for &(a, b) in &pairs {
        let (a1, b1) = ends(a);
        let (a2, b2) = ends(b);
        for (x, y) in [(a1, b2), (b1, a2)] {
            let (rx, ry) = (find(&mut uf, x), find(&mut uf, y));
            if rx != ry {
                uf[rx.max(ry)] = rx.min(ry);
            }
        }
    }
    let mut node_of_root: BTreeMap<usize, u32> = BTreeMap::new();
    let mut node_of_pt = vec![0u32; total_pts];
    for x in 0..total_pts {
        let r = find(&mut uf, x);
        let n = node_of_root.len() as u32;
        node_of_pt[x] = *node_of_root.entry(r).or_insert(n);
    }
    let n_nodes = node_of_root.len();
    let mut arc_of = vec![(NONE, false); slot_of.len()];
    let mut arcs: Vec<TArc> = Vec::new();
    let mut partner = vec![usize::MAX; slot_of.len()];
    for &(a, b) in &pairs {
        partner[a] = b;
        partner[b] = a;
    }
    for x in 0..slot_of.len() {
        if arc_of[x].0 != NONE {
            continue;
        }
        let (p, k, i) = slot_of[x];
        let v = sides[p][k][i] as f64;
        let len = (v + rng.range_f64(-0.45, 0.45)).max(0.05);
        let (s0, s1) = ends(x);
        let e = [node_of_pt[s0], node_of_pt[s1]];
        let id = arcs.len() as u32;
        if partner[x] == usize::MAX {
            arcs.push(arc(e, len, ArcKind::Boundary, false, 0));
            arc_of[x] = (id, false);
        } else {
            arcs.push(arc(e, len, ArcKind::Separatrix, false, NONE));
            arc_of[x] = (id, false);
            arc_of[partner[x]] = (id, true);
        }
    }
    let mut patches = Vec::new();
    let mut x = 0usize;
    for s in &sides {
        let mut ps: [Vec<(u32, bool)>; 4] = Default::default();
        for k in 0..4 {
            for _ in 0..s[k].len() {
                ps[k].push(arc_of[x]);
                x += 1;
            }
        }
        patches.push(TPatch { sides: ps });
    }
    ensure(arcs.len() == na, || format!("bidirected generator made {} arcs, not {na}", arcs.len()))?;
    let chi = n_nodes as i64 - arcs.len() as i64 + patches.len() as i64;
    let nodes = vec![TNode { kind: NodeKind::Regular, index4: 0, chain: NONE }; n_nodes];
    finish(TMesh { nodes, arcs, patches, rings: Vec::new(), chi: chi as i32 })
}

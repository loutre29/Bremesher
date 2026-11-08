//! Test support: layouts built from axis-aligned rectangle dissections of a uv domain, lifted
//! to 3D by an arbitrary map, and a QComb builder (the union-find construction of DESIGN 6.11
//! step 4, written here so the tests do not depend on aq_quantize).
#![allow(dead_code, clippy::needless_range_loop)]

use std::collections::{BTreeMap, BTreeSet};

use aq_core::{
    ArcKind, Layout, LayoutHealth, NONE, NodeKind, QComb, SplitMix64, TArc, TMesh, TNode, TPatch, Topology,
    Transition, TriMesh, V2, V3,
};

pub struct RectSpec<'a> {
    /// Rectangles in cell units [x0, y0, x1, y1].
    pub rects: Vec<[i64; 4]>,
    pub cell: f64,
    /// Interior vertex jitter, fraction of the cell.
    pub jitter: f64,
    pub seed: u64,
    /// Diagonal of cell (ci, cj): true = (i, j)-(i+1, j+1), false = (i+1, j)-(i, j+1).
    /// None = random.
    pub diag: Option<&'a dyn Fn(i64, i64) -> bool>,
}

impl<'a> RectSpec<'a> {
    pub fn new(rects: Vec<[i64; 4]>, cell: f64) -> Self {
        RectSpec { rects, cell, jitter: 0.0, seed: 1, diag: None }
    }
}

pub struct Case {
    pub lay: Layout,
    /// uv of each node.
    pub node_uv: Vec<V2>,
    /// uv per W' vertex (single chart).
    pub uv: Vec<V2>,
}

fn snap(x: f64) -> f64 {
    (x * 16_777_216.0).round() / 16_777_216.0
}

pub fn rect_layout(spec: &RectSpec, lift: &dyn Fn(V2) -> V3) -> Case {
    let cell = spec.cell;
    let mut rng = SplitMix64::new(spec.seed);
    // Cells.
    let mut cell_rect: BTreeMap<(i64, i64), u32> = BTreeMap::new();
    for (r, re) in spec.rects.iter().enumerate() {
        for cj in re[1]..re[3] {
            for ci in re[0]..re[2] {
                assert!(cell_rect.insert((ci, cj), r as u32).is_none(), "rectangles overlap");
            }
        }
    }
    let on_line = |i: i64, j: i64| -> bool {
        spec.rects.iter().any(|r| {
            ((i == r[0] || i == r[2]) && j >= r[1] && j <= r[3]) || ((j == r[1] || j == r[3]) && i >= r[0] && i <= r[2])
        })
    };
    let mut vid: BTreeMap<(i64, i64), u32> = BTreeMap::new();
    for &(ci, cj) in cell_rect.keys() {
        for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
            let n = vid.len() as u32;
            vid.entry((ci + di, cj + dj)).or_insert(n);
        }
    }
    let mut uv = vec![[0.0; 2]; vid.len()];
    for (&(i, j), &v) in &vid {
        let mut p = [i as f64 * cell, j as f64 * cell];
        if spec.jitter > 0.0 && !on_line(i, j) {
            p[0] = snap(p[0] + spec.jitter * cell * (2.0 * rng.next_f64() - 1.0));
            p[1] = snap(p[1] + spec.jitter * cell * (2.0 * rng.next_f64() - 1.0));
        }
        uv[v as usize] = p;
    }
    let mut tris = Vec::new();
    let mut tri_rect = Vec::new();
    for (&(ci, cj), &r) in &cell_rect {
        let a = vid[&(ci, cj)];
        let b = vid[&(ci + 1, cj)];
        let c = vid[&(ci + 1, cj + 1)];
        let d = vid[&(ci, cj + 1)];
        let main = match spec.diag {
            Some(f) => f(ci, cj),
            None => rng.next_u64() & 1 == 0,
        };
        if main {
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        } else {
            tris.push([a, b, d]);
            tris.push([b, c, d]);
        }
        tri_rect.push(r);
        tri_rect.push(r);
    }
    let pos: Vec<V3> = uv.iter().map(|&p| lift(p)).collect();
    let wmesh = TriMesh { pos, tris };
    let wtopo = Topology::build(&wmesh).expect("W' topology");
    let nf = wmesh.tris.len();
    let wuv: Vec<[V2; 3]> = wmesh.tris.iter().map(|t| [uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]]).collect();

    // Nodes: rectangle corners.
    let mut corner_count: BTreeMap<(i64, i64), u32> = BTreeMap::new();
    for r in &spec.rects {
        for c in [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])] {
            *corner_count.entry(c).or_insert(0) += 1;
        }
    }
    let node_pts: Vec<(i64, i64)> = corner_count.keys().copied().collect();
    let node_of: BTreeMap<(i64, i64), u32> = node_pts.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
    // Sides of each rect as node sequences.
    let side_nodes = |r: &[i64; 4], k: usize| -> Vec<u32> {
        let mut v: Vec<(i64, u32)> = Vec::new();
        for (&(x, y), &n) in &node_of {
            match k {
                0 if y == r[1] && x >= r[0] && x <= r[2] => v.push((x, n)),
                1 if x == r[2] && y >= r[1] && y <= r[3] => v.push((y, n)),
                2 if y == r[3] && x >= r[0] && x <= r[2] => v.push((-x, n)),
                3 if x == r[0] && y >= r[1] && y <= r[3] => v.push((-y, n)),
                _ => {}
            }
        }
        v.sort();
        v.into_iter().map(|x| x.1).collect()
    };
    let mut arc_of: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut arc_ends: Vec<[u32; 2]> = Vec::new();
    let mut arc_uses: Vec<u32> = Vec::new();
    let mut patches = Vec::new();
    for r in &spec.rects {
        let mut sides: [Vec<(u32, bool)>; 4] = Default::default();
        for (k, side) in sides.iter_mut().enumerate() {
            let ns = side_nodes(r, k);
            for w in ns.windows(2) {
                let key = (w[0].min(w[1]), w[0].max(w[1]));
                let a = *arc_of.entry(key).or_insert_with(|| {
                    arc_ends.push([key.0, key.1]);
                    arc_uses.push(0);
                    (arc_ends.len() - 1) as u32
                });
                arc_uses[a as usize] += 1;
                side.push((a, w[0] != arc_ends[a as usize][0]));
            }
        }
        patches.push(TPatch { sides });
    }
    // Arc vertex chains.
    let mut arc_verts = Vec::new();
    let mut arc_s = Vec::new();
    let mut arcs = Vec::new();
    let mut vert_chain = vec![NONE; wmesh.pos.len()];
    for (a, e) in arc_ends.iter().enumerate() {
        let p0 = node_pts[e[0] as usize];
        let p1 = node_pts[e[1] as usize];
        let steps = (p1.0 - p0.0).abs() + (p1.1 - p0.1).abs();
        let (dx, dy) = ((p1.0 - p0.0).signum(), (p1.1 - p0.1).signum());
        let vs: Vec<u32> = (0..=steps).map(|k| vid[&(p0.0 + k * dx, p0.1 + k * dy)]).collect();
        let mut s = vec![0.0];
        for w in vs.windows(2) {
            let (u0, u1) = (uv[w[0] as usize], uv[w[1] as usize]);
            let d = (u1[0] - u0[0]).abs() + (u1[1] - u0[1]).abs();
            s.push(s[s.len() - 1] + d);
        }
        let len = s[s.len() - 1];
        let bnd = arc_uses[a] == 1;
        if bnd {
            for &v in &vs {
                vert_chain[v as usize] = 0;
            }
        }
        arcs.push(TArc {
            ends: *e,
            len,
            kind: if bnd { ArcKind::Boundary } else { ArcKind::Separatrix },
            lb: 1,
            weight: if bnd { 4.0 } else { 1.0 },
            wrap: false,
            chain: if bnd { 0 } else { NONE },
        });
        arc_verts.push(vs);
        arc_s.push(s);
    }
    let mut node_bnd = vec![false; node_pts.len()];
    for (a, e) in arc_ends.iter().enumerate() {
        if arc_uses[a] == 1 {
            node_bnd[e[0] as usize] = true;
            node_bnd[e[1] as usize] = true;
        }
    }
    let nodes: Vec<TNode> = node_pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let c = corner_count[p];
            let corner = node_bnd[i] && (c == 1 || c == 3);
            TNode {
                kind: if corner { NodeKind::BoundaryCorner } else { NodeKind::Regular },
                index4: if corner { if c == 1 { 1 } else { -1 } } else { 0 },
                chain: if node_bnd[i] { 0 } else { NONE },
            }
        })
        .collect();
    let node_vert: Vec<u32> = node_pts.iter().map(|p| vid[p]).collect();
    let chi = nodes.len() as i32 - arcs.len() as i32 + patches.len() as i32;
    let tmesh = TMesh { nodes, arcs, patches, rings: Vec::new(), chi };
    tmesh.validate().expect("T-mesh");
    let np = spec.rects.len();
    let mut patch_faces = vec![Vec::new(); np];
    for (f, &r) in tri_rect.iter().enumerate() {
        patch_faces[r as usize].push(f as u32);
    }
    let patch_chart_face = patch_faces.iter().map(|fs| fs[0]).collect();
    let side_lens: Vec<[f64; 4]> = tmesh
        .patches
        .iter()
        .map(|p| {
            let mut l = [0.0; 4];
            for k in 0..4 {
                l[k] = p.sides[k].iter().map(|&(a, _)| tmesh.arcs[a as usize].len).sum();
            }
            l
        })
        .collect();
    let lay = Layout {
        wmesh,
        wtopo,
        wuv,
        wtrans: vec![Transition::IDENTITY; 3 * nf],
        parent_face: (0..nf as u32).collect(),
        tmesh,
        node_vert,
        arc_verts,
        arc_s,
        patch_faces,
        patch_chart_face,
        vert_chain,
        health: LayoutHealth::from_sides(&side_lens),
    };
    lay.validate(nf).expect("layout");
    let node_uv = node_pts.iter().map(|p| [p.0 as f64 * cell, p.1 as f64 * cell]).collect();
    Case { lay, node_uv, uv }
}

/// Moves the faces selected by `pick` to the chart `tr(uv)` and sets the transitions on the
/// halfedges between the two charts (C10: across he from f to g, `phi_g = T_he(phi_f)`).
pub fn rechart(lay: &mut Layout, pick: &dyn Fn(u32) -> bool, tr: Transition) {
    let nf = lay.wmesh.tris.len();
    let sel: Vec<bool> = (0..nf as u32).map(pick).collect();
    for f in 0..nf {
        if sel[f] {
            for k in 0..3 {
                lay.wuv[f][k] = tr.apply(lay.wuv[f][k]);
            }
        }
    }
    for h in 0..3 * nf {
        let tw = lay.wtopo.twin[h];
        if tw == NONE {
            continue;
        }
        let (f, g) = (h / 3, tw as usize / 3);
        lay.wtrans[h] = match (sel[f], sel[g]) {
            (true, false) => tr.inverse(),
            (false, true) => tr,
            _ => Transition::IDENTITY,
        };
    }
}

/// n per arc: round(len), with overrides.
pub fn round_n(t: &TMesh, over: &[(u32, u32)]) -> Vec<u32> {
    let mut n: Vec<u32> = t.arcs.iter().map(|a| a.len.round() as u32).collect();
    for &(a, v) in over {
        n[a as usize] = v;
    }
    n
}

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

/// QComb of `t` with arc lengths `n` (same element order and union rules as aq_quantize).
pub fn build_comb(t: &TMesh, n: &[u32]) -> QComb {
    let nn = t.nodes.len();
    let na = t.arcs.len();
    let mut dims = Vec::new();
    for pa in &t.patches {
        let mut s = [0u32; 4];
        for (k, side) in pa.sides.iter().enumerate() {
            s[k] = side.iter().map(|&(a, _)| n[a as usize]).sum();
        }
        assert!(s[0] == s[2] && s[1] == s[3], "unbalanced patch {s:?}");
        dims.push([s[0], s[1]]);
    }
    let mut total = nn;
    let mut arc_base = Vec::new();
    for &x in n {
        arc_base.push(total);
        total += x as usize + 1;
    }
    let mut patch_base = Vec::new();
    for d in &dims {
        patch_base.push(total);
        total += (d[0] as usize + 1) * (d[1] as usize + 1);
    }
    let mut par: Vec<u32> = (0..total as u32).collect();
    for (a, arc) in t.arcs.iter().enumerate() {
        let b = arc_base[a] as u32;
        union(&mut par, arc.ends[0], b);
        union(&mut par, b + n[a], arc.ends[1]);
    }
    for (p, pa) in t.patches.iter().enumerate() {
        let [w, h] = dims[p];
        let base = patch_base[p] as u32;
        for (k, side) in pa.sides.iter().enumerate() {
            let mut pos = 0u32;
            for &(a, rev) in side {
                let m = n[a as usize];
                for s in 0..=m {
                    let tp = pos + s;
                    let (i, j) = match k {
                        0 => (tp, 0),
                        1 => (w, tp),
                        2 => (w - tp, h),
                        _ => (0, h - tp),
                    };
                    let e = arc_base[a as usize] as u32 + if rev { m - s } else { s };
                    union(&mut par, base + j * (w + 1) + i, e);
                }
                pos += m;
            }
        }
    }
    let mut id_of_root = vec![NONE; total];
    let mut elem = vec![0u32; total];
    let mut nv = 0u32;
    for e in 0..total {
        let r = find(&mut par, e as u32) as usize;
        if id_of_root[r] == NONE {
            id_of_root[r] = nv;
            nv += 1;
        }
        elem[e] = id_of_root[r];
    }
    let node_id = (0..nn).map(|i| elem[i]).collect();
    let arc_ids = (0..na).map(|a| (0..=n[a] as usize).map(|s| elem[arc_base[a] + s]).collect()).collect();
    let mut patch_ids = Vec::new();
    let mut quads = Vec::new();
    let mut quad_patch = Vec::new();
    for (p, d) in dims.iter().enumerate() {
        let [w, h] = *d;
        let cnt = (w as usize + 1) * (h as usize + 1);
        let ids: Vec<u32> = elem[patch_base[p]..patch_base[p] + cnt].to_vec();
        let g = |i: u32, j: u32| ids[(j * (w + 1) + i) as usize];
        for j in 0..h {
            for i in 0..w {
                quads.push([g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)]);
                quad_patch.push(p as u32);
            }
        }
        patch_ids.push(ids);
    }
    let q = QComb { n_verts: nv, node_id, arc_ids, patch_dims: dims, patch_ids, quads, quad_patch };
    q.validate(t).expect("QComb");
    q
}

pub fn plane(p: V2) -> V3 {
    [p[0], p[1], 0.0]
}

/// Square [0, 8]^2 onto a hemisphere of radius 16/pi (each side maps to a quarter of the
/// equator, of length 8), through the elliptical square-to-disk map.
pub fn hemisphere(p: V2) -> V3 {
    let r = 16.0 / std::f64::consts::PI;
    let a = p[0] / 4.0 - 1.0;
    let b = p[1] / 4.0 - 1.0;
    let x = a * (1.0 - 0.5 * b * b).sqrt();
    let y = b * (1.0 - 0.5 * a * a).sqrt();
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    [r * x, r * y, r * z]
}

pub fn cylinder(radius: f64) -> impl Fn(V2) -> V3 {
    move |p: V2| [radius * (p[0] / radius).cos(), radius * (p[0] / radius).sin(), p[1]]
}

/// Mean |corner angle - 90 degrees| over all quad corners, in degrees.
pub fn mean_corner_dev_deg(m: &aq_core::QuadMesh) -> f64 {
    let mut s = 0.0;
    let mut c = 0.0;
    for f in 0..m.n_faces() {
        let fv = m.face(f);
        for k in 0..4 {
            let v = m.pos[fv[k] as usize];
            let a = aq_core::math::sub(m.pos[fv[(k + 1) % 4] as usize], v);
            let b = aq_core::math::sub(m.pos[fv[(k + 3) % 4] as usize], v);
            let ang = aq_core::math::angle_between(a, b).to_degrees();
            s += (ang - 90.0).abs();
            c += 1.0;
        }
    }
    s / c
}

/// Number of undirected edges of a polygon mesh.
pub fn n_edges(m: &aq_core::QuadMesh) -> usize {
    let mut e: BTreeSet<(u32, u32)> = BTreeSet::new();
    for f in 0..m.n_faces() {
        let fv = m.face(f);
        for k in 0..fv.len() {
            let (a, b) = (fv[k], fv[(k + 1) % fv.len()]);
            e.insert((a.min(b), a.max(b)));
        }
    }
    e.len()
}

/// Distance from p to segment ab.
pub fn seg_dist(p: V3, a: V3, b: V3) -> f64 {
    use aq_core::math::*;
    let d = sub(b, a);
    let l2 = norm2(d);
    let t = if l2 > 0.0 { (dot(sub(p, a), d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    dist(p, add(a, scale(d, t)))
}

#![allow(dead_code)]

use aq_core::math;
use aq_core::{Bvh, CtlState, FeatureKind, Projector, TriMesh, V3};
use aq_prep::{ComponentMesh, Ingested, Proxy, ProxyParams, build_proxy, ingest, split_components};

pub fn ctl() -> CtlState {
    CtlState::new()
}

/// Ingest without features.
pub fn ingest_plain(m: TriMesh) -> Ingested {
    let st = ctl();
    ingest(m, &[], &[], &st.unbounded()).unwrap()
}

/// Edges whose dihedral angle exceeds `deg` (the CLI's `--features auto`), as Hard pairs.
pub fn auto_features(m: &TriMesh, deg: f64) -> (Vec<[u32; 2]>, Vec<u8>) {
    let mut e: Vec<(u32, u32, usize)> = Vec::new();
    for (f, t) in m.tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            e.push((a.min(b), a.max(b), f));
        }
    }
    e.sort_unstable();
    let mut pairs = Vec::new();
    let lim = deg.to_radians().cos();
    let mut i = 0;
    while i < e.len() {
        let mut j = i;
        while j < e.len() && e[j].0 == e[i].0 && e[j].1 == e[i].1 {
            j += 1;
        }
        if j - i == 2 {
            let n1 = m.face_normal(e[i].2 as u32);
            let n2 = m.face_normal(e[i + 1].2 as u32);
            if let (Some(n1), Some(n2)) = (n1, n2)
                && math::dot(n1, n2) < lim
            {
                pairs.push([e[i].0, e[i].1]);
            }
        }
        i = j;
    }
    let kinds = vec![FeatureKind::Hard as u8; pairs.len()];
    (pairs, kinds)
}

/// Components scaled to quad units for a total target of `n_target` quads, and h0.
pub fn quad_units(ing: &Ingested, n_target: f64) -> (Vec<ComponentMesh>, f64) {
    let comps = split_components(ing);
    let a: f64 = comps.iter().map(|c| c.area).sum();
    let h0 = (a / n_target).sqrt();
    (comps.iter().map(|c| c.scaled(1.0 / h0)).collect(), h0)
}

/// Proxy of a component with a constant length, projecting onto the component.
pub fn proxy_const(c: &ComponentMesh, len: f64, p: &ProxyParams) -> Result<Proxy, aq_core::AqError> {
    let (fr, bvh) = c.source_geometry()?;
    let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
    let st = ctl();
    build_proxy(c, &proj, &|_| len, p, &st.unbounded())
}

/// One-sided distance from `a` (vertices plus 3 interior samples per face) to `b`.
pub fn one_sided(a: &TriMesh, b: &TriMesh) -> f64 {
    let bvh = Bvh::build(b);
    let mut mx: f64 = 0.0;
    for p in &a.pos {
        mx = mx.max(bvh.closest(*p).2);
    }
    let w = [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]];
    for f in 0..a.tris.len() as u32 {
        let [p0, p1, p2] = a.tri_pos(f);
        for wi in &w {
            mx = mx.max(bvh.closest(math::bary_point(p0, p1, p2, *wi)).2);
        }
    }
    mx
}

pub fn hausdorff(a: &TriMesh, b: &TriMesh) -> f64 {
    one_sided(a, b).max(one_sided(b, a))
}

/// Unique edge lengths.
pub fn edge_lengths(m: &TriMesh) -> Vec<f64> {
    let mut e: Vec<(u32, u32)> = Vec::new();
    for t in &m.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.sort_unstable();
    e.dedup();
    e.iter().map(|&(a, b)| math::dist(m.pos[a as usize], m.pos[b as usize])).collect()
}

pub fn min_angle_deg(m: &TriMesh, f: u32) -> f64 {
    let [a, b, c] = m.tri_pos(f);
    let ang = |p: V3, q: V3, r: V3| math::angle_between(math::sub(q, p), math::sub(r, p));
    ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b)).to_degrees()
}

/// 1-to-4 subdivision.
pub fn subdivide(m: &TriMesh) -> TriMesh {
    use std::collections::BTreeMap;
    let mut pos = m.pos.clone();
    let mut mid: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut tris = Vec::with_capacity(4 * m.tris.len());
    let mut get = |a: u32, b: u32, pos: &mut Vec<V3>| -> u32 {
        let k = (a.min(b), a.max(b));
        *mid.entry(k).or_insert_with(|| {
            pos.push(math::scale(math::add(pos[a as usize], pos[b as usize]), 0.5));
            (pos.len() - 1) as u32
        })
    };
    for t in &m.tris {
        let ab = get(t[0], t[1], &mut pos);
        let bc = get(t[1], t[2], &mut pos);
        let ca = get(t[2], t[0], &mut pos);
        tris.push([t[0], ab, ca]);
        tris.push([ab, t[1], bc]);
        tris.push([ca, bc, t[2]]);
        tris.push([ab, bc, ca]);
    }
    TriMesh { pos, tris }
}

/// Signed volume of a closed mesh.
pub fn volume(m: &TriMesh) -> f64 {
    m.tris
        .iter()
        .map(|t| {
            let (a, b, c) = (m.pos[t[0] as usize], m.pos[t[1] as usize], m.pos[t[2] as usize]);
            math::dot(a, math::cross(b, c)) / 6.0
        })
        .sum()
}

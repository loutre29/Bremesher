//! DESIGN 6.4 tests: chain classification (kinks), feature isolation on random networks.

mod common;

use aq_core::{FeatureKind, NONE, SplitMix64, Topology, TriMesh};
use aq_prep::{
    CornerRule, Proxy, ProxyParams, build_feature_set, classify_chains, count_bridge_edges,
    count_multi_chain_faces, isolate_features, split_components,
};
use aq_testkit::{icosphere, plane_grid};
use common::*;

/// n x n grid of cell `c` whose diagonals all run from (i, j) to (i+1, j+1).
fn ne_grid(n: u32, c: f64) -> TriMesh {
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let mut pos = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            pos.push([i as f64 * c, j as f64 * c, 0.0]);
        }
    }
    let mut tris = Vec::new();
    for j in 0..n {
        for i in 0..n {
            tris.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            tris.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    TriMesh { pos, tris }
}

/// Marks the polyline through grid points `pts` (consecutive points must be grid edges).
fn polyline_proxy(n: u32, c: f64, pts: &[(u32, u32)]) -> Proxy {
    let m = ne_grid(n, c);
    let t = Topology::build(&m).unwrap();
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let mut kinds = vec![FeatureKind::None; 3 * m.tris.len()];
    for w in pts.windows(2) {
        let (a, b) = (id(w[0].0, w[0].1), id(w[1].0, w[1].1));
        let h = t.find_he(&m, a, b).or_else(|| t.find_he(&m, b, a)).unwrap();
        kinds[h as usize] = FeatureKind::Hard;
        let tw = t.twin[h as usize];
        if tw != NONE {
            kinds[tw as usize] = FeatureKind::Hard;
        }
    }
    let zero = |_: u32| 0.0;
    let rule = CornerRule { corner_deg: 50.0, sym_corner_deg: 70.0, window: &zero, fixed: None };
    let fs = build_feature_set(&m, &t, kinds, &rule).unwrap();
    Proxy::from_mesh(m, fs).unwrap()
}

/// (start, direction, steps); only the first segment's start is used.
type Seg = ((u32, u32), (i32, i32), u32);

fn path(segs: &[Seg]) -> Vec<(u32, u32)> {
    let mut out = vec![segs[0].0];
    for &(_, d, k) in segs {
        for _ in 0..k {
            let (x, y) = *out.last().unwrap();
            out.push(((x as i32 + d.0) as u32, (y as i32 + d.1) as u32));
        }
    }
    out
}

/// Number of interior corners of the single Hard chain (excluding its start).
fn kink_corners(px: &Proxy) -> usize {
    let hard: Vec<_> = px.feats.chains.iter().filter(|c| c.kind == FeatureKind::Hard).collect();
    assert_eq!(hard.len(), 1);
    assert!(!hard[0].closed);
    assert_eq!(hard[0].corners[0], 0);
    hard[0].corners.len() - 1
}

#[test]
fn kink_45_is_not_a_corner_90_is() {
    let (n, c) = (60, 0.05);
    let h = vec![1.0; ((n + 1) * (n + 1)) as usize];
    let p = ProxyParams::default();
    // 45 degree kink: east then north-east.
    let mut px = polyline_proxy(n, c, &path(&[((10, 30), (1, 0), 20), ((0, 0), (1, 1), 20)]));
    classify_chains(&mut px, &h, &p);
    px.validate().unwrap();
    assert_eq!(kink_corners(&px), 0, "45 degree kink");
    // 90 degree kink: east then north.
    let mut px = polyline_proxy(n, c, &path(&[((10, 30), (1, 0), 20), ((0, 0), (0, 1), 20)]));
    classify_chains(&mut px, &h, &p);
    assert_eq!(kink_corners(&px), 1, "90 degree kink");
    let ch = px.feats.chains.iter().find(|c| c.kind == FeatureKind::Hard).unwrap();
    assert_eq!(ch.corners, vec![0, 20]);
    // A rounded 90 degree turn (two 45 degree kinks one edge apart): one corner with the
    // 0.5 h window, none with a per-vertex angle.
    let pts = path(&[((10, 20), (1, 0), 20), ((0, 0), (1, 1), 1), ((0, 0), (0, 1), 20)]);
    let mut px = polyline_proxy(n, c, &pts);
    assert_eq!(kink_corners(&px), 0, "per-vertex turning angle");
    classify_chains(&mut px, &h, &p);
    assert_eq!(kink_corners(&px), 1, "windowed turning angle");
    // A U-turn made of two 90 degree kinks 1.0 (two windows) apart: two corners.
    let pts = path(&[((10, 10), (1, 0), 20), ((0, 0), (0, 1), 20), ((0, 0), (-1, 0), 20)]);
    let mut px = polyline_proxy(n, c, &pts);
    classify_chains(&mut px, &h, &p);
    assert_eq!(kink_corners(&px), 2);
}

#[test]
fn kinks_survive_the_proxy() {
    // Same polylines on the source, remeshed at len 0.05 (h = 0.15).
    let (n, c) = (60, 0.05);
    for (dir, want) in [((1, 1), 0usize), ((0, 1), 1)] {
        let src = polyline_proxy(n, c, &path(&[((10, 30), (1, 0), 20), ((0, 0), dir, 20)]));
        let pairs: Vec<[u32; 2]> = src.feats.chains[1..]
            .iter()
            .chain(src.feats.chains.iter())
            .filter(|c| c.kind == FeatureKind::Hard)
            .take(1)
            .flat_map(|c| c.hes.iter().map(|&h| [src.topo.tail(&src.mesh, h), src.topo.head(&src.mesh, h)]))
            .collect();
        let kinds = vec![1u8; pairs.len()];
        let st = ctl();
        let ing = aq_prep::ingest(src.mesh.clone(), &pairs, &kinds, &st.unbounded()).unwrap();
        let comp = &split_components(&ing)[0];
        let mut px = proxy_const(comp, 0.05, &ProxyParams { max_tris: 0, ..ProxyParams::default() }).unwrap();
        let h = vec![0.15; px.mesh.pos.len()];
        classify_chains(&mut px, &h, &ProxyParams::default());
        px.validate().unwrap();
        assert_eq!(kink_corners(&px), want, "kink direction {dir:?}");
    }
}

/// Random walks over the edges of `m`, marked Hard.
fn random_network(m: &TriMesh, walks: usize, steps: usize, seed: u64) -> Proxy {
    let t = Topology::build(m).unwrap();
    let mut rng = SplitMix64::new(seed);
    let mut kinds = vec![FeatureKind::None; 3 * m.tris.len()];
    for _ in 0..walks {
        let mut v = rng.below(m.pos.len() as u64) as u32;
        for _ in 0..steps {
            let out: Vec<u32> = t.fan_out(v).collect();
            let h = out[rng.below(out.len() as u64) as usize];
            if t.twin[h as usize] != NONE {
                kinds[h as usize] = FeatureKind::Hard;
                kinds[t.twin[h as usize] as usize] = FeatureKind::Hard;
            }
            v = t.head(m, h);
        }
    }
    let zero = |_: u32| 0.0;
    let rule = CornerRule { corner_deg: 50.0, sym_corner_deg: 70.0, window: &zero, fixed: None };
    let fs = build_feature_set(m, &t, kinds, &rule).unwrap();
    Proxy::from_mesh(m.clone(), fs).unwrap()
}

#[test]
fn isolation_on_random_chain_networks() {
    let plane = plane_grid(24, 24, 0.1, 0.4, 0.25, 4).unwrap();
    let sphere = icosphere(3, 1.0).unwrap();
    for (mi, m) in [plane, sphere].iter().enumerate() {
        for seed in 0..6u64 {
            let mut px = random_network(m, 12, 15, seed + 100 * mi as u64);
            let bridges0 = count_bridge_edges(&px.mesh, &px.topo, &px.feats);
            let multi0 = count_multi_chain_faces(&px.mesh, &px.feats);
            assert!(bridges0 > 0, "the network should have bridge edges");
            let chain_he0 = px.feats.he_kind.iter().filter(|&&k| k == FeatureKind::Hard).count();
            let corners0 = px.feats.vert_corner.iter().filter(|&&x| x).count();
            let chi0 = px.topo.euler_char(&px.mesh, 0);
            let n = isolate_features(&mut px).unwrap();
            px.validate().unwrap();
            assert_eq!(count_bridge_edges(&px.mesh, &px.topo, &px.feats), 0);
            assert_eq!(count_multi_chain_faces(&px.mesh, &px.feats), 0);
            assert!(n as usize >= bridges0.min(1));
            assert_eq!(px.feats.he_kind.iter().filter(|&&k| k == FeatureKind::Hard).count(), chain_he0);
            assert_eq!(px.feats.vert_corner.iter().filter(|&&x| x).count(), corners0);
            assert_eq!(px.topo.euler_char(&px.mesh, 0), chi0);
            let _ = multi0;
        }
    }
}

#[test]
fn faces_with_three_chain_edges_get_a_centroid() {
    // A single triangle whose 3 edges are Hard (a closed chain around it) inside a grid.
    let m = ne_grid(6, 1.0);
    let t = Topology::build(&m).unwrap();
    let f = 2 * (3 * 6 + 3); // a face near the middle
    let mut kinds = vec![FeatureKind::None; 3 * m.tris.len()];
    for k in 0..3 {
        let h = 3 * f + k;
        kinds[h] = FeatureKind::Hard;
        kinds[t.twin[h] as usize] = FeatureKind::Hard;
    }
    let zero = |_: u32| 0.0;
    let rule = CornerRule { corner_deg: 50.0, sym_corner_deg: 70.0, window: &zero, fixed: None };
    let fs = build_feature_set(&m, &t, kinds, &rule).unwrap();
    let mut px = Proxy::from_mesh(m, fs).unwrap();
    assert!(count_multi_chain_faces(&px.mesh, &px.feats) >= 1);
    let n = isolate_features(&mut px).unwrap();
    assert!(n >= 1);
    px.validate().unwrap();
    assert_eq!(count_multi_chain_faces(&px.mesh, &px.feats), 0);
    assert_eq!(count_bridge_edges(&px.mesh, &px.topo, &px.feats), 0);
    let hard: Vec<_> = px.feats.chains.iter().filter(|c| c.kind == FeatureKind::Hard).collect();
    assert_eq!(hard.len(), 1);
    assert!(hard[0].closed);
    assert_eq!(hard[0].hes.len(), 3);
}

//! DESIGN 6.4 tests: ingest repairs, invalid input, orientation, feature mapping.

mod common;

use aq_core::{ErrKind, FeatureKind, NONE, SplitMix64, Topology, TriMesh, math};
use aq_prep::{ingest, split_components};
use aq_testkit::{hemisphere, icosphere, plane_grid};
use common::*;

fn check_manifold(m: &TriMesh) -> Topology {
    m.validate().unwrap();
    let t = Topology::build(m).unwrap();
    t.validate(m).unwrap();
    t
}

#[test]
fn bowtie_vertex_is_split() {
    // Two triangles touching at vertex 0 only.
    let m = TriMesh {
        pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, -1.0, 0.0]],
        tris: vec![[0, 1, 2], [0, 3, 4]],
    };
    let ing = ingest_plain(m);
    assert_eq!(ing.report.split_verts, 1);
    assert_eq!(ing.mesh.pos.len(), 6);
    check_manifold(&ing.mesh);
    assert_eq!(ing.report.components, 2);
    ing.feats.validate(&ing.mesh, &ing.topo).unwrap();
    // The copy maps back to input vertex 0.
    assert_eq!(ing.vert_src.iter().filter(|&&v| v == 0).count(), 2);
}

#[test]
fn bowtie_between_two_fans() {
    // Two 6-triangle fans (hexagons) sharing their centre vertex.
    let mut pos = vec![[0.0, 0.0, 0.0]];
    let mut tris = Vec::new();
    for (side, z) in [(0u32, 1.0f64), (1, -1.0)] {
        let base = pos.len() as u32;
        for i in 0..6 {
            let a = std::f64::consts::TAU * i as f64 / 6.0;
            pos.push([a.cos(), a.sin(), z * 0.5 * (1.0 + side as f64 * 0.0)]);
        }
        for i in 0..6u32 {
            tris.push([0, base + i, base + (i + 1) % 6]);
        }
    }
    let ing = ingest_plain(TriMesh { pos, tris });
    assert_eq!(ing.report.split_verts, 1);
    let t = check_manifold(&ing.mesh);
    assert_eq!(t.n_comp, 2);
    assert_eq!(t.boundary_loops.len(), 2);
}

#[test]
fn edge_with_three_faces() {
    let mut m = plane_grid(4, 4, 1.0, 0.0, 0.0, 1).unwrap();
    let t0 = Topology::build(&m).unwrap();
    let loops0 = t0.boundary_loops.len();
    // A fin on the interior edge (6, 12) (vertex id = j * 5 + i): find an interior edge.
    let (a, b) = {
        let mut found = None;
        for h in 0..t0.twin.len() as u32 {
            if t0.twin[h as usize] != NONE {
                let (a, b) = (t0.tail(&m, h), t0.head(&m, h));
                if !t0.is_bnd_vert[a as usize] && !t0.is_bnd_vert[b as usize] {
                    found = Some((a, b));
                    break;
                }
            }
        }
        found.unwrap()
    };
    let mid = math::scale(math::add(m.pos[a as usize], m.pos[b as usize]), 0.5);
    m.pos.push([mid[0], mid[1], 1.0]);
    let apex = (m.pos.len() - 1) as u32;
    m.tris.push([a, b, apex]);
    let ing = ingest_plain(m);
    let t = check_manifold(&ing.mesh);
    assert!(ing.report.split_edges >= 1);
    assert_eq!(t.boundary_loops.len(), loops0 + 1, "one extra boundary loop");
    assert_eq!(ing.mesh.tris.len(), 33);
}

#[test]
fn sliver_in_closed_mesh_is_collapsed() {
    let mut m = icosphere(3, 1.0).unwrap();
    let nf0 = m.tris.len();
    let t0 = Topology::build(&m).unwrap();
    // Replace the face (b, a, d) across edge a->b of face 0 by a sliver (b, a, m) and two faces.
    let h = 0u32;
    let tw = t0.twin[h as usize];
    let (a, b) = (t0.tail(&m, h), t0.head(&m, h));
    let g = (tw / 3) as usize;
    let d = t0.tail(&m, Topology::prev(tw));
    let pa = m.pos[a as usize];
    let pb = m.pos[b as usize];
    let len = math::dist(pa, pb);
    // Midpoint pushed off the segment so that the sliver area is about 1e-16.
    let e = math::scale(math::sub(pb, pa), 1.0 / len);
    let n = math::normalize(math::cross(e, [0.3, 0.5, 0.7])).unwrap();
    let off = 2.0 * 1e-16 / len;
    let pm = math::add(math::scale(math::add(pa, pb), 0.5), math::scale(n, off));
    m.pos.push(pm);
    let mv = (m.pos.len() - 1) as u32;
    m.tris[g] = [b, a, mv];
    m.tris.push([mv, a, d]);
    m.tris.push([b, mv, d]);
    let area = {
        let c = math::cross(math::sub(pa, pb), math::sub(pm, pb));
        0.5 * math::norm(c)
    };
    assert!(area < 1e-15 && area > 0.0, "sliver area {area}");
    let ing = ingest_plain(m);
    let t = check_manifold(&ing.mesh);
    assert!(ing.report.collapsed_faces >= 1);
    assert_eq!(ing.report.dropped_faces, 0);
    assert_eq!(t.euler_char(&ing.mesh, 0), 2);
    assert!(t.boundary_loops.is_empty(), "no holes");
    assert_eq!(ing.mesh.tris.len(), nf0);
}

#[test]
fn invalid_input() {
    let st = ctl();
    let c = st.unbounded();
    let good = icosphere(1, 1.0).unwrap();
    let mut nan = good.clone();
    nan.pos[3][1] = f64::NAN;
    assert_eq!(ingest(nan, &[], &[], &c).unwrap_err().kind, ErrKind::InvalidInput);
    let mut inf = good.clone();
    inf.pos[0][0] = f64::INFINITY;
    assert_eq!(ingest(inf, &[], &[], &c).unwrap_err().kind, ErrKind::InvalidInput);
    let mut oor = good.clone();
    oor.tris[2][1] = 9999;
    assert_eq!(ingest(oor, &[], &[], &c).unwrap_err().kind, ErrKind::InvalidInput);
    let empty = TriMesh { pos: good.pos.clone(), tris: Vec::new() };
    assert_eq!(ingest(empty, &[], &[], &c).unwrap_err().kind, ErrKind::InvalidInput);
    assert_eq!(ingest(TriMesh::default(), &[], &[], &c).unwrap_err().kind, ErrKind::InvalidInput);
    let e = ingest(good.clone(), &[[0, 99999]], &[1], &c).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    assert_eq!(ingest(good.clone(), &[[0, 1]], &[], &c).unwrap_err().kind, ErrKind::InvalidInput);
    assert_eq!(ingest(good, &[[0, 1]], &[9], &c).unwrap_err().kind, ErrKind::InvalidInput);
}

#[test]
fn random_flips_are_reoriented() {
    let orig = icosphere(3, 1.0).unwrap();
    let mut rng = SplitMix64::new(7);
    for frac in [0.3, 1.0] {
        let mut m = orig.clone();
        let mut flipped = 0;
        for t in m.tris.iter_mut() {
            if rng.next_f64() < frac {
                *t = [t[0], t[2], t[1]];
                flipped += 1;
            }
        }
        let ing = ingest_plain(m);
        check_manifold(&ing.mesh);
        assert_eq!(ing.report.flipped_faces, flipped);
        assert!(volume(&ing.mesh) > 0.0);
        // Every face has the original orientation (same cyclic order).
        for (f, t) in ing.mesh.tris.iter().enumerate() {
            let o = orig.tris[ing.face_src[f] as usize];
            let t = t.map(|v| ing.vert_src[v as usize]);
            let rot = |x: [u32; 3], k: usize| [x[k], x[(k + 1) % 3], x[(k + 2) % 3]];
            assert!((0..3).any(|k| rot(t, k) == o), "face {f} not re-oriented");
        }
    }
}

#[test]
fn feature_pairs_are_routed_or_dropped() {
    // Vertex id of grid point (i, j) is j * 11 + i.
    let m = plane_grid(10, 10, 1.0, 0.0, 0.0, 3).unwrap();
    let id = |i: u32, j: u32| j * 11 + i;
    let st = ctl();
    // Distance 2 rings along a row: rerouted along 2 edges.
    let ing = ingest(m.clone(), &[[id(3, 5), id(5, 5)]], &[1], &st.unbounded()).unwrap();
    assert_eq!(ing.report.rerouted_features, 1);
    assert_eq!(ing.report.dropped_features, 0);
    ing.feats.validate(&ing.mesh, &ing.topo).unwrap();
    let hard: Vec<_> = ing.feats.chains.iter().filter(|c| c.kind == FeatureKind::Hard).collect();
    assert_eq!(hard.len(), 1);
    assert_eq!(hard[0].hes.len(), 2);
    // Distance 5 rings: dropped and counted.
    let ing = ingest(m.clone(), &[[id(2, 5), id(7, 5)]], &[1], &st.unbounded()).unwrap();
    assert_eq!(ing.report.dropped_features, 1);
    assert_eq!(ing.report.rerouted_features, 0);
    assert!(ing.feats.chains.iter().all(|c| c.kind == FeatureKind::Boundary));
    // An edge pair marks both halves.
    let ing = ingest(m, &[[id(3, 5), id(4, 5)]], &[1], &st.unbounded()).unwrap();
    let n_hard = ing.feats.he_kind.iter().filter(|&&k| k == FeatureKind::Hard).count();
    assert_eq!(n_hard, 2);
}

#[test]
fn boundary_symmetry_wins() {
    let s = hemisphere(8, 1.0).unwrap();
    let st = ctl();
    let ing = ingest(s.mesh.clone(), &s.feat_pairs, &s.feat_kind, &st.unbounded()).unwrap();
    ing.feats.validate(&ing.mesh, &ing.topo).unwrap();
    assert_eq!(ing.feats.chains.len(), 1);
    let c = &ing.feats.chains[0];
    assert_eq!(c.kind, FeatureKind::Symmetry);
    assert!(c.closed);
    assert!(c.corners.is_empty());
    // A plain boundary gets Boundary.
    let ing = ingest_plain(plane_grid(3, 3, 1.0, 0.0, 0.0, 0).unwrap());
    assert_eq!(ing.feats.chains.len(), 1);
    assert_eq!(ing.feats.chains[0].kind, FeatureKind::Boundary);
    assert_eq!(ing.feats.chains[0].corners.len(), 4);
}

#[test]
fn duplicate_faces_and_repeated_indices() {
    let mut m = icosphere(2, 1.0).unwrap();
    let nf = m.tris.len();
    let t = m.tris[5];
    m.tris.push(t);
    m.tris.push([t[0], t[2], t[1]]);
    m.tris.push([1, 1, 2]);
    let ing = ingest_plain(m);
    check_manifold(&ing.mesh);
    assert_eq!(ing.mesh.tris.len(), nf);
    assert_eq!(ing.report.dropped_faces, 2);
    assert_eq!(ing.report.collapsed_faces, 1);
}

#[test]
fn components_are_split() {
    let a = icosphere(2, 1.0).unwrap();
    let mut b = plane_grid(3, 3, 1.0, 0.0, 0.0, 0).unwrap();
    for p in b.pos.iter_mut() {
        p[2] += 5.0;
    }
    let m = aq_testkit::merge(&[a.clone(), b.clone()]);
    let ing = ingest_plain(m);
    assert_eq!(ing.report.components, 2);
    let comps = split_components(&ing);
    assert_eq!(comps.len(), 2);
    assert_eq!(comps[0].chi, 2);
    assert!(comps[0].closed);
    assert_eq!(comps[1].chi, 1);
    assert!(!comps[1].closed);
    assert_eq!(comps[0].mesh.tris.len(), a.tris.len());
    for c in &comps {
        c.topo.validate(&c.mesh).unwrap();
        c.feats.validate(&c.mesh, &c.topo).unwrap();
        assert!((c.area - c.mesh.total_area()).abs() < 1e-12);
    }
    // Source maps compose to the input.
    assert_eq!(comps[1].face_src[0] as usize, a.tris.len());
    assert!(comps[1].vert_src.iter().all(|&v| v as usize >= a.pos.len()));
}

#[test]
fn ingest_performance_300k() {
    if cfg!(debug_assertions) {
        return;
    }
    let base = aq_testkit::load_testdata("noisy_blob").unwrap();
    // noisy_blob x4 (one 1-to-4 subdivision, 328k triangles) is the DESIGN perf input.
    let m4 = subdivide(&base);
    assert!(m4.tris.len() >= 300_000);
    let st = ctl();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let ing = ingest(m4.clone(), &[], &[], &st.unbounded()).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        assert_eq!(ing.mesh.tris.len(), m4.tris.len());
    }
    println!("ingest 328k: {best:.3} s");
    assert!(best < 0.5, "ingest took {best} s");
}

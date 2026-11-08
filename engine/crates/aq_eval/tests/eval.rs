//! DESIGN 6.15 tests of aq_eval.

mod common;

use std::path::PathBuf;

use aq_core::math;
use aq_core::{QuadMesh, Topology, TriMesh};
use aq_eval::{bisect, fan_triangulate, hausdorff, loop_closure, mirror_weld, poly_from_tris, quad_stats, vertex_distance};
use common::{cube_sphere, cylinder_grid, helix_grid, torus_grid};

fn engine_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

// ---------------------------------------------------------------- quad_stats parity

fn check_fixture(name: &str) {
    let fx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(format!("stats_{name}.json"));
    let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&fx_path).unwrap()).unwrap();
    let q = aq_io::read_obj_poly(&engine_dir().join("baselines").join(format!("{name}.obj"))).unwrap();
    let s = quad_stats(&q);
    let raw = &fx["raw"];
    let tol = 1e-6;
    assert_eq!(s.faces as u64, raw["faces"].as_u64().unwrap(), "{name} faces");
    assert_eq!(s.irregular as u64, raw["irregular_verts"].as_u64().unwrap(), "{name} irregular");
    assert_eq!(s.nonmanifold_edges as u64, raw["nonmanifold_edges"].as_u64().unwrap(), "{name} nonmanifold");
    assert_eq!(s.boundary_edges as u64, raw["boundary_edges"].as_u64().unwrap(), "{name} boundary");
    for (key, val) in [("quad_ratio", s.quad_ratio), ("edge_len_cv", s.edge_cv), ("mean_angle_dev_deg", s.angle_dev_deg)] {
        let want = raw[key].as_f64().unwrap();
        assert!((val - want).abs() <= tol, "{name} {key}: rust {val} vs script {want}");
    }
    let sizes: Vec<(String, u64)> = s.face_sizes.iter().map(|(k, v)| (k.to_string(), *v as u64)).collect();
    let want_sizes: Vec<(String, u64)> =
        raw["face_sizes"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    assert_eq!(sizes, want_sizes, "{name} face_sizes");
    let val: Vec<(String, u64)> = s.valence.iter().map(|(k, v)| (k.clone(), *v as u64)).collect();
    let want_val: Vec<(String, u64)> =
        raw["valence"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    assert_eq!(val, want_val, "{name} valence");
    // The rounded dict is exactly what the script prints.
    assert_eq!(s.to_py_json(), fx["py"], "{name} rounded output");
}

#[test]
fn stats_parity_torus_qf2000() {
    check_fixture("torus_qf2000");
}

#[test]
fn stats_parity_retopo_a_qf2000() {
    check_fixture("retopo_a_qf2000");
}

#[test]
fn stats_parity_all_baselines() {
    for name in [
        "basemesh_female_qf5000",
        "cube_sharp_qf2000",
        "cylinder_open_qf2000",
        "double_torus_qf2000",
        "noisy_blob_qf5000",
        "retopo_b_qf2000",
        "retopo_c_qf2000",
        "sculpt_cylinder_qf2000",
        "suzanne_sub2_qf2000",
    ] {
        check_fixture(name);
    }
}

#[test]
fn stats_small_cases() {
    // One square: 4 boundary vertices of valence 2 (regular), 4 boundary edges, 0 angle dev.
    let mut q = QuadMesh::new();
    q.pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    q.vflag = vec![0; 4];
    q.push_face(&[0, 1, 2, 3], 0);
    let s = quad_stats(&q);
    assert_eq!(s.faces, 1);
    assert_eq!(s.quad_ratio, 1.0);
    assert_eq!(s.irregular, 0);
    assert_eq!(s.valence.get("b2"), Some(&4));
    assert_eq!(s.boundary_edges, 4);
    assert_eq!(s.edge_cv, 0.0);
    assert!(s.angle_dev_deg.abs() < 1e-12);

    // Cube sphere: 8 valence-3 poles, closed, all quads.
    let cs = cube_sphere(6, 1.0);
    cs.validate().unwrap();
    let s = quad_stats(&cs);
    assert_eq!(s.faces, 6 * 36);
    assert_eq!(s.irregular, 8);
    assert_eq!(s.valence.get("3"), Some(&8));
    assert_eq!(s.boundary_edges, 0);
    assert_eq!(s.nonmanifold_edges, 0);

    // A mixed mesh: quad ratio and face sizes.
    let mut m = QuadMesh::new();
    m.pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.5, 0.0]];
    m.vflag = vec![0; 5];
    m.push_face(&[0, 1, 2, 3], 0);
    m.push_face(&[1, 4, 2], 0);
    let s = quad_stats(&m);
    assert_eq!(s.quad_ratio, 0.5);
    assert_eq!(s.face_sizes.get(&3), Some(&1));
    assert_eq!(s.face_sizes.get(&4), Some(&1));
    assert_eq!(s.boundary_edges, 5);
    // Empty mesh does not panic.
    let e = quad_stats(&QuadMesh::new());
    assert_eq!(e.faces, 0);
}

// ---------------------------------------------------------------- Hausdorff

#[test]
fn hausdorff_of_a_mesh_with_itself_is_zero() {
    let q = cube_sphere(8, 1.0);
    let input = fan_triangulate(&q);
    let h = hausdorff(&q, &input, 16);
    assert!(h.q_to_in_max < 1e-12 && h.q_to_in_mean < 1e-12, "{h:?}");
    assert!(h.in_to_q_max < 1e-12 && h.in_to_q_mean < 1e-12, "{h:?}");
    assert!(h.symmetric() < 1e-12);
    let (vm, vmean) = vertex_distance(&q, &input);
    assert!(vm < 1e-12 && vmean < 1e-12);

    // A triangle mesh read from testdata against itself.
    let t = aq_io::read_obj_tri(&engine_dir().join("testdata").join("torus.obj")).unwrap();
    let h = hausdorff(&poly_from_tris(&t), &t, 4);
    assert!(h.symmetric() < 1e-12, "{h:?}");
}

#[test]
fn hausdorff_concentric_spheres_and_determinism() {
    let q = cube_sphere(8, 1.1);
    let input = fan_triangulate(&cube_sphere(16, 1.0));
    let h = hausdorff(&q, &input, 8);
    // Outer vertices lie at 1.1, the inner surface at most chord error below 1.
    assert!(h.q_to_in_max > 0.09 && h.q_to_in_max < 0.13, "{h:?}");
    assert!(h.in_to_q_max > 0.05 && h.in_to_q_max < 0.13, "{h:?}");
    assert!(h.q_to_in_mean <= h.q_to_in_max && h.in_to_q_mean <= h.in_to_q_max);
    let s = h.scaled(2.0);
    assert_eq!(s.q_to_in_max, 2.0 * h.q_to_in_max);

    // Bit-identical results for any thread count.
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| hausdorff(&q, &input, 8))
    };
    let a = run(1);
    let b = run(4);
    assert_eq!(a.q_to_in_max.to_bits(), b.q_to_in_max.to_bits());
    assert_eq!(a.q_to_in_mean.to_bits(), b.q_to_in_mean.to_bits());
    assert_eq!(a.in_to_q_max.to_bits(), b.in_to_q_max.to_bits());
    assert_eq!(a.in_to_q_mean.to_bits(), b.in_to_q_mean.to_bits());

    // Empty sides.
    let e = hausdorff(&QuadMesh::new(), &input, 4);
    assert_eq!(e.q_to_in_max, 0.0);
    assert!(e.in_to_q_max.is_infinite());
}

// ---------------------------------------------------------------- loop closure

#[test]
fn perfect_torus_grid_is_fully_closed() {
    let q = torus_grid(48, 16, 3.0, 1.0);
    q.validate().unwrap();
    let r = loop_closure(&q, None);
    assert_eq!(r.loops, 48 + 16);
    assert_eq!(r.closed, 48 + 16);
    assert_eq!(r.closed_frac, 1.0);
    assert_eq!(r.helix_suspects, 0);
    // About z only the 16 rings around the axis count, all closed, no helix.
    let r = loop_closure(&q, Some([0.0, 0.0, 1.0]));
    assert_eq!(r.loops, 16);
    assert_eq!(r.closed_frac, 1.0);
    assert_eq!(r.helix_suspects, 0);
}

#[test]
fn open_cylinder_rings_close_and_axial_lines_do_not() {
    let (n, rows) = (24, 10);
    let q = cylinder_grid(n, rows, 1.0);
    q.validate().unwrap();
    let r = loop_closure(&q, Some([0.0, 0.0, 1.0]));
    assert_eq!(r.loops, rows + 1, "{r:?}");
    assert_eq!(r.closed_frac, 1.0);
    assert_eq!(r.helix_suspects, 0);
    let r = loop_closure(&q, None);
    assert_eq!(r.loops, rows + 1 + n);
    assert_eq!(r.closed, rows + 1);
}

#[test]
fn helix_is_detected() {
    let q = helix_grid(24, 6, 1.0);
    q.validate().unwrap();
    let r = loop_closure(&q, Some([0.0, 0.0, 1.0]));
    assert!(r.helix_suspects >= 1, "{r:?}");
    assert!(r.closed_frac < 1.0, "{r:?}");
}

#[test]
fn cube_sphere_lines_stop_at_poles() {
    let q = cube_sphere(6, 1.0);
    let r = loop_closure(&q, None);
    // Grid lines off the cube edges wrap around 4 faces (3 families x 5 lines); the 12 cube
    // edges are open lines between poles.
    assert_eq!(r.closed, 15, "{r:?}");
    assert_eq!(r.loops, 15 + 12, "{r:?}");
}

// ---------------------------------------------------------------- bisect and mirror weld

fn check_closed_quadmesh(q: &QuadMesh) {
    q.validate().unwrap();
    let s = quad_stats(q);
    assert_eq!(s.boundary_edges, 0);
    assert_eq!(s.nonmanifold_edges, 0);
}

#[test]
fn bisect_then_mirror_weld_quad_sphere_has_no_open_edges() {
    let qs = cube_sphere(8, 1.0);
    let tri = fan_triangulate(&qs);
    let plane = [1.0, 0.0, 0.0, 0.0];
    let (half, cut) = bisect(&tri, plane, true);
    half.validate().unwrap();
    let topo = Topology::build(&half).unwrap();
    assert_eq!(topo.boundary_loops.len(), 1);
    assert_eq!(topo.boundary_loops[0].len(), cut.len());
    assert_eq!(cut.len(), 4 * 8);
    for e in &cut {
        for &v in e {
            assert_eq!(half.pos[v as usize][0], 0.0);
        }
    }
    assert!(half.pos.iter().all(|p| p[0] >= 0.0));
    assert_eq!(half.tris.len(), tri.tris.len() / 2);

    let (welded, rep) = mirror_weld(&poly_from_tris(&half), plane, 1e-9);
    assert_eq!(rep.open_edges, 0, "{rep:?}");
    assert_eq!(rep.seam_open_edges, 0);
    assert_eq!(rep.nonmanifold_edges, 0);
    assert_eq!(rep.seam_verts, cut.len() as u32);
    check_closed_quadmesh(&welded);
    assert_eq!(welded.n_faces(), tri.tris.len());
    let t = Topology::build(&fan_triangulate(&welded)).unwrap();
    assert_eq!(t.euler_char(&fan_triangulate(&welded), 0), 2);

    // Negative side, same counts.
    let (neg, cut_n) = bisect(&tri, plane, false);
    assert_eq!(neg.tris.len(), half.tris.len());
    assert_eq!(cut_n.len(), cut.len());
    assert!(neg.pos.iter().all(|p| p[0] <= 0.0));
}

#[test]
fn bisect_off_grid_clips_and_welds() {
    let tri = fan_triangulate(&cube_sphere(10, 1.0));
    let c = 0.1234;
    let plane = [2.0, 0.0, 0.0, -2.0 * c]; // x = c, unnormalized on purpose
    let (half, cut) = bisect(&tri, plane, true);
    half.validate().unwrap();
    let topo = Topology::build(&half).unwrap();
    assert_eq!(topo.boundary_loops.len(), 1);
    assert_eq!(topo.boundary_loops[0].len(), cut.len());
    assert!(half.pos.iter().all(|p| p[0] >= c - 1e-12));
    for e in &cut {
        for &v in e {
            assert!((half.pos[v as usize][0] - c).abs() < 1e-12);
        }
    }
    // Every output triangle is CCW-consistent with the input orientation (outward normals).
    for f in 0..half.tris.len() as u32 {
        let [a, b, cc] = half.tri_pos(f);
        let n = half.face_cross(f);
        let ctr = math::scale(math::add(math::add(a, b), cc), 1.0 / 3.0);
        assert!(math::dot(n, ctr) > 0.0);
    }
    // Area: the spherical cap x >= c of the faceted sphere.
    let (_, rep) = mirror_weld(&poly_from_tris(&half), plane, 1e-9);
    assert_eq!(rep.open_edges, 0, "{rep:?}");
    assert_eq!(rep.nonmanifold_edges, 0);
}

#[test]
fn mirror_weld_of_a_quad_half_sphere_is_clean() {
    let qs = cube_sphere(8, 1.0);
    // Half by selection: quads with every vertex at x >= 0.
    let mut half = QuadMesh::new();
    half.pos = qs.pos.clone();
    half.vflag = vec![0; qs.pos.len()];
    for f in 0..qs.n_faces() {
        let fv = qs.face(f);
        if fv.iter().all(|&v| qs.pos[v as usize][0] >= 0.0) {
            half.push_face(fv, 0);
        }
    }
    let (welded, rep) = mirror_weld(&half, [1.0, 0.0, 0.0, 0.0], 1e-9);
    assert_eq!(rep.open_edges, 0, "{rep:?}");
    assert_eq!(rep.seam_poles, 0, "{rep:?}");
    check_closed_quadmesh(&welded);
    assert!(welded.is_all_quads());
    let s = quad_stats(&welded);
    assert_eq!(s.faces, 6 * 64);
    assert_eq!(s.irregular, 8);
    // Same vertices as the full sphere (the mirrored quads start at another corner, so their
    // fan diagonals differ and the surfaces only agree up to the quad bending).
    let full = fan_triangulate(&qs);
    let (vmax, _) = vertex_distance(&welded, &full);
    assert!(vmax < 1e-12, "{vmax}");
    let h = hausdorff(&welded, &full, 4);
    assert!(h.symmetric() < 1e-2, "{h:?}");

    // Dropping a face of the half leaves a hole in both copies: open edges are reported.
    let mut broken = half.clone();
    broken.face_off.pop();
    broken.face_verts.truncate(*broken.face_off.last().unwrap() as usize);
    broken.face_comp.pop();
    let (_, rep) = mirror_weld(&broken, [1.0, 0.0, 0.0, 0.0], 1e-9);
    assert!(rep.open_edges > 0);
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let empty = TriMesh::default();
    let (m, c) = bisect(&empty, [1.0, 0.0, 0.0, 0.0], true);
    assert!(m.tris.is_empty() && c.is_empty());
    let bad = TriMesh { pos: vec![[0.0; 3], [1.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]], tris: vec![[0, 1, 2], [0, 1, 7]] };
    let _ = bisect(&bad, [1.0, 0.0, 0.0, -0.5], true);
    let _ = bisect(&bad, [0.0, 0.0, 0.0, 1.0], true);
    let mut q = QuadMesh::new();
    q.pos = vec![[0.0; 3]; 3];
    q.vflag = vec![0; 3];
    q.push_face(&[0, 1, 9, 2], 0);
    let _ = quad_stats(&q);
    let _ = loop_closure(&q, Some([0.0, 0.0, 1.0]));
    let _ = loop_closure(&q, Some([0.0, 0.0, 0.0]));
    let _ = mirror_weld(&q, [1.0, 0.0, 0.0, 0.0], 1e-9);
    let _ = hausdorff(&q, &bad, 3);
}

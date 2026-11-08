//! DESIGN 6.15 tests of aq_io: OBJ reading and writing, debug dumps.

use std::path::PathBuf;

use aq_core::{
    ArcKind, Chain, CutArc, CutGraph, ErrKind, FeatureKind, FeatureSet, Frames, Layout, QComb, QuadMesh,
    SeamlessParam, SplitMix64, TArc, TMesh, TNode, TPatch, Topology, TriMesh,
};
use aq_io::*;

fn engine_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("aq_io_tests").join(name)
}

/// n x n planar grid of 2n^2 CCW triangles in z = 0.
fn grid(n: u32) -> TriMesh {
    let mut pos = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            pos.push([i as f64, j as f64, 0.0]);
        }
    }
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let mut tris = Vec::new();
    for j in 0..n {
        for i in 0..n {
            tris.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            tris.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    TriMesh { pos, tris }
}

#[test]
fn parse_statements_suffixes_and_negative_indices() {
    let text = "# comment\n\
        o thing\n\
        v 0 0 0\n\
        v 1 0 0 1.0\n\
        v 1 1 0 0.5 0.5 0.5\n\
        v 0 1 0\n\
        vt 0 0\n\
        vn 0 0 1\n\
        g group\n\
        s off\n\
        f 1/1/1 2/2/1 3//1 4\n\
        l 1 2\n\
        f -4 -3 -2\n\
        f 1 2\n\
        \n";
    let d = parse_obj(text).unwrap();
    assert_eq!(d.pos.len(), 4);
    assert_eq!(d.pos[1], [1.0, 0.0, 0.0]);
    assert_eq!(d.n_faces(), 2);
    assert_eq!(d.face(0), &[0, 1, 2, 3]);
    assert_eq!(d.face(1), &[0, 1, 2]);
    assert_eq!(d.skipped_faces, 1);
    let t = d.to_tri_mesh();
    assert_eq!(t.tris, vec![[0, 1, 2], [0, 2, 3], [0, 1, 2]]);
    let q = d.to_quad_mesh();
    q.validate().unwrap_err(); // the triangle repeats a directed edge of the quad
    assert_eq!(q.n_faces(), 2);
    assert_eq!(q.vflag.len(), 4);
    assert_eq!(q.face_comp, vec![0, 0]);
}

#[test]
fn parse_errors_are_invalid_input() {
    for bad in ["v 0 0\n", "v 0 0 x\n", "v 0 0 0\nf 1 2 3\n", "v 0 0 0\nf 0 1 1\n", "v 0 0 0\nf -2 1 1\n", "f a b c\n"] {
        let e = parse_obj(bad).unwrap_err();
        assert_eq!(e.kind, ErrKind::InvalidInput, "{bad:?}: {e}");
    }
    let e = read_obj(&tmp("does_not_exist.obj")).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    // Empty input is a valid, empty mesh.
    let d = parse_obj("").unwrap();
    assert_eq!(d.n_faces(), 0);
    assert!(d.to_quad_mesh().validate().is_ok());
}

#[test]
fn write_read_round_trip_is_bit_exact() {
    let mut rng = SplitMix64::new(7);
    let mut m = grid(6);
    for p in &mut m.pos {
        for c in p.iter_mut() {
            *c += (rng.next_f64() - 0.5) * 1e-3 + 1e-300;
        }
    }
    m.pos[0] = [1.0 / 3.0, -2.5e-17, 123456789.12345679];
    let path = tmp("round_trip_tri.obj");
    write_obj_tri(&path, &m).unwrap();
    let back = read_obj_tri(&path).unwrap();
    assert_eq!(back.tris, m.tris);
    for (a, b) in back.pos.iter().zip(&m.pos) {
        for k in 0..3 {
            assert_eq!(a[k].to_bits(), b[k].to_bits());
        }
    }

    let mut q = QuadMesh::new();
    q.pos = m.pos.clone();
    q.vflag = vec![0; q.pos.len()];
    q.push_face(&[0, 1, 8, 7], 0);
    q.push_face(&[1, 2, 9, 8], 0);
    q.push_face(&[2, 3, 10], 0);
    let path = tmp("round_trip_quad.obj");
    write_obj_quad(&path, &q).unwrap();
    let back = read_obj_poly(&path).unwrap();
    assert_eq!(back.face_off, q.face_off);
    assert_eq!(back.face_verts, q.face_verts);
    assert_eq!(back.pos, q.pos);
}

#[test]
fn reads_every_testdata_and_baseline_mesh() {
    let mut n = 0;
    for dir in ["testdata", "baselines"] {
        for e in std::fs::read_dir(engine_dir().join(dir)).unwrap() {
            let p = e.unwrap().path();
            if p.extension().and_then(|x| x.to_str()) != Some("obj") {
                continue;
            }
            let d = read_obj(&p).unwrap();
            assert!(d.n_faces() > 0, "{}", p.display());
            let t = d.to_tri_mesh();
            t.validate().unwrap();
            if dir == "testdata" {
                assert!(d.face_off.windows(2).all(|w| w[1] - w[0] == 3), "{} is not triangulated", p.display());
            } else {
                let q = d.to_quad_mesh();
                assert!(q.is_all_quads(), "{}", p.display());
                q.validate().unwrap();
            }
            n += 1;
        }
    }
    assert_eq!(n, 12 + 11);
}

fn count_prefix(text: &str, prefix: &str) -> usize {
    text.lines().filter(|l| l.starts_with(prefix)).count()
}

#[test]
fn dumps_are_valid_obj_and_json() {
    let m = grid(4);
    let t = Topology::build(&m).unwrap();
    let fr = Frames::new(&m, &t).unwrap();
    let nf = m.tris.len();
    let nh = 3 * nf;
    let dd = DumpDir::new(tmp("dumps"));

    // proxy.obj with one boundary chain.
    let mut fs = FeatureSet::new(&m, &t);
    let bl = t.boundary_loops[0].clone();
    for &h in &bl {
        fs.he_chain[h as usize] = 0;
    }
    fs.chains.push(Chain { hes: bl.clone(), kind: FeatureKind::Boundary, closed: true, corners: vec![] });
    let p = dd.proxy(&m, Some(&fs)).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(count_prefix(&text, "f "), nf);
    assert!(text.contains("g chain_0_Boundary"));
    assert_eq!(count_prefix(&text, "l "), 1);
    assert_eq!(read_obj_tri(&p).unwrap().tris, m.tris);

    // field.obj: two segments per face.
    let theta = vec![0.3; nf];
    let p = dd.field(&m, &fr, &theta, 0.0).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(count_prefix(&text, "l "), 2 * nf);
    assert_eq!(count_prefix(&text, "v "), 4 * nf);
    assert!(text.contains("g u") && text.contains("g v"));
    parse_obj(&text).unwrap();

    // cones.obj: grouped points.
    let p = dd.cones(&m.pos, &[(3, 1), (7, -1), (9, 1), (99999, 1)]).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(count_prefix(&text, "p "), 3);
    assert!(text.contains("g index4_+1") && text.contains("g index4_-1"));
    parse_obj(&text).unwrap();

    // cut.obj.
    let cut = CutGraph {
        is_cut: vec![false; nh],
        he_arc: vec![(aq_core::NONE, 0); nh],
        arcs: vec![CutArc { hes: vec![bl[0], bl[1]], nodes: [0, 1], r: 1 }],
        node_verts: vec![0, 2],
        theta_combed: theta.clone(),
        r: vec![0; nh],
        comb_offset: vec![0; nf],
    };
    let p = dd.cut(&m, &cut).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains("g arc_0_r1"));
    assert_eq!(count_prefix(&text, "l "), 1);
    assert_eq!(count_prefix(&text, "p "), 2);
    parse_obj(&text).unwrap();

    // param.obj: one vt per corner, uv = xy.
    let uv: Vec<[[f64; 2]; 3]> = m.tris.iter().map(|t| t.map(|v| [m.pos[v as usize][0], m.pos[v as usize][1]])).collect();
    let mut folded = vec![false; nf];
    folded[5] = true;
    let param = SeamlessParam { uv, folded, ..Default::default() };
    let p = dd.param(&m, &param).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(count_prefix(&text, "vt "), 3 * nf);
    assert!(text.contains("g folded"));
    let d = parse_obj(&text).unwrap();
    assert_eq!(d.n_faces(), nf);

    // layout.obj and quant.json on a one-patch T-mesh.
    let tm = TMesh {
        nodes: vec![TNode::default(); 4],
        arcs: (0..4)
            .map(|a| TArc { ends: [a, (a + 1) % 4], len: 4.0, kind: ArcKind::Boundary, lb: 1, weight: 1.0, wrap: false, chain: 0 })
            .collect(),
        patches: vec![TPatch { sides: [vec![(0, false)], vec![(1, false)], vec![(2, false)], vec![(3, false)]] }],
        rings: vec![],
        chi: 1,
    };
    let corners = [0u32, 4, 24, 20];
    let side = |a: u32, b: u32, step: i32| -> Vec<u32> {
        let mut v = vec![a];
        let mut x = a as i32;
        while x != b as i32 {
            x += step;
            v.push(x as u32);
        }
        v
    };
    let arc_verts = vec![side(0, 4, 1), side(4, 24, 5), side(24, 20, -1), side(20, 0, -5)];
    let layout = Layout {
        wmesh: m.clone(),
        wtopo: t.clone(),
        parent_face: (0..nf as u32).collect(),
        tmesh: tm.clone(),
        node_vert: corners.to_vec(),
        arc_s: arc_verts.iter().map(|v| (0..v.len()).map(|i| i as f64).collect()).collect(),
        arc_verts,
        patch_faces: vec![(0..nf as u32 - 2).collect()],
        patch_chart_face: vec![0],
        vert_chain: vec![aq_core::NONE; m.pos.len()],
        ..Default::default()
    };
    let p = dd.layout(&layout).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(count_prefix(&text, "l "), 4);
    assert!(text.contains("g patch_0") && text.contains("g unassigned") && text.contains("g arc_3_Boundary_n4.000"));
    assert_eq!(parse_obj(&text).unwrap().n_faces(), nf);

    let qc = QComb { patch_dims: vec![[4, 4]], quads: vec![[0, 1, 2, 3]; 16], n_verts: 25, ..Default::default() };
    let p = dd.quant(&tm, Some(&[4, 4, 4, 3]), Some(&qc)).unwrap();
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    assert_eq!(j["arcs"].as_array().unwrap().len(), 4);
    assert_eq!(j["arcs"][3]["n"], 3);
    assert_eq!(j["patches"][0]["dims"], serde_json::json!([4, 4]));
    assert_eq!(j["n_quads"], 16);
    assert_eq!(j["sum_abs_dev"], 1.0);
    let p = dd.quant(&tm, None, None).unwrap();
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    assert!(j["arcs"][0]["n"].is_null());

    // patches.obj.
    let mut q = QuadMesh::new();
    q.pos = m.pos.clone();
    q.vflag = vec![0; q.pos.len()];
    q.push_face(&[0, 1, 6, 5], 0);
    q.push_face(&[1, 2, 7, 6], 0);
    q.push_face(&[2, 3, 8, 7], 0);
    let p = dd.patches(&q, &[1, 0]).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains("g patch_0") && text.contains("g patch_1") && text.contains("g unassigned"));
    assert_eq!(parse_obj(&text).unwrap().n_faces(), 3);

    let p = dd.quads("post.obj", &q).unwrap();
    assert_eq!(read_obj_poly(&p).unwrap().face_verts, q.face_verts);
}

#[test]
fn dumps_tolerate_broken_inputs() {
    let m = TriMesh { pos: vec![[0.0; 3]; 2], tris: vec![[0, 1, 5]] };
    let mut out: Vec<u8> = Vec::new();
    dump_proxy(&mut out, &m, None).unwrap();
    dump_cones(&mut out, &m.pos, &[(10, 1)]).unwrap();
    dump_param(&mut out, &m, &SeamlessParam::default()).unwrap();
    dump_layout(&mut out, &Layout::default()).unwrap();
    dump_quant(&mut out, &TMesh::default(), None, None).unwrap();
    let mut q = QuadMesh::new();
    q.push_face(&[0, 1, 2, 3], 0);
    dump_patches(&mut out, &q, &[]).unwrap();
    assert!(write_quad_obj(&mut out, &q).is_ok());
}

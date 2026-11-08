//! Every contract type has a `validate()`: positive and negative cases on small instances.

mod common;
use common::*;

use aq_core::param::FAMILY_NONE;
use aq_core::tol::snap_dyadic;
use aq_core::{
    ArcKind, Chain, CurvField, CutArc, CutGraph, FeatureKind, FeatureSet, LayoutHealth, LoopCand,
    NONE, NodeKind, QComb, QuadMesh, SeamlessParam, SizeField, SurfacePoint, TArc, TMesh, TNode,
    TPatch, Transition, TriMesh,
};

#[test]
fn trimesh_and_quadmesh() {
    let m = icosphere(1, 1.0);
    m.validate().unwrap();
    let mut bad = m.clone();
    bad.pos[0][1] = f64::NAN;
    assert!(bad.validate().is_err());
    let mut bad = m.clone();
    bad.tris[0][0] = 9999;
    assert!(bad.validate().is_err());

    // 2x1 quad strip.
    let mut q = QuadMesh::new();
    q.pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]];
    q.vflag = vec![0; 6];
    q.push_face(&[0, 1, 4, 3], 0);
    q.push_face(&[1, 2, 5, 4], 0);
    q.validate().unwrap();
    assert!(q.is_all_quads());
    let mut bad = q.clone();
    bad.push_face(&[0, 1, 4, 3], 0); // duplicate directed edges
    bad.vflag = vec![0; 6];
    assert!(bad.validate().is_err());
}

#[test]
fn feature_set() {
    let c = ctx(open_cylinder(1.0, 1.0, 8, 2));
    let mut fs = FeatureSet::new(&c.m, &c.t);
    fs.validate(&c.m, &c.t).unwrap();
    // One boundary chain from the first boundary loop.
    let lp = c.t.boundary_loops[0].clone();
    for &h in &lp {
        fs.he_chain[h as usize] = 0;
    }
    fs.chains.push(Chain { hes: lp.clone(), kind: FeatureKind::Boundary, closed: true, corners: vec![] });
    fs.validate(&c.m, &c.t).unwrap();
    let mut bad = fs.clone();
    bad.chains[0].hes.swap(0, 1);
    assert!(bad.validate(&c.m, &c.t).is_err());
    let mut bad = fs.clone();
    bad.he_kind[lp[0] as usize] = FeatureKind::None;
    assert!(bad.validate(&c.m, &c.t).is_err());
    let mut bad = fs.clone();
    bad.chains[0].corners = vec![3, 1];
    assert!(bad.validate(&c.m, &c.t).is_err());
}

#[test]
fn curv_size_fields() {
    let c = ctx(icosphere(2, 1.0));
    let nf = c.m.tris.len();
    let nv = c.m.pos.len();
    let cf = CurvField {
        k1: vec![1.0; nf],
        k2: vec![-0.5; nf],
        phi: vec![0.0; nf],
        conf: vec![0.5; nf],
        kappa_hat: vec![1.0; nv],
    };
    cf.validate(nf, nv).unwrap();
    let mut bad = cf.clone();
    bad.k2[0] = 2.0;
    assert!(bad.validate(nf, nv).is_err());
    let mut sf = SizeField { h_vert: vec![0.5; nv], h_face: vec![0.5; nf], n_pred: 0.0 };
    sf.n_pred = sf.predicted_count(&c.fr);
    sf.validate(&c.fr, nv).unwrap();
    sf.n_pred *= 1.01;
    assert!(sf.validate(&c.fr, nv).is_err());
}

/// Cylinder seamless param: u = column index (around), v = ring index; the seam between the
/// last column and column 0 is cut with translation (-n, 0).
fn cylinder_param(c: &Ctx, n: u32) -> SeamlessParam {
    let nf = c.m.tris.len();
    let nh = 3 * nf;
    let col_row = |v: u32| (v % n, v / n);
    let mut uv = vec![[[0.0; 2]; 3]; nf];
    for f in 0..nf {
        let t = c.m.tris[f];
        let cols: Vec<u32> = t.iter().map(|&v| col_row(v).0).collect();
        let wraps = cols.contains(&0) && cols.contains(&(n - 1));
        for k in 0..3 {
            let (col, row) = col_row(t[k]);
            let u = if wraps && col == 0 { n } else { col };
            uv[f][k] = [snap_dyadic(u as f64), snap_dyadic(row as f64)];
        }
    }
    let mut trans = vec![Transition::IDENTITY; nh];
    let mut is_cut = vec![false; nh];
    for h in 0..nh as u32 {
        let tw = c.t.twin[h as usize];
        if tw == NONE {
            continue;
        }
        let (a, b) = (c.t.tail(&c.m, h), c.t.head(&c.m, h));
        // Seam edges: both endpoints in column 0 (vertical edges at angle 0).
        if col_row(a).0 == 0 && col_row(b).0 == 0 {
            is_cut[h as usize] = true;
            // f is in the wrapping column (u = n) if its third vertex is in column n-1.
            let f = (h / 3) as usize;
            let in_last = c.m.tris[f].iter().any(|&v| col_row(v).0 == n - 1);
            trans[h as usize] = if in_last {
                Transition { r: 0, t: [-(n as f64), 0.0] }
            } else {
                Transition { r: 0, t: [n as f64, 0.0] }
            };
        }
    }
    // Boundary edges are U (v constant).
    let he_family = (0..nh).map(|h| if c.t.twin[h] == NONE { 0 } else { FAMILY_NONE }).collect();
    SeamlessParam {
        uv,
        trans,
        is_cut,
        cones: vec![],
        he_family,
        loops_exact: vec![],
        folded: vec![false; nf],
        report: Default::default(),
    }
}

#[test]
fn seamless_param() {
    let n = 12;
    let c = ctx(open_cylinder(2.0, 3.0, n, 3));
    let sp = cylinder_param(&c, n);
    sp.validate(&c.m, &c.t).unwrap();
    // Break one transition by a quantum.
    let mut bad = sp.clone();
    let h = bad.is_cut.iter().position(|&x| x).unwrap();
    bad.trans[h].t[0] += 1.0 / 16_777_216.0;
    assert!(bad.validate(&c.m, &c.t).is_err());
    // Non-dyadic uv.
    let mut bad = sp.clone();
    bad.uv[0][0][0] += 1e-9;
    assert!(bad.validate(&c.m, &c.t).is_err());
    // Iso chain broken.
    let mut bad = sp.clone();
    let hb = (0..c.t.twin.len()).find(|&h| c.t.twin[h] == NONE).unwrap();
    bad.he_family[hb] = 1;
    assert!(bad.validate(&c.m, &c.t).is_err());
    // Out of bound uv.
    let mut bad = sp.clone();
    bad.uv[0][0][0] = 2f64.powi(28);
    assert_eq!(bad.validate(&c.m, &c.t).unwrap_err().kind, aq_core::ErrKind::Numerical);
}

#[test]
fn cut_graph_and_loops() {
    let n = 8;
    let c = ctx(open_cylinder(1.0, 1.0, n, 1));
    let nf = c.m.tris.len();
    let nh = 3 * nf;
    // Cut the seam (column-0 vertical edge, from vertex 0 to vertex n).
    let mut cg = CutGraph {
        is_cut: vec![false; nh],
        he_arc: vec![(NONE, 0); nh],
        arcs: vec![],
        node_verts: vec![0, n],
        theta_combed: vec![0.0; nf],
        r: vec![0; nh],
        comb_offset: vec![0; nf],
    };
    let ha = c.t.find_he(&c.m, 0, n).unwrap();
    let hb = c.t.twin[ha as usize];
    cg.is_cut[ha as usize] = true;
    cg.is_cut[hb as usize] = true;
    cg.he_arc[ha as usize] = (0, 1);
    cg.he_arc[hb as usize] = (0, -1);
    cg.arcs.push(CutArc { hes: vec![ha], nodes: [0, n], r: 0 });
    cg.validate(&c.m, &c.t, None).unwrap();
    let mut bad = cg.clone();
    bad.r[ha as usize] = 1;
    assert!(bad.validate(&c.m, &c.t, None).is_err());

    // A loop around the cylinder: with one ring of faces the dual graph is a cycle.
    let start = 0u32;
    let mut hes = Vec::new();
    let mut f = start;
    let mut entry = NONE;
    loop {
        let h = (0..3).map(|k| 3 * f + k).find(|&h| c.t.twin[h as usize] != NONE && h != entry).unwrap();
        hes.push(h);
        entry = c.t.twin[h as usize];
        f = entry / 3;
        if f == start {
            break;
        }
        assert!(hes.len() <= nf);
    }
    assert_eq!(hes.len(), nf);
    let lc = LoopCand {
        hes,
        start_face: start,
        seed: SurfacePoint { face: start, bary: [1.0 / 3.0; 3] },
        branch: 0,
        length: 6.0,
        ring: true,
    };
    lc.validate(&c.m, &c.t).unwrap();
    let mut bad = lc.clone();
    bad.hes.pop();
    assert!(bad.validate(&c.m, &c.t).is_err());
}

fn rect_tmesh() -> TMesh {
    // Nodes 0..4 CCW corners; arcs 0: 0->1, 1: 1->2, 2: 2->3, 3: 3->0.
    let node = TNode { kind: NodeKind::BoundaryCorner, index4: 1, chain: NONE };
    let arc = |a: u32, b: u32, len: f64| TArc { ends: [a, b], len, kind: ArcKind::Boundary, lb: 1, weight: 4.0, wrap: false, chain: NONE };
    TMesh {
        nodes: vec![node.clone(), node.clone(), node.clone(), node],
        arcs: vec![arc(0, 1, 2.0), arc(1, 2, 1.0), arc(2, 3, 2.0), arc(3, 0, 1.0)],
        patches: vec![TPatch { sides: [vec![(0, false)], vec![(1, false)], vec![(2, false)], vec![(3, false)]] }],
        rings: vec![],
        chi: 1,
    }
}

#[test]
fn tmesh_and_qcomb() {
    let t = rect_tmesh();
    t.validate().unwrap();
    assert_eq!(t.patch_corners(0), [0, 1, 2, 3]);
    let mut bad = t.clone();
    bad.patches[0].sides[2] = vec![(2, true)];
    assert!(bad.validate().is_err());
    let mut bad = t.clone();
    bad.chi = 2;
    assert!(bad.validate().is_err());

    // 2 x 1 grid: ids j*(W+1)+i.
    let q = QComb {
        n_verts: 6,
        node_id: vec![0, 2, 5, 3],
        arc_ids: vec![vec![0, 1, 2], vec![2, 5], vec![5, 4, 3], vec![3, 0]],
        patch_dims: vec![[2, 1]],
        patch_ids: vec![vec![0, 1, 2, 3, 4, 5]],
        quads: vec![[0, 1, 4, 3], [1, 2, 5, 4]],
        quad_patch: vec![0, 0],
    };
    q.validate(&t).unwrap();
    let mut bad = q.clone();
    bad.quads[1] = [1, 4, 5, 2];
    assert!(bad.validate(&t).is_err());
    let mut bad = q.clone();
    bad.arc_ids[0] = vec![0, 1, 3];
    assert!(bad.validate(&t).is_err());

    let h = LayoutHealth::from_sides(&[[2.0, 1.0, 2.0, 1.0], [3.0, 1.0, 2.0, 1.0]]);
    assert_eq!(h.per_patch[0], 0.0);
    assert!((h.per_patch[1] - 1.0 / 3.0).abs() < 1e-15);
    assert_eq!(h.frac_above, 0.5);
    h.validate().unwrap();
}

#[test]
fn layout_validate() {
    // W' = a unit square split in 2 triangles, one patch, 4 boundary arcs.
    let wmesh = TriMesh {
        pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        tris: vec![[0, 1, 2], [0, 2, 3]],
    };
    let wtopo = aq_core::Topology::build(&wmesh).unwrap();
    let mut tm = rect_tmesh();
    for a in tm.arcs.iter_mut() {
        a.len = 1.0;
    }
    let lay = aq_core::Layout {
        wuv: vec![[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        wtrans: vec![Transition::IDENTITY; 6],
        parent_face: vec![0, 0],
        tmesh: tm,
        node_vert: vec![0, 1, 2, 3],
        arc_verts: vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
        arc_s: vec![vec![0.0, 1.0]; 4],
        patch_faces: vec![vec![0, 1]],
        patch_chart_face: vec![0],
        vert_chain: vec![NONE; 4],
        health: LayoutHealth::from_sides(&[[1.0; 4]]),
        wmesh,
        wtopo,
    };
    lay.validate(1).unwrap();
    let mut bad = lay.clone();
    bad.arc_verts[1] = vec![1, 3, 2];
    bad.arc_s[1] = vec![0.0, 0.5, 1.0];
    // 1-3 is not an edge of W'.
    assert!(bad.validate(1).is_err());
    let mut bad = lay.clone();
    bad.patch_faces = vec![vec![0]];
    assert!(bad.validate(1).is_err());
}

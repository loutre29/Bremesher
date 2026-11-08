//! DESIGN 6.1: Euler characteristic and topology identities.

mod common;
use common::*;

use aq_core::math;
use aq_core::{ErrKind, NONE, Topology, TriMesh};

fn signed_volume(m: &TriMesh) -> f64 {
    m.tris
        .iter()
        .map(|t| {
            let (a, b, c) = (m.pos[t[0] as usize], m.pos[t[1] as usize], m.pos[t[2] as usize]);
            math::dot(a, math::cross(b, c)) / 6.0
        })
        .sum()
}

#[test]
fn euler_characteristic() {
    let c = ctx(icosphere(4, 1.0));
    assert_eq!(c.m.tris.len(), 5120);
    assert_eq!(c.t.n_comp, 1);
    assert_eq!(c.t.euler_char(&c.m, 0), 2);
    assert!(c.t.boundary_loops.is_empty());
    assert!(signed_volume(&c.m) > 0.0, "icosphere must be outward CCW");

    let c = ctx(torus(3.0, 1.0, 48, 16));
    assert_eq!(c.t.euler_char(&c.m, 0), 0);
    assert!(signed_volume(&c.m) > 0.0);

    let c = ctx(double_torus());
    assert_eq!(c.t.n_comp, 1);
    assert_eq!(c.t.euler_char(&c.m, 0), -2);
    assert!(signed_volume(&c.m) > 0.0);

    let c = ctx(open_cylinder(1.0, 3.0, 24, 8));
    assert_eq!(c.t.euler_char(&c.m, 0), 0);
    assert_eq!(c.t.boundary_loops.len(), 2);
    assert_eq!(c.t.n_boundary_loops(0), 2);
    for lp in &c.t.boundary_loops {
        assert_eq!(lp.len(), 24);
    }
}

#[test]
fn topology_identities() {
    for m in [icosphere(3, 1.0), torus(3.0, 1.0, 20, 10), double_torus(), open_cylinder(1.0, 2.0, 12, 4)] {
        let c = ctx(m);
        let nh = c.t.twin.len() as u32;
        let mut n_out = vec![0usize; c.m.pos.len()];
        for h in 0..nh {
            assert_eq!(Topology::next(Topology::next(Topology::next(h))), h, "next^3 != id");
            assert_eq!(Topology::prev(Topology::next(h)), h);
            assert_eq!(Topology::face(h), h / 3);
            let tw = c.t.twin[h as usize];
            if tw != NONE {
                assert_eq!(c.t.twin[tw as usize], h, "twin(twin h) != h");
                assert_eq!(c.t.tail(&c.m, h), c.t.head(&c.m, tw));
                assert_eq!(c.t.is_canonical(h), h < tw);
            } else {
                assert!(c.t.is_canonical(h));
            }
            n_out[c.t.tail(&c.m, h) as usize] += 1;
        }
        for v in 0..c.m.pos.len() as u32 {
            // The fan covers every outgoing halfedge, in CCW order.
            let fan: Vec<u32> = c.t.fan_out(v).collect();
            assert_eq!(fan.len(), n_out[v as usize]);
            for w in fan.windows(2) {
                assert_eq!(c.t.ccw_out(w[0]), w[1]);
                assert_eq!(c.t.cw_out(w[1]), w[0]);
            }
            for &h in &fan {
                assert_eq!(c.t.tail(&c.m, h), v);
            }
            if c.t.is_bnd_vert[v as usize] {
                assert_eq!(c.t.twin[fan[0] as usize], NONE, "vert_he must be the boundary he");
                assert_eq!(c.t.ccw_out(*fan.last().unwrap()), NONE);
            }
            // H_in: interior, head v, and crossing goes CCW (to the face of the next fan he).
            let hin: Vec<u32> = c.t.incoming(&c.m, v).collect();
            let expect = if c.t.is_bnd_vert[v as usize] { fan.len() - 1 } else { fan.len() };
            assert_eq!(hin.len(), expect);
            for &h in &hin {
                assert_eq!(c.t.head(&c.m, h), v);
                assert_ne!(c.t.twin[h as usize], NONE);
            }
            for (i, &h) in hin.iter().enumerate() {
                let g = c.t.twin[h as usize] / 3;
                assert_eq!(g, fan[(i + 1) % fan.len()] / 3, "crossing H_in is one CCW step");
            }
        }
    }
}

#[test]
fn topology_errors() {
    // Flipped face: directed edge twice.
    let mut m = icosphere(1, 1.0);
    let t0 = m.tris[0];
    m.tris[0] = [t0[0], t0[2], t0[1]];
    assert_eq!(Topology::build(&m).unwrap_err().kind, ErrKind::Topology);
    // Edge with 3 faces.
    let m = TriMesh {
        pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        tris: vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
    };
    assert_eq!(Topology::build(&m).unwrap_err().kind, ErrKind::Topology);
    // Bowtie vertex: two triangles sharing only vertex 0.
    let m = TriMesh {
        pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, -1.0, 0.0]],
        tris: vec![[0, 1, 2], [0, 3, 4]],
    };
    assert_eq!(Topology::build(&m).unwrap_err().kind, ErrKind::Topology);
    // Out of range and repeated index.
    let m = TriMesh { pos: vec![[0.0; 3]; 3], tris: vec![[0, 1, 5]] };
    assert_eq!(Topology::build(&m).unwrap_err().kind, ErrKind::InvalidInput);
    let m = TriMesh { pos: vec![[0.0; 3]; 3], tris: vec![[0, 1, 1]] };
    assert_eq!(Topology::build(&m).unwrap_err().kind, ErrKind::InvalidInput);
    // Empty mesh is fine for the topology.
    let t = Topology::build(&TriMesh::default()).unwrap();
    assert_eq!(t.n_comp, 0);
}

#[test]
fn components() {
    let mut m = icosphere(1, 1.0);
    let t2 = torus(3.0, 1.0, 8, 6);
    let off = m.pos.len() as u32;
    m.pos.extend_from_slice(&t2.pos);
    m.tris.extend(t2.tris.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    let c = ctx(m);
    assert_eq!(c.t.n_comp, 2);
    assert_eq!(c.t.euler_char(&c.m, 0), 2);
    assert_eq!(c.t.euler_char(&c.m, 1), 0);
}

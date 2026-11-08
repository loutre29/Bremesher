//! Hand-built motorcycle instances through the public `trace_all` / `embed` API.

mod common;

use aq_core::{CtlState, V2};
use aq_layout::{Curve, CurveKind, Emitter, LayoutParams, Pt, Seed, curves, embed, geom, trace_all};
use aq_testkit::*;
use common::*;

/// Ray emitter at the vertex nearest to p along direction d (global plane chart).
fn ray(c: &ParamCase, p: V2, d: u8) -> Emitter {
    let (v, f) = vertex_near(c, p);
    let ems = curves::emitters_for(&px(c), &c.param, &[], &[Seed { vert: v, face: f, dir: d }]).unwrap();
    let mut e = *ems.iter().find(|e| e.vert == v).unwrap();
    e.kind = CurveKind::Separatrix;
    e
}

fn run(c: &ParamCase, ems: &[Emitter]) -> (Vec<Curve>, Vec<Curve>) {
    let st = CtlState::new();
    let ctl = st.unbounded();
    let p = LayoutParams::default();
    let base = curves::constraint_curves(&px(c), &c.param, &[], &p, &ctl).unwrap();
    let tr = trace_all(&px(c), &c.param, &base, ems, &p, &ctl).unwrap();
    (base, tr)
}

fn end_uv(c: &ParamCase, tr: &Curve) -> V2 {
    let pc = tr.pieces.last().unwrap();
    let tri = &c.param.uv[pc.face as usize];
    pc.line.point(geom::aval(tri, pc.line, pc.end))
}

fn vuv(c: &ParamCase, p: V2) -> V2 {
    let (v, f) = vertex_near(c, p);
    let k = c.surface.mesh.tris[f as usize].iter().position(|&x| x == v).unwrap();
    c.param.uv[f as usize][k]
}

fn embeds(c: &ParamCase, base: &[Curve], tr: &[Curve]) {
    let mut all = base.to_vec();
    all.extend_from_slice(tr);
    let e = embed(&px(c), &c.param, &all).unwrap();
    e.wmesh.validate().unwrap();
    e.wtopo.validate(&e.wmesh).unwrap();
}

fn grid() -> ParamCase {
    plane_param_with(0.0, 1.0, 16.0, ParamOpts::aligned(1)).unwrap()
}

#[test]
fn later_arrival_stops() {
    let c = grid();
    // A: (2,5) +u reaches (8,5) at t = 6; B: (8,1) +v reaches it at t = 4.
    let ems = [ray(&c, [2.0, 5.0], 0), ray(&c, [8.0, 1.0], 1)];
    let (base, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[0]), [8.0, 5.0]);
    assert_eq!(end_uv(&c, &tr[1]), [8.0, 16.0]);
    embeds(&c, &base, &tr);
}

#[test]
fn stopped_trace_does_not_block() {
    let c = grid();
    // C: (12,9) -v crosses v = 5 at t = 4, where A would have been at t = 10 had it not
    // stopped at (8,5) at t = 6.
    let ems = [ray(&c, [2.0, 5.0], 0), ray(&c, [8.0, 1.0], 1), ray(&c, [12.0, 9.0], 3)];
    let (base, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[0]), [8.0, 5.0]);
    assert_eq!(end_uv(&c, &tr[2]), [12.0, 0.0]);
    embeds(&c, &base, &tr);
    // C emitted far enough that A arrives first: C stops on A instead.
    let ems = [ray(&c, [2.0, 5.0], 0), ray(&c, [8.0, 1.0], 1), ray(&c, [6.0, 13.0], 3)];
    let (_, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[2]), [6.0, 5.0]);
}

#[test]
fn equal_times_higher_id_stops() {
    let c = grid();
    // Both reach (8,5) at t = 4.
    let ems = [ray(&c, [4.0, 5.0], 0), ray(&c, [8.0, 1.0], 1)];
    let (_, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[0]), [16.0, 5.0]);
    assert_eq!(end_uv(&c, &tr[1]), [8.0, 5.0]);
    let ems = [ray(&c, [8.0, 1.0], 1), ray(&c, [4.0, 5.0], 0)];
    let (_, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[0]), [8.0, 16.0]);
    assert_eq!(end_uv(&c, &tr[1]), [8.0, 5.0]);
}

#[test]
fn head_on_at_vertex_and_edge_midpoint() {
    let c = grid();
    let ems = [ray(&c, [2.0, 5.0], 0), ray(&c, [10.0, 5.0], 2)];
    let (base, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[0]), [6.0, 5.0]);
    assert_eq!(end_uv(&c, &tr[1]), [6.0, 5.0]);
    assert!(matches!(tr[0].pieces.last().unwrap().end, Pt::V(_)));
    embeds(&c, &base, &tr);
    let ems = [ray(&c, [2.0, 5.0], 0), ray(&c, [11.0, 5.0], 2)];
    let (base, tr) = run(&c, &ems);
    assert_eq!(end_uv(&c, &tr[0]), [6.5, 5.0]);
    assert_eq!(end_uv(&c, &tr[1]), [6.5, 5.0]);
    embeds(&c, &base, &tr);
    // Vertical head-on on an odd side length.
    let c15 = plane_param_with(0.0, 1.0, 15.0, ParamOpts::aligned(1)).unwrap();
    let ems = [ray(&c15, [7.0, 0.0], 1), ray(&c15, [7.0, 15.0], 3)];
    let (base, tr) = run(&c15, &ems);
    assert!(ems[0].kind == CurveKind::Separatrix);
    assert_eq!(end_uv(&c15, &tr[0]), [7.0, 7.5]);
    assert_eq!(end_uv(&c15, &tr[1]), [7.0, 7.5]);
    embeds(&c15, &base, &tr);
}

#[test]
fn truncation_inside_a_face_on_a_jittered_grid() {
    let c = plane_param_with(0.0, 1.0, 16.0, ParamOpts { res: 1, jitter: 0.25, seed: 5 }).unwrap();
    let a = vuv(&c, [2.0, 5.0]);
    let b = vuv(&c, [8.0, 1.0]);
    let ems = [ray(&c, [2.0, 5.0], 0), ray(&c, [8.0, 1.0], 1)];
    let (base, tr) = run(&c, &ems);
    // A stops at the exact crossing (b.u, a.v), strictly inside a face: its placed chord was
    // truncated there.
    assert_eq!(end_uv(&c, &tr[0]), [b[0], a[1]]);
    assert!(matches!(tr[0].pieces.last().unwrap().end, Pt::X(_)));
    assert_eq!(end_uv(&c, &tr[1])[1], 16.0);
    embeds(&c, &base, &tr);
}

#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::collections::BTreeSet;

use aq_core::exact::orient2d;
use aq_core::{AqError, CtlState, Layout, LoopCand, NONE, V2, math};
use aq_layout::{LayoutParams, LayoutStats, ProxyView, Seed, build_layout_with};
use aq_testkit::ParamCase;

pub fn px(c: &ParamCase) -> ProxyView<'_> {
    ProxyView { mesh: &c.surface.mesh, topo: &c.surface.topo, feats: &c.surface.feats }
}

pub fn build_p(c: &ParamCase, loops: bool, seeds: &[Seed], p: &LayoutParams) -> Result<(Layout, LayoutStats), AqError> {
    let st = CtlState::new();
    let ctl = st.unbounded();
    let lp: &[LoopCand] = if loops { &c.loops } else { &[] };
    build_layout_with(&px(c), &c.param, lp, seeds, p, &ctl)
}

pub fn build(c: &ParamCase, loops: bool, seeds: &[Seed]) -> (Layout, LayoutStats) {
    match build_p(c, loops, seeds, &LayoutParams::default()) {
        Ok(x) => x,
        Err(e) => panic!("build_layout failed: {e}"),
    }
}

/// The proxy vertex whose uv (in any incident face chart) is nearest to p, and one face of it.
pub fn vertex_near(c: &ParamCase, p: V2) -> (u32, u32) {
    let mut best = (f64::INFINITY, 0u32, 0u32);
    for (f, tri) in c.param.uv.iter().enumerate() {
        for k in 0..3 {
            let d = (tri[k][0] - p[0]).hypot(tri[k][1] - p[1]);
            let v = c.surface.mesh.tris[f][k];
            if d < best.0 || (d == best.0 && v < best.1) {
                best = (d, v, f as u32);
            }
        }
    }
    (best.1, best.2)
}

/// Checks shared by every valid layout: the contract validators, W' quality (manifold,
/// orientation, exact transitions, area per parent face) and the T-mesh Euler relation.
pub fn check_layout(c: &ParamCase, l: &Layout) {
    let m = &c.surface.mesh;
    if let Err(e) = l.validate(m.tris.len()) {
        panic!("Layout::validate: {e}");
    }
    let t = &l.tmesh;
    assert_eq!(t.nodes.len() as i64 - t.arcs.len() as i64 + t.patches.len() as i64, t.chi as i64);
    let chi: i64 = (0..c.surface.topo.n_comp).map(|k| c.surface.topo.euler_char(m, k)).sum();
    assert_eq!(t.chi as i64, chi);
    let w = &l.wmesh;
    // Exact inherited transitions.
    for h in 0..3 * w.tris.len() {
        let tw = l.wtopo.twin[h];
        if tw == NONE {
            continue;
        }
        let tr = l.wtrans[h];
        assert_eq!(l.wtrans[tw as usize], tr.inverse(), "wtrans not inverse on twin at {h}");
        let (f, k) = (h / 3, h % 3);
        let (g, kg) = (tw as usize / 3, tw as usize % 3);
        let a = l.wuv[f][k];
        let b = l.wuv[f][(k + 1) % 3];
        assert_eq!(tr.apply(a), l.wuv[g][(kg + 1) % 3], "transition not exact at W' halfedge {h}");
        assert_eq!(tr.apply(b), l.wuv[g][kg], "transition not exact at W' halfedge {h}");
    }
    // Orientation (non-folded parents) and 3D area per parent face.
    let mut child_area = vec![0.0; m.tris.len()];
    for (wf, &pf) in l.parent_face.iter().enumerate() {
        child_area[pf as usize] += w.face_area(wf as u32);
        if !c.param.folded[pf as usize] {
            let u = l.wuv[wf];
            assert!(orient2d(u[0], u[1], u[2]) > 0.0, "W' face {wf} not positively oriented in uv");
            let n0 = w.face_cross(wf as u32);
            let n1 = m.face_cross(pf);
            assert!(math::dot(n0, n1) > 0.0, "W' face {wf} flips its parent orientation");
        }
    }
    for (f, a) in child_area.iter().enumerate() {
        let pa = m.face_area(f as u32);
        assert!((a - pa).abs() <= 1e-9 * pa.max(1e-12), "face {f}: children area {a} parent {pa}");
    }
    // Patch areas sum to the mesh area.
    let total: f64 = l.patch_faces.iter().flatten().map(|&f| w.face_area(f)).sum();
    let ma = m.total_area();
    assert!((total - ma).abs() <= 1e-9 * ma, "patch areas {total} vs mesh {ma}");
}

/// No two W' vertices share a 3D position or are used twice as node vertices.
pub fn check_no_duplicates(l: &Layout) {
    let mut seen = BTreeSet::new();
    for p in &l.wmesh.pos {
        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        assert!(seen.insert(key), "duplicate W' position {p:?}");
    }
    let nv: BTreeSet<u32> = l.node_vert.iter().copied().collect();
    assert_eq!(nv.len(), l.node_vert.len(), "duplicate node vertices");
}

/// Number of times each arc occurs on patch sides.
pub fn occurrences(l: &Layout) -> Vec<u32> {
    l.tmesh.arc_occurrences()
}

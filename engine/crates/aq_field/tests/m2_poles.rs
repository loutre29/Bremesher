//! WP-23 (DESIGN 6.6 M2) tests: same-sign repulsion, energy-guarded far cancellation, the
//! pole sanity cap predicate, the corner merge and the merge of -1 pairs into -2 cones.
#![allow(clippy::needless_range_loop)]

mod common;

use aq_core::{CrossField, CurvField};
use aq_field::{FieldParams, MAX_EDIT_DELTA, edit_singularities, excess_poles, field_energy, resolve_fixed_matching, solve_field};
use aq_testkit::{FieldKind, PlantChart, Surface, analytic_field};
use common::*;

/// The planted-pair plane of `planted.rs`: a jittered rotated grid, h = 1, a gently rotating
/// curvature target of confidence `conf`.
fn plane_conf(conf: f64) -> (Surface, CurvField, CrossField) {
    let angle = 0.2;
    let s = Surface::new(aq_testkit::plane_grid(42, 42, 1.0 / 3.0, angle, 0.15, 9).unwrap()).unwrap();
    let mut c = zero_curv(&s);
    for f in 0..s.n_faces() {
        let x = centroid(&s.mesh, f as u32);
        let g = angle + 0.15 * (x[0] / 3.0).sin();
        c.phi[f] = angle_in_face(&s.frames, f as u32, [g.cos(), g.sin(), 0.0]);
        c.conf[f] = conf;
    }
    let st = ctl_state();
    let (base, _) = solve_field(&px(&s), &c, &size(&s, 1.0), &FieldParams::default(), &st.unbounded()).unwrap();
    assert!(interior_cones(&s, &base).is_empty());
    (s, c, base)
}

fn plane() -> (Surface, CurvField, CrossField) {
    plane_conf(0.5)
}

/// The interior vertex nearest the centre of the plane, and the interior vertex whose edge
/// distance from it is closest to `d`.
fn centre_and_partner(s: &Surface, d: f64) -> (u32, u32) {
    let n = s.n_verts() as f64;
    let mid = s.mesh.pos.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
    let c = (0..s.n_verts() as u32)
        .min_by(|&a, &b| {
            let (pa, pb) = (s.mesh.pos[a as usize], s.mesh.pos[b as usize]);
            aq_core::math::dist(pa, mid).total_cmp(&aq_core::math::dist(pb, mid)).then(a.cmp(&b))
        })
        .unwrap();
    let dc = edge_dijkstra(s, &[c]);
    let b = (0..s.n_verts() as u32)
        .filter(|&v| v != c && !s.topo.is_bnd_vert[v as usize])
        .min_by(|&x, &y| (dc[x as usize] - d).abs().total_cmp(&(dc[y as usize] - d).abs()).then(x.cmp(&y)))
        .unwrap();
    (c, b)
}

fn planted(s: &Surface, base: &CrossField, c: &CurvField, sings: Vec<(u32, i8)>) -> CrossField {
    let kind = FieldKind::Planted { base: Box::new(FieldKind::Theta(base.theta.clone())), sings, chart: PlantChart::Plane };
    let mut f = analytic_field(s, &kind).unwrap();
    resolve_fixed_matching(&px(s), c, &size(s, 1.0), &mut f, &FieldParams::default()).unwrap();
    f
}

/// Two +1 cones on adjacent vertices: the repulsion moves one of them to at least 0.7 h from
/// every other cone, and nothing cancels them (no negative partner inside the plane).
#[test]
fn repulsion_separates_adjacent_same_sign_cones() {
    let (s, c, base) = plane();
    let (a, _) = centre_and_partner(&s, 0.0);
    let b = s.topo.fan_out(a).map(|h| s.topo.head(&s.mesh, h)).min().unwrap();
    let mut f = planted(&s, &base, &c, vec![(a, 1), (b, 1)]);
    assert_eq!((f.index4[a as usize], f.index4[b as usize]), (1, 1));
    // Cancellation off, so only the repulsion acts (the plane's boundary leaves a -1 far away).
    let p = FieldParams { cancel_dist: 0.0, cancel_dist_lowconf: 0.0, far_cancel: 0.0, ..FieldParams::default() };
    let st = ctl_state();
    let rep = edit_singularities(&px(&s), &c, &size(&s, 1.0), &mut f, &p, &st.unbounded()).unwrap();
    assert_eq!(rep.merged_neg, 0, "positive cones never merge");
    let plus: Vec<u32> = interior_cones(&s, &f).iter().filter(|&&(_, i)| i == 1).map(|&(v, _)| v).collect();
    assert_eq!(plus.len(), 2, "cones {:?}", interior_cones(&s, &f));
    let d = edge_dijkstra(&s, &[plus[0]])[plus[1] as usize];
    assert!(d >= 0.7, "same-sign cones only {d} h apart after the repulsion ({rep:?})");
    assert!(rep.pushed >= 1);
}

/// A +- pair 4 h apart on a plane of low confidence (0.1, below the 0.2 of step 5): outside
/// the 2.5 h low-confidence distance, so only the energy-guarded far cancellation removes it,
/// at a small energy cost. On the same plane with confidence 0.5 (a coherent target) the far
/// cancellation leaves it alone.
#[test]
fn far_pair_is_cancelled_when_the_field_pays_for_it() {
    for (conf, want) in [(0.1, 1u32), (0.5, 0)] {
        let (s, c, base) = plane_conf(conf);
        let sz = size(&s, 1.0);
        let e_clean = field_energy(&px(&s), &c, &sz, &base, &FieldParams::default()).unwrap();
        let (a, b) = centre_and_partner(&s, 4.0);
        let st = ctl_state();
        // Off: the pair stays.
        let mut f = planted(&s, &base, &c, vec![(a, 1), (b, -1)]);
        let off = FieldParams { far_cancel: 0.0, ..FieldParams::default() };
        let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &off, &st.unbounded()).unwrap();
        assert_eq!(interior_cones(&s, &f).len(), 2, "{rep:?}");
        assert_eq!(rep.far_cancelled, 0);
        // Default.
        let mut f = planted(&s, &base, &c, vec![(a, 1), (b, -1)]);
        let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &FieldParams::default(), &st.unbounded()).unwrap();
        assert_eq!(rep.far_cancelled, want, "conf {conf}: {rep:?}");
        assert_eq!(interior_cones(&s, &f).len(), 2 - 2 * want as usize, "conf {conf}: {rep:?}");
        if want == 1 {
            let md = f.delta.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(md <= MAX_EDIT_DELTA, "|delta| {md}");
            let e = field_energy(&px(&s), &c, &sz, &f, &FieldParams::default()).unwrap();
            assert!(e < 1.05 * e_clean, "energy {e} against the clean {e_clean}");
        }
    }
}

/// The energy guard is honoured: with a threshold no pair can meet, the far pair is tried,
/// reverted and kept.
#[test]
fn far_pair_is_kept_when_its_energy_share_exceeds_the_threshold() {
    let (s, c, base) = plane_conf(0.1);
    let sz = size(&s, 1.0);
    let (a, b) = centre_and_partner(&s, 4.0);
    let mut f = planted(&s, &base, &c, vec![(a, 1), (b, -1)]);
    let p = FieldParams { far_cancel_energy: -1e6, ..FieldParams::default() }; // demands an energy drop no pair can give
    let st = ctl_state();
    let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &p, &st.unbounded()).unwrap();
    let mut want = vec![(a, 1i8), (b, -1)];
    want.sort_unstable();
    assert_eq!(interior_cones(&s, &f), want, "{rep:?}");
    assert_eq!(rep.far_cancelled, 0);
    assert!(rep.reverted >= 1, "the far pair was never tried: {rep:?}");
}

/// A cone at the apex of a conical bump (2 quarter turns of Gaussian curvature within 1.5 h)
/// is geometric: the far cancellation never removes it, while the same pair on the flat
/// plane goes (see `far_pair_is_cancelled_when_the_field_pays_for_it`).
#[test]
fn geometric_cone_is_never_far_cancelled() {
    // slope 0: the flat plane, where the pair goes; slope sqrt(3): apex angle defect pi (two
    // quarter turns), where the apex cone stays.
    for (slope, want_far) in [(0.0, 1u32), (3f64.sqrt(), 0)] {
        let mut m = aq_testkit::plane_grid(42, 42, 1.0 / 3.0, 0.0, 0.0, 9).unwrap();
        let n = m.pos.len() as f64;
        let mid = m.pos.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, 0.0]);
        let rxy = |q: &[f64; 3]| (q[0] - mid[0]).hypot(q[1] - mid[1]);
        let apex = (0..m.pos.len() as u32).min_by(|&x, &y| rxy(&m.pos[x as usize]).total_cmp(&rxy(&m.pos[y as usize])).then(x.cmp(&y))).unwrap();
        for q in m.pos.iter_mut() {
            q[2] = slope * (2.0 - rxy(q)).max(0.0);
        }
        let s = Surface::new(m).unwrap();
        let c = zero_curv(&s);
        let sz = size(&s, 1.0);
        let da = edge_dijkstra(&s, &[apex]);
        let b = (0..s.n_verts() as u32)
            .filter(|&v| !s.topo.is_bnd_vert[v as usize])
            .min_by(|&x, &y| (da[x as usize] - 5.5).abs().total_cmp(&(da[y as usize] - 5.5).abs()).then(x.cmp(&y)))
            .unwrap();
        let st = ctl_state();
        let p0 = FieldParams { far_cancel: 0.0, ..FieldParams::default() };
        let (base, _) = solve_field(&px(&s), &c, &sz, &p0, &st.unbounded()).unwrap();
        assert!(interior_cones(&s, &base).is_empty());
        let mut f = planted(&s, &base, &c, vec![(apex, 1), (b, -1)]);
        assert_eq!((f.index4[apex as usize], f.index4[b as usize]), (1, -1));
        let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &FieldParams::default(), &st.unbounded()).unwrap();
        assert_eq!(rep.far_cancelled, want_far, "slope {slope}: {rep:?}");
        let apex_cone = interior_cones(&s, &f).iter().any(|&(v, i)| i == 1 && da[v as usize] < 1.0);
        assert_eq!(apex_cone, want_far == 0, "slope {slope}: cones {:?}", interior_cones(&s, &f));
    }
}

#[test]
fn pole_cap_threshold() {
    // max(1.5 S_ref, S_ref + 8)
    assert!(!excess_poles(16, 8));
    assert!(excess_poles(17, 8));
    assert!(!excess_poles(30, 20));
    assert!(excess_poles(31, 20));
    assert!(!excess_poles(8, 0));
    assert!(excess_poles(9, 0));
}

/// Cube with Hard chains: a -1 cone planted next to a corner (within 0.5 h, off the chains)
/// with its +1 partner far away. The -1 merges into the +1 corner (merged index 0) before the
/// push-off could move it; Poincare-Hopf holds and no free cone is left within 0.5 h of a
/// corner.
#[test]
fn cone_next_to_a_corner_merges_into_it() {
    let sm = aq_testkit::cube_sharp(12, 4.0, 0.0, 1).unwrap();
    let s = Surface::from_synth(&sm).unwrap();
    let c = zero_curv(&s);
    let sz = size(&s, 1.0);
    let st = ctl_state();
    let p = FieldParams { far_cancel: 0.0, ..FieldParams::default() };
    let (base, _) = solve_field(&px(&s), &c, &sz, &p, &st.unbounded()).unwrap();
    assert_eq!(base.cones().len(), 8);
    let corner = (0..s.n_verts() as u32).find(|&v| s.feats.vert_corner[v as usize]).unwrap();
    let chain_vert: Vec<bool> = {
        let mut cv = vec![false; s.n_verts()];
        for h in 0..3 * s.n_faces() {
            if s.feats.he_chain[h] != aq_core::NONE {
                cv[s.topo.tail(&s.mesh, h as u32) as usize] = true;
                cv[s.topo.head(&s.mesh, h as u32) as usize] = true;
            }
        }
        cv
    };
    let dc = edge_dijkstra(&s, &[corner]);
    let near = (0..s.n_verts() as u32)
        .filter(|&v| !chain_vert[v as usize] && dc[v as usize] < 0.5)
        .min_by(|&x, &y| dc[x as usize].total_cmp(&dc[y as usize]).then(x.cmp(&y)))
        .expect("a free vertex within 0.5 h of the corner");
    // The free vertex of the same cube face (reachable without crossing a chain) farthest
    // from the corner, and the free path to it.
    let (far, path) = free_path_farthest(&s, &chain_vert, near);
    assert!(dc[far as usize] > 2.0);
    let mut f = base.clone();
    aq_field::try_edit_path(&px(&s), &mut f, &path, -1).unwrap();
    resolve_fixed_matching(&px(&s), &c, &sz, &mut f, &p).unwrap();
    assert_eq!(f.index4[near as usize], -1, "planting failed");
    assert_eq!(f.index4[corner as usize], 1);
    let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &p, &st.unbounded()).unwrap();
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
    assert_eq!(rep.corner_merged, 1, "{rep:?}");
    assert_eq!(f.index4[corner as usize], 0);
    assert_eq!(f.index4[near as usize], 0);
    let corners: Vec<u32> = (0..s.n_verts() as u32).filter(|&v| s.feats.vert_corner[v as usize]).collect();
    let dcs = edge_dijkstra(&s, &corners);
    for (v, _) in f.cones() {
        if !s.feats.vert_corner[v as usize] {
            assert!(dcs[v as usize] >= 0.5, "free cone {v} {} h from a corner", dcs[v as usize]);
        }
    }
}

/// Dijkstra from `a` over free (non-chain) vertices: the farthest reached vertex and the path
/// from `a` to it.
fn free_path_farthest(s: &Surface, chain_vert: &[bool], a: u32) -> (u32, Vec<u32>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let nv = s.n_verts();
    let mut d = vec![f64::INFINITY; nv];
    let mut pred = vec![u32::MAX; nv];
    let mut q = BinaryHeap::new();
    d[a as usize] = 0.0;
    q.push(Reverse((0u64, a)));
    while let Some(Reverse((db, v))) = q.pop() {
        let dv = f64::from_bits(db);
        if dv > d[v as usize] {
            continue;
        }
        for h in s.topo.fan_out(v) {
            let w = s.topo.head(&s.mesh, h);
            if chain_vert[w as usize] || s.topo.is_bnd_vert[w as usize] {
                continue;
            }
            let nd = dv + aq_core::math::dist(s.mesh.pos[v as usize], s.mesh.pos[w as usize]);
            if nd < d[w as usize] {
                d[w as usize] = nd;
                pred[w as usize] = v;
                q.push(Reverse((nd.to_bits(), w)));
            }
        }
    }
    let far = (0..nv as u32).filter(|&v| d[v as usize].is_finite()).max_by(|&x, &y| d[x as usize].total_cmp(&d[y as usize]).then(y.cmp(&x))).unwrap();
    let mut path = vec![far];
    while *path.last().unwrap() != a {
        path.push(pred[*path.last().unwrap() as usize]);
    }
    path.reverse();
    (far, path)
}

/// The cancellation and merge distances off or on, the rest default.
fn merge_params(merge: f64) -> FieldParams {
    FieldParams { cancel_dist: 0.0, cancel_dist_lowconf: 0.0, far_cancel: 0.0, merge_neg: merge, ..FieldParams::default() }
}

/// Interior cones within `r` h of vertex `a` (the plane's boundary pushes the +1 cones that
/// planted -1 cones need far away, so the tests look at the neighbourhood of the plant only).
fn cones_near(s: &Surface, f: &CrossField, a: u32, r: f64) -> Vec<(u32, i8)> {
    let d = edge_dijkstra(s, &[a]);
    interior_cones(s, f).into_iter().filter(|&(v, _)| d[v as usize] <= r).collect()
}

/// Two -1 cones about 1 h apart (a saddle that needs two negative quarters): with the merge
/// they become one -2 cone (an interior valence 6 pole) between them, the index is conserved,
/// and the edited edges keep |delta| <= pi/3. Without it, two -1 cones stay (the repulsion
/// keeps them at least 0.7 h apart).
#[test]
fn close_negative_pair_merges_into_one_valence_six_cone() {
    let (s, c, base) = plane();
    let (a, b) = centre_and_partner(&s, 1.0);
    let sz = size(&s, 1.0);
    let st = ctl_state();
    // Off.
    let mut f = planted(&s, &base, &c, vec![(a, -1), (b, -1)]);
    let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &merge_params(0.0), &st.unbounded()).unwrap();
    let cones = cones_near(&s, &f, a, 3.0);
    assert_eq!(cones.len(), 2, "{cones:?} {rep:?}");
    assert!(cones.iter().all(|&(_, i)| i == -1));
    assert_eq!(rep.merged_neg, 0);
    // On.
    let mut f = planted(&s, &base, &c, vec![(a, -1), (b, -1)]);
    let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &merge_params(1.5), &st.unbounded()).unwrap();
    f.validate(&s.mesh, &s.topo, &s.frames).unwrap();
    let cones = cones_near(&s, &f, a, 3.0);
    assert_eq!(rep.merged_neg, 1, "{rep:?}");
    assert_eq!(cones.len(), 1, "{cones:?}");
    let (m, i) = cones[0];
    assert_eq!(i, -2);
    let (da, db) = (edge_dijkstra(&s, &[a]), edge_dijkstra(&s, &[b]));
    assert!(da[m as usize] <= 1.5 && db[m as usize] <= 1.5, "merged cone {} / {} h from the pair", da[m as usize], db[m as usize]);
    let md = f.delta.iter().fold(0.0f64, |x, d| x.max(d.abs()));
    assert!(md <= MAX_EDIT_DELTA + 1e-9, "|delta| {md}");
    assert!(aq_field::interior_index_ok(-2) && !aq_field::interior_index_ok(2) && !aq_field::interior_index_ok(-3));
}

/// Two -1 cones farther apart than the merge distance stay apart.
#[test]
fn distant_negative_pair_does_not_merge() {
    let (s, c, base) = plane();
    let (a, b) = centre_and_partner(&s, 3.0);
    let mut f = planted(&s, &base, &c, vec![(a, -1), (b, -1)]);
    let st = ctl_state();
    let rep = edit_singularities(&px(&s), &c, &size(&s, 1.0), &mut f, &merge_params(1.5), &st.unbounded()).unwrap();
    assert_eq!(rep.merged_neg, 0, "{rep:?}");
    let cones = cones_near(&s, &f, a, 4.0);
    assert_eq!(cones, { let mut w = vec![(a, -1i8), (b, -1)]; w.sort_unstable(); w }, "{rep:?}");
}

/// Cancellation comes first: -1, -1 and +1 cones in a row (each 1 h from the next, the +1 at
/// one end) leave one -1 cone, never a -2 cone next to a +1.
#[test]
fn cancellation_beats_merge() {
    let (s, c, base) = plane();
    let (a, b) = centre_and_partner(&s, 1.0);
    let db = edge_dijkstra(&s, &[b]);
    let da = edge_dijkstra(&s, &[a]);
    // The interior vertex about 1 h from b and 2 h from a.
    let e = (0..s.n_verts() as u32)
        .filter(|&v| !s.topo.is_bnd_vert[v as usize])
        .min_by(|&x, &y| {
            let sx = (db[x as usize] - 1.0).abs() + (da[x as usize] - 2.0).abs();
            let sy = (db[y as usize] - 1.0).abs() + (da[y as usize] - 2.0).abs();
            sx.total_cmp(&sy).then(x.cmp(&y))
        })
        .unwrap();
    let mut f = planted(&s, &base, &c, vec![(a, -1), (b, -1), (e, 1)]);
    let p = FieldParams { far_cancel: 0.0, ..FieldParams::default() };
    let st = ctl_state();
    let rep = edit_singularities(&px(&s), &c, &size(&s, 1.0), &mut f, &p, &st.unbounded()).unwrap();
    let cones = cones_near(&s, &f, a, 3.0);
    assert_eq!(rep.merged_neg, 0, "{rep:?}");
    assert!(rep.cancelled >= 1, "{rep:?}");
    assert_eq!(cones.len(), 1, "{cones:?}");
    assert_eq!(cones[0].1, -1);
    assert!(f.index4.iter().all(|&i| i > -2), "a -2 cone was made");
}

// M2 integration (WP-20 + WP-23), CHANGES.md "M2 integration (WP-20 + WP-23)".

/// `s` with h = `h_near` within `r` h (edges) of vertex `v` and 1 elsewhere; `n_pred` is
/// recomputed so the field is valid.
fn size_step(s: &Surface, v: u32, r: f64, h_near: f64) -> aq_core::SizeField {
    let d = edge_dijkstra(s, &[v]);
    let h_vert: Vec<f64> = (0..s.n_verts()).map(|u| if d[u] <= r { h_near } else { 1.0 }).collect();
    let h_face: Vec<f64> =
        s.mesh.tris.iter().map(|t| (h_vert[t[0] as usize] + h_vert[t[1] as usize] + h_vert[t[2] as usize]) / 3.0).collect();
    let n_pred = h_face.iter().zip(&s.frames.area).map(|(h, a)| a / (h * h)).sum();
    aq_core::SizeField { h_vert, h_face, n_pred }
}

/// The far pair of `far_pair_is_cancelled_when_the_field_pays_for_it` (confidence 0.1), but
/// with its +1 cone inside a region of h = 0.8 (a thickness-capped rim) and its -1 cone at
/// h = 1: the cones' sizes differ by more than 1.15, so the pair absorbs a size change and the
/// far cancellation keeps it. A step of 0.9 (within 1.15) does not stop it.
#[test]
fn far_pair_across_a_size_step_is_kept() {
    for (h_near, want) in [(0.8, 0u32), (0.9, 1)] {
        let (s, c, base) = plane_conf(0.1);
        let (a, b) = centre_and_partner(&s, 4.0);
        let sz = size_step(&s, a, 1.0, h_near);
        assert_eq!((sz.h_vert[a as usize], sz.h_vert[b as usize]), (h_near, 1.0));
        let mut f = planted(&s, &base, &c, vec![(a, 1), (b, -1)]);
        let st = ctl_state();
        let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &FieldParams::default(), &st.unbounded()).unwrap();
        assert_eq!(rep.far_cancelled, want, "h {h_near}: {rep:?}");
    }
}

/// `FieldParams::keep_out`: the close -1 pair of `close_negative_pair_merges_into_one_valence_six_cone`
/// is not merged when a keep-out point (an S7 fold of a failed run) lies next to it, and is
/// merged when the point is far away.
#[test]
fn keep_out_point_blocks_the_merge_next_to_it() {
    let (s, c, base) = plane();
    let (a, b) = centre_and_partner(&s, 1.0);
    let sz = size(&s, 1.0);
    let st = ctl_state();
    let pa = s.mesh.pos[a as usize];
    for (point, want) in [([pa[0] + 0.3, pa[1], pa[2]], 0u32), ([pa[0] + 8.0, pa[1], pa[2]], 1)] {
        let mut f = planted(&s, &base, &c, vec![(a, -1), (b, -1)]);
        let p = FieldParams { keep_out: vec![point], ..merge_params(1.5) };
        let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &p, &st.unbounded()).unwrap();
        assert_eq!(rep.merged_neg, want, "keep-out at {point:?}: {rep:?}");
        let cones = cones_near(&s, &f, a, 3.0);
        assert_eq!(cones.len(), 2 - want as usize, "{cones:?}");
    }
}

/// A -1 cone next to a 90 degree corner of a square's boundary (index4 +1, one quad corner)
/// does not merge into it: the merged corner would have index4 0, a flat 180 degree quad
/// corner at a 90 degree surface angle (ratio 2 > 1.5). The slit of retopo_c that took two
/// cones and folded S7 is the same rule at 367 degrees.
#[test]
fn cone_next_to_a_square_corner_does_not_merge_into_the_boundary() {
    let s = Surface::new(aq_testkit::plane_grid(42, 42, 1.0 / 3.0, 0.0, 0.0, 9).unwrap()).unwrap();
    let c = zero_curv(&s);
    let sz = size(&s, 1.0);
    let st = ctl_state();
    let p = FieldParams { far_cancel: 0.0, ..FieldParams::default() };
    let (base, _) = solve_field(&px(&s), &c, &sz, &p, &st.unbounded()).unwrap();
    // The boundary corners carry the +1 quarters of the square.
    let corner = (0..s.n_verts() as u32).find(|&v| s.topo.is_bnd_vert[v as usize] && base.index4[v as usize] == 1).expect("a +1 boundary corner");
    let chain_vert: Vec<bool> = (0..s.n_verts()).map(|v| s.topo.is_bnd_vert[v]).collect();
    let dc = edge_dijkstra(&s, &[corner]);
    let near = (0..s.n_verts() as u32)
        .filter(|&v| !chain_vert[v as usize] && dc[v as usize] < 1.0)
        .min_by(|&x, &y| dc[x as usize].total_cmp(&dc[y as usize]).then(x.cmp(&y)))
        .expect("an interior vertex within 1 h of the corner");
    let (far, path) = free_path_farthest(&s, &chain_vert, near);
    assert!(dc[far as usize] > 4.0);
    let mut f = base.clone();
    aq_field::try_edit_path(&px(&s), &mut f, &path, -1).unwrap();
    resolve_fixed_matching(&px(&s), &c, &sz, &mut f, &p).unwrap();
    assert_eq!(f.index4[near as usize], -1, "planting failed");
    let rep = edit_singularities(&px(&s), &c, &sz, &mut f, &p, &st.unbounded()).unwrap();
    assert_eq!(rep.boundary_merged, 0, "{rep:?}");
    assert_eq!(f.index4[corner as usize], 1, "the corner keeps its single quad corner");
}

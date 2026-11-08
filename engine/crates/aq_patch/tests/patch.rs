//! DESIGN 6.12 tests for aq_patch (WP-13).
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

mod common;

use std::time::Instant;

use aq_core::math;
use aq_core::{CtlState, ErrKind, FeatureKind, QuadMesh, SplitMix64, Topology, Transition, TriMesh, V2, V3};
use aq_patch::{
    MapKind, PatchParams, count_flips, floater_mvc, mesh_patches, mesh_patches_full, mesh_patches_with, presplit,
    sd_barrier, sd_barrier_trace, sd_energy,
};
use common::*;

fn run(case: &Case, n: &[u32]) -> aq_patch::PatchOutput {
    let q = build_comb(&case.lay.tmesh, n);
    let st = CtlState::new();
    mesh_patches_full(&case.lay, &q, &[], &PatchParams::default(), &st.unbounded()).expect("mesh_patches")
}

/// Every quad corner Jacobian positive against the reference normals of the patch grids.
fn corner_jacobians_positive(out: &aq_patch::PatchOutput) -> bool {
    for g in &out.grids {
        let [w, h] = g.dims;
        if w == 0 || h == 0 {
            continue;
        }
        let at = |i: u32, j: u32| (j * (w + 1) + i) as usize;
        for j in 0..h {
            for i in 0..w {
                let c = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                for k in 0..4 {
                    let v = g.pos[c[k]];
                    let a = math::sub(g.pos[c[(k + 1) % 4]], v);
                    let b = math::sub(g.pos[c[(k + 3) % 4]], v);
                    if !(math::dot(math::cross(a, b), g.normal[c[k]]) > 0.0) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn is_manifold_quad_mesh(m: &QuadMesh) -> bool {
    m.validate().is_ok() && m.is_all_quads()
}

// ---------------------------------------------------------------------------------------
// Hemisphere disk patch with sides (8, 8).
// ---------------------------------------------------------------------------------------

fn hemisphere_case(seed: u64) -> Case {
    let mut spec = RectSpec::new(vec![[0, 0, 32, 32]], 0.25);
    spec.jitter = 0.2;
    spec.seed = seed;
    rect_layout(&spec, &hemisphere)
}

fn check_hemisphere(case: &Case, out: &aq_patch::PatchOutput) {
    let m = &out.mesh;
    assert_eq!(m.n_faces(), 64);
    assert_eq!(m.pos.len(), 81);
    assert!(is_manifold_quad_mesh(m));
    assert!(corner_jacobians_positive(out));
    // Every boundary vertex on its W' polyline within 1e-9.
    let lay = &case.lay;
    let mut n_bnd = 0;
    for (v, &fl) in m.vflag.iter().enumerate() {
        if fl & (1 << 26) == 0 {
            continue;
        }
        n_bnd += 1;
        let p = m.pos[v];
        let d = lay
            .arc_verts
            .iter()
            .flat_map(|vs| vs.windows(2))
            .map(|w| seg_dist(p, lay.wmesh.pos[w[0] as usize], lay.wmesh.pos[w[1] as usize]))
            .fold(f64::INFINITY, f64::min);
        assert!(d < 1e-9, "boundary vertex {v} is {d} away from its polyline");
    }
    assert_eq!(n_bnd, 32);
    // Interior vertices lie on the hemisphere's W' (radius about 5.09, within the chord error).
    let r = 16.0 / std::f64::consts::PI;
    for p in &m.pos {
        let d = math::norm(*p);
        assert!(d <= r + 1e-9 && d > r * 0.99, "vertex off the surface: {d}");
    }
}

#[test]
fn hemisphere_disk_patch() {
    let case = hemisphere_case(7);
    let n = round_n(&case.lay.tmesh, &[]);
    assert_eq!(n, vec![8, 8, 8, 8]);
    let out = run(&case, &n);
    check_hemisphere(&case, &out);
    assert_eq!(out.report.field_aligned, 1, "{:?}", out.report);
    // vflag: 4 corner nodes, 32 boundary vertices on chain 0 (Boundary).
    let m = &out.mesh;
    let corners = m.vflag.iter().filter(|&&f| f & (1 << 24) != 0).count();
    assert_eq!(corners, 4);
    for &f in &m.vflag {
        if f & (1 << 26) != 0 {
            assert_eq!(f & 0xFFFFF, 1);
            assert_eq!((f >> 20) & 0xF, FeatureKind::Boundary as u32);
        } else {
            assert_eq!(f, 0);
        }
    }
}

#[test]
fn hemisphere_with_transitions_inside_the_patch() {
    // Half of the faces live in another chart (quarter rotation plus translation); the map
    // must be the same as without the cut.
    let base = hemisphere_case(11);
    let n = round_n(&base.lay.tmesh, &[]);
    let ref_out = run(&base, &n);
    let mut cut = hemisphere_case(11);
    let uvs = cut.uv.clone();
    let tris = cut.lay.wmesh.tris.clone();
    let pick = move |f: u32| {
        let t = tris[f as usize];
        (uvs[t[0] as usize][0] + uvs[t[1] as usize][0] + uvs[t[2] as usize][0]) / 3.0 < 4.0
    };
    rechart(&mut cut.lay, &pick, Transition { r: 1, t: [8.0, -3.0] });
    cut.lay.validate(cut.lay.wmesh.tris.len()).unwrap();
    let out = run(&cut, &n);
    check_hemisphere(&cut, &out);
    for (a, b) in ref_out.mesh.pos.iter().zip(&out.mesh.pos) {
        assert!(math::dist(*a, *b) < 1e-9);
    }
    // Chart face in the moved region, too.
    let mut cut2 = hemisphere_case(11);
    let uvs = cut2.uv.clone();
    let tris = cut2.lay.wmesh.tris.clone();
    let pick2 = move |f: u32| {
        let t = tris[f as usize];
        (uvs[t[0] as usize][1] + uvs[t[1] as usize][1] + uvs[t[2] as usize][1]) / 3.0 < 2.0
    };
    rechart(&mut cut2.lay, &pick2, Transition { r: 3, t: [0.5, 2.0] });
    let out2 = run(&cut2, &n);
    for (a, b) in ref_out.mesh.pos.iter().zip(&out2.mesh.pos) {
        assert!(math::dist(*a, *b) < 1e-9);
    }
}

// ---------------------------------------------------------------------------------------
// Floater on random star-shaped disks, barrier on random folded maps.
// ---------------------------------------------------------------------------------------

struct RandDisk {
    mesh: TriMesh,
    topo: Topology,
    bnd: Vec<u32>,
}

/// Random star-shaped disk: a center vertex and K jittered rings of M vertices, quads
/// between rings split along random diagonals, a random height field.
fn random_disk(rng: &mut SplitMix64) -> RandDisk {
    let k = 1 + rng.below(6) as usize;
    let m = 8 + rng.below(33) as usize;
    let (amp, freq, ph) = (rng.range_f64(0.0, 0.45), 1 + rng.below(5), rng.range_f64(0.0, 6.3));
    let (hx, hy) = (rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0));
    let radius = |th: f64| 1.0 + amp * (freq as f64 * th + ph).sin();
    let mut pos = vec![[0.0, 0.0, 0.0]];
    for r in 1..=k {
        for i in 0..m {
            let th = std::f64::consts::TAU * (i as f64 + rng.range_f64(-0.3, 0.3)) / m as f64;
            let rr = (r as f64 + if r < k { rng.range_f64(-0.3, 0.3) } else { 0.0 }) / k as f64 * radius(th);
            let (x, y) = (rr * th.cos(), rr * th.sin());
            pos.push([x, y, 0.3 * (hx * x * x + hy * y * y + (3.0 * x).sin() * 0.2)]);
        }
    }
    let vid = |r: usize, i: usize| (1 + (r - 1) * m + i % m) as u32;
    let mut tris = Vec::new();
    for i in 0..m {
        tris.push([0, vid(1, i), vid(1, i + 1)]);
    }
    for r in 1..k {
        for i in 0..m {
            let (a, b, c, d) = (vid(r, i), vid(r + 1, i), vid(r + 1, i + 1), vid(r, i + 1));
            if rng.next_u64() & 1 == 0 {
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            } else {
                tris.push([a, b, d]);
                tris.push([b, c, d]);
            }
        }
    }
    let mesh = TriMesh { pos, tris };
    let topo = Topology::build(&mesh).unwrap();
    let bnd = (0..m).map(|i| vid(k, i)).collect();
    RandDisk { mesh, topo, bnd }
}

/// Boundary on the unit circle or on the square [0, 4]^2 (corners at quarter indices).
fn boundary_uv(d: &RandDisk, square: bool) -> Vec<(u32, V2)> {
    let m = d.bnd.len();
    d.bnd
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let t = i as f64 / m as f64;
            let uv = if square {
                let s = 16.0 * t;
                match (s / 4.0).floor() as u32 {
                    0 => [s, 0.0],
                    1 => [4.0, s - 4.0],
                    2 => [12.0 - s, 4.0],
                    _ => [0.0, 16.0 - s],
                }
            } else {
                let a = std::f64::consts::TAU * t;
                [a.cos(), a.sin()]
            };
            (v, uv)
        })
        .collect()
}

#[test]
fn floater_random_star_disks() {
    let mut rng = SplitMix64::new(2024);
    let mut faces = 0;
    for it in 0..1000 {
        let d = random_disk(&mut rng);
        let square = it % 2 == 1;
        let b = boundary_uv(&d, square);
        let uv = floater_mvc(&d.mesh, &d.topo, &b).unwrap();
        for &(v, p) in &b {
            assert_eq!(uv[v as usize], p);
        }
        assert_eq!(count_flips(&d.mesh, &uv), 0, "disk {it} has flipped triangles");
        faces += d.mesh.tris.len();
    }
    assert!(faces > 50_000);
}

#[test]
fn floater_rejects_missing_boundary() {
    let mut rng = SplitMix64::new(5);
    let d = random_disk(&mut rng);
    let mut b = boundary_uv(&d, false);
    b.pop();
    let e = floater_mvc(&d.mesh, &d.topo, &b).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
}

/// Random orientation-preserving targets with a strong twist (rotation varying with the
/// position) and anisotropy, in the C4 frame of each face.
fn twisted_targets(d: &RandDisk, rng: &mut SplitMix64) -> Vec<[[f64; 2]; 2]> {
    let twist = rng.range_f64(2.0, 6.0);
    let (s1, s2) = (rng.range_f64(0.5, 2.0), rng.range_f64(0.5, 2.0));
    d.mesh
        .tris
        .iter()
        .map(|t| {
            let c = [0, 1, 2].map(|k| d.mesh.pos[t[k] as usize]);
            let x = (c[0][0] + c[1][0] + c[2][0]) / 3.0;
            let y = (c[0][1] + c[1][1] + c[2][1]) / 3.0;
            let a = twist * (x * x + y * y).sqrt() + rng.range_f64(-0.3, 0.3);
            let (co, si) = (a.cos(), a.sin());
            [[co * s1, -si * s2], [si * s1, co * s2]]
        })
        .collect()
}

#[test]
fn barrier_random_folded_maps() {
    let mut rng = SplitMix64::new(77);
    let mut progressed = 0;
    for it in 0..200 {
        let d = random_disk(&mut rng);
        let b = boundary_uv(&d, it % 2 == 0);
        let fixed: Vec<bool> = {
            let mut f = vec![false; d.mesh.pos.len()];
            for &(v, _) in &b {
                f[v as usize] = true;
            }
            f
        };
        let targets = twisted_targets(&d, &mut rng);
        // A folded initial map: the boundary in place, interior vertices scattered.
        let mut uv = vec![[0.0, 0.0]; d.mesh.pos.len()];
        for &(v, p) in &b {
            uv[v as usize] = p;
        }
        for (v, x) in uv.iter_mut().enumerate() {
            if !fixed[v] {
                *x = [rng.range_f64(-1.0, 4.0), rng.range_f64(-1.0, 4.0)];
            }
        }
        // Make sure it is folded: swap two interior vertices if needed.
        if count_flips(&d.mesh, &uv) == 0 {
            uv[0] = [uv[0][0] + 50.0, uv[0][1]];
        }
        assert!(count_flips(&d.mesh, &uv) > 0, "case {it} is not folded");
        let tr = sd_barrier_trace(&d.mesh, &d.topo, &mut uv, &targets, &fixed, 20).unwrap();
        assert!(tr.restarted);
        assert_eq!(count_flips(&d.mesh, &uv), 0, "case {it} ends with flips");
        for w in tr.energies.windows(2) {
            assert!(w[1] < w[0], "case {it}: energy increased {} -> {}", w[0], w[1]);
        }
        assert_eq!(tr.energies.len() as u32, tr.iters + 1);
        let e_end = sd_energy(&d.mesh, &uv, &targets);
        assert!((e_end - tr.energies[tr.energies.len() - 1]).abs() <= 1e-9 * e_end);
        for &(v, p) in &b {
            assert_eq!(uv[v as usize], p, "fixed vertex moved");
        }
        if tr.iters > 0 {
            progressed += 1;
        }
    }
    assert!(progressed >= 190, "only {progressed} of 200 runs decreased the energy");
}

#[test]
fn barrier_from_valid_map_keeps_it_valid() {
    let mut rng = SplitMix64::new(3);
    for _ in 0..20 {
        let d = random_disk(&mut rng);
        let b = boundary_uv(&d, true);
        let mut uv = floater_mvc(&d.mesh, &d.topo, &b).unwrap();
        let mut fixed = vec![false; d.mesh.pos.len()];
        for &(v, _) in &b {
            fixed[v as usize] = true;
        }
        let targets = twisted_targets(&d, &mut rng);
        let e0 = sd_energy(&d.mesh, &uv, &targets);
        let it = sd_barrier(&d.mesh, &d.topo, &mut uv, &targets, &fixed, 20).unwrap();
        assert!(it <= 20);
        assert_eq!(count_flips(&d.mesh, &uv), 0);
        assert!(sd_energy(&d.mesh, &uv, &targets) <= e0);
    }
}

// ---------------------------------------------------------------------------------------
// Two patches with a T-junction; zero arcs and zero-height patches.
// ---------------------------------------------------------------------------------------

fn bump(p: V2) -> V3 {
    [p[0], p[1], 0.4 * (0.7 * p[0]).sin() * (0.5 * p[1]).cos()]
}

fn assert_no_duplicates(m: &QuadMesh) {
    let mut ps: Vec<(V3, usize)> = m.pos.iter().copied().zip(0..).collect();
    ps.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
    for i in 0..ps.len() {
        for j in i + 1..ps.len() {
            if ps[j].0[0] - ps[i].0[0] > 1e-6 {
                break;
            }
            assert!(math::dist(ps[i].0, ps[j].0) > 1e-6, "duplicate vertices {} and {}", ps[i].1, ps[j].1);
        }
    }
}

#[test]
fn two_patches_with_a_t_junction() {
    // Bottom patch 8 x 2, top patch 4 x 4 on its left half: node (4, 2) is a T-node of the
    // bottom patch's top side.
    let mut spec = RectSpec::new(vec![[0, 0, 32, 8], [0, 8, 16, 24]], 0.25);
    spec.jitter = 0.2;
    spec.seed = 9;
    let case = rect_layout(&spec, &bump);
    let n = round_n(&case.lay.tmesh, &[]);
    let out = run(&case, &n);
    let m = &out.mesh;
    assert_eq!(m.n_faces(), 32);
    assert_eq!(m.pos.len(), 9 * 3 + 5 * 5 - 5);
    assert!(is_manifold_quad_mesh(m));
    assert_no_duplicates(m);
    assert!(corner_jacobians_positive(&out));
    // Euler characteristic of a disk.
    assert_eq!(m.pos.len() as i64 - n_edges(m) as i64 + m.n_faces() as i64, 1);
    // The T-node is a regular boundary vertex of the quad mesh with 3 quads.
    let tnode = case.node_uv.iter().position(|p| *p == [4.0, 2.0]).unwrap();
    let q = build_comb(&case.lay.tmesh, &n);
    let id = q.node_id[tnode];
    let deg = (0..m.n_faces()).filter(|&f| m.face(f).contains(&id)).count();
    assert_eq!(deg, 3);
}

#[test]
fn zero_height_patch_is_glued() {
    // Bottom 4 x 2, thin 4 x 0.25 (quantized to height 0), top 4 x 1.75 (height 2).
    let mut spec = RectSpec::new(vec![[0, 0, 16, 8], [0, 8, 16, 9], [0, 9, 16, 16]], 0.25);
    spec.jitter = 0.2;
    let case = rect_layout(&spec, &bump);
    let n = round_n(&case.lay.tmesh, &[]);
    assert!(n.contains(&0));
    let out = run(&case, &n);
    let m = &out.mesh;
    assert_eq!(out.report.zero_patches, 1);
    assert_eq!(m.n_faces(), 16);
    assert_eq!(m.pos.len(), 25);
    assert_eq!(n_edges(m), 40);
    assert!(is_manifold_quad_mesh(m));
    assert!(corner_jacobians_positive(&out));
    assert_eq!(out.grids[1].kind, None);
}

#[test]
fn zero_arc_t_offsets() {
    // Two rows of 2 patches whose separators are offset by 0.25: the short arc between the
    // two T-nodes is quantized to 0 and merges them. Case B: the zero arc ends the sides it
    // lies on; case C: it starts them.
    for split in [9i64, 7] {
        let rects = vec![[0, 0, 8, 8], [8, 0, 16, 8], [0, 8, split, 16], [split, 8, 16, 16]];
        let mut spec = RectSpec::new(rects, 0.25);
        spec.jitter = 0.2;
        spec.seed = split as u64;
        let case = rect_layout(&spec, &bump);
        let n = round_n(&case.lay.tmesh, &[]);
        assert_eq!(n.iter().filter(|&&x| x == 0).count(), 1);
        let out = run(&case, &n);
        let m = &out.mesh;
        assert_eq!(out.report.zero_patches, 0);
        assert_eq!(m.n_faces(), 16, "split {split}");
        assert_eq!(m.pos.len(), 25, "split {split}");
        assert_eq!(n_edges(m), 40, "split {split}");
        assert!(is_manifold_quad_mesh(m));
        assert_no_duplicates(m);
        assert!(corner_jacobians_positive(&out));
        // The merged vertex is the average of the two T-nodes.
        let a = case.node_uv.iter().position(|p| *p == [2.0, 2.0]).unwrap();
        let b = case.node_uv.iter().position(|p| *p == [split as f64 * 0.25, 2.0]).unwrap();
        let q = build_comb(&case.lay.tmesh, &n);
        assert_eq!(q.node_id[a], q.node_id[b]);
        let pa = case.lay.wmesh.pos[case.lay.node_vert[a] as usize];
        let pb = case.lay.wmesh.pos[case.lay.node_vert[b] as usize];
        let mid = math::scale(math::add(pa, pb), 0.5);
        assert!(math::dist(m.pos[q.node_id[a] as usize], mid) < 1e-15);
    }
}

// ---------------------------------------------------------------------------------------
// Shared arc samples, pre-split, cylinder, barrier in the pipeline.
// ---------------------------------------------------------------------------------------

#[test]
fn shared_arc_samples_are_bitwise_identical() {
    let mut spec = RectSpec::new(vec![[0, 0, 22, 26], [22, 0, 41, 26]], 0.125);
    spec.jitter = 0.2;
    spec.seed = 4;
    let lift = cylinder(2.0);
    let case = rect_layout(&spec, &lift);
    let n = round_n(&case.lay.tmesh, &[]);
    let out = run(&case, &n);
    let t = &case.lay.tmesh;
    // The shared arc: side 1 of patch 0, side 3 of patch 1.
    let (a, rev0) = t.patches[0].sides[1][0];
    let (a1, rev1) = t.patches[1].sides[3][0];
    assert_eq!(a, a1);
    assert_ne!(rev0, rev1);
    let g0 = &out.grids[0];
    let g1 = &out.grids[1];
    let (w0, h) = (g0.dims[0], g0.dims[1]);
    assert_eq!(h, g1.dims[1]);
    let w1 = g1.dims[0];
    let smp = &out.samples[a as usize];
    assert_eq!(smp.pos.len() as u32, h + 1);
    for j in 0..=h {
        let p0 = g0.pos[(j * (w0 + 1) + w0) as usize];
        let p1 = g1.pos[(j * (w1 + 1)) as usize];
        let id0 = g0.ids[(j * (w0 + 1) + w0) as usize];
        let id1 = g1.ids[(j * (w1 + 1)) as usize];
        assert_eq!(id0, id1);
        assert_eq!(p0.map(f64::to_bits), p1.map(f64::to_bits));
        let s = if rev0 { smp.pos[(h - j) as usize] } else { smp.pos[j as usize] };
        assert_eq!(p0.map(f64::to_bits), s.map(f64::to_bits));
        assert_eq!(out.mesh.pos[id0 as usize].map(f64::to_bits), s.map(f64::to_bits));
    }
    // Samples lie on the W' polyline at the right parametric length.
    let s = &case.lay.arc_s[a as usize];
    let len = s[s.len() - 1];
    for (j, &(va, vb, tt)) in smp.seg.iter().enumerate() {
        let i = case.lay.arc_verts[a as usize].iter().position(|&v| v == va).unwrap();
        assert_eq!(case.lay.arc_verts[a as usize][i + 1], vb);
        let sj = s[i] + tt * (s[i + 1] - s[i]);
        assert!((sj - j as f64 * len / h as f64).abs() < 1e-12);
    }
    assert!(is_manifold_quad_mesh(&out.mesh));
    assert!(corner_jacobians_positive(&out));
}

#[test]
fn presplit_boundary_only_triangles() {
    // Square disk whose boundary vertex (1, 0) bulges outward in 3D: the ear
    // (0,0) (1,0) (2,0) has its 3 vertices on side 0 (collinear in the rectangle).
    let uv: Vec<V2> = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [1.0, 2.0],
        [0.0, 2.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let mut pos: Vec<V3> = uv.iter().map(|p| [p[0], p[1], 0.0]).collect();
    pos[1] = [1.0, -0.3, 0.0];
    let tris = vec![
        [0, 1, 2],
        [0, 2, 8],
        [2, 3, 8],
        [3, 4, 8],
        [4, 5, 8],
        [5, 6, 8],
        [6, 7, 8],
        [7, 0, 8],
    ];
    let mesh = TriMesh { pos, tris };
    let topo = Topology::build(&mesh).unwrap();
    let on_b: Vec<bool> = (0..9).map(|v| v != 8).collect();
    let ps = presplit(&mesh, &topo, &uv, &on_b).unwrap();
    // Only (0, 2) is an interior edge joining two boundary vertices.
    assert_eq!(ps.n_split, 1);
    assert_eq!(ps.mesh.tris.len(), 10);
    assert_eq!(ps.mesh.pos.len(), 10);
    let flag = |v: u32| (v as usize) < 9 && on_b[v as usize];
    for t in &ps.mesh.tris {
        assert!(!(flag(t[0]) && flag(t[1]) && flag(t[2])), "boundary-only triangle {t:?} left");
    }
    assert_eq!(ps.uv[9], [1.0, 0.0]);
    assert_eq!(ps.mesh.pos[9], [1.0, 0.0, 0.0]);
    for (f, &p) in ps.parent.iter().enumerate() {
        let a = mesh.face_normal(p).unwrap();
        let b = ps.mesh.face_normal(f as u32).unwrap();
        assert!(math::dot(a, b) > 1.0 - 1e-12);
    }
    // Floater on the split disk with the rectangle boundary: no flip (without the split, the
    // ear is degenerate).
    let pt = Topology::build(&ps.mesh).unwrap();
    let b: Vec<(u32, V2)> = (0..8).map(|v| (v, uv[v as usize])).collect();
    let fl = floater_mvc(&ps.mesh, &pt, &b).unwrap();
    assert_eq!(count_flips(&ps.mesh, &fl), 0);
    let fl0 = floater_mvc(&mesh, &topo, &b).unwrap();
    assert!(count_flips(&mesh, &fl0) > 0);
}

#[test]
fn boundary_only_triangles_in_the_pipeline() {
    // 2 x 2 cells of size 1, every diagonal avoiding the center: 4 ears.
    let away = |ci: i64, cj: i64| (ci + cj) % 2 == 1;
    let mut spec = RectSpec::new(vec![[0, 0, 2, 2]], 1.0);
    spec.diag = Some(&away);
    let case = rect_layout(&spec, &bump);
    let out = run(&case, &[2, 2, 2, 2]);
    assert_eq!(out.report.split_edges, 4);
    assert_eq!(out.mesh.n_faces(), 4);
    assert!(corner_jacobians_positive(&out));
    // A strip one cell thick: every triangle has its 3 vertices on the boundary.
    let spec = RectSpec::new(vec![[0, 0, 3, 1]], 1.0);
    let case = rect_layout(&spec, &bump);
    let n = round_n(&case.lay.tmesh, &[]);
    let side1 = case.lay.tmesh.patches[0].sides[1][0].0;
    let side3 = case.lay.tmesh.patches[0].sides[3][0].0;
    let n = {
        let mut n = n;
        n[side1 as usize] = 2;
        n[side3 as usize] = 2;
        n
    };
    let out = run(&case, &n);
    // 2 inner rungs plus 3 diagonals.
    assert_eq!(out.report.split_edges, 2 + 3);
    assert_eq!(out.mesh.n_faces(), 6);
    assert!(is_manifold_quad_mesh(&out.mesh));
    assert!(corner_jacobians_positive(&out));
}

#[test]
fn cylinder_patch_field_aligned_corner_angles() {
    // 6.375 x 4.375 sector of a cylinder of radius 3, quantized to 6 x 4 (stretch), jittered.
    let mut spec = RectSpec::new(vec![[0, 0, 51, 35]], 0.125);
    spec.jitter = 0.2;
    spec.seed = 31;
    let lift = cylinder(3.0);
    let case = rect_layout(&spec, &lift);
    let n = round_n(&case.lay.tmesh, &[]);
    assert_eq!(n.iter().max(), Some(&6));
    let out = run(&case, &n);
    assert_eq!(out.report.field_aligned, 1);
    assert_eq!(out.mesh.n_faces(), 24);
    let dev = mean_corner_dev_deg(&out.mesh);
    assert!(dev < 3.0, "mean corner angle deviation {dev} degrees");
    assert!(corner_jacobians_positive(&out));
}

/// An 8 x 8 patch whose seamless uv is twisted about the center by `amp * exp(-r^2 / width)`
/// radians (a bad param: the targets rotate inside the patch).
fn twisted_case(amp: f64, width: f64) -> Case {
    let mut spec = RectSpec::new(vec![[0, 0, 32, 32]], 0.25);
    spec.jitter = 0.1;
    let case0 = rect_layout(&spec, &bump);
    let mut lay = case0.lay.clone();
    let twist = |p: V2| -> V2 {
        let (x, y) = (p[0] - 4.0, p[1] - 4.0);
        let r = (x * x + y * y).sqrt();
        let a = amp * (-r * r / width).exp();
        let (c, s) = (a.cos(), a.sin());
        [4.0 + c * x - s * y, 4.0 + s * x + c * y]
    };
    for f in 0..lay.wuv.len() {
        for k in 0..3 {
            lay.wuv[f][k] = twist(lay.wuv[f][k]);
        }
    }
    Case { lay, node_uv: case0.node_uv, uv: case0.uv }
}

#[test]
fn folded_least_squares_goes_to_the_barrier() {
    // A twist of 4 rad: the field-aligned map flips triangles, the barrier map is kept.
    let case = twisted_case(4.0, 2.0);
    let n = round_n(&case.lay.tmesh, &[]);
    let out = run(&case, &n);
    assert_eq!(out.report.field_aligned, 0, "{:?}", out.report);
    assert_eq!(out.report.barrier, 1, "{:?}", out.report);
    assert_eq!(out.report.remeshed, 0);
    assert_eq!(out.grids[0].kind, Some(MapKind::Barrier));
    assert_eq!(out.mesh.n_faces(), 64);
    assert!(corner_jacobians_positive(&out));
    assert!(is_manifold_quad_mesh(&out.mesh));
}

#[test]
fn failed_final_check_remeshes_with_floater() {
    // A wide twist of 2.5 rad: the field-aligned map is injective but its sampled quads are
    // not all positive against the surface normal, so the patch is re-meshed with Floater.
    let case = twisted_case(2.5, 4.0);
    let n = round_n(&case.lay.tmesh, &[]);
    let out = run(&case, &n);
    assert_eq!(out.report.remeshed, 1, "{:?}", out.report);
    assert_eq!(out.report.floater, 1);
    assert_eq!(out.grids[0].kind, Some(MapKind::Floater));
    assert_eq!(out.mesh.n_faces(), 64);
    assert!(corner_jacobians_positive(&out));
    assert!(is_manifold_quad_mesh(&out.mesh));
}

// ---------------------------------------------------------------------------------------
// Determinism, errors, cancel, performance.
// ---------------------------------------------------------------------------------------

fn grid_case(k: i64, cells: i64, cell: f64, seed: u64) -> Case {
    let mut rects = Vec::new();
    for j in 0..k {
        for i in 0..k {
            rects.push([i * cells, j * cells, (i + 1) * cells, (j + 1) * cells]);
        }
    }
    let mut spec = RectSpec::new(rects, cell);
    spec.jitter = 0.2;
    spec.seed = seed;
    rect_layout(&spec, &|p: V2| [p[0], p[1], 3.0 * (p[0] / 10.0).sin() * (p[1] / 13.0).cos()])
}

#[test]
fn deterministic_across_thread_counts() {
    let case = grid_case(4, 12, 0.25, 8);
    let n = round_n(&case.lay.tmesh, &[]);
    let q = build_comb(&case.lay.tmesh, &n);
    let mut results = Vec::new();
    for threads in [1, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let st = CtlState::new();
        let r = pool.install(|| mesh_patches(&case.lay, &q, &PatchParams::default(), &st.unbounded())).unwrap();
        results.push(r);
    }
    for r in &results[1..] {
        assert_eq!(r.0, results[0].0);
        assert_eq!(r.1, results[0].1);
        for (a, b) in r.0.pos.iter().zip(&results[0].0.pos) {
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        }
    }
}

#[test]
fn errors_and_cancel() {
    let case = grid_case(2, 8, 0.25, 1);
    let n = round_n(&case.lay.tmesh, &[]);
    let q = build_comb(&case.lay.tmesh, &n);
    let st = CtlState::new();
    st.cancel();
    let e = mesh_patches(&case.lay, &q, &PatchParams::default(), &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
    assert_eq!(e.stage, aq_core::Stage::Patch);
    // A QComb of another T-mesh.
    let other = grid_case(1, 8, 0.25, 1);
    let q2 = build_comb(&other.lay.tmesh, &round_n(&other.lay.tmesh, &[]));
    let st = CtlState::new();
    let e = mesh_patches(&case.lay, &q2, &PatchParams::default(), &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    // Bad parameters.
    let prm = PatchParams { sd_iters: 20, eps_zero: 0.0 };
    let e = mesh_patches(&case.lay, &q, &prm, &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    // Chain kinds override.
    let (m, _) = mesh_patches_with(&case.lay, &q, &[FeatureKind::Symmetry], &PatchParams::default(), &st.unbounded())
        .unwrap();
    let b = m.vflag.iter().find(|&&f| f & (1 << 26) != 0).unwrap();
    assert_eq!((b >> 20) & 0xF, FeatureKind::Symmetry as u32);
}

#[test]
fn t_offset_rows_on_a_bumpy_surface() {
    // Three rows of 2 patches with T-offsets (heights 3, 2 and 2.5 rounded to 3), on a bumpy
    // surface.
    let rects = vec![
        [0, 0, 8, 12],
        [8, 0, 24, 12],
        [0, 12, 12, 20],
        [12, 12, 24, 20],
        [0, 20, 20, 30],
        [20, 20, 24, 30],
    ];
    let mut spec = RectSpec::new(rects, 0.25);
    spec.jitter = 0.2;
    let case = rect_layout(&spec, &bump);
    let n = round_n(&case.lay.tmesh, &[]);
    let out = run(&case, &n);
    assert!(is_manifold_quad_mesh(&out.mesh));
    assert!(corner_jacobians_positive(&out));
    let m = &out.mesh;
    assert_eq!(m.pos.len() as i64 - n_edges(m) as i64 + m.n_faces() as i64, 1);
}

#[test]
fn performance_5k_quads_from_100k_faces() {
    if cfg!(debug_assertions) {
        eprintln!("performance test skipped in debug builds");
        return;
    }
    // 10 x 10 patches of 7 x 7 quads; W' = 220 x 220 cells = 96800 triangles.
    let case = grid_case(10, 22, 7.0 / 22.0, 3);
    let nf = case.lay.wmesh.tris.len();
    assert!(nf >= 96_000);
    let n = round_n(&case.lay.tmesh, &[]);
    let q = build_comb(&case.lay.tmesh, &n);
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..3 {
        let st = CtlState::new();
        let t0 = Instant::now();
        let r = mesh_patches(&case.lay, &q, &PatchParams::default(), &st.unbounded()).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        out = Some(r);
    }
    let (m, rep) = out.unwrap();
    assert_eq!(m.n_faces(), 4900);
    eprintln!("aq_patch: {} quads from {nf} W' faces in {best:.3} s ({rep:?})", m.n_faces());
    assert!(best < 1.5, "took {best} s");
}

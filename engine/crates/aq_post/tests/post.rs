//! DESIGN 6.13 tests for aq_post (WP-14).
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;

use aq_core::math;
use aq_core::{Bvh, CtlState, ErrKind, FeatureKind, NONE, QuadMesh, SplitMix64, TriMesh, V3};
use aq_post::{
    CleanupParams, PostParams, PostProxy, VF_BOUNDARY, VF_CONE, VF_CORNER, cleanup, cleanup_with,
    dist_to_polyline, relax,
};
use aq_testkit::{Surface, icosphere};

// ---------------------------------------------------------------- helpers

/// Proxy grid over [x0, x0 + nx*cell] x [y0, y0 + ny*cell] mapped by `f`, split as
/// [a, b, c], [a, c, d] per cell, with its PL evaluator.
struct GridProxy {
    nx: u32,
    ny: u32,
    cell: f64,
    x0: f64,
    y0: f64,
    mesh: TriMesh,
}

impl GridProxy {
    fn new(nx: u32, ny: u32, cell: f64, x0: f64, y0: f64, f: impl Fn(f64, f64) -> V3) -> Self {
        let mut pos = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                pos.push(f(x0 + i as f64 * cell, y0 + j as f64 * cell));
            }
        }
        let id = |i: u32, j: u32| j * (nx + 1) + i;
        let mut tris = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        GridProxy { nx, ny, cell, x0, y0, mesh: TriMesh { pos, tris } }
    }

    fn vid(&self, i: u32, j: u32) -> u32 {
        j * (self.nx + 1) + i
    }

    /// The PL surface at domain point (x, y).
    fn eval(&self, x: f64, y: f64) -> V3 {
        let u = (x - self.x0) / self.cell;
        let v = (y - self.y0) / self.cell;
        let i = (u.floor().max(0.0) as u32).min(self.nx - 1);
        let j = (v.floor().max(0.0) as u32).min(self.ny - 1);
        let s = u - i as f64;
        let t = v - j as f64;
        let p = |a: u32| self.mesh.pos[a as usize];
        let (a, b, c, d) = (self.vid(i, j), self.vid(i + 1, j), self.vid(i + 1, j + 1), self.vid(i, j + 1));
        if s >= t {
            // [a, b, c]: a + s (b - a) + t (c - b)
            let w = [1.0 - s, s - t, t];
            math::bary_point(p(a), p(b), p(c), w)
        } else {
            // [a, c, d]: a + s (c - d) + t (d - a)
            let w = [1.0 - t, s, t - s];
            math::bary_point(p(a), p(c), p(d), w)
        }
    }
}

/// Quad grid (nx x ny cells), vertex (i, j) = j*(nx+1)+i at `f(i, j)`, CCW seen from +z.
fn quad_grid(nx: u32, ny: u32, mut f: impl FnMut(u32, u32) -> V3) -> QuadMesh {
    let mut q = QuadMesh::new();
    for j in 0..=ny {
        for i in 0..=nx {
            q.pos.push(f(i, j));
            q.vflag.push(0);
        }
    }
    let id = |i: u32, j: u32| j * (nx + 1) + i;
    for j in 0..ny {
        for i in 0..nx {
            q.push_face(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], 0);
        }
    }
    q
}

fn edges(q: &QuadMesh) -> Vec<(u32, u32)> {
    let mut e = Vec::new();
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        for k in 0..fv.len() {
            let (a, b) = (fv[k], fv[(k + 1) % fv.len()]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.sort_unstable();
    e.dedup();
    e
}

fn edge_cv(q: &QuadMesh) -> f64 {
    let l: Vec<f64> =
        edges(q).iter().map(|&(a, b)| math::dist(q.pos[a as usize], q.pos[b as usize])).collect();
    let mean = l.iter().sum::<f64>() / l.len() as f64;
    let var = l.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / l.len() as f64;
    var.sqrt() / mean
}

/// Smallest cosine between a quad corner normal and `nref(face centroid)`.
fn min_corner_cos(q: &QuadMesh, nref: impl Fn(V3) -> V3) -> f64 {
    let mut m = f64::INFINITY;
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        let n = fv.len();
        let mut c = [0.0; 3];
        for &v in fv {
            c = math::add(c, q.pos[v as usize]);
        }
        let r = nref(math::scale(c, 1.0 / n as f64));
        for k in 0..n {
            let p = q.pos[fv[k] as usize];
            let a = q.pos[fv[(k + 1) % n] as usize];
            let b = q.pos[fv[(k + n - 1) % n] as usize];
            let cn = math::cross(math::sub(a, p), math::sub(b, p));
            m = m.min(math::dot(cn, r) / math::norm(cn));
        }
    }
    m
}

/// Faces per vertex.
fn valences(q: &QuadMesh) -> Vec<u32> {
    let mut v = vec![0u32; q.n_verts()];
    for &w in &q.face_verts {
        v[w as usize] += 1;
    }
    v
}

fn chain_points(s: &Surface, ci: usize) -> Vec<V3> {
    let c = &s.feats.chains[ci];
    let mut pts = vec![s.mesh.pos[s.topo.tail(&s.mesh, c.hes[0]) as usize]];
    for &h in &c.hes {
        pts.push(s.mesh.pos[s.topo.head(&s.mesh, h) as usize]);
    }
    pts
}

/// Planar test bed: proxy = flat square [0, n]^2 (cells of 0.5) with its boundary chain, quad
/// grid of unit cells. Interior vertices jittered by up to `jit`, boundary vertices slid along
/// the boundary by up to `jit`, the 4 corners fixed. Returns (surface, quads, chain_of_vert).
fn planar_case(n: u32, jit: f64, seed: u64) -> (Surface, QuadMesh, Vec<u32>) {
    let gp = GridProxy::new(2 * n, 2 * n, 0.5, 0.0, 0.0, |x, y| [x, y, 0.0]);
    let s = Surface::new(gp.mesh).unwrap();
    assert_eq!(s.feats.chains.len(), 1, "one boundary chain");
    let mut rng = SplitMix64::new(seed);
    let nf = n as f64;
    let mut q = quad_grid(n, n, |i, j| {
        let jx = jit * rng.range_f64(-1.0, 1.0);
        let jy = jit * rng.range_f64(-1.0, 1.0);
        let mut x = i as f64;
        let mut y = j as f64;
        if i > 0 && i < n {
            x += jx;
        }
        if j > 0 && j < n {
            y += jy;
        }
        [x.clamp(0.0, nf), y.clamp(0.0, nf), 0.0]
    });
    let mut chain = vec![NONE; q.n_verts()];
    for j in 0..=n {
        for i in 0..=n {
            let v = (j * (n + 1) + i) as usize;
            let bi = i == 0 || i == n;
            let bj = j == 0 || j == n;
            if bi || bj {
                chain[v] = 0;
                q.vflag[v] |= VF_BOUNDARY | 1;
            }
            if bi && bj {
                q.vflag[v] |= VF_CORNER;
            }
        }
    }
    (s, q, chain)
}

fn run_relax(s: &Surface, q: &mut QuadMesh, chain: &[u32], p: &PostParams) -> aq_post::RelaxReport {
    let bvh = Bvh::build(&s.mesh);
    let px = PostProxy::new(&s.mesh, &s.topo, &s.frames, &s.feats, &bvh);
    let st = CtlState::new();
    relax(q, &px, chain, p, &st.unbounded()).unwrap()
}

// ---------------------------------------------------------------- relaxation

#[test]
fn jittered_planar_grid_edge_cv_halves() {
    let (s, mut q, chain) = planar_case(16, 0.3, 7);
    let cv0 = edge_cv(&q);
    let rep = run_relax(&s, &mut q, &chain, &PostParams::default());
    let cv1 = edge_cv(&q);
    println!("edge cv {cv0:.4} -> {cv1:.4}, {rep:?}");
    assert!(cv0 > 0.1, "jitter too small: {cv0}");
    assert!(cv1 <= 0.5 * cv0, "edge CV {cv0} -> {cv1}");
    q.validate().unwrap();
    assert!(min_corner_cos(&q, |_| [0.0, 0.0, 1.0]) > 0.5);
    for v in 0..q.n_verts() {
        assert!(q.pos[v][2].abs() < 1e-12);
    }
    assert_eq!(rep.iters, 8);
    assert!(rep.chain_verts > 0 && rep.interior_verts > 0);
}

#[test]
fn perfect_grid_moves_less_than_1e_12() {
    let (s, mut q, chain) = planar_case(12, 0.0, 1);
    let before = q.pos.clone();
    let rep = run_relax(&s, &mut q, &chain, &PostParams::default());
    let mut m = 0.0f64;
    for v in 0..q.n_verts() {
        m = m.max(math::dist(before[v], q.pos[v]));
    }
    println!("max move {m:e}, {rep:?}");
    assert!(m < 1e-12, "perfect grid moved by {m}");
}

/// Creased, warped surface with a curved Hard chain across it and two boundary chains.
#[test]
fn chain_vertices_stay_on_polylines() {
    let n = 16u32;
    let warp = |x: f64, y: f64| [x, y + 0.6 * (0.4 * x).sin(), 0.3 * y.abs()];
    let gp = GridProxy::new(2 * n, 2 * n, 0.5, 0.0, -8.0, warp);
    let crease_j = n; // proxy row y = 0
    let feat: Vec<([u32; 2], FeatureKind)> = (0..2 * n)
        .map(|i| ([gp.vid(i, crease_j), gp.vid(i + 1, crease_j)], FeatureKind::Hard))
        .collect();
    let s = Surface::with_features(gp.mesh.clone(), &feat, 50.0, 70.0).unwrap();
    let polys: Vec<Vec<V3>> = (0..s.feats.chains.len()).map(|c| chain_points(&s, c)).collect();
    let hard = s.feats.chains.iter().position(|c| c.kind == FeatureKind::Hard).unwrap();

    let mut rng = SplitMix64::new(11);
    let nf = n as f64;
    let mid = n / 2;
    let mut dom = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let mut x = i as f64;
            let mut y = j as f64 - 8.0;
            let jx = 0.3 * rng.range_f64(-1.0, 1.0);
            let jy = 0.3 * rng.range_f64(-1.0, 1.0);
            let bi = i == 0 || i == n;
            let bj = j == 0 || j == n;
            if !bi {
                x += jx;
            }
            if !bj && j != mid {
                y += jy;
            }
            dom.push([x.clamp(0.0, nf), y.clamp(-8.0, 8.0)]);
        }
    }
    let mut q = quad_grid(n, n, |i, j| {
        let d = dom[(j * (n + 1) + i) as usize];
        gp.eval(d[0], d[1])
    });
    let mut chain = vec![NONE; q.n_verts()];
    for j in 0..=n {
        for i in 0..=n {
            let v = (j * (n + 1) + i) as usize;
            let bi = i == 0 || i == n;
            let bj = j == 0 || j == n;
            let on_crease = j == mid;
            if !(bi || bj || on_crease) {
                continue;
            }
            let p = q.pos[v];
            let ci = if on_crease && !bi {
                hard
            } else {
                (0..polys.len())
                    .filter(|&c| c != hard)
                    .find(|&c| dist_to_polyline(p, &polys[c]) < 1e-12)
                    .unwrap()
            };
            assert!(dist_to_polyline(p, &polys[ci]) < 1e-12);
            chain[v] = ci as u32;
            q.vflag[v] |= ci as u32 + 1;
            if bi || bj {
                q.vflag[v] |= VF_BOUNDARY;
            }
            if (bi && bj) || (bi && on_crease) {
                q.vflag[v] |= VF_CORNER;
            }
        }
    }
    let before = q.pos.clone();
    let rep = run_relax(&s, &mut q, &chain, &PostParams::default());
    println!("{rep:?}");
    let mut worst = 0.0f64;
    let mut moved = 0;
    for v in 0..q.n_verts() {
        if chain[v] == NONE {
            continue;
        }
        let d = dist_to_polyline(q.pos[v], &polys[chain[v] as usize]);
        worst = worst.max(d);
        if math::dist(before[v], q.pos[v]) > 1e-3 {
            moved += 1;
        }
    }
    println!("worst chain distance {worst:e}, moved chain vertices {moved}");
    assert!(worst < 1e-9, "chain vertex off its polyline by {worst}");
    assert!(moved > 10, "chain vertices did not slide");
    // Corners did not move.
    for v in 0..q.n_verts() {
        if q.vflag[v] & VF_CORNER != 0 {
            assert_eq!(before[v], q.pos[v]);
        }
    }
    q.validate().unwrap();
    // No corner turned past 60 degrees from the proxy normal.
    let bvh = Bvh::build(&s.mesh);
    let m = min_corner_cos(&q, |c| s.frames.n[bvh.closest(c).0.face as usize]);
    assert!(m > 0.5, "min corner cos {m}");
}

/// Cube-sphere quad mesh (n x n per cube face) on a sphere of radius r, outward CCW faces.
fn cube_sphere_quads(n: u32, r: f64) -> QuadMesh {
    let mut ids: BTreeMap<[u32; 3], u32> = BTreeMap::new();
    let mut q = QuadMesh::new();
    let mut vid = |k: [u32; 3], q: &mut QuadMesh| -> u32 {
        if let Some(&i) = ids.get(&k) {
            return i;
        }
        let c = k.map(|x| -1.0 + 2.0 * x as f64 / n as f64);
        let p = math::scale(math::normalize(c).unwrap(), r);
        q.pos.push(p);
        q.vflag.push(0);
        let i = q.pos.len() as u32 - 1;
        ids.insert(k, i);
        i
    };
    for axis in 0..3 {
        for side in [0u32, n] {
            for j in 0..n {
                for i in 0..n {
                    let key = |a: u32, b: u32| {
                        let mut k = [0u32; 3];
                        k[axis] = side;
                        k[(axis + 1) % 3] = a;
                        k[(axis + 2) % 3] = b;
                        k
                    };
                    let mut f = [
                        vid(key(i, j), &mut q),
                        vid(key(i + 1, j), &mut q),
                        vid(key(i + 1, j + 1), &mut q),
                        vid(key(i, j + 1), &mut q),
                    ];
                    let p: Vec<V3> = f.iter().map(|&v| q.pos[v as usize]).collect();
                    let nn = math::cross(math::sub(p[2], p[0]), math::sub(p[3], p[1]));
                    let c = math::add(math::add(p[0], p[1]), math::add(p[2], p[3]));
                    if math::dot(nn, c) < 0.0 {
                        f.reverse();
                    }
                    q.push_face(&f, 0);
                }
            }
        }
    }
    q
}

fn sphere_case(seed: u64) -> (Surface, QuadMesh) {
    let r = 5.0;
    let s = Surface::new(icosphere(5, r).unwrap()).unwrap();
    let mut q = cube_sphere_quads(8, r);
    let mut rng = SplitMix64::new(seed);
    for p in q.pos.iter_mut() {
        let d = [rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0)];
        let nrm = math::normalize(*p).unwrap();
        let t = math::sub(d, math::scale(nrm, math::dot(d, nrm)));
        *p = math::scale(math::normalize(math::add(*p, math::scale(t, 0.3))).unwrap(), r);
    }
    (s, q)
}

#[test]
fn sphere_output_has_no_inversions() {
    let (s, mut q) = sphere_case(3);
    q.validate().unwrap();
    let radial = |c: V3| math::normalize(c).unwrap();
    let m0 = min_corner_cos(&q, radial);
    let cv0 = edge_cv(&q);
    let chain = vec![NONE; q.n_verts()];
    let rep = run_relax(&s, &mut q, &chain, &PostParams::default());
    let m1 = min_corner_cos(&q, radial);
    let cv1 = edge_cv(&q);
    println!("min corner cos {m0:.3} -> {m1:.3}, edge cv {cv0:.3} -> {cv1:.3}, {rep:?}");
    assert!(m1 > 0.0, "inverted corner after relax: {m1}");
    assert!(m1 > 0.5);
    assert!(cv1 < cv0);
    q.validate().unwrap();
    // Every vertex lies on the proxy.
    let bvh = Bvh::build(&s.mesh);
    for p in &q.pos {
        assert!(bvh.closest(*p).2 < 1e-9);
    }
}

#[test]
fn relax_is_deterministic_across_thread_counts() {
    let mut outs = Vec::new();
    for threads in [1usize, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let (s, mut q) = sphere_case(5);
        let chain = vec![NONE; q.n_verts()];
        pool.install(|| run_relax(&s, &mut q, &chain, &PostParams::default()));
        outs.push(q.pos.iter().flat_map(|p| p.map(f64::to_bits)).collect::<Vec<u64>>());
    }
    assert_eq!(outs[0], outs[1]);
}

/// Two patches (x < 8 and x > 8) share the arc x = 8, which carries a kink (a tent in x
/// with its apex at y = 8); the warp is linear across each patch.
#[test]
fn kinked_arc_between_two_patches_straightens() {
    let n = 16u32;
    let gp = GridProxy::new(2 * n, 2 * n, 0.5, 0.0, 0.0, |x, y| [x, y, 0.0]);
    let s = Surface::new(gp.mesh).unwrap();
    let a = 3.0;
    let mut q = quad_grid(n, n, |i, j| {
        let (x, y) = (i as f64, j as f64);
        let dx = a * (1.0 - (y - 8.0).abs() / 8.0) * (1.0 - (x - 8.0).abs() / 8.0);
        [x + dx, y, 0.0]
    });
    let col = n / 2;
    let kink = |q: &QuadMesh| -> f64 {
        let mut m = 0.0f64;
        for j in 1..n {
            let p = |jj: u32| q.pos[(jj * (n + 1) + col) as usize];
            let e0 = math::sub(p(j), p(j - 1));
            let e1 = math::sub(p(j + 1), p(j));
            m = m.max(math::angle_between(e0, e1));
        }
        m
    };
    let k0 = kink(&q);
    let chain = vec![NONE; q.n_verts()];
    let rep = run_relax(&s, &mut q, &chain, &PostParams::default());
    let k1 = kink(&q);
    println!("kink {:.2} deg -> {:.2} deg, {rep:?}", k0.to_degrees(), k1.to_degrees());
    assert!(k0.to_degrees() > 30.0);
    assert!(k1 <= 0.5 * k0, "kink {k0} -> {k1}");
    q.validate().unwrap();
}

#[test]
fn nodes_are_frozen_then_cones_move_and_corners_never() {
    let (s, mut q, chain) = planar_case(10, 0.3, 21);
    let cone = 3 * 11 + 4;
    let corner = 6 * 11 + 6;
    q.vflag[cone] |= VF_CONE;
    q.vflag[corner] |= VF_CORNER;
    let before = q.pos.clone();
    let mut q4 = q.clone();
    run_relax(&s, &mut q4, &chain, &PostParams { iters: 4, freeze_nodes_iters: 4 });
    assert_eq!(q4.pos[cone], before[cone]);
    assert_eq!(q4.pos[corner], before[corner]);
    assert_ne!(q4.pos[cone - 1], before[cone - 1]);
    let rep = run_relax(&s, &mut q, &chain, &PostParams::default());
    assert_ne!(q.pos[cone], before[cone]);
    assert_eq!(q.pos[corner], before[corner]);
    assert_eq!(rep.cone_verts, 1);
}

#[test]
fn relax_errors() {
    let (s, mut q, chain) = planar_case(4, 0.2, 2);
    let bvh = Bvh::build(&s.mesh);
    let px = PostProxy::new(&s.mesh, &s.topo, &s.frames, &s.feats, &bvh);
    let st = CtlState::new();
    let p = PostParams::default();
    let e = relax(&mut q, &px, &chain[1..], &p, &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    let mut bad = chain.clone();
    bad[0] = 99;
    let e = relax(&mut q, &px, &bad, &p, &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::InvalidInput);
    st.cancel();
    let e = relax(&mut q, &px, &chain, &p, &st.unbounded()).unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
    let mut broken = q.clone();
    broken.face_verts[0] = 10_000;
    let st = CtlState::new();
    assert!(relax(&mut broken, &px, &chain, &p, &st.unbounded()).is_err());
}

/// Budget check (DESIGN 7.4: Post 1 s for 5k quads): about 5.4k quads on an 82k-face proxy.
#[test]
fn relax_5k_quads_is_fast() {
    if cfg!(debug_assertions) {
        return;
    }
    let r = 15.0;
    let s = Surface::new(icosphere(6, r).unwrap()).unwrap();
    let mut q = cube_sphere_quads(30, r);
    let mut rng = SplitMix64::new(9);
    for p in q.pos.iter_mut() {
        let d = [rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0)];
        let nrm = math::normalize(*p).unwrap();
        let t = math::sub(d, math::scale(nrm, math::dot(d, nrm)));
        *p = math::scale(math::normalize(math::add(*p, math::scale(t, 0.2))).unwrap(), r);
    }
    let bvh = Bvh::build(&s.mesh);
    let px = PostProxy::new(&s.mesh, &s.topo, &s.frames, &s.feats, &bvh);
    let st = CtlState::new();
    let chain = vec![NONE; q.n_verts()];
    let t0 = std::time::Instant::now();
    let rep = relax(&mut q, &px, &chain, &PostParams::default(), &st.unbounded()).unwrap();
    let mut c = cleanup(&mut q);
    c.vert_map.clear();
    let dt = t0.elapsed().as_secs_f64();
    println!("{} quads: relax + cleanup {dt:.3} s, {rep:?}, {c:?}", q.n_faces());
    assert!(dt < 1.0, "post took {dt} s");
    assert!(min_corner_cos(&q, |c| math::normalize(c).unwrap()) > 0.5);
}

// ---------------------------------------------------------------- cleanup

#[test]
fn planted_doublet_is_removed() {
    let n = 4u32;
    let mut q = quad_grid(n, n, |i, j| [i as f64, j as f64, 0.0]);
    // Split cell (1, 1) = [p0, p1, p2, p3] along p0-p2 with a vertex v at its center.
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let (p0, p1, p2, p3) = (id(1, 1), id(2, 1), id(2, 2), id(1, 2));
    let cell = (n + 1) as usize; // face index of cell (1, 1)
    assert_eq!(q.face(cell), &[p0, p1, p2, p3]);
    q.pos.push([1.5, 1.5, 0.0]);
    q.vflag.push(0);
    let v = q.pos.len() as u32 - 1;
    let mut faces: Vec<Vec<u32>> = (0..q.n_faces()).map(|f| q.face(f).to_vec()).collect();
    faces[cell] = vec![p0, p1, p2, v];
    faces.push(vec![p2, p3, p0, v]);
    let mut planted = QuadMesh::new();
    planted.pos = q.pos.clone();
    planted.vflag = q.vflag.clone();
    for f in &faces {
        planted.push_face(f, 0);
    }
    planted.validate().unwrap();
    assert_eq!(planted.n_faces(), 17);
    let rep = cleanup(&mut planted);
    println!("{rep:?}");
    assert_eq!(rep.doublets, 1);
    assert_eq!(rep.faces_before, 17);
    assert_eq!(rep.faces_after, 16);
    assert_eq!(rep.verts_after, 25);
    assert_eq!(rep.vert_map[v as usize], NONE);
    planted.validate().unwrap();
    assert!(planted.is_all_quads());
    let val = valences(&planted);
    for j in 1..n {
        for i in 1..n {
            assert_eq!(val[id(i, j) as usize], 4);
        }
    }
    assert!(min_corner_cos(&planted, |_| [0.0, 0.0, 1.0]) > 0.99);
}

#[test]
fn chained_doublets_and_protected_vertices() {
    // A doublet on a protected (chain) vertex stays.
    let n = 4u32;
    let q0 = quad_grid(n, n, |i, j| [i as f64, j as f64, 0.0]);
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let (p0, p1, p2, p3) = (id(1, 1), id(2, 1), id(2, 2), id(1, 2));
    let cell = (n + 1) as usize;
    let mut q = QuadMesh::new();
    q.pos = q0.pos.clone();
    q.vflag = q0.vflag.clone();
    q.pos.push([1.5, 1.5, 0.0]);
    q.vflag.push(5); // chain id 4
    let v = q.pos.len() as u32 - 1;
    for f in 0..q0.n_faces() {
        if f == cell {
            q.push_face(&[p0, p1, p2, v], 0);
        } else {
            q.push_face(q0.face(f), 0);
        }
    }
    q.push_face(&[p2, p3, p0, v], 0);
    let rep = cleanup(&mut q);
    assert_eq!(rep.doublets, 0);
    assert_eq!(rep.faces_after, 17);
    // An invalid mesh is skipped.
    let mut broken = q.clone();
    broken.face_verts[0] = 10_000;
    let rep = cleanup(&mut broken);
    assert!(rep.skipped);
}

#[test]
fn small_quad_3_3_diagonal_is_collapsed() {
    let n = 6u32;
    let id = |i: u32, j: u32| j * (n + 1) + i;
    let g = quad_grid(n, n, |i, j| [i as f64, j as f64, 0.0]);
    // Split vertex m = (3, 3) into a (left) and c (right) with a new quad [b, c, d, a],
    // b = (3, 2), d = (3, 4).
    let m = id(3, 3);
    let (b, d) = (id(3, 2), id(3, 4));
    let mut q = QuadMesh::new();
    q.pos = g.pos.clone();
    q.vflag = g.vflag.clone();
    q.pos[m as usize] = [2.9, 3.0, 0.0];
    let a = m;
    q.pos.push([3.1, 3.0, 0.0]);
    q.vflag.push(VF_CONE);
    q.vflag[a as usize] = VF_CONE;
    let c = q.pos.len() as u32 - 1;
    for f in 0..g.n_faces() {
        let fv = g.face(f);
        let right = fv.iter().any(|&w| w == id(4, 3));
        let mapped: Vec<u32> = fv.iter().map(|&w| if w == m && right { c } else { w }).collect();
        q.push_face(&mapped, 0);
    }
    q.push_face(&[b, c, d, a], 0);
    q.validate().unwrap();
    let val = valences(&q);
    assert_eq!((val[a as usize], val[c as usize], val[b as usize], val[d as usize]), (3, 3, 5, 5));
    let rep = cleanup(&mut q);
    println!("{rep:?}");
    assert_eq!(rep.diag_collapses, 1);
    assert_eq!(rep.faces_after, 36);
    assert_eq!(rep.verts_after, 49);
    q.validate().unwrap();
    // One of a and c survives at the midpoint, the other is removed.
    let (ma, mc) = (rep.vert_map[a as usize], rep.vert_map[c as usize]);
    assert!((ma == NONE) != (mc == NONE));
    let na = if ma == NONE { mc } else { ma } as usize;
    assert!(math::dist(q.pos[na], [3.0, 3.0, 0.0]) < 1e-12);
    assert_eq!(q.vflag[na] & VF_CONE, 0);
    let val = valences(&q);
    for j in 1..n {
        for i in 1..n {
            let o = rep.vert_map[id(i, j) as usize];
            let nv = if o == NONE { na } else { o as usize };
            assert_eq!(val[nv], 4);
        }
    }
    assert!(min_corner_cos(&q, |_| [0.0, 0.0, 1.0]) > 0.99);

    // The same quad above the area threshold stays.
    let mut q2 = QuadMesh::new();
    q2.pos = g.pos.clone();
    q2.vflag = g.vflag.clone();
    q2.pos[m as usize] = [2.9, 3.0, 0.0];
    q2.pos.push([3.1, 3.0, 0.0]);
    q2.vflag.push(0);
    for f in 0..g.n_faces() {
        let fv = g.face(f);
        let right = fv.iter().any(|&w| w == id(4, 3));
        let mapped: Vec<u32> = fv.iter().map(|&w| if w == m && right { c } else { w }).collect();
        q2.push_face(&mapped, 0);
    }
    q2.push_face(&[b, c, d, a], 0);
    let rep = cleanup_with(&mut q2, &CleanupParams { small_area: 0.1, ..Default::default() });
    assert_eq!(rep.diag_collapses, 0);
    assert_eq!(rep.faces_after, 37);
}

#[test]
fn clean_grid_is_untouched() {
    let mut q = quad_grid(5, 5, |i, j| [i as f64, j as f64, 0.0]);
    let q0 = q.clone();
    let rep = cleanup(&mut q);
    assert_eq!(rep.doublets + rep.diag_collapses, 0);
    assert_eq!(q, q0);
}

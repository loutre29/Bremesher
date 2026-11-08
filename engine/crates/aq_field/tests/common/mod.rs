//! Test support for aq_field: proxy views, a test-local quad-scale curvature estimate (the
//! DESIGN 6.5 normal-cycle tensor with implicit diffusion; aq_curv is a separate work package),
//! and a light local repair of the testdata meshes standing in for aq_prep (bowtie vertices
//! split, faces with two or more boundary edges split).
#![allow(dead_code, clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};

use aq_core::cross::wrap_pi;
use aq_core::math;
use aq_core::{CrossField, CtlState, CurvField, Frames, NONE, SizeField, Topology, TriMesh, V3};
use aq_field::ProxyRef;
use aq_linalg::{Chol, Csc, cotan_laplacian, lumped_mass};
use aq_testkit::Surface;

pub fn px(s: &Surface) -> ProxyRef<'_> {
    ProxyRef::new(&s.mesh, &s.topo, &s.frames, &s.feats)
}

pub fn ctl_state() -> CtlState {
    CtlState::new()
}

/// Zero curvature (conf 0 everywhere).
pub fn zero_curv(s: &Surface) -> CurvField {
    CurvField {
        k1: vec![0.0; s.n_faces()],
        k2: vec![0.0; s.n_faces()],
        phi: vec![0.0; s.n_faces()],
        conf: vec![0.0; s.n_faces()],
        kappa_hat: vec![0.0; s.n_verts()],
    }
}

pub fn size(s: &Surface, h: f64) -> SizeField {
    aq_testkit::uniform_size(s, h).unwrap()
}

pub fn centroid(m: &TriMesh, f: u32) -> V3 {
    let [a, b, c] = m.tri_pos(f);
    math::scale(math::add(math::add(a, b), c), 1.0 / 3.0)
}

/// Angle of the 3D direction d in the frame of face f.
pub fn angle_in_face(fr: &Frames, f: u32, d: V3) -> f64 {
    let a = fr.to_face_2d(f, d);
    a[1].atan2(a[0])
}

/// |theta - a| modulo pi/2, in [0, pi/4].
pub fn cross_dev(theta: f64, a: f64) -> f64 {
    wrap_pi(4.0 * (theta - a)).abs() / 4.0
}

/// Cones (index4 != 0) that are not on the boundary.
pub fn interior_cones(s: &Surface, f: &CrossField) -> Vec<(u32, i8)> {
    f.cones().into_iter().filter(|&(v, _)| !s.topo.is_bnd_vert[v as usize]).collect()
}

/// The DESIGN 6.5 curvature at scale h (uniform h, c_s = 1), computed locally for the tests.
pub fn estimate_curv(s: &Surface, h: f64) -> CurvField {
    let (m, t, fr) = (&s.mesh, &s.topo, &s.frames);
    let nv = m.pos.len();
    let nf = m.tris.len();
    // T0 per vertex, 6 components xx xy xz yy yz zz.
    let mut t0 = vec![[0.0f64; 6]; nv];
    for hh in 0..3 * nf {
        let tw = t.twin[hh];
        if tw == NONE || (tw as usize) < hh {
            continue;
        }
        let (f, g) = (hh / 3, tw as usize / 3);
        let (a, b) = (t.tail(m, hh as u32), t.head(m, hh as u32));
        let e = math::sub(m.pos[b as usize], m.pos[a as usize]);
        let l = math::norm(e);
        let u = math::scale(e, 1.0 / l);
        let beta = math::dot(math::cross(fr.n[f], fr.n[g]), u).atan2(math::dot(fr.n[f], fr.n[g]));
        let c = 0.5 * 0.5 * beta * l;
        let ten = [u[0] * u[0], u[0] * u[1], u[0] * u[2], u[1] * u[1], u[1] * u[2], u[2] * u[2]];
        for v in [a, b] {
            for k in 0..6 {
                t0[v as usize][k] += c * ten[k];
            }
        }
    }
    let mass = lumped_mass(m, t, fr);
    for v in 0..nv {
        if mass[v] > 0.0 {
            for k in 0..6 {
                t0[v][k] /= mass[v];
            }
        }
    }
    let l = cotan_laplacian(m, t, fr, None);
    let md = Csc::from_diag(&mass).unwrap();
    let a = Csc::lin_comb(&md, 1.0, &l, h * h).unwrap();
    let ch = Chol::new(&a).unwrap();
    let mut tv = vec![[0.0f64; 6]; nv];
    for k in 0..6 {
        let mut b: Vec<f64> = (0..nv).map(|v| mass[v] * t0[v][k]).collect();
        ch.try_solve(&mut b).unwrap();
        for v in 0..nv {
            tv[v][k] = b[v];
        }
    }
    let full = |x: &[f64; 6]| [[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]];
    let mut cf = CurvField::default();
    for f in 0..nf {
        let mut x = [0.0f64; 6];
        for &v in &m.tris[f] {
            for k in 0..6 {
                x[k] += tv[v as usize][k] / 3.0;
            }
        }
        let tm = full(&x);
        let q = |a: V3, b: V3| -> f64 {
            let mut s = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    s += a[i] * tm[i][j] * b[j];
                }
            }
            s
        };
        let (e1, e2) = (fr.e1[f], fr.e2[f]);
        let (t00, t01, t11) = (q(e1, e1), q(e1, e2), q(e2, e2));
        // C = J T J^T with J the 90 degree rotation.
        let (a_, b_, d_) = (t11, -t01, t00);
        let mean = 0.5 * (a_ + d_);
        let r = (0.25 * (a_ - d_) * (a_ - d_) + b_ * b_).sqrt();
        let (l1, l2) = (mean + r, mean - r);
        let ang1 = 0.5 * (2.0 * b_).atan2(a_ - d_);
        let (k1, k2, phi) = if l1.abs() >= l2.abs() { (l1, l2, ang1) } else { (l2, l1, ang1 + FRAC_PI_2) };
        let dk = (k1 - k2).abs();
        let sx = ((h * dk - 0.02) / (0.25 - 0.02)).clamp(0.0, 1.0);
        let conf = (sx * sx * (3.0 - 2.0 * sx) * dk / (k1.abs() + k2.abs() + 1e-12)).clamp(0.0, 1.0);
        cf.k1.push(k1);
        cf.k2.push(k2);
        cf.phi.push(phi);
        cf.conf.push(conf);
    }
    cf.kappa_hat = tv
        .iter()
        .map(|x| (x[0] * x[0] + x[3] * x[3] + x[5] * x[5] + 2.0 * (x[1] * x[1] + x[2] * x[2] + x[4] * x[4])).sqrt())
        .collect();
    cf.validate(nf, nv).unwrap();
    cf
}

/// Splits non-manifold (bowtie) vertices: each edge-connected fan gets its own vertex.
pub fn split_bowties(m: &mut TriMesh) -> usize {
    let nv = m.pos.len();
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (f, t) in m.tris.iter().enumerate() {
        for &v in t {
            inc[v as usize].push(f as u32);
        }
    }
    let mut splits = 0;
    for v in 0..nv {
        let faces = inc[v].clone();
        if faces.len() < 2 {
            continue;
        }
        // Union faces sharing an edge (v, x).
        let mut parent: Vec<usize> = (0..faces.len()).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut by_other: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, &f) in faces.iter().enumerate() {
            for &x in &m.tris[f as usize] {
                if x != v as u32 {
                    by_other.entry(x).or_default().push(i);
                }
            }
        }
        for list in by_other.values() {
            for w in list.windows(2) {
                let (a, b) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        let mut roots: BTreeMap<usize, u32> = BTreeMap::new();
        for i in 0..faces.len() {
            let r = find(&mut parent, i);
            let next = roots.len();
            let id = *roots.entry(r).or_insert_with(|| if next == 0 { v as u32 } else { u32::MAX });
            let id = if id == u32::MAX {
                let nid = m.pos.len() as u32;
                m.pos.push(m.pos[v]);
                roots.insert(r, nid);
                splits += 1;
                nid
            } else {
                id
            };
            let f = faces[i] as usize;
            for k in 0..3 {
                if m.tris[f][k] == v as u32 {
                    m.tris[f][k] = id;
                }
            }
        }
    }
    splits
}

/// Splits every face with two or more boundary edges (the aq_prep isolation postcondition for
/// boundary chains): the interior edge of a face with two boundary edges is split at its
/// midpoint, a face with three gets its centroid.
pub fn split_ears(m: &mut TriMesh) -> usize {
    let mut count = 0;
    for _pass in 0..20 {
        let t = Topology::build(m).unwrap();
        let mut done_any = false;
        let mut touched = vec![false; m.tris.len()];
        let faces: Vec<(usize, usize)> = (0..m.tris.len())
            .map(|f| (f, (0..3).filter(|&k| t.twin[3 * f + k] == NONE).count()))
            .filter(|&(_, n)| n >= 2)
            .collect();
        for (f, n) in faces {
            if touched[f] {
                continue;
            }
            if n == 3 {
                let [a, b, c] = m.tris[f];
                let p = centroid(m, f as u32);
                let o = m.pos.len() as u32;
                m.pos.push(p);
                m.tris[f] = [a, b, o];
                m.tris.push([b, c, o]);
                m.tris.push([c, a, o]);
                touched[f] = true;
                done_any = true;
                count += 1;
                continue;
            }
            let k = (0..3).find(|&k| t.twin[3 * f + k] != NONE).unwrap();
            let h = 3 * f + k;
            let g = t.twin[h] as usize / 3;
            if touched[g] {
                continue;
            }
            let a = m.tris[f][k];
            let b = m.tris[f][(k + 1) % 3];
            let c = m.tris[f][(k + 2) % 3];
            let kg = (t.twin[h] % 3) as usize; // in g: b -> a
            let d = m.tris[g][(kg + 2) % 3];
            let mid = math::scale(math::add(m.pos[a as usize], m.pos[b as usize]), 0.5);
            let o = m.pos.len() as u32;
            m.pos.push(mid);
            m.tris[f] = [a, o, c];
            m.tris.push([o, b, c]);
            m.tris[g] = [b, o, d];
            m.tris.push([o, a, d]);
            touched[f] = true;
            touched[g] = true;
            done_any = true;
            count += 1;
        }
        if !done_any {
            return count;
        }
    }
    count
}

/// A testdata mesh repaired locally and scaled so that its mean edge length is h/3 with h = 1
/// (the proxy scale of DESIGN 6.4).
pub fn testdata_surface(name: &str) -> Surface {
    let mut m = aq_testkit::load_testdata(name).unwrap();
    // Drop faces with a repeated index.
    m.tris.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    split_bowties(&mut m);
    split_ears(&mut m);
    let mut sum = 0.0;
    let mut n = 0usize;
    for t in &m.tris {
        for k in 0..3 {
            sum += math::dist(m.pos[t[k] as usize], m.pos[t[(k + 1) % 3] as usize]);
            n += 1;
        }
    }
    let scale = 1.0 / (3.0 * sum / n as f64);
    for p in m.pos.iter_mut() {
        *p = math::scale(*p, scale);
    }
    // Remove unused vertices.
    let mut used = vec![false; m.pos.len()];
    for t in &m.tris {
        for &v in t {
            used[v as usize] = true;
        }
    }
    let mut map = vec![u32::MAX; m.pos.len()];
    let mut pos = Vec::new();
    for (v, &u) in used.iter().enumerate() {
        if u {
            map[v] = pos.len() as u32;
            pos.push(m.pos[v]);
        }
    }
    let tris = m.tris.iter().map(|t| [map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]]).collect();
    Surface::new(TriMesh { pos, tris }).unwrap()
}

/// Smallest geodesic (Dijkstra on edges) distances from `src` to every vertex.
pub fn edge_dijkstra(s: &Surface, src: &[u32]) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let (m, t) = (&s.mesh, &s.topo);
    let nv = m.pos.len();
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); nv];
    for h in 0..3 * m.tris.len() as u32 {
        let (a, b) = (t.tail(m, h), t.head(m, h));
        let l = math::dist(m.pos[a as usize], m.pos[b as usize]);
        adj[a as usize].push((b, l));
        adj[b as usize].push((a, l));
    }
    let mut d = vec![f64::INFINITY; nv];
    let mut q = BinaryHeap::new();
    for &v in src {
        d[v as usize] = 0.0;
        q.push(Reverse((0u64, v)));
    }
    while let Some(Reverse((db, v))) = q.pop() {
        let dv = f64::from_bits(db);
        if dv > d[v as usize] {
            continue;
        }
        for &(w, l) in &adj[v as usize] {
            let nd = dv + l;
            if nd < d[w as usize] {
                d[w as usize] = nd;
                q.push(Reverse((nd.to_bits(), w)));
            }
        }
    }
    d
}

pub const DEG: f64 = PI / 180.0;

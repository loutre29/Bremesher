#![allow(dead_code, clippy::neg_cmp_op_on_partial_ord)]

use std::collections::BTreeMap;

use aq_core::math;
use aq_core::{Bvh, CtlState, Projector, QuadMesh, TriMesh, V3};
use aq_fallback::{SfParams, VFLAG_SF, sf_quadrangulate};
use aq_prep::{ComponentMesh, ingest, split_components};

/// Components of an ingested mesh, scaled to quad units for a total of `n_target` quads
/// (h0 = sqrt(A / N), as the engine does), with the per-component targets.
pub fn components(m: TriMesh, pairs: &[[u32; 2]], kinds: &[u8], n_target: f64) -> Vec<(ComponentMesh, u32)> {
    let st = CtlState::new();
    let ing = ingest(m, pairs, kinds, &st.unbounded()).unwrap();
    let comps = split_components(&ing);
    let a: f64 = comps.iter().map(|c| c.area).sum();
    let h0 = (a / n_target).sqrt();
    comps
        .iter()
        .map(|c| {
            let s = c.scaled(1.0 / h0);
            let t = (n_target * c.area / a).round().max(1.0) as u32;
            (s, t)
        })
        .collect()
}

/// The single component of a mesh.
pub fn single(m: TriMesh, pairs: &[[u32; 2]], kinds: &[u8], n_target: f64) -> ComponentMesh {
    let mut c = components(m, pairs, kinds, n_target);
    assert_eq!(c.len(), 1, "expected one component");
    c.pop().unwrap().0
}

pub fn run(c: &ComponentMesh, target: u32) -> QuadMesh {
    let (fr, bvh) = c.source_geometry().unwrap();
    let proj = Projector::new(&c.mesh, &c.topo, &fr, &bvh);
    let st = CtlState::new();
    sf_quadrangulate(c, &proj, &SfParams { target, relax_iters: 5 }, &st.unbounded()).unwrap()
}

/// Number of fans per vertex that are not a single connected fan.
pub fn nonmanifold_vertices(q: &QuadMesh) -> u32 {
    // Corners per vertex: (prev, next).
    let mut corners: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        let n = fv.len();
        for i in 0..n {
            corners.entry(fv[i]).or_default().push((fv[(i + n - 1) % n], fv[(i + 1) % n]));
        }
    }
    let mut bad = 0;
    for cs in corners.values() {
        // Union-find over corners that share an edge at the vertex.
        let n = cs.len();
        let mut par: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut y = x;
            while p[y] != r {
                let z = p[y];
                p[y] = r;
                y = z;
            }
            r
        }
        for i in 0..n {
            for j in 0..n {
                if i != j && cs[i].0 == cs[j].1 {
                    let (a, b) = (find(&mut par, i), find(&mut par, j));
                    par[a] = b;
                }
            }
        }
        let roots = (0..n).filter(|&i| find(&mut par, i) == i).count();
        if roots != 1 {
            bad += 1;
        }
    }
    bad
}

/// Euler characteristic V - E + F (referenced vertices only).
pub fn euler(q: &QuadMesh) -> i64 {
    let mut e: Vec<(u32, u32)> = Vec::new();
    let mut used = vec![false; q.n_verts()];
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        for i in 0..fv.len() {
            let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
            e.push((a.min(b), a.max(b)));
            used[a as usize] = true;
        }
    }
    e.sort_unstable();
    e.dedup();
    used.iter().filter(|&&u| u).count() as i64 - e.len() as i64 + q.n_faces() as i64
}

/// Flipped quads: the quad normal `(p2 - p0) x (p3 - p1)` is not positive against the source
/// normal at the closest point to the quad centroid (a reversed face), or a corner does not
/// turn positively about the quad normal (a folded or concave corner).
pub fn flipped_quads(q: &QuadMesh, src: &TriMesh) -> u32 {
    let bvh = Bvh::build(src);
    let mut bad = 0;
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        let p: Vec<V3> = fv.iter().map(|&v| q.pos[v as usize]).collect();
        let k = p.len();
        let mut c = [0.0; 3];
        for &x in &p {
            c = math::add(c, x);
        }
        let c = math::scale(c, 1.0 / k as f64);
        let (sp, _, _) = bvh.closest(c);
        let nq = if k == 4 {
            math::cross(math::sub(p[2], p[0]), math::sub(p[3], p[1]))
        } else {
            let mut n = [0.0; 3];
            for i in 0..k {
                n = math::add(n, math::cross(p[i], p[(i + 1) % k]));
            }
            n
        };
        let reversed = src.face_normal(sp.face).is_none_or(|n| !(math::dot(nq, n) > 0.0));
        let folded = (0..k).any(|i| {
            let e1 = math::sub(p[(i + 1) % k], p[i]);
            let e0 = math::sub(p[(i + k - 1) % k], p[i]);
            !(math::dot(math::cross(e1, e0), nq) > 0.0)
        });
        if reversed || folded {
            bad += 1;
        }
    }
    bad
}

/// Max distance from quad vertices to the source surface.
pub fn max_vertex_dist(q: &QuadMesh, src: &TriMesh) -> f64 {
    let bvh = Bvh::build(src);
    q.pos.iter().map(|&p| bvh.closest(p).2).fold(0.0, f64::max)
}

/// The DESIGN 6.14 checks: 100% quads, manifold, chi, 0 flips, count within 25%.
pub fn check(name: &str, q: &QuadMesh, c: &ComponentMesh, target: u32) {
    check_with(name, q, c, target, 0.0, 0.25);
}

/// [`check`] with a tolerated fraction of flipped quads and a count tolerance.
pub fn check_with(name: &str, q: &QuadMesh, c: &ComponentMesh, target: u32, flip_frac: f64, count_tol: f64) -> u32 {
    q.validate().unwrap_or_else(|e| panic!("{name}: invalid quad mesh: {e}"));
    assert!(q.is_all_quads(), "{name}: not all quads");
    assert_eq!(nonmanifold_vertices(q), 0, "{name}: non-manifold vertices");
    assert_eq!(euler(q), c.chi, "{name}: Euler characteristic");
    let flips = flipped_quads(q, &c.mesh);
    assert!(flips as f64 <= flip_frac * q.n_faces() as f64, "{name}: {flips} flipped quads");
    assert!(q.vflag.iter().all(|&f| f & VFLAG_SF != 0), "{name}: SF bit");
    let n = q.n_faces() as f64;
    let err = (n - target as f64).abs() / target as f64;
    let d = max_vertex_dist(q, &c.mesh);
    eprintln!(
        "{name}: target {target}, quads {} ({:+.1}%), flips {flips}, max vertex dist {d:.2e}",
        q.n_faces(),
        100.0 * (n / target as f64 - 1.0)
    );
    assert!(err <= count_tol, "{name}: count {n} vs target {target}");
    assert!(d < 0.05, "{name}: vertices off the surface ({d})");
    flips
}

/// Edges whose dihedral angle exceeds `deg` (the CLI's `--features auto`), as Hard pairs.
pub fn auto_features(m: &TriMesh, deg: f64) -> (Vec<[u32; 2]>, Vec<u8>) {
    let mut e: Vec<(u32, u32, usize)> = Vec::new();
    for (f, t) in m.tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            e.push((a.min(b), a.max(b), f));
        }
    }
    e.sort_unstable();
    let mut pairs = Vec::new();
    let lim = deg.to_radians().cos();
    let mut i = 0;
    while i < e.len() {
        let mut j = i;
        while j < e.len() && e[j].0 == e[i].0 && e[j].1 == e[i].1 {
            j += 1;
        }
        if j - i == 2 {
            let n1 = m.face_normal(e[i].2 as u32);
            let n2 = m.face_normal(e[i + 1].2 as u32);
            if let (Some(n1), Some(n2)) = (n1, n2)
                && math::dot(n1, n2) < lim
            {
                pairs.push([e[i].0, e[i].1]);
            }
        }
        i = j;
    }
    let kinds = vec![aq_core::FeatureKind::Hard as u8; pairs.len()];
    (pairs, kinds)
}

/// Mean |corner angle - 90| over quads, in degrees.
pub fn angle_dev_deg(q: &QuadMesh) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        for i in 0..4 {
            let p = q.pos[fv[i] as usize];
            let a = math::sub(q.pos[fv[(i + 1) % 4] as usize], p);
            let b = math::sub(q.pos[fv[(i + 3) % 4] as usize], p);
            s += (math::angle_between(a, b).to_degrees() - 90.0).abs();
            n += 1.0;
        }
    }
    s / n
}

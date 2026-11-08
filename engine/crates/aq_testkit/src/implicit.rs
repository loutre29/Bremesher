//! Double torus (DESIGN 6.2): an implicit genus-2 surface, the smooth union of two tori, turned
//! into triangles by marching tetrahedra (Kuhn subdivision of a regular grid, combinatorial
//! orientation, so the result is a closed, consistently oriented manifold), then cleaned by a
//! small isotropic remesh: edge collapses under the link condition, Delaunay flips, tangential
//! smoothing and Newton projection back onto the implicit surface. chi = -2.

use std::collections::BTreeMap;

use aq_core::math;
use aq_core::{AqError, Topology, TriMesh, V3};

use crate::util::{bad, ensure};

const BIG_R: f64 = 1.0;
const TUBE_R: f64 = 0.4;
const OFFSET: f64 = 0.75;
const BLEND: f64 = 0.25;

fn torus_sdf(p: V3, cx: f64) -> f64 {
    let rho = (p[0] - cx).hypot(p[1]);
    (rho - BIG_R).hypot(p[2]) - TUBE_R
}

/// The implicit function (negative inside): polynomial smooth min of two torus distances.
pub fn double_torus_sdf(p: V3) -> f64 {
    let a = torus_sdf(p, -OFFSET);
    let b = torus_sdf(p, OFFSET);
    let h = (0.5 + 0.5 * (b - a) / BLEND).clamp(0.0, 1.0);
    b + (a - b) * h - BLEND * h * (1.0 - h)
}

fn grad(p: V3) -> V3 {
    let e = 1e-6;
    let mut g = [0.0; 3];
    for k in 0..3 {
        let mut a = p;
        let mut b = p;
        a[k] += e;
        b[k] -= e;
        g[k] = (double_torus_sdf(a) - double_torus_sdf(b)) / (2.0 * e);
    }
    g
}

fn project(mut p: V3) -> V3 {
    for _ in 0..6 {
        let g = grad(p);
        let g2 = math::norm2(g);
        if g2 < 1e-20 {
            break;
        }
        p = math::sub(p, math::scale(g, double_torus_sdf(p) / g2));
    }
    p
}

/// [`double_torus_with`] with cell 0.1 and 4 cleanup iterations (about 9k faces).
pub fn double_torus() -> Result<TriMesh, AqError> {
    double_torus_with(0.1, 4)
}

/// Double torus: tori of radii (1, 0.4) centred at x = -0.75 and x = +0.75, blended; marching
/// tetrahedra on a grid of spacing `cell`, then `iters` cleanup iterations. Deterministic.
pub fn double_torus_with(cell: f64, iters: u32) -> Result<TriMesh, AqError> {
    if !(cell.is_finite() && (0.01..=0.3).contains(&cell)) {
        return Err(bad("double_torus: cell must lie in [0.01, 0.3]"));
    }
    let m = marching_tets(cell)?;
    let mut mm = MutMesh::new(m);
    for _ in 0..iters {
        mm.collapse_short(0.45 * cell);
        mm.flip_delaunay(6);
        mm.smooth(2, 0.5);
    }
    mm.collapse_short(0.3 * cell);
    mm.flip_delaunay(6);
    mm.smooth(1, 0.5);
    let out = mm.compact();
    out.validate()?;
    let t = Topology::build(&out)?;
    ensure(t.n_comp == 1 && t.boundary_loops.is_empty(), || "double torus is not one closed surface".into())?;
    let chi = t.euler_char(&out, 0);
    ensure(chi == -2, || format!("double torus has chi {chi}"))?;
    Ok(out)
}

fn marching_tets(cell: f64) -> Result<TriMesh, AqError> {
    let lo = [-(OFFSET + BIG_R + TUBE_R) - 2.0 * cell, -(BIG_R + TUBE_R) - 2.0 * cell, -TUBE_R - 2.0 * cell];
    let n: [usize; 3] = [
        ((-2.0 * lo[0]) / cell).ceil() as usize,
        ((-2.0 * lo[1]) / cell).ceil() as usize,
        ((-2.0 * lo[2]) / cell).ceil() as usize,
    ];
    let (n0, n1) = (n[0] + 1, n[1] + 1);
    let nid = |i: usize, j: usize, k: usize| (i + n0 * (j + n1 * k)) as u64;
    let npos = |i: usize, j: usize, k: usize| -> V3 {
        [lo[0] + i as f64 * cell, lo[1] + j as f64 * cell, lo[2] + k as f64 * cell]
    };
    let mut val = vec![0.0f64; n0 * n1 * (n[2] + 1)];
    for k in 0..=n[2] {
        for j in 0..=n[1] {
            for i in 0..=n[0] {
                let mut f = double_torus_sdf(npos(i, j, k));
                if f.abs() < 1e-9 * cell {
                    f = 1e-9 * cell;
                }
                val[nid(i, j, k) as usize] = f;
            }
        }
    }
    let mut ids: BTreeMap<(u64, u64), u32> = BTreeMap::new();
    let mut pos: Vec<V3> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    // Kuhn tets: (0, e_a, e_a + e_b, 111) for the 6 axis permutations, with their parity.
    const PERMS: [([usize; 3], bool); 6] = [
        ([0, 1, 2], true),
        ([1, 2, 0], true),
        ([2, 0, 1], true),
        ([0, 2, 1], false),
        ([2, 1, 0], false),
        ([1, 0, 2], false),
    ];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                for &(perm, even) in &PERMS {
                    let mut c = [0usize; 3];
                    let mut tet = [[0usize; 3]; 4];
                    tet[0] = [i, j, k];
                    for s in 0..3 {
                        c[perm[s]] = 1;
                        tet[s + 1] = [i + c[0], j + c[1], k + c[2]];
                    }
                    let tid: [u64; 4] = tet.map(|p| nid(p[0], p[1], p[2]));
                    let inside: [bool; 4] = tid.map(|x| val[x as usize] < 0.0);
                    let cnt = inside.iter().filter(|&&b| b).count();
                    if cnt == 0 || cnt == 4 {
                        continue;
                    }
                    let mut ep = |a: usize, b: usize, pos: &mut Vec<V3>| -> u32 {
                        let (ia, ib) = (tid[a], tid[b]);
                        let key = (ia.min(ib), ia.max(ib));
                        *ids.entry(key).or_insert_with(|| {
                            let (pa, pb) = (tet[a], tet[b]);
                            let (fa, fb) = (val[ia as usize], val[ib as usize]);
                            let s = fa / (fa - fb);
                            pos.push(math::lerp(npos(pa[0], pa[1], pa[2]), npos(pb[0], pb[1], pb[2]), s));
                            (pos.len() - 1) as u32
                        })
                    };
                    // Parity of a vertex sequence of the tet, combined with the tet orientation.
                    let positive = |seq: [usize; 4]| -> bool {
                        let mut inv_count = 0;
                        for x in 0..4 {
                            for y in x + 1..4 {
                                if seq[x] > seq[y] {
                                    inv_count += 1;
                                }
                            }
                        }
                        (inv_count % 2 == 0) == even
                    };
                    if cnt == 1 || cnt == 3 {
                        let lone = (0..4).find(|&x| inside[x] == (cnt == 1)).unwrap_or(0);
                        let o: Vec<usize> = (0..4).filter(|&x| x != lone).collect();
                        let (mut b, mut c2) = (o[1], o[2]);
                        if !positive([lone, o[0], b, c2]) {
                            std::mem::swap(&mut b, &mut c2);
                        }
                        let t = [ep(lone, o[0], &mut pos), ep(lone, b, &mut pos), ep(lone, c2, &mut pos)];
                        // Normal points away from the lone vertex: outward iff it is inside.
                        tris.push(if cnt == 1 { t } else { [t[0], t[2], t[1]] });
                    } else {
                        let ins: Vec<usize> = (0..4).filter(|&x| inside[x]).collect();
                        let out: Vec<usize> = (0..4).filter(|&x| !inside[x]).collect();
                        let (i1, i2) = (ins[0], ins[1]);
                        let (mut o1, mut o2) = (out[0], out[1]);
                        if !positive([i1, i2, o1, o2]) {
                            std::mem::swap(&mut o1, &mut o2);
                        }
                        let q = [ep(i1, o1, &mut pos), ep(i1, o2, &mut pos), ep(i2, o2, &mut pos), ep(i2, o1, &mut pos)];
                        tris.push([q[0], q[1], q[2]]);
                        tris.push([q[0], q[2], q[3]]);
                    }
                }
            }
        }
    }
    Ok(TriMesh { pos, tris })
}

/// A small mutable triangle mesh for the cleanup (closed surfaces only).
struct MutMesh {
    pos: Vec<V3>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    vf: Vec<Vec<u32>>,
}

impl MutMesh {
    fn new(m: TriMesh) -> Self {
        let mut vf = vec![Vec::new(); m.pos.len()];
        for (f, t) in m.tris.iter().enumerate() {
            for &v in t {
                vf[v as usize].push(f as u32);
            }
        }
        let alive = vec![true; m.tris.len()];
        MutMesh { pos: m.pos, tris: m.tris, alive, vf }
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut n: Vec<u32> = self.vf[v as usize]
            .iter()
            .flat_map(|&f| self.tris[f as usize])
            .filter(|&w| w != v)
            .collect();
        n.sort_unstable();
        n.dedup();
        n
    }

    fn well_oriented(&self, p: [V3; 3]) -> bool {
        let c = math::cross(math::sub(p[1], p[0]), math::sub(p[2], p[0]));
        let cen = math::scale(math::add(math::add(p[0], p[1]), p[2]), 1.0 / 3.0);
        let g = grad(cen);
        let (nc, ng) = (math::norm(c), math::norm(g));
        nc > 1e-14 && math::dot(c, g) > 0.4 * nc * ng
    }

    fn edges_by_length(&self) -> Vec<(f64, u32, u32)> {
        let mut e = Vec::new();
        for (f, t) in self.tris.iter().enumerate() {
            if !self.alive[f] {
                continue;
            }
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if a < b {
                    e.push((math::dist(self.pos[a as usize], self.pos[b as usize]), a, b));
                }
            }
        }
        e.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        e
    }

    fn collapse_short(&mut self, lmin: f64) {
        for (_, a, b) in self.edges_by_length() {
            if self.vf[a as usize].is_empty() || self.vf[b as usize].is_empty() {
                continue;
            }
            if math::dist(self.pos[a as usize], self.pos[b as usize]) >= lmin {
                continue;
            }
            self.try_collapse(a, b);
        }
    }

    fn try_collapse(&mut self, a: u32, b: u32) -> bool {
        let shared: Vec<u32> = self.vf[a as usize]
            .iter()
            .copied()
            .filter(|&f| self.tris[f as usize].contains(&b))
            .collect();
        if shared.len() != 2 {
            return false;
        }
        let (na, nb) = (self.neighbours(a), self.neighbours(b));
        if na.len() < 4 || nb.len() < 4 {
            return false;
        }
        let common = na.iter().filter(|x| nb.binary_search(x).is_ok()).count();
        if common != 2 {
            return false;
        }
        let p = project(math::scale(math::add(self.pos[a as usize], self.pos[b as usize]), 0.5));
        let mut faces: Vec<u32> = self.vf[a as usize].iter().chain(&self.vf[b as usize]).copied().collect();
        faces.sort_unstable();
        faces.dedup();
        for &f in &faces {
            if shared.contains(&f) {
                continue;
            }
            let t = self.tris[f as usize];
            let q = t.map(|v| if v == a || v == b { p } else { self.pos[v as usize] });
            if !self.well_oriented(q) {
                return false;
            }
        }
        for &f in &shared {
            self.alive[f as usize] = false;
        }
        for &f in &faces {
            if shared.contains(&f) {
                continue;
            }
            for v in self.tris[f as usize].iter_mut() {
                if *v == b {
                    *v = a;
                }
            }
        }
        let keep: Vec<u32> = faces.into_iter().filter(|f| !shared.contains(f)).collect();
        for &f in &shared {
            for &v in &self.tris[f as usize] {
                self.vf[v as usize].retain(|&x| x != f);
            }
        }
        self.vf[a as usize] = keep;
        self.vf[b as usize].clear();
        self.pos[a as usize] = p;
        true
    }

    fn flip_delaunay(&mut self, passes: u32) {
        for _ in 0..passes {
            let mut changed = false;
            for (_, a, b) in self.edges_by_length().into_iter().rev() {
                if self.try_flip(a, b) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn try_flip(&mut self, a: u32, b: u32) -> bool {
        let find = |x: u32, y: u32| -> Option<(u32, u32)> {
            self.vf[x as usize].iter().find_map(|&f| {
                let t = self.tris[f as usize];
                (0..3).find(|&k| t[k] == x && t[(k + 1) % 3] == y).map(|k| (f, t[(k + 2) % 3]))
            })
        };
        let (Some((f1, c)), Some((f2, d))) = (find(a, b), find(b, a)) else { return false };
        if c == d || self.neighbours(c).binary_search(&d).is_ok() {
            return false;
        }
        if self.neighbours(a).len() <= 3 || self.neighbours(b).len() <= 3 {
            return false;
        }
        let p = |v: u32| self.pos[v as usize];
        let ang = |o: u32, x: u32, y: u32| math::angle_between(math::sub(p(x), p(o)), math::sub(p(y), p(o)));
        if ang(c, a, b) + ang(d, b, a) <= std::f64::consts::PI + 1e-9 {
            return false;
        }
        let t1 = [a, d, c];
        let t2 = [d, b, c];
        if !self.well_oriented(t1.map(p)) || !self.well_oriented(t2.map(p)) {
            return false;
        }
        self.tris[f1 as usize] = t1;
        self.tris[f2 as usize] = t2;
        self.vf[a as usize].retain(|&f| f != f2);
        self.vf[b as usize].retain(|&f| f != f1);
        self.vf[c as usize].push(f2);
        self.vf[d as usize].push(f1);
        true
    }

    fn smooth(&mut self, rounds: u32, lambda: f64) {
        for _ in 0..rounds {
            let old = self.pos.clone();
            for v in 0..self.pos.len() as u32 {
                if self.vf[v as usize].is_empty() {
                    continue;
                }
                let nb = self.neighbours(v);
                let c = math::scale(
                    nb.iter().fold([0.0; 3], |s, &w| math::add(s, old[w as usize])),
                    1.0 / nb.len() as f64,
                );
                let p = old[v as usize];
                let n = math::normalize(grad(p)).unwrap_or([0.0, 0.0, 1.0]);
                let d = math::sub(c, p);
                let d = math::sub(d, math::scale(n, math::dot(d, n)));
                let q = project(math::add(p, math::scale(d, lambda)));
                // Keep the move only if no incident face degrades.
                let ok = self.vf[v as usize].iter().all(|&f| {
                    let t = self.tris[f as usize];
                    self.well_oriented(t.map(|w| if w == v { q } else { self.pos[w as usize] }))
                });
                if ok {
                    self.pos[v as usize] = q;
                }
            }
        }
    }

    fn compact(&self) -> TriMesh {
        let mut map = vec![u32::MAX; self.pos.len()];
        let mut pos = Vec::new();
        let mut tris = Vec::new();
        for (f, t) in self.tris.iter().enumerate() {
            if !self.alive[f] {
                continue;
            }
            let nt = t.map(|v| {
                if map[v as usize] == u32::MAX {
                    map[v as usize] = pos.len() as u32;
                    pos.push(self.pos[v as usize]);
                }
                map[v as usize]
            });
            tris.push(nt);
        }
        TriMesh { pos, tris }
    }
}

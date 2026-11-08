//! Small deterministic mesh generators for the aq_core tests (aq_testkit depends on aq_core,
//! so these tests carry their own).
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use aq_core::math;
use aq_core::{Frames, SplitMix64, Topology, TriMesh, V3};

pub struct Ctx {
    pub m: TriMesh,
    pub t: Topology,
    pub fr: Frames,
}

pub fn ctx(m: TriMesh) -> Ctx {
    m.validate().unwrap();
    let t = Topology::build(&m).unwrap();
    t.validate(&m).unwrap();
    let fr = Frames::new(&m, &t).unwrap();
    fr.validate(&m, &t).unwrap();
    Ctx { m, t, fr }
}

/// Icosphere of the given subdivision level and radius (level 4: 5120 faces).
pub fn icosphere(level: u32, radius: f64) -> TriMesh {
    let g = (1.0 + 5f64.sqrt()) / 2.0;
    let mut pos: Vec<V3> = vec![
        [-1.0, g, 0.0],
        [1.0, g, 0.0],
        [-1.0, -g, 0.0],
        [1.0, -g, 0.0],
        [0.0, -1.0, g],
        [0.0, 1.0, g],
        [0.0, -1.0, -g],
        [0.0, 1.0, -g],
        [g, 0.0, -1.0],
        [g, 0.0, 1.0],
        [-g, 0.0, -1.0],
        [-g, 0.0, 1.0],
    ];
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for p in pos.iter_mut() {
        *p = math::scale(math::normalize(*p).unwrap(), radius);
    }
    for _ in 0..level {
        let mut cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut mid = |a: u32, b: u32, pos: &mut Vec<V3>| -> u32 {
            let key = (a.min(b), a.max(b));
            if let Some(&i) = cache.get(&key) {
                return i;
            }
            let p = math::scale(math::add(pos[a as usize], pos[b as usize]), 0.5);
            let p = math::scale(math::normalize(p).unwrap(), radius);
            pos.push(p);
            let i = (pos.len() - 1) as u32;
            cache.insert(key, i);
            i
        };
        let mut nt = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = mid(a, b, &mut pos);
            let bc = mid(b, c, &mut pos);
            let ca = mid(c, a, &mut pos);
            nt.push([a, ab, ca]);
            nt.push([b, bc, ab]);
            nt.push([c, ca, bc]);
            nt.push([ab, bc, ca]);
        }
        tris = nt;
    }
    TriMesh { pos, tris }
}

/// Radial noise of relative amplitude `amp`.
pub fn add_radial_noise(m: &mut TriMesh, amp: f64, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for p in m.pos.iter_mut() {
        let s = 1.0 + amp * rng.range_f64(-1.0, 1.0);
        *p = math::scale(*p, s);
    }
}

/// Torus with major radius `big_r`, minor radius `r`, nu x nv quads split in 2.
pub fn torus(big_r: f64, r: f64, nu: u32, nv: u32) -> TriMesh {
    let mut pos = Vec::new();
    for i in 0..nu {
        let a = TAU * i as f64 / nu as f64;
        for j in 0..nv {
            let b = TAU * j as f64 / nv as f64;
            let rr = big_r + r * b.cos();
            pos.push([rr * a.cos(), rr * a.sin(), r * b.sin()]);
        }
    }
    let id = |i: u32, j: u32| (i % nu) * nv + (j % nv);
    let mut tris = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    TriMesh { pos, tris }
}

/// Open cylinder (prism) of radius `rad` along z in [0, len], `n` around, `m` along.
/// Outward CCW. Every quad of the prism is planar and rectangular.
pub fn open_cylinder(rad: f64, len: f64, n: u32, m: u32) -> TriMesh {
    let mut pos = Vec::new();
    for j in 0..=m {
        let z = len * j as f64 / m as f64;
        for i in 0..n {
            let a = TAU * i as f64 / n as f64;
            pos.push([rad * a.cos(), rad * a.sin(), z]);
        }
    }
    let id = |i: u32, j: u32| j * n + (i % n);
    let mut tris = Vec::new();
    for j in 0..m {
        for i in 0..n {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    TriMesh { pos, tris }
}

/// Planar grid in z = 0 on [x0, x1] x [y0, y1] with nx x ny cells, interior vertices jittered
/// by `jitter` (fraction of the cell size), diagonals alternating. Normal +z.
pub fn plane_grid(x0: f64, x1: f64, y0: f64, y1: f64, nx: u32, ny: u32, jitter: f64, seed: u64) -> TriMesh {
    let mut rng = SplitMix64::new(seed);
    let dx = (x1 - x0) / nx as f64;
    let dy = (y1 - y0) / ny as f64;
    let mut pos = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let mut x = x0 + (x1 - x0) * (i as f64 / nx as f64);
            let mut y = y0 + (y1 - y0) * (j as f64 / ny as f64);
            if i > 0 && i < nx && j > 0 && j < ny {
                x += jitter * dx * rng.range_f64(-0.5, 0.5);
                y += jitter * dy * rng.range_f64(-0.5, 0.5);
            }
            pos.push([x, y, 0.0]);
        }
    }
    let id = |i: u32, j: u32| j * (nx + 1) + i;
    let mut tris = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            if (i + j) % 2 == 0 {
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            } else {
                tris.push([a, b, d]);
                tris.push([b, c, d]);
            }
        }
    }
    TriMesh { pos, tris }
}

/// Surface of the cube [-1, 1]^3 with n x n quads per face (split in 2), outward CCW.
/// Returns the mesh and, per face, the outward axis (0..6: +x, -x, +y, -y, +z, -z).
pub fn cube(n: u32) -> (TriMesh, Vec<u8>) {
    let mut ids: BTreeMap<(i64, i64, i64), u32> = BTreeMap::new();
    let mut pos: Vec<V3> = Vec::new();
    let mut tris = Vec::new();
    let mut axis_of = Vec::new();
    let n_i = n as i64;
    let mut vid = |p: (i64, i64, i64), pos: &mut Vec<V3>| -> u32 {
        *ids.entry(p).or_insert_with(|| {
            let s = 1.0 / n as f64;
            pos.push([p.0 as f64 * s - 1.0, p.1 as f64 * s - 1.0, p.2 as f64 * s - 1.0]);
            (pos.len() - 1) as u32
        })
    };
    // Lattice coordinates 0..=2n. For each face: fixed axis, fixed value, two in-plane axes
    // (u, v) ordered so that u x v is the outward normal.
    let faces: [(usize, i64, usize, usize, u8); 6] = [
        (0, 2 * n_i, 1, 2, 0), // +x: y x z = x
        (0, 0, 2, 1, 1),       // -x: z x y = -x
        (1, 2 * n_i, 2, 0, 2), // +y: z x x = y
        (1, 0, 0, 2, 3),       // -y: x x z = -y
        (2, 2 * n_i, 0, 1, 4), // +z: x x y = z
        (2, 0, 1, 0, 5),       // -z: y x x = -z
    ];
    for &(fa, fv, ua, va, code) in &faces {
        for i in 0..2 * n_i {
            for j in 0..2 * n_i {
                let mk = |a: i64, b: i64| {
                    let mut c = [0i64; 3];
                    c[fa] = fv;
                    c[ua] = a;
                    c[va] = b;
                    (c[0], c[1], c[2])
                };
                let a = vid(mk(i, j), &mut pos);
                let b = vid(mk(i + 1, j), &mut pos);
                let c = vid(mk(i + 1, j + 1), &mut pos);
                let d = vid(mk(i, j + 1), &mut pos);
                tris.push([a, b, c]);
                tris.push([a, c, d]);
                axis_of.push(code);
                axis_of.push(code);
            }
        }
    }
    // Lattice step is 1/n on [-1,1] with 2n steps per side.
    (TriMesh { pos, tris }, axis_of)
}

/// Voxel surface (polycube) of a set of unit cells, outward CCW, each square split in 2.
pub fn polycube(cells: &[(i64, i64, i64)]) -> TriMesh {
    let set: std::collections::BTreeSet<(i64, i64, i64)> = cells.iter().copied().collect();
    let mut ids: BTreeMap<(i64, i64, i64), u32> = BTreeMap::new();
    let mut pos: Vec<V3> = Vec::new();
    let mut tris = Vec::new();
    let mut vid = |p: (i64, i64, i64), pos: &mut Vec<V3>| -> u32 {
        *ids.entry(p).or_insert_with(|| {
            pos.push([p.0 as f64, p.1 as f64, p.2 as f64]);
            (pos.len() - 1) as u32
        })
    };
    for &(x, y, z) in &set {
        // (direction, the 4 corners CCW seen from outside)
        let quads: [((i64, i64, i64), [(i64, i64, i64); 4]); 6] = [
            ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ];
        for (d, q) in quads {
            if set.contains(&(x + d.0, y + d.1, z + d.2)) {
                continue;
            }
            let v: Vec<u32> = q.iter().map(|o| vid((x + o.0, y + o.1, z + o.2), &mut pos)).collect();
            tris.push([v[0], v[1], v[2]]);
            tris.push([v[0], v[2], v[3]]);
        }
    }
    TriMesh { pos, tris }
}

/// Genus-2 surface: a 3 x 5 x 1 voxel plate with two holes (chi = -2).
pub fn double_torus() -> TriMesh {
    let mut cells = Vec::new();
    for x in 0..3 {
        for y in 0..5 {
            if x == 1 && (y == 1 || y == 3) {
                continue;
            }
            cells.push((x, y, 0));
        }
    }
    polycube(&cells)
}

/// Random theta per face in [-lim, lim].
pub fn random_theta(nf: usize, lim: f64, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..nf).map(|_| rng.range_f64(-lim, lim)).collect()
}

/// Angle in the frame of face f of the 3D direction d.
pub fn angle_in_face(fr: &Frames, f: usize, d: V3) -> f64 {
    let a = fr.to_face_2d(f as u32, d);
    a[1].atan2(a[0])
}

pub fn centroid(m: &TriMesh, f: usize) -> V3 {
    let [a, b, c] = m.tri_pos(f as u32);
    math::scale(math::add(math::add(a, b), c), 1.0 / 3.0)
}

pub const PI_: f64 = PI;

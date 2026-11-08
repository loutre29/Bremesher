//! Small deterministic mesh generators for the aq_linalg tests (adapted from the aq_core test
//! helpers; aq_linalg does not depend on aq_testkit).
#![allow(dead_code, clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::f64::consts::TAU;

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
    let fr = Frames::new(&m, &t).unwrap();
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
pub fn plane_grid(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    nx: u32,
    ny: u32,
    jitter: f64,
    seed: u64,
) -> TriMesh {
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

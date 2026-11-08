//! Synthetic quad meshes for the aq_eval tests (kept local so the tests do not depend on
//! aq_testkit's progress).
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use aq_core::math;
use aq_core::{QuadMesh, V3};

fn finish(pos: Vec<V3>, quads: &[[u32; 4]]) -> QuadMesh {
    let mut q = QuadMesh::new();
    q.vflag = vec![0; pos.len()];
    q.pos = pos;
    for f in quads {
        q.push_face(f, 0);
    }
    q
}

/// Orients a quad so that its normal points along `out`.
fn orient(f: [u32; 4], pos: &[V3], out: V3) -> [u32; 4] {
    let p: Vec<V3> = f.iter().map(|&v| pos[v as usize]).collect();
    let n = math::cross(math::sub(p[2], p[0]), math::sub(p[3], p[1]));
    if math::dot(n, out) >= 0.0 { f } else { [f[0], f[3], f[2], f[1]] }
}

/// Cube subdivided n x n per face, projected onto the sphere of radius r. With n even, the
/// planes x = 0, y = 0, z = 0 run along grid lines and their vertices have that coordinate
/// exactly 0. Eight valence 3 poles at the cube corners.
pub fn cube_sphere(n: i32, r: f64) -> QuadMesh {
    let mut ids: BTreeMap<[i32; 3], u32> = BTreeMap::new();
    let mut pos: Vec<V3> = Vec::new();
    let mut quads: Vec<[u32; 4]> = Vec::new();
    let mut vid = |l: [i32; 3], pos: &mut Vec<V3>| -> u32 {
        *ids.entry(l).or_insert_with(|| {
            let c = l.map(|x| -1.0 + 2.0 * x as f64 / n as f64);
            let p = math::normalize(c).unwrap_or(c);
            pos.push(math::scale(p, r));
            (pos.len() - 1) as u32
        })
    };
    for axis in 0..3 {
        for side in [0, n] {
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            for i in 0..n {
                for j in 0..n {
                    let mut corners = [[0i32; 3]; 4];
                    for (k, (di, dj)) in [(0, 0), (1, 0), (1, 1), (0, 1)].iter().enumerate() {
                        corners[k][axis] = side;
                        corners[k][b] = i + di;
                        corners[k][c] = j + dj;
                    }
                    let f = corners.map(|l| vid(l, &mut pos));
                    let mut out = [0.0; 3];
                    out[axis] = if side == 0 { -1.0 } else { 1.0 };
                    quads.push(orient(f, &pos, out));
                }
            }
        }
    }
    finish(pos, &quads)
}

/// Torus grid around the z axis: nu segments around z (u), nv around the tube (v).
pub fn torus_grid(nu: u32, nv: u32, big_r: f64, r: f64) -> QuadMesh {
    let mut pos = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (u, v) = (TAU * i as f64 / nu as f64, TAU * j as f64 / nv as f64);
            pos.push([(big_r + r * v.cos()) * u.cos(), (big_r + r * v.cos()) * u.sin(), r * v.sin()]);
        }
    }
    let id = |i: u32, j: u32| (i % nu) * nv + (j % nv);
    let mut quads = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (u, v) = (TAU * (i as f64 + 0.5) / nu as f64, TAU * (j as f64 + 0.5) / nv as f64);
            let out = [v.cos() * u.cos(), v.cos() * u.sin(), v.sin()];
            quads.push(orient([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], &pos, out));
        }
    }
    finish(pos, &quads)
}

/// Open cylinder about z: n vertices per ring, rows quad rows of height dz, radius chosen so
/// the ring edges have length about dz.
pub fn cylinder_grid(n: u32, rows: u32, dz: f64) -> QuadMesh {
    let r = n as f64 * dz / TAU;
    let mut pos = Vec::new();
    for row in 0..=rows {
        for j in 0..n {
            let a = TAU * j as f64 / n as f64;
            pos.push([r * a.cos(), r * a.sin(), row as f64 * dz]);
        }
    }
    let id = |row: u32, j: u32| row * n + (j % n);
    let mut quads = Vec::new();
    for row in 0..rows {
        for j in 0..n {
            let a = TAU * (j as f64 + 0.5) / n as f64;
            quads.push(orient([id(row, j), id(row, j + 1), id(row + 1, j + 1), id(row + 1, j)], &pos, [a.cos(), a.sin(), 0.0]));
        }
    }
    finish(pos, &quads)
}

/// A helical strip: vertex k sits at angle 2pi k / n and height dz k / n, quads
/// (k, k+1, k+1+n, k+n). The "circumferential" edge line is a helix of pitch dz.
pub fn helix_grid(n: u32, turns: u32, dz: f64) -> QuadMesh {
    let r = n as f64 * dz / TAU;
    let total = (turns + 1) * n + 1;
    let pos: Vec<V3> = (0..total)
        .map(|k| {
            let a = TAU * k as f64 / n as f64;
            [r * a.cos(), r * a.sin(), dz * k as f64 / n as f64]
        })
        .collect();
    let mut quads = Vec::new();
    for k in 0..total {
        if k + 1 + n < total {
            let a = TAU * (k as f64 + 0.5) / n as f64;
            quads.push(orient([k, k + 1, k + 1 + n, k + n], &pos, [a.cos(), a.sin(), 0.0]));
        }
    }
    finish(pos, &quads)
}

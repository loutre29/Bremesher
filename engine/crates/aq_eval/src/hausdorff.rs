//! Sampled two-sided Hausdorff distance between a quad mesh and the input triangle mesh
//! (DESIGN 6.15).
//!
//! A polygon is measured through its fan triangulation from its first vertex (the same
//! surface as [`crate::fan_triangulate`]). Samples of one side: every used vertex, plus
//! `samples_per_face` points per triangle on a deterministic low-discrepancy (R2) pattern.
//! The max is over all samples; the mean is area weighted over the triangle samples (each
//! weighs area / samples_per_face), or the plain vertex mean when `samples_per_face == 0`.
//! Distances are closest-point distances through `aq_core::Bvh`, in the units of the
//! positions given (input units); [`Hausdorff::scaled`] converts, for example to h with
//! `scaled(1.0 / h)`.
//!
//! Deterministic for any thread count: faces are processed in fixed chunks and the chunk
//! results are reduced in order.

use aq_core::math;
use aq_core::{Bvh, QuadMesh, TriMesh, V3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::poly::fan_triangulate;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hausdorff {
    pub q_to_in_max: f64,
    pub q_to_in_mean: f64,
    pub in_to_q_max: f64,
    pub in_to_q_mean: f64,
}

impl Hausdorff {
    /// Symmetric Hausdorff distance (max of both directions).
    pub fn symmetric(&self) -> f64 {
        self.q_to_in_max.max(self.in_to_q_max)
    }

    /// Every field multiplied by `s`.
    pub fn scaled(&self, s: f64) -> Hausdorff {
        Hausdorff {
            q_to_in_max: self.q_to_in_max * s,
            q_to_in_mean: self.q_to_in_mean * s,
            in_to_q_max: self.in_to_q_max * s,
            in_to_q_mean: self.in_to_q_mean * s,
        }
    }
}

const CHUNK: usize = 512;

fn valid_tris(m: &TriMesh) -> Vec<[V3; 3]> {
    let nv = m.pos.len();
    m.tris
        .iter()
        .filter(|t| t.iter().all(|&v| (v as usize) < nv))
        .map(|t| [m.pos[t[0] as usize], m.pos[t[1] as usize], m.pos[t[2] as usize]])
        .collect()
}

fn used_verts(m: &TriMesh) -> Vec<V3> {
    let nv = m.pos.len();
    let mut used = vec![false; nv];
    for t in &m.tris {
        if t.iter().all(|&v| (v as usize) < nv) {
            for &v in t {
                used[v as usize] = true;
            }
        }
    }
    (0..nv).filter(|&v| used[v]).map(|v| m.pos[v]).collect()
}

/// Barycentric weights of sample k (R2 sequence folded into the triangle).
fn r2_bary(k: u32) -> V3 {
    const G: f64 = 1.324_717_957_244_746;
    let a1 = 1.0 / G;
    let a2 = 1.0 / (G * G);
    let mut x = (0.5 + a1 * k as f64).fract();
    let mut y = (0.5 + a2 * k as f64).fract();
    if x + y > 1.0 {
        x = 1.0 - x;
        y = 1.0 - y;
    }
    [1.0 - x - y, x, y]
}

/// One direction: samples of `src` measured against `dst`. Returns (max, mean).
fn one_side(src: &TriMesh, dst: &Bvh, samples_per_face: u32) -> (f64, f64) {
    let verts = used_verts(src);
    let tris = valid_tris(src);
    if verts.is_empty() {
        return (0.0, 0.0);
    }
    if dst.n_faces() == 0 {
        return (f64::INFINITY, f64::INFINITY);
    }
    // Vertices.
    let vparts: Vec<(f64, f64)> = verts
        .par_chunks(CHUNK)
        .map(|ch| {
            let mut mx = 0.0f64;
            let mut sum = 0.0;
            for &p in ch {
                let d = dst.closest(p).2;
                mx = mx.max(d);
                sum += d;
            }
            (mx, sum)
        })
        .collect();
    let mut max = 0.0f64;
    let mut vsum = 0.0;
    for (m, s) in vparts {
        max = max.max(m);
        vsum += s;
    }
    if samples_per_face == 0 || tris.is_empty() {
        return (max, vsum / verts.len() as f64);
    }
    let bary: Vec<V3> = (0..samples_per_face).map(r2_bary).collect();
    let tparts: Vec<(f64, f64, f64)> = tris
        .par_chunks(CHUNK)
        .map(|ch| {
            let mut mx = 0.0f64;
            let mut wsum = 0.0;
            let mut asum = 0.0;
            for t in ch {
                let area = 0.5 * math::norm(math::cross(math::sub(t[1], t[0]), math::sub(t[2], t[0])));
                let w = area / samples_per_face as f64;
                for b in &bary {
                    let p = math::bary_point(t[0], t[1], t[2], *b);
                    let d = dst.closest(p).2;
                    mx = mx.max(d);
                    wsum += w * d;
                }
                asum += area;
            }
            (mx, wsum, asum)
        })
        .collect();
    let mut wsum = 0.0;
    let mut asum = 0.0;
    for (m, w, a) in tparts {
        max = max.max(m);
        wsum += w;
        asum += a;
    }
    let mean = if asum > 0.0 && wsum.is_finite() { wsum / asum } else { vsum / verts.len() as f64 };
    (max, mean)
}

/// Sampled Hausdorff distances between q and the input, both directions. An empty side
/// gives 0 from it and infinity towards it.
pub fn hausdorff(q: &QuadMesh, input: &TriMesh, samples_per_face: u32) -> Hausdorff {
    let qt = fan_triangulate(q);
    let bvh_in = Bvh::build(input);
    let bvh_q = Bvh::build(&qt);
    let (q_to_in_max, q_to_in_mean) = one_side(&qt, &bvh_in, samples_per_face);
    let (in_to_q_max, in_to_q_mean) = one_side(input, &bvh_q, samples_per_face);
    Hausdorff { q_to_in_max, q_to_in_mean, in_to_q_max, in_to_q_mean }
}

/// Distance from every used vertex of q to the input surface: (max, mean). This is the
/// "distance from output vertices to the input" gate of DESIGN 10.3.
pub fn vertex_distance(q: &QuadMesh, input: &TriMesh) -> (f64, f64) {
    let qt = fan_triangulate(q);
    one_side(&qt, &Bvh::build(input), 0)
}

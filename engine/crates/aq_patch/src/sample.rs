//! Grid sampling of a patch map (DESIGN 6.12 "Sampling").
//!
//! The mapped triangles are put in a uniform bucket grid over [0, W] x [0, H] (about one
//! triangle per cell). A grid point is located with the exact orient2d: the containing
//! triangle with the lowest id wins, so a point on an edge or a vertex always picks the same
//! face. The point is lifted through its barycentric coordinates on the disk positions.

use aq_core::exact::orient2d;
use aq_core::{TriMesh, V2, V3};

use crate::geom::cross_tri;

pub(crate) struct Locator {
    nx: usize,
    ny: usize,
    sx: f64,
    sy: f64,
    ptr: Vec<usize>,
    items: Vec<u32>,
}

impl Locator {
    pub fn new(m: &TriMesh, uv: &[V2], w: f64, h: f64) -> Self {
        let nf = m.tris.len().max(1) as f64;
        let aspect = (w / h).clamp(1e-6, 1e6);
        let nx = ((nf * aspect).sqrt().ceil() as usize).clamp(1, 4096);
        let ny = ((nf / aspect).sqrt().ceil() as usize).clamp(1, 4096);
        let sx = nx as f64 / w;
        let sy = ny as f64 / h;
        let cell = |x: f64, n: usize, s: f64| -> usize {
            let c = (x * s).floor();
            if c.is_nan() || c < 0.0 { 0 } else { (c as usize).min(n - 1) }
        };
        let mut ranges = Vec::with_capacity(m.tris.len());
        let mut count = vec![0usize; nx * ny + 1];
        for t in &m.tris {
            let p = [uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]];
            let (mut x0, mut x1, mut y0, mut y1) = (p[0][0], p[0][0], p[0][1], p[0][1]);
            for q in &p[1..] {
                x0 = x0.min(q[0]);
                x1 = x1.max(q[0]);
                y0 = y0.min(q[1]);
                y1 = y1.max(q[1]);
            }
            let r = (cell(x0, nx, sx), cell(x1, nx, sx), cell(y0, ny, sy), cell(y1, ny, sy));
            for cy in r.2..=r.3 {
                for cx in r.0..=r.1 {
                    count[cy * nx + cx + 1] += 1;
                }
            }
            ranges.push(r);
        }
        for i in 1..count.len() {
            count[i] += count[i - 1];
        }
        let ptr = count.clone();
        let mut fill = count;
        let mut items = vec![0u32; *ptr.last().unwrap_or(&0)];
        for (f, r) in ranges.iter().enumerate() {
            for cy in r.2..=r.3 {
                for cx in r.0..=r.1 {
                    let c = cy * nx + cx;
                    items[fill[c]] = f as u32;
                    fill[c] += 1;
                }
            }
        }
        Locator { nx, ny, sx, sy, ptr, items }
    }

    /// Lowest-id triangle containing p (closed triangle, exact test).
    pub fn locate(&self, m: &TriMesh, uv: &[V2], p: V2) -> Option<u32> {
        let cx = (p[0] * self.sx).floor();
        let cy = (p[1] * self.sy).floor();
        if !(cx.is_finite() && cy.is_finite()) {
            return None;
        }
        let cx = (cx.max(0.0) as usize).min(self.nx - 1);
        let cy = (cy.max(0.0) as usize).min(self.ny - 1);
        let c = cy * self.nx + cx;
        for &f in &self.items[self.ptr[c]..self.ptr[c + 1]] {
            let t = m.tris[f as usize];
            let (a, b, cc) = (uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]);
            if orient2d(a, b, p) >= 0.0 && orient2d(b, cc, p) >= 0.0 && orient2d(cc, a, p) >= 0.0 {
                return Some(f);
            }
        }
        None
    }
}

/// Barycentric coordinates of p in the mapped triangle f (clamped to the triangle).
pub(crate) fn bary(m: &TriMesh, uv: &[V2], f: u32, p: V2) -> V3 {
    let t = m.tris[f as usize];
    let (a, b, c) = (uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]);
    let tot = cross_tri(a, b, c);
    let mut l = [cross_tri(p, b, c) / tot, cross_tri(a, p, c) / tot, cross_tri(a, b, p) / tot];
    for x in l.iter_mut() {
        if !(*x > 0.0) {
            *x = 0.0;
        }
    }
    let s = l[0] + l[1] + l[2];
    if s > 0.0 { [l[0] / s, l[1] / s, l[2] / s] } else { [1.0 / 3.0; 3] }
}

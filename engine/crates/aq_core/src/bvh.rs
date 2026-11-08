//! Closest-point BVH (DESIGN 6.1): median split over centroids, at most 4 triangles per
//! leaf, stack traversal (no recursion). Deterministic: ties are broken by face id.

use crate::error::AqError;
use crate::loops::SurfacePoint;
use crate::math;
use crate::mesh::{NONE, TriMesh, V3};

const LEAF: usize = 4;

#[derive(Clone, Debug)]
struct Node {
    lo: V3,
    hi: V3,
    /// Leaf: first index into `order`; inner: index of the left child (right = left + 1).
    first: u32,
    /// Leaf: number of triangles (> 0); inner: 0.
    count: u32,
}

/// Bounding volume hierarchy over a copy of the triangle positions.
#[derive(Clone, Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
    tris: Vec<[V3; 3]>,
}

fn box_of(tris: &[[V3; 3]], ids: &[u32]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in ids {
        for p in &tris[i as usize] {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    (lo, hi)
}

fn box_dist2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut d = 0.0;
    for k in 0..3 {
        let e = if p[k] < lo[k] {
            lo[k] - p[k]
        } else if p[k] > hi[k] {
            p[k] - hi[k]
        } else {
            0.0
        };
        d += e * e;
    }
    d
}

/// Closest point of triangle (a, b, c) to p (Ericson, Real-Time Collision Detection 5.1.5).
/// Returns barycentric weights for (a, b, c) and the point. Degenerate triangles fall back
/// to the closest point on their edges.
pub fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> (V3, V3) {
    use math::{dot, sub};
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ([1.0, 0.0, 0.0], a);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return ([0.0, 1.0, 0.0], b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let den = d1 - d3;
        if den > 0.0 {
            let v = d1 / den;
            return ([1.0 - v, v, 0.0], math::lerp(a, b, v));
        }
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return ([0.0, 0.0, 1.0], c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let den = d2 - d6;
        if den > 0.0 {
            let w = d2 / den;
            return ([1.0 - w, 0.0, w], math::lerp(a, c, w));
        }
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let den = (d4 - d3) + (d5 - d6);
        if den > 0.0 {
            let w = (d4 - d3) / den;
            return ([0.0, 1.0 - w, w], math::lerp(b, c, w));
        }
    }
    let den = va + vb + vc;
    if den > 0.0 && den.is_finite() {
        let v = vb / den;
        let w = vc / den;
        let bary = [1.0 - v - w, v, w];
        return (bary, math::bary_point(a, b, c, bary));
    }
    // Degenerate triangle: best of the three edges.
    let seg = |x: V3, y: V3| -> (f64, f64) {
        let d = sub(y, x);
        let l2 = dot(d, d);
        let t = if l2 > 0.0 { (dot(sub(p, x), d) / l2).clamp(0.0, 1.0) } else { 0.0 };
        (math::dist2(p, math::lerp(x, y, t)), t)
    };
    let (dab, tab) = seg(a, b);
    let (dbc, tbc) = seg(b, c);
    let (dca, tca) = seg(c, a);
    if dab <= dbc && dab <= dca {
        ([1.0 - tab, tab, 0.0], math::lerp(a, b, tab))
    } else if dbc <= dca {
        ([0.0, 1.0 - tbc, tbc], math::lerp(b, c, tbc))
    } else {
        ([tca, 0.0, 1.0 - tca], math::lerp(c, a, tca))
    }
}

impl Bvh {
    /// Builds the BVH (copies the triangle positions). An allocation failure aborts like any
    /// `Vec` allocation; use [`Bvh::try_build`] where `ErrKind::Budget` is wanted.
    pub fn build(m: &TriMesh) -> Self {
        Self::try_build(m).unwrap_or_default()
    }

    /// Fallible build: an allocation failure gives `ErrKind::Budget`; an index out of range
    /// gives `ErrKind::InvalidInput`.
    pub fn try_build(m: &TriMesh) -> Result<Self, AqError> {
        let nf = m.tris.len();
        let mut tris = crate::alloc::try_with_capacity(nf)?;
        for (f, t) in m.tris.iter().enumerate() {
            let mut q = [[0.0; 3]; 3];
            for k in 0..3 {
                q[k] = *m.pos.get(t[k] as usize).ok_or_else(|| {
                    AqError::invalid_input(format!("Bvh: face {f} index out of range"))
                })?;
            }
            tris.push(q);
        }
        let mut order: Vec<u32> = crate::alloc::try_with_capacity(nf)?;
        order.extend(0..nf as u32);
        let cent: Vec<V3> = tris
            .iter()
            .map(|t: &[V3; 3]| {
                [
                    (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                    (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                    (t[0][2] + t[1][2] + t[2][2]) / 3.0,
                ]
            })
            .collect();
        let mut nodes: Vec<Node> = crate::alloc::try_with_capacity(2 * nf.div_ceil(LEAF) + 1)?;
        if nf == 0 {
            return Ok(Bvh { nodes, order, tris });
        }
        let (lo, hi) = box_of(&tris, &order);
        nodes.push(Node { lo, hi, first: 0, count: nf as u32 });
        // Work stack of (node index, start, end).
        let mut stack: Vec<(usize, usize, usize)> = vec![(0, 0, nf)];
        while let Some((ni, s, e)) = stack.pop() {
            let n = e - s;
            if n <= LEAF {
                nodes[ni].first = s as u32;
                nodes[ni].count = n as u32;
                continue;
            }
            // Largest centroid extent.
            let mut clo = [f64::INFINITY; 3];
            let mut chi = [f64::NEG_INFINITY; 3];
            for &i in &order[s..e] {
                let c = cent[i as usize];
                for k in 0..3 {
                    clo[k] = clo[k].min(c[k]);
                    chi[k] = chi[k].max(c[k]);
                }
            }
            let mut axis = 0;
            for k in 1..3 {
                if chi[k] - clo[k] > chi[axis] - clo[axis] {
                    axis = k;
                }
            }
            let mid = n / 2;
            order[s..e].select_nth_unstable_by(mid, |&a, &b| {
                cent[a as usize][axis].total_cmp(&cent[b as usize][axis]).then(a.cmp(&b))
            });
            let left = nodes.len();
            let (llo, lhi) = box_of(&tris, &order[s..s + mid]);
            let (rlo, rhi) = box_of(&tris, &order[s + mid..e]);
            nodes.push(Node { lo: llo, hi: lhi, first: 0, count: 0 });
            nodes.push(Node { lo: rlo, hi: rhi, first: 0, count: 0 });
            nodes[ni].first = left as u32;
            nodes[ni].count = 0;
            stack.push((left + 1, s + mid, e));
            stack.push((left, s, s + mid));
        }
        Ok(Bvh { nodes, order, tris })
    }

    pub fn n_faces(&self) -> usize {
        self.tris.len()
    }

    /// Closest point on the mesh to p: (surface point, position, Euclidean distance). Ties are
    /// broken by the lowest face id. For an empty mesh: face NONE, position p, distance
    /// infinity.
    pub fn closest(&self, p: V3) -> (SurfacePoint, V3, f64) {
        let mut best_d2 = f64::INFINITY;
        let mut best = (SurfacePoint { face: NONE, bary: [0.0; 3] }, p);
        if self.nodes.is_empty() {
            return (best.0, best.1, f64::INFINITY);
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if box_dist2(n.lo, n.hi, p) > best_d2 {
                continue;
            }
            if n.count > 0 {
                for &f in &self.order[n.first as usize..(n.first + n.count) as usize] {
                    let t = &self.tris[f as usize];
                    let (bary, q) = closest_on_triangle(p, t[0], t[1], t[2]);
                    let d2 = math::dist2(p, q);
                    if d2 < best_d2 || (d2 == best_d2 && f < best.0.face) {
                        best_d2 = d2;
                        best = (SurfacePoint { face: f, bary }, q);
                    }
                }
            } else {
                let l = n.first;
                let r = l + 1;
                let dl = {
                    let c = &self.nodes[l as usize];
                    box_dist2(c.lo, c.hi, p)
                };
                let dr = {
                    let c = &self.nodes[r as usize];
                    box_dist2(c.lo, c.hi, p)
                };
                // Visit the nearer child first (pushed last).
                if dl <= dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        (best.0, best.1, best_d2.sqrt())
    }

    /// Closest point among an explicit list of faces (brute force), same tie rule.
    pub fn closest_among(&self, p: V3, faces: &[u32]) -> Option<(SurfacePoint, V3, f64)> {
        let mut best: Option<(SurfacePoint, V3, f64)> = None;
        for &f in faces {
            let t = self.tris.get(f as usize)?;
            let (bary, q) = closest_on_triangle(p, t[0], t[1], t[2]);
            let d2 = math::dist2(p, q);
            let better = match &best {
                None => true,
                Some((sp, _, bd2)) => d2 < *bd2 || (d2 == *bd2 && f < sp.face),
            };
            if better {
                best = Some((SurfacePoint { face: f, bary }, q, d2));
            }
        }
        best.map(|(sp, q, d2)| (sp, q, d2.sqrt()))
    }
}

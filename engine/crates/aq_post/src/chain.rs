//! Chain polylines of the proxy and local closest-point queries along them.

use aq_core::math;
use aq_core::{Chain, Topology, TriMesh, V3};

/// Polyline of one proxy chain: `pts[i] = tail(hes[i])`, plus the last head.
pub(crate) struct Polyline {
    pub pts: Vec<V3>,
    /// Cumulative arc length at each point (`s[0] = 0`).
    pub s: Vec<f64>,
    /// Face of each segment's halfedge (projection hint).
    pub seg_face: Vec<u32>,
    pub closed: bool,
}

/// A point on a polyline: segment, local parameter in [0, 1], arc length, position.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OnPoly {
    pub seg: u32,
    pub t: f64,
    pub pos: V3,
}

/// Closest point of segment [a, b] to p: (parameter, point, squared distance).
#[inline]
fn closest_on_segment(p: V3, a: V3, b: V3) -> (f64, V3, f64) {
    let d = math::sub(b, a);
    let l2 = math::dot(d, d);
    let u = if l2 > 0.0 { (math::dot(math::sub(p, a), d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = if u <= 0.0 {
        a
    } else if u >= 1.0 {
        b
    } else {
        math::lerp(a, b, u)
    };
    (u, q, math::dist2(p, q))
}

impl Polyline {
    pub fn new(m: &TriMesh, t: &Topology, c: &Chain) -> Option<Self> {
        let nf = m.tris.len() as u32;
        let first = *c.hes.first()?;
        if first / 3 >= nf {
            return None;
        }
        let mut pts = Vec::with_capacity(c.hes.len() + 1);
        let mut seg_face = Vec::with_capacity(c.hes.len());
        pts.push(m.pos.get(t.tail(m, first) as usize).copied()?);
        for &h in &c.hes {
            if h / 3 >= nf {
                return None;
            }
            pts.push(m.pos.get(t.head(m, h) as usize).copied()?);
            seg_face.push(h / 3);
        }
        let mut s = Vec::with_capacity(pts.len());
        s.push(0.0);
        for i in 1..pts.len() {
            let l = s[i - 1] + math::dist(pts[i - 1], pts[i]);
            s.push(l);
        }
        Some(Polyline { pts, s, seg_face, closed: c.closed })
    }

    #[inline]
    pub fn n_seg(&self) -> usize {
        self.seg_face.len()
    }

    #[inline]
    pub fn length(&self) -> f64 {
        self.s.last().copied().unwrap_or(0.0)
    }

    /// Arc length of a point.
    #[inline]
    pub fn arc(&self, o: &OnPoly) -> f64 {
        let i = o.seg as usize;
        self.s[i] + o.t * (self.s[i + 1] - self.s[i])
    }

    fn eval(&self, p: V3, i: usize, best: &mut Option<(OnPoly, f64)>) {
        let (u, q, d2) = closest_on_segment(p, self.pts[i], self.pts[i + 1]);
        let better = match best {
            None => true,
            Some((_, bd)) => d2 < *bd,
        };
        if better {
            *best = Some((OnPoly { seg: i as u32, t: u, pos: q }, d2));
        }
    }

    /// Global closest point (ties: lowest segment).
    pub fn closest(&self, p: V3) -> Option<(OnPoly, f64)> {
        let mut best = None;
        for i in 0..self.n_seg() {
            self.eval(p, i, &mut best);
        }
        best.map(|(o, d2)| (o, d2.sqrt()))
    }

    /// Closest point to p among the segments within arc-length distance `r` of `cur`
    /// (wrapping on a closed chain). Keeps a vertex from jumping to a distant part of a
    /// folded chain.
    pub fn closest_local(&self, p: V3, cur: &OnPoly, r: f64) -> Option<OnPoly> {
        let n = self.n_seg();
        if n == 0 {
            return None;
        }
        let len = self.length();
        let t0 = self.arc(cur);
        let c = (cur.seg as usize).min(n - 1);
        let mut best = None;
        self.eval(p, c, &mut best);
        // Forward.
        let mut i = c;
        for _ in 1..n {
            let mut nx = i + 1;
            if nx == n {
                if !self.closed {
                    break;
                }
                nx = 0;
            }
            if nx == c {
                break;
            }
            let mut d = self.s[nx] - t0;
            if self.closed && d < 0.0 {
                d += len;
            }
            if d > r {
                break;
            }
            self.eval(p, nx, &mut best);
            i = nx;
        }
        // Backward.
        let mut i = c;
        for _ in 1..n {
            let pv = if i == 0 {
                if !self.closed {
                    break;
                }
                n - 1
            } else {
                i - 1
            };
            if pv == c {
                break;
            }
            let mut d = t0 - self.s[pv + 1];
            if self.closed && d < 0.0 {
                d += len;
            }
            if d > r {
                break;
            }
            self.eval(p, pv, &mut best);
            i = pv;
        }
        best.map(|(o, _)| o)
    }
}

/// Distance from p to a polyline given as a point list (tests and diagnostics).
pub fn dist_to_polyline(p: V3, pts: &[V3]) -> f64 {
    let mut best = f64::INFINITY;
    for w in pts.windows(2) {
        let (_, _, d2) = closest_on_segment(p, w[0], w[1]);
        best = best.min(d2);
    }
    if pts.len() == 1 {
        best = math::dist2(p, pts[0]);
    }
    best.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_loop() -> Polyline {
        // Closed unit-step loop around [0, 4]^2, 16 segments.
        let mut pts = Vec::new();
        for i in 0..4 {
            pts.push([i as f64, 0.0, 0.0]);
        }
        for i in 0..4 {
            pts.push([4.0, i as f64, 0.0]);
        }
        for i in 0..4 {
            pts.push([4.0 - i as f64, 4.0, 0.0]);
        }
        for i in 0..4 {
            pts.push([0.0, 4.0 - i as f64, 0.0]);
        }
        pts.push([0.0, 0.0, 0.0]);
        let mut s = vec![0.0];
        for i in 1..pts.len() {
            s.push(s[i - 1] + math::dist(pts[i - 1], pts[i]));
        }
        Polyline { seg_face: vec![0; pts.len() - 1], pts, s, closed: true }
    }

    #[test]
    fn local_search_wraps_on_closed_chains() {
        let pl = square_loop();
        let (cur, d) = pl.closest([0.2, -0.1, 0.0]).unwrap_or_else(|| unreachable!());
        assert_eq!(cur.seg, 0);
        assert!((d - 0.1).abs() < 1e-15);
        // Target just before the start, on the last segment (0, 1) -> (0, 0).
        let o = pl.closest_local([-0.1, 0.5, 0.0], &cur, 1.5).unwrap_or_else(|| unreachable!());
        assert_eq!(o.seg, 15);
        assert!(math::dist(o.pos, [0.0, 0.5, 0.0]) < 1e-15);
        // A far part of the loop is out of the window even when it is closer.
        let o = pl.closest_local([2.0, 3.9, 0.0], &cur, 0.5).unwrap_or_else(|| unreachable!());
        assert!(o.seg <= 1 || o.seg == 15);
    }

    #[test]
    fn distance_to_polyline() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        assert!((dist_to_polyline([0.5, 0.3, 0.0], &pts) - 0.3).abs() < 1e-15);
        assert!((dist_to_polyline([2.0, 0.5, 0.0], &pts) - 1.0).abs() < 1e-15);
    }
}

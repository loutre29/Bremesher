//! Exact iso-line geometry inside one face chart (DESIGN 6.10 "Principle").
//!
//! A curve inside face f is the iso-line `p[q] = c` of chart f, travelled along the other
//! axis `a`. Every comparison of points along a line, and of a line with the face corners,
//! is decided exactly: corner and line coordinates are dyadic and the remaining tests go
//! through Shewchuk's orient2d (`aq_core::exact`).
//!
//! Simulation of simplicity: the line is perturbed by an infinitesimal toward the left of the
//! travel direction. Rotations preserve "left", so the perturbation is carried through every
//! transition for free, and a corner lying exactly on the line is always on the right side.

use std::cmp::Ordering;

use aq_core::exact::orient2d;
use aq_core::{Transition, V2};

/// Direction codes: 0 = +u, 1 = +v, 2 = -u, 3 = -v (CCW order, rotations add to the code).
pub const DIR_ALL: u8 = 255;

/// Unit axis vector of a direction code.
#[inline]
pub fn dir_vec(d: u8) -> V2 {
    match d % 4 {
        0 => [1.0, 0.0],
        1 => [0.0, 1.0],
        2 => [-1.0, 0.0],
        _ => [0.0, -1.0],
    }
}

/// An oriented iso-line of one chart: travel direction code `d` and transverse constant `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub d: u8,
    pub c: f64,
}

impl Line {
    /// Varying axis.
    #[inline]
    pub fn a(&self) -> usize {
        (self.d & 1) as usize
    }
    /// Constant (transverse) axis.
    #[inline]
    pub fn q(&self) -> usize {
        1 - self.a()
    }
    /// +1 when the travel goes toward increasing `a`.
    #[inline]
    pub fn s(&self) -> f64 {
        if self.d < 2 { 1.0 } else { -1.0 }
    }
    /// Sign of the SoS perturbation along +q (left of the travel direction).
    #[inline]
    pub fn pert(&self) -> i8 {
        match self.d {
            0 | 3 => 1,
            _ => -1,
        }
    }
    pub fn reversed(&self) -> Line {
        Line { d: (self.d + 2) % 4, c: self.c }
    }
    /// The same line in the chart reached through transition `tr` (C10): the direction
    /// rotates by `-r` and `c' = sigma c + t[q']`. Exact for dyadic inputs.
    pub fn map(&self, tr: &Transition) -> Line {
        let d = (self.d + 4 - tr.r % 4) % 4;
        let mut e = [0.0; 2];
        e[self.q()] = 1.0;
        let m = tr.apply_vec(e);
        let q2 = 1 - (d & 1) as usize;
        Line { d, c: m[q2] * self.c + tr.t[q2] + 0.0 }
    }
    /// The chart point with varying coordinate x.
    #[inline]
    pub fn point(&self, x: f64) -> V2 {
        let mut p = [0.0; 2];
        p[self.a()] = x + 0.0;
        p[self.q()] = self.c + 0.0;
        p
    }
    /// Side of a corner with SoS: +1 above the perturbed line along q, -1 below.
    #[inline]
    pub fn side(&self, p: V2) -> i8 {
        let v = p[self.q()];
        if v > self.c {
            1
        } else if v < self.c {
            -1
        } else {
            -self.pert()
        }
    }
}

/// A point on a line inside one face: a corner, the crossing with a face edge (edge k runs
/// from corner k to corner k+1), or a dyadic value of the varying coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pt {
    V(u8),
    E(u8),
    X(f64),
}

#[inline]
fn sgn(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

#[inline]
fn ord(s: i32) -> Ordering {
    s.cmp(&0)
}

/// Approximate varying coordinate of a point (exact for V and X).
pub fn aval(t: &[V2; 3], l: Line, p: Pt) -> f64 {
    let (a, q) = (l.a(), l.q());
    match p {
        Pt::V(k) => t[k as usize % 3][a],
        Pt::X(x) => x,
        Pt::E(k) => {
            let ea = t[k as usize % 3];
            let eb = t[(k as usize + 1) % 3];
            let den = eb[q] - ea[q];
            if den == 0.0 {
                return ea[a];
            }
            let lam = ((l.c - ea[q]) / den).clamp(0.0, 1.0);
            ea[a] + lam * (eb[a] - ea[a])
        }
    }
}

/// Exact sign of `(crossing of edge k with l)_a - x`.
fn cmp_e_val(t: &[V2; 3], l: Line, k: u8, x: f64) -> Ordering {
    let ea = t[k as usize % 3];
    let eb = t[(k as usize + 1) % 3];
    let xp = l.point(x);
    let o = sgn(orient2d(ea, eb, xp));
    let saq = if l.a() == 0 { 1 } else { -1 };
    let sb = sgn(eb[l.q()] - ea[l.q()]);
    ord(o * saq * sb)
}

fn val(t: &[V2; 3], l: Line, p: Pt) -> f64 {
    match p {
        Pt::V(k) => t[k as usize % 3][l.a()],
        Pt::X(x) => x,
        Pt::E(_) => f64::NAN,
    }
}

/// Exact order of two points of line l by their varying coordinate.
pub fn cmp_a(t: &[V2; 3], l: Line, p1: Pt, p2: Pt) -> Ordering {
    match (p1, p2) {
        (Pt::E(k1), Pt::E(k2)) => {
            if k1 == k2 {
                return Ordering::Equal;
            }
            let (k1, k2) = (k1 % 3, k2 % 3);
            let (lc, mc, nc) = if (k1 + 1) % 3 == k2 { (k2, k1, (k2 + 1) % 3) } else { (k1, (k1 + 1) % 3, k2) };
            let lp = t[lc as usize];
            let s = (if l.a() == 0 { 1 } else { -1 })
                * sgn(orient2d(lp, t[mc as usize], t[nc as usize]))
                * sgn(l.c - lp[l.q()]);
            ord(s)
        }
        (Pt::E(k), o) => cmp_e_val(t, l, k, val(t, l, o)),
        (o, Pt::E(k)) => cmp_e_val(t, l, k, val(t, l, o)).reverse(),
        _ => val(t, l, p1).partial_cmp(&val(t, l, p2)).unwrap_or(Ordering::Equal),
    }
}

/// Exact order of two points along the travel direction of l.
#[inline]
pub fn cmp_travel(t: &[V2; 3], l: Line, p1: Pt, p2: Pt) -> Ordering {
    let o = cmp_a(t, l, p1, p2);
    if l.s() > 0.0 { o } else { o.reverse() }
}

/// The exact crossing point of edge k with the (perturbed) line, as a corner when the
/// corner lies on the exact line.
pub fn edge_pt(t: &[V2; 3], l: Line, k: u8) -> Pt {
    let ea = t[k as usize % 3];
    let eb = t[(k as usize + 1) % 3];
    if ea[l.q()] == l.c {
        Pt::V(k % 3)
    } else if eb[l.q()] == l.c {
        Pt::V((k + 1) % 3)
    } else {
        Pt::E(k % 3)
    }
}

/// The edges crossed by the perturbed line (0 or 2 of them under SoS).
pub fn crossed(t: &[V2; 3], l: Line) -> Vec<u8> {
    let s = [l.side(t[0]), l.side(t[1]), l.side(t[2])];
    (0..3u8).filter(|&k| s[k as usize] != s[(k as usize + 1) % 3]).collect()
}

/// Chord of a face: entry edge and point, exit edge and point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chord {
    pub entry: u8,
    pub p0: Pt,
    pub exit: u8,
    pub p1: Pt,
}

/// The chord entered through edge `entry`.
pub fn chord_from_edge(t: &[V2; 3], l: Line, entry: u8) -> Option<Chord> {
    let c = crossed(t, l);
    if c.len() != 2 {
        return None;
    }
    let exit = if c[0] == entry {
        c[1]
    } else if c[1] == entry {
        c[0]
    } else {
        return None;
    };
    Some(Chord { entry, p0: edge_pt(t, l, entry), exit, p1: edge_pt(t, l, exit) })
}

/// The chord leaving corner `kv` (the line passes through it).
pub fn chord_from_vertex(t: &[V2; 3], l: Line, kv: u8) -> Option<Chord> {
    let c = crossed(t, l);
    if c.len() != 2 {
        return None;
    }
    let p = [edge_pt(t, l, c[0]), edge_pt(t, l, c[1])];
    let at = [p[0] == Pt::V(kv), p[1] == Pt::V(kv)];
    let (i, j) = match at {
        [true, false] => (0, 1),
        [false, true] => (1, 0),
        _ => return None,
    };
    if cmp_travel(t, l, p[i], p[j]) != Ordering::Less {
        return None;
    }
    Some(Chord { entry: c[i], p0: p[i], exit: c[j], p1: p[j] })
}

/// The chord of a free line, oriented by the travel direction.
pub fn chord_free(t: &[V2; 3], l: Line) -> Option<Chord> {
    let c = crossed(t, l);
    if c.len() != 2 {
        return None;
    }
    let p = [edge_pt(t, l, c[0]), edge_pt(t, l, c[1])];
    match cmp_travel(t, l, p[0], p[1]) {
        Ordering::Less => Some(Chord { entry: c[0], p0: p[0], exit: c[1], p1: p[1] }),
        Ordering::Greater => Some(Chord { entry: c[1], p0: p[1], exit: c[0], p1: p[0] }),
        Ordering::Equal => None,
    }
}

/// Face edge a chord runs along (both ends are distinct corners), or 255.
pub fn on_edge_of(p0: Pt, p1: Pt) -> u8 {
    match (p0, p1) {
        (Pt::V(i), Pt::V(j)) if i != j => {
            if j == (i + 1) % 3 {
                i
            } else {
                j
            }
        }
        _ => 255,
    }
}

/// Exact angular order of nonzero direction vectors, CCW from +u.
pub fn cmp_angle(d1: V2, d2: V2) -> Ordering {
    let half = |d: V2| if d[1] > 0.0 || (d[1] == 0.0 && d[0] > 0.0) { 0 } else { 1 };
    let (h1, h2) = (half(d1), half(d2));
    if h1 != h2 {
        return h1.cmp(&h2);
    }
    let o = orient2d([0.0, 0.0], d1, d2);
    if o > 0.0 {
        Ordering::Less
    } else if o < 0.0 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Direction code of a chart vector by its dominant axis.
pub fn dir_code(d: V2) -> u8 {
    if d[0].abs() >= d[1].abs() {
        if d[0] >= 0.0 { 0 } else { 2 }
    } else if d[1] > 0.0 {
        1
    } else {
        3
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn line_map_roundtrip() {
        let tr = Transition { r: 1, t: [3.0, -2.0] };
        for d in 0..4u8 {
            let l = Line { d, c: 0.75 };
            let m = l.map(&tr);
            let p = l.point(1.25);
            let pm = tr.apply(p);
            assert_eq!(pm[m.q()], m.c);
            let dm = tr.apply_vec(dir_vec(d));
            assert_eq!(dm, dir_vec(m.d));
            assert_eq!(m.map(&tr.inverse()), l);
        }
    }

    #[test]
    fn edge_compare_exact() {
        let t = [[0.0, 0.0], [4.0, 1.0], [1.0, 3.0]];
        let l = Line { d: 0, c: 0.5 };
        let ch = chord_free(&t, l).unwrap();
        assert_eq!(ch.p0, Pt::E(2));
        assert_eq!(ch.p1, Pt::E(0));
        let x0 = aval(&t, l, ch.p0);
        let x1 = aval(&t, l, ch.p1);
        assert!(x0 < x1);
        assert_eq!(cmp_a(&t, l, ch.p0, ch.p1), Ordering::Less);
        assert_eq!(cmp_a(&t, l, ch.p1, Pt::X(1.999)), Ordering::Greater);
        assert_eq!(cmp_a(&t, l, ch.p1, Pt::X(2.0)), Ordering::Equal);
        assert_eq!(cmp_a(&t, l, Pt::X(2.0), ch.p1), Ordering::Equal);
    }

    #[test]
    fn sos_vertex_pass() {
        let t = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        // Along the bottom edge, travelling +u: the face is on the left, so it is crossed.
        let l = Line { d: 0, c: 0.0 };
        let ch = chord_from_vertex(&t, l, 0).unwrap();
        assert_eq!((ch.p0, ch.p1), (Pt::V(0), Pt::V(1)));
        assert_eq!(on_edge_of(ch.p0, ch.p1), 0);
        // Travelling -u, the perturbation goes below: the face is not crossed.
        assert!(crossed(&t, l.reversed()).is_empty());
    }
}

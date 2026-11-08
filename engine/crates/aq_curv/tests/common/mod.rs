#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use aq_core::{CtlState, TriMesh, V3, math};
use aq_curv::ProxyRef;
use aq_testkit::Surface;

pub fn surf(m: TriMesh) -> Surface {
    Surface::new(m).unwrap()
}

pub fn view(s: &Surface) -> ProxyRef<'_> {
    ProxyRef::new(&s.mesh, &s.topo, &s.frames, &s.feats)
}

pub fn ctl_state() -> CtlState {
    CtlState::new()
}

pub fn centroid(m: &TriMesh, f: usize) -> V3 {
    let [a, b, c] = m.tri_pos(f as u32);
    math::scale(math::add(math::add(a, b), c), 1.0 / 3.0)
}

/// Angle of the 3D direction d in the frame of face f.
pub fn angle_in_face(s: &Surface, f: usize, d: V3) -> f64 {
    let fr = &s.frames;
    math::dot(d, fr.e2[f]).atan2(math::dot(d, fr.e1[f]))
}

/// Difference of two line directions (mod pi), in [0, pi/2].
pub fn line_angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(PI);
    if d > FRAC_PI_2 { PI - d } else { d }
}

/// Circumferential unit direction about the z axis at x.
pub fn circ_dir(x: V3) -> V3 {
    let r = x[0].hypot(x[1]);
    [-x[1] / r, x[0] / r, 0.0]
}

/// Open cylinder of radius r along z on [0, len] with edges of about `edge`.
pub fn cylinder(r: f64, len: f64, edge: f64, jitter: f64, seed: u64) -> TriMesh {
    let na = ((TAU * r / edge).round() as u32).max(3);
    let nl = ((len / edge).round() as u32).max(1);
    aq_testkit::cylinder_open(r, len, na, nl, jitter, seed).unwrap()
}

/// Shifts a mesh.
pub fn translate(m: &mut TriMesh, d: V3) {
    for p in m.pos.iter_mut() {
        *p = math::add(*p, d);
    }
}

/// Max and min of a slice.
pub fn minmax(x: &[f64]) -> (f64, f64) {
    x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)))
}

/// Fraction of values satisfying a predicate.
pub fn frac(it: impl Iterator<Item = bool>) -> f64 {
    let (mut n, mut k) = (0usize, 0usize);
    for b in it {
        n += 1;
        if b {
            k += 1;
        }
    }
    if n == 0 { 0.0 } else { k as f64 / n as f64 }
}

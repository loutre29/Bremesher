//! C4 face frames and C5 discrete Levi-Civita transport.
//!
//! * C4: `e1 = normalize(p1 - p0)`, `e2 = n x e1` (right-handed), angles CCW about n from e1.
//!   `edge_angle(he) = atan2(d.e2, d.e1)` with `d = pos[head] - pos[tail]`, in the frame of
//!   `face(he)`.
//! * C5: for a canonical interior halfedge he (f = face(he), g = face(twin)):
//!   `kappa[he] = wrap_pi(edge_angle(twin) + pi - edge_angle(he))` and
//!   `kappa[twin] = -kappa[he]` exactly. A tangent direction at angle a in f has angle
//!   `a + kappa[he]` in g. Gauss-Bonnet: `sum_{he in H_in(v)} kappa[he] == K(v) mod 2pi`.

use std::f64::consts::{PI, TAU};

use crate::cross::wrap_pi;
use crate::error::{AqError, ensure};
use crate::math;
use crate::mesh::{NONE, TriMesh, V2, V3};
use crate::topo::Topology;

/// Per-face frames, areas, corner angles and transport angles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frames {
    pub e1: Vec<V3>,
    pub e2: Vec<V3>,
    pub n: Vec<V3>,
    pub area: Vec<f64>,
    /// Per corner `3f + k`: the interior angle of face f at vertex `tris[f][k]`.
    pub corner_angle: Vec<f64>,
    /// Per halfedge (C5), 0 on the boundary.
    pub kappa: Vec<f64>,
}

impl Frames {
    /// Computes the frames. A face whose normal or first edge cannot be normalized (exactly
    /// zero area, or non-finite positions) gives `ErrKind::Numerical`.
    pub fn new(m: &TriMesh, t: &Topology) -> Result<Self, AqError> {
        let nf = m.tris.len();
        ensure(t.twin.len() == 3 * nf, || "Frames::new: topology does not match mesh".into())?;
        let mut e1 = crate::alloc::try_with_capacity(nf)?;
        let mut e2 = crate::alloc::try_with_capacity(nf)?;
        let mut n = crate::alloc::try_with_capacity(nf)?;
        let mut area = crate::alloc::try_with_capacity(nf)?;
        let mut corner_angle = crate::alloc::try_with_capacity(3 * nf)?;
        for f in 0..nf as u32 {
            let [p0, p1, p2] = m.tri_pos(f);
            let c = math::cross(math::sub(p1, p0), math::sub(p2, p0));
            let nf_ = math::normalize(c).ok_or_else(|| {
                AqError::numerical(format!("face {f} has zero area or non-finite positions"))
            })?;
            let a = math::normalize(math::sub(p1, p0))
                .ok_or_else(|| AqError::numerical(format!("face {f} has a zero-length edge")))?;
            let b = math::cross(nf_, a);
            e1.push(a);
            e2.push(b);
            n.push(nf_);
            area.push(0.5 * math::norm(c));
            let p = [p0, p1, p2];
            for k in 0..3 {
                let u = math::sub(p[(k + 1) % 3], p[k]);
                let w = math::sub(p[(k + 2) % 3], p[k]);
                corner_angle.push(math::angle_between(u, w));
            }
        }
        let mut fr = Frames { e1, e2, n, area, corner_angle, kappa: Vec::new() };
        let mut kappa = crate::alloc::try_vec(3 * nf, 0.0f64)?;
        for h in 0..(3 * nf) as u32 {
            let tw = t.twin[h as usize];
            if tw == NONE || tw < h {
                continue;
            }
            let k = wrap_pi(fr.edge_angle(m, tw) + PI - fr.edge_angle(m, h));
            kappa[h as usize] = k;
            kappa[tw as usize] = -k;
        }
        fr.kappa = kappa;
        Ok(fr)
    }

    /// C4: angle of halfedge h in the frame of face(h), in (-pi, pi].
    pub fn edge_angle(&self, m: &TriMesh, h: u32) -> f64 {
        let f = (h / 3) as usize;
        let t = m.tris[f];
        let k = (h % 3) as usize;
        let d = math::sub(m.pos[t[(k + 1) % 3] as usize], m.pos[t[k] as usize]);
        math::dot(d, self.e2[f]).atan2(math::dot(d, self.e1[f]))
    }

    /// Sum of the corner angles at v (omega).
    pub fn angle_sum(&self, t: &Topology, v: u32) -> f64 {
        t.fan_out(v).map(|h| self.corner_angle[h as usize]).sum()
    }

    /// K(v) = 2pi - omega for an interior vertex, pi - omega on the boundary (0 for an
    /// isolated vertex).
    pub fn angle_defect(&self, t: &Topology, v: u32) -> f64 {
        if t.vert_he.get(v as usize).copied().unwrap_or(NONE) == NONE {
            return 0.0;
        }
        let omega = self.angle_sum(t, v);
        if t.is_bnd_vert[v as usize] { PI - omega } else { TAU - omega }
    }

    /// Coordinates of the 3D vector d in the frame of face f: `(d.e1, d.e2)`.
    pub fn to_face_2d(&self, f: u32, d: V3) -> V2 {
        let f = f as usize;
        [math::dot(d, self.e1[f]), math::dot(d, self.e2[f])]
    }

    /// The 3D vector `a0 e1 + a1 e2` of face f.
    pub fn from_face_2d(&self, f: u32, a: V2) -> V3 {
        let f = f as usize;
        math::add(math::scale(self.e1[f], a[0]), math::scale(self.e2[f], a[1]))
    }

    /// The 3D unit direction at angle `a` (CCW from e1) in face f.
    pub fn dir_at(&self, f: u32, a: f64) -> V3 {
        self.from_face_2d(f, [a.cos(), a.sin()])
    }

    /// Face-frame 2D coordinates of the 3 corners of f, relative to corner 0.
    pub fn face_coords(&self, m: &TriMesh, f: u32) -> [V2; 3] {
        let [p0, p1, p2] = m.tri_pos(f);
        [[0.0, 0.0], self.to_face_2d(f, math::sub(p1, p0)), self.to_face_2d(f, math::sub(p2, p0))]
    }

    /// Checks sizes, orthonormality (1e-12) and `kappa[twin] == -kappa[he]` bitwise.
    pub fn validate(&self, m: &TriMesh, t: &Topology) -> Result<(), AqError> {
        let nf = m.tris.len();
        ensure(
            self.e1.len() == nf
                && self.e2.len() == nf
                && self.n.len() == nf
                && self.area.len() == nf
                && self.corner_angle.len() == 3 * nf
                && self.kappa.len() == 3 * nf,
            || "Frames sizes do not match the mesh".into(),
        )?;
        for f in 0..nf {
            let (a, b, c) = (self.e1[f], self.e2[f], self.n[f]);
            let ok = (math::norm(a) - 1.0).abs() < 1e-12
                && (math::norm(b) - 1.0).abs() < 1e-12
                && (math::norm(c) - 1.0).abs() < 1e-12
                && math::dot(a, b).abs() < 1e-12
                && math::dot(a, c).abs() < 1e-12
                && math::dot(b, c).abs() < 1e-12;
            ensure(ok, || format!("frame of face {f} is not orthonormal"))?;
            ensure(self.area[f].is_finite() && self.area[f] > 0.0, || format!("area of {f}"))?;
        }
        for h in 0..3 * nf {
            let tw = t.twin[h];
            if tw == NONE {
                ensure(self.kappa[h] == 0.0, || format!("kappa of boundary he {h} != 0"))?;
            } else {
                ensure(self.kappa[tw as usize].to_bits() == (-self.kappa[h]).to_bits(), || {
                    format!("kappa[twin] != -kappa at he {h}")
                })?;
            }
        }
        Ok(())
    }
}

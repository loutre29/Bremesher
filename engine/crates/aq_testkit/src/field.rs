//! Analytic cross fields, curvature and size fields (DESIGN 6.2 "Analytic contracts").
//!
//! [`analytic_field`] evaluates a 3D direction per face (at the centroid), turns it into the
//! face-frame angle theta (C4, C6), then snaps every face carrying a boundary or chain edge to
//! that edge (the nearest of `edge_angle + k pi/2`, flagged `hard`). Without the snap a field
//! that is not exactly boundary-aligned has a non-integer boundary index (C9). The matching,
//! residuals and indices come from [`CrossField::from_theta`], and the result is validated
//! (Poincare-Hopf included).

use std::f64::consts::{FRAC_PI_2, TAU};

use aq_core::cross::wrap_pi;
use aq_core::math;
use aq_core::{AqError, CrossField, CurvField, NONE, SizeField, V2, V3};

use crate::surface::Surface;
use crate::util::{bad, centroid, ensure, tangent_basis, vertex_normals};

/// Chart used to plant singularities ([`FieldKind::Planted`]).
#[derive(Clone, Debug, PartialEq)]
pub enum PlantChart {
    /// Orthogonal projection onto the tangent plane at the first planted vertex (planar meshes).
    Plane,
    /// Stereographic projection of the sphere `center` from the antipode of the planted
    /// vertices' mean direction (orientation preserving, smooth everywhere else).
    Sphere { center: V3 },
}

/// Analytic field kinds. Directions are tangent directions of branch 0 (the u-axis, C6).
#[derive(Clone, Debug, PartialEq)]
pub enum FieldKind {
    /// The global direction `(cos a, sin a, 0)` projected into each face (a constant field on a
    /// plane in z = 0).
    Constant { angle: f64 },
    /// Cylinder along z through the origin: branch 0 circumferential (`z x r`), branch 1 axial.
    CylinderCircumferential,
    /// Cylinder along z: branch 0 along the helix that advances `pitch` along z per turn.
    CylinderTwisted { pitch: f64 },
    /// Torus about z: branch 0 along the big circles (around z), branch 1 along the meridians.
    TorusCircumferential,
    /// Torus about z: [`FieldKind::TorusCircumferential`] rotated by `twist` radians.
    TorusTwisted { twist: f64 },
    /// Cube or cube-sphere centred at the origin: aligned with the cube axes (by the dominant
    /// axis of the face centroid), 8 cones of +1 at the corners.
    CubeAligned,
    /// `base` plus planted singularities: for each (vertex, index4), theta gets
    /// `index4 * arg(sigma(x) - sigma(p)) / 4` with sigma the chart. A pair (+1, -1) is a
    /// planted +- pair, a single (v, 2) a +2 vertex (its compensation lies at the chart's
    /// infinity: outside the mesh on a plane, at the antipode on a sphere).
    Planted { base: Box<FieldKind>, sings: Vec<(u32, i8)>, chart: PlantChart },
    /// Raw per-face theta (still snapped to boundaries and chains).
    Theta(Vec<f64>),
}

impl FieldKind {
    /// A planted +- pair: index4 +1 at `plus`, -1 at `minus`, on top of `base`.
    pub fn planted_pair(base: FieldKind, plus: u32, minus: u32, chart: PlantChart) -> Self {
        FieldKind::Planted { base: Box::new(base), sings: vec![(plus, 1), (minus, -1)], chart }
    }
}

fn angle_of(s: &Surface, f: u32, d: V3) -> f64 {
    let a = s.frames.to_face_2d(f, d);
    if a[0].hypot(a[1]) < 1e-14 { 0.0 } else { a[1].atan2(a[0]) }
}

fn phi_hat(c: V3) -> V3 {
    math::normalize([-c[1], c[0], 0.0]).unwrap_or([1.0, 0.0, 0.0])
}

/// Raw (unsnapped) theta per face.
fn raw_theta(s: &Surface, kind: &FieldKind) -> Result<Vec<f64>, AqError> {
    let nf = s.n_faces();
    let mut out = Vec::with_capacity(nf);
    match kind {
        FieldKind::Theta(v) => {
            if v.len() != nf {
                return Err(bad("FieldKind::Theta has the wrong length"));
            }
            return Ok(v.clone());
        }
        FieldKind::Planted { base, sings, chart } => {
            let mut th = raw_theta(s, base)?;
            let pts: Vec<(V3, i8)> = sings
                .iter()
                .map(|&(v, i)| {
                    s.mesh.pos.get(v as usize).map(|p| (*p, i)).ok_or_else(|| bad("planted vertex out of range"))
                })
                .collect::<Result<_, _>>()?;
            let Some(&(p0, _)) = pts.first() else { return Ok(th) };
            let sigma: Box<dyn Fn(V3) -> V2> = match chart {
                PlantChart::Plane => {
                    let n = vertex_normals(&s.mesh);
                    let n0 = n[sings[0].0 as usize];
                    let (t1, t2) = tangent_basis(if math::norm(n0) > 0.5 { n0 } else { [0.0, 0.0, 1.0] });
                    Box::new(move |x: V3| {
                        let d = math::sub(x, p0);
                        [math::dot(d, t1), math::dot(d, t2)]
                    })
                }
                PlantChart::Sphere { center } => {
                    let c = *center;
                    let mean = pts.iter().fold([0.0; 3], |a, (p, _)| math::add(a, math::sub(*p, c)));
                    let cd = math::normalize(mean).ok_or_else(|| bad("planted points straddle the centre"))?;
                    let (t1, t2) = tangent_basis(cd);
                    Box::new(move |x: V3| {
                        let q = math::normalize(math::sub(x, c)).unwrap_or(cd);
                        let den = (1.0 + math::dot(q, cd)).max(1e-12);
                        [math::dot(q, t1) / den, math::dot(q, t2) / den]
                    })
                }
            };
            let sp: Vec<(V2, i8)> = pts.iter().map(|(p, i)| (sigma(*p), *i)).collect();
            for (f, t) in th.iter_mut().enumerate() {
                let x = sigma(centroid(&s.mesh, f as u32));
                for (p, i) in &sp {
                    *t += *i as f64 * (x[1] - p[1]).atan2(x[0] - p[0]) / 4.0;
                }
            }
            return Ok(th);
        }
        _ => {}
    }
    for f in 0..nf as u32 {
        let c = centroid(&s.mesh, f);
        let th = match kind {
            FieldKind::Constant { angle } => angle_of(s, f, [angle.cos(), angle.sin(), 0.0]),
            FieldKind::CylinderCircumferential | FieldKind::TorusCircumferential => angle_of(s, f, phi_hat(c)),
            FieldKind::CylinderTwisted { pitch } => {
                let rho = c[0].hypot(c[1]);
                let d = math::add(math::scale(phi_hat(c), TAU * rho), [0.0, 0.0, *pitch]);
                angle_of(s, f, d)
            }
            FieldKind::TorusTwisted { twist } => angle_of(s, f, phi_hat(c)) + twist,
            FieldKind::CubeAligned => {
                let a = (0..3).max_by(|&i, &j| c[i].abs().total_cmp(&c[j].abs())).unwrap_or(0);
                let mut d = [0.0; 3];
                d[(a + 1) % 3] = 1.0;
                angle_of(s, f, d)
            }
            FieldKind::Theta(_) | FieldKind::Planted { .. } => 0.0,
        };
        out.push(th);
    }
    Ok(out)
}

/// Snaps theta to the boundary and chain edges of each face; returns the hard flags.
pub(crate) fn snap_to_constraints(s: &Surface, theta: &mut [f64]) -> Result<Vec<bool>, AqError> {
    let nf = s.n_faces();
    let mut hard = vec![false; nf];
    for f in 0..nf {
        let cons: Vec<u32> = (0..3)
            .map(|k| (3 * f + k) as u32)
            .filter(|&h| s.topo.twin[h as usize] == NONE || s.feats.he_chain[h as usize] != NONE)
            .collect();
        let Some(&h0) = cons.first() else { continue };
        let a0 = s.frames.edge_angle(&s.mesh, h0);
        let k = ((theta[f] - a0) / FRAC_PI_2).round();
        theta[f] = a0 + k * FRAC_PI_2;
        hard[f] = true;
        for &h in &cons[1..] {
            let a = s.frames.edge_angle(&s.mesh, h);
            let dev = wrap_pi(4.0 * (a - theta[f])).abs() / 4.0;
            ensure(dev < 1e-9, || {
                format!("face {f}: constraint edges not orthogonal ({:.3e} rad), cannot align", dev)
            })?;
        }
    }
    Ok(hard)
}

/// Analytic cross field on `s` (see the module doc). Validated, Poincare-Hopf included.
pub fn analytic_field(s: &Surface, kind: &FieldKind) -> Result<CrossField, AqError> {
    let mut theta = raw_theta(s, kind)?;
    let hard = snap_to_constraints(s, &mut theta)?;
    let cf = CrossField::from_theta(&s.mesh, &s.topo, &s.frames, theta, hard)?;
    cf.validate(&s.mesh, &s.topo, &s.frames)?;
    Ok(cf)
}

/// Analytic principal curvatures ([`analytic_curv`]).
#[derive(Clone, Debug, PartialEq)]
pub enum CurvKind {
    Plane,
    Sphere { center: V3, radius: f64 },
    /// Cylinder along z through the origin.
    Cylinder { radius: f64 },
    /// Torus about z.
    Torus { big_r: f64, r: f64 },
}

/// DESIGN 6.5 confidence with h = 1.
fn confidence(k1: f64, k2: f64) -> f64 {
    let d = (k1 - k2).abs();
    let s = ((d - 0.02) / (0.25 - 0.02)).clamp(0.0, 1.0);
    let c = s * s * (3.0 - 2.0 * s) * d / (k1.abs() + k2.abs() + 1e-12);
    c.clamp(0.0, 1.0)
}

/// (k1, k2, k1 direction) at point x.
fn principal(kind: &CurvKind, x: V3) -> (f64, f64, V3) {
    match kind {
        CurvKind::Plane => (0.0, 0.0, [1.0, 0.0, 0.0]),
        CurvKind::Sphere { radius, .. } => (1.0 / radius, 1.0 / radius, [1.0, 0.0, 0.0]),
        CurvKind::Cylinder { radius } => (1.0 / radius, 0.0, phi_hat(x)),
        CurvKind::Torus { big_r, r } => {
            let rho = x[0].hypot(x[1]);
            let psi = x[2].atan2(rho - big_r);
            let radial = if rho > 0.0 { [x[0] / rho, x[1] / rho, 0.0] } else { [1.0, 0.0, 0.0] };
            let mer = math::add(math::scale(radial, -psi.sin()), [0.0, 0.0, psi.cos()]);
            let kt = 1.0 / r;
            let kl = psi.cos() / (big_r + r * psi.cos());
            if kt.abs() >= kl.abs() { (kt, kl, mer) } else { (kl, kt, phi_hat(x)) }
        }
    }
}

/// Analytic [`CurvField`] (k1, k2, phi from the exact surface at the face centroids, conf by
/// the DESIGN 6.5 formula with h = 1, kappa_hat at the vertices). Validated.
pub fn analytic_curv(s: &Surface, kind: &CurvKind) -> Result<CurvField, AqError> {
    let nf = s.n_faces();
    let mut cf = CurvField::default();
    for f in 0..nf as u32 {
        let (k1, k2, d) = principal(kind, centroid(&s.mesh, f));
        cf.k1.push(k1);
        cf.k2.push(k2);
        cf.phi.push(angle_of(s, f, d));
        cf.conf.push(if matches!(kind, CurvKind::Sphere { .. }) { 0.0 } else { confidence(k1, k2) });
    }
    cf.kappa_hat = s
        .mesh
        .pos
        .iter()
        .map(|&p| {
            let (k1, k2, _) = principal(kind, p);
            k1.hypot(k2)
        })
        .collect();
    cf.validate(nf, s.n_verts())?;
    Ok(cf)
}

/// Uniform [`SizeField`] of size `h` with `n_pred = sum A / h^2`. Validated.
pub fn uniform_size(s: &Surface, h: f64) -> Result<SizeField, AqError> {
    if !(h.is_finite() && h > 0.0) {
        return Err(bad("uniform_size needs h > 0"));
    }
    let mut sf = SizeField { h_vert: vec![h; s.n_verts()], h_face: vec![h; s.n_faces()], n_pred: 0.0 };
    sf.n_pred = sf.predicted_count(&s.frames);
    sf.validate(&s.frames, s.n_verts())?;
    Ok(sf)
}


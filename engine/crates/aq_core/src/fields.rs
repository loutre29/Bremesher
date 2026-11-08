//! Curvature and size fields (DESIGN 5 `aq_core::fields`, 6.5).

use crate::error::{AqError, ensure};
use crate::frames::Frames;

/// Curvature at the quad scale (S3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CurvField {
    /// Per face, signed principal curvatures with |k1| >= |k2|.
    pub k1: Vec<f64>,
    pub k2: Vec<f64>,
    /// Per face, angle of the k1 direction in the face frame.
    pub phi: Vec<f64>,
    /// Per face, in [0, 1].
    pub conf: Vec<f64>,
    /// Per vertex, sqrt(k1^2 + k2^2).
    pub kappa_hat: Vec<f64>,
}

impl CurvField {
    /// Checks sizes, finiteness, |k1| >= |k2|, conf in [0, 1] and kappa_hat >= 0.
    pub fn validate(&self, n_faces: usize, n_verts: usize) -> Result<(), AqError> {
        ensure(
            self.k1.len() == n_faces
                && self.k2.len() == n_faces
                && self.phi.len() == n_faces
                && self.conf.len() == n_faces
                && self.kappa_hat.len() == n_verts,
            || "CurvField sizes".into(),
        )?;
        for f in 0..n_faces {
            ensure(
                self.k1[f].is_finite() && self.k2[f].is_finite() && self.phi[f].is_finite(),
                || format!("CurvField face {f} not finite"),
            )?;
            ensure(self.k1[f].abs() >= self.k2[f].abs(), || format!("|k1| < |k2| at face {f}"))?;
            ensure((0.0..=1.0).contains(&self.conf[f]), || format!("conf[{f}] outside [0,1]"))?;
        }
        for (v, k) in self.kappa_hat.iter().enumerate() {
            ensure(k.is_finite() && *k >= 0.0, || format!("kappa_hat[{v}] invalid"))?;
        }
        Ok(())
    }
}

/// Target size h (quad units) per vertex and face; `n_pred = sum_f A_f / h_f^2`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SizeField {
    pub h_vert: Vec<f64>,
    pub h_face: Vec<f64>,
    pub n_pred: f64,
}

impl SizeField {
    /// `sum_f A_f / h_f^2`.
    pub fn predicted_count(&self, fr: &Frames) -> f64 {
        fr.area.iter().zip(&self.h_face).map(|(a, h)| a / (h * h)).sum()
    }

    /// Checks sizes, h finite and positive, and `n_pred` equal to `sum A/h^2` within 1e-6
    /// relative.
    pub fn validate(&self, fr: &Frames, n_verts: usize) -> Result<(), AqError> {
        ensure(self.h_face.len() == fr.area.len() && self.h_vert.len() == n_verts, || {
            "SizeField sizes".into()
        })?;
        for (i, h) in self.h_vert.iter().chain(&self.h_face).enumerate() {
            ensure(h.is_finite() && *h > 0.0, || format!("SizeField entry {i} invalid: {h}"))?;
        }
        let n = self.predicted_count(fr);
        ensure((n - self.n_pred).abs() <= 1e-6 * n.abs().max(1.0), || {
            format!("SizeField.n_pred = {} but sum A/h^2 = {n}", self.n_pred)
        })
    }
}

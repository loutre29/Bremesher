//! Quad mesh statistics with the exact rules of `tools/quad_stats.py` (DESIGN 6.15).
//!
//! * Irregular: an interior vertex of valence other than 4, or a boundary vertex of valence
//!   not in {2, 3}. Valence = number of unique edges at the vertex; a boundary vertex is an
//!   endpoint of an edge used by exactly one face side. Vertices used by no face are ignored.
//! * Valence keys: "N" for interior vertices, "bN" for boundary vertices.
//! * Edge CV: population standard deviation over the mean of the unique edge lengths.
//! * Angle deviation: mean of |corner angle - 90 deg| over every corner of every 4-sided
//!   face (corners with a zero-length side are skipped).
//!
//! [`quad_stats`] returns unrounded values; [`QuadStats::py_rounded`] and
//! [`QuadStats::to_py_json`] reproduce the script's rounding and key names.

use std::collections::BTreeMap;

use aq_core::QuadMesh;
use aq_core::math;
use serde::{Deserialize, Serialize};

use crate::poly::{edge_census, is_irregular, valid_faces};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QuadStats {
    pub faces: u32,
    pub quad_ratio: f64,
    pub face_sizes: BTreeMap<u32, u32>,
    pub irregular: u32,
    pub valence: BTreeMap<String, u32>,
    pub edge_cv: f64,
    pub angle_dev_deg: f64,
    pub nonmanifold_edges: u32,
    pub boundary_edges: u32,
}

/// Python's `round(x, n)`: correctly rounded to n decimals (exact ties to even), which is
/// what Rust's `{:.n}` formatting does.
pub fn py_round(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.n$}").parse().unwrap_or(x)
}

impl QuadStats {
    /// The values as the script prints them: quad_ratio and edge_cv to 4 decimals, the
    /// angle deviation to 2.
    pub fn py_rounded(&self) -> QuadStats {
        QuadStats {
            quad_ratio: py_round(self.quad_ratio, 4),
            edge_cv: py_round(self.edge_cv, 4),
            angle_dev_deg: py_round(self.angle_dev_deg, 2),
            ..self.clone()
        }
    }

    /// The script's dict (rounded, with its key names: `irregular_verts`, `edge_len_cv`,
    /// `mean_angle_dev_deg`).
    pub fn to_py_json(&self) -> serde_json::Value {
        let r = self.py_rounded();
        let sizes: serde_json::Map<String, serde_json::Value> =
            r.face_sizes.iter().map(|(k, v)| (k.to_string(), (*v).into())).collect();
        let val: serde_json::Map<String, serde_json::Value> =
            r.valence.iter().map(|(k, v)| (k.clone(), (*v).into())).collect();
        serde_json::json!({
            "faces": r.faces,
            "quad_ratio": r.quad_ratio,
            "face_sizes": sizes,
            "irregular_verts": r.irregular,
            "valence": val,
            "edge_len_cv": r.edge_cv,
            "mean_angle_dev_deg": r.angle_dev_deg,
            "nonmanifold_edges": r.nonmanifold_edges,
            "boundary_edges": r.boundary_edges,
        })
    }
}

/// Statistics of a polygon mesh (any face sizes). Faces with an invalid CSR range or an
/// out-of-range vertex are skipped. An edgeless mesh gives edge_cv 0 (the script divides by
/// zero there).
pub fn quad_stats(q: &QuadMesh) -> QuadStats {
    let faces = valid_faces(q);
    let mut face_sizes = BTreeMap::new();
    for fv in &faces {
        *face_sizes.entry(fv.len() as u32).or_insert(0u32) += 1;
    }
    let n_faces = faces.len() as u32;
    let quads = face_sizes.get(&4).copied().unwrap_or(0);
    let c = edge_census(q.pos.len(), &faces);

    let mut valence: BTreeMap<String, u32> = BTreeMap::new();
    let mut irregular = 0u32;
    for v in 0..q.pos.len() {
        let k = c.valence[v];
        if k == 0 {
            continue;
        }
        let b = c.boundary[v];
        let key = if b { format!("b{k}") } else { k.to_string() };
        *valence.entry(key).or_insert(0) += 1;
        if is_irregular(k, b) {
            irregular += 1;
        }
    }

    let lens: Vec<f64> = c.edges.iter().map(|&([a, b], _)| math::dist(q.pos[a as usize], q.pos[b as usize])).collect();
    let edge_cv = if lens.is_empty() {
        0.0
    } else {
        let n = lens.len() as f64;
        let mean = lens.iter().sum::<f64>() / n;
        let var = lens.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        var.sqrt() / mean
    };

    let mut dev_sum = 0.0;
    let mut dev_n = 0usize;
    for fv in &faces {
        if fv.len() != 4 {
            continue;
        }
        for i in 0..4 {
            let p = q.pos[fv[(i + 3) % 4] as usize];
            let o = q.pos[fv[i] as usize];
            let r = q.pos[fv[(i + 1) % 4] as usize];
            let u = math::sub(p, o);
            let w = math::sub(r, o);
            let (nu, nw) = (math::norm(u), math::norm(w));
            if nu > 0.0 && nw > 0.0 {
                let cs = (math::dot(u, w) / (nu * nw)).clamp(-1.0, 1.0);
                dev_sum += (cs.acos().to_degrees() - 90.0).abs();
                dev_n += 1;
            }
        }
    }

    QuadStats {
        faces: n_faces,
        quad_ratio: quads as f64 / n_faces.max(1) as f64,
        face_sizes,
        irregular,
        valence,
        edge_cv,
        angle_dev_deg: dev_sum / dev_n.max(1) as f64,
        nonmanifold_edges: c.edges.iter().filter(|e| e.1 > 2).count() as u32,
        boundary_edges: c.edges.iter().filter(|e| e.1 == 1).count() as u32,
    }
}

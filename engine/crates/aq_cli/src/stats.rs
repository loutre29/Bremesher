//! The stats report (DESIGN 9 "Stats report"): `autoquad stats` and the `stats` block of a
//! remesh report.

use serde_json::{Value, json};

use aq_core::{QuadMesh, TriMesh, V3};
use aq_eval::{
    Hausdorff, LoopReport, WeldReport, hausdorff, loop_closure, quad_stats, vertex_distance,
};

/// Hausdorff samples per triangle (plus every vertex).
pub const HAUSDORFF_SAMPLES: u32 = 4;

/// Everything the stats block shows, unrendered.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub q: aq_eval::QuadStats,
    pub target: Option<u32>,
    pub count_err: Option<f64>,
    /// In input units.
    pub haus: Option<Hausdorff>,
    /// Output vertices to the input: (max, mean), input units.
    pub vdist: Option<(f64, f64)>,
    /// Input bounding box diagonal and the quad size h = sqrt(A_input / N).
    pub diag: f64,
    pub h: f64,
    pub loops: Option<(V3, LoopReport)>,
    pub weld: Option<WeldReport>,
}

/// Computes the stats of `q` against an optional input (same units).
pub fn compute(
    q: &QuadMesh,
    input: Option<&TriMesh>,
    target: Option<u32>,
    axis: Option<V3>,
    weld: Option<WeldReport>,
) -> Stats {
    let qs = quad_stats(q);
    let mut s = Stats {
        q: qs,
        target,
        weld,
        ..Stats::default()
    };
    if let Some(t) = target
        && t > 0
    {
        s.count_err = Some((s.q.faces as f64 - t as f64) / t as f64);
    }
    if let Some(m) = input {
        s.diag = m.bbox_diag();
        let n = target
            .filter(|&t| t > 0)
            .map_or(s.q.faces.max(1) as f64, |t| t as f64);
        s.h = (m.total_area() / n).sqrt();
        s.haus = Some(hausdorff(q, m, HAUSDORFF_SAMPLES));
        s.vdist = Some(vertex_distance(q, m));
    }
    if let Some(a) = axis {
        s.loops = Some((a, loop_closure(q, Some(a))));
    }
    s
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn haus_json(h: &Hausdorff) -> Value {
    json!({
        "q_to_in_max": num(h.q_to_in_max),
        "q_to_in_mean": num(h.q_to_in_mean),
        "in_to_q_max": num(h.in_to_q_max),
        "in_to_q_mean": num(h.in_to_q_mean),
        "symmetric": num(h.symmetric()),
    })
}

impl Stats {
    /// The JSON block.
    pub fn to_json(&self) -> Value {
        let mut v = serde_json::to_value(&self.q).unwrap_or(json!({}));
        if let Value::Object(m) = &mut v {
            if let Some(t) = self.target {
                m.insert(
                    "count".into(),
                    json!({"target": t, "result": self.q.faces, "err": self.count_err.map_or(Value::Null, num)}),
                );
            }
            if let Some(h) = &self.haus {
                let pct = if self.diag > 0.0 {
                    100.0 / self.diag
                } else {
                    f64::NAN
                };
                let inv_h = if self.h > 0.0 { 1.0 / self.h } else { f64::NAN };
                m.insert(
                    "hausdorff".into(),
                    json!({
                        "units": haus_json(h),
                        "pct_diag": haus_json(&h.scaled(pct)),
                        "h": haus_json(&h.scaled(inv_h)),
                        "diag": num(self.diag),
                        "quad_size_h": num(self.h),
                    }),
                );
            }
            if let Some((mx, mean)) = self.vdist {
                let inv_h = if self.h > 0.0 { 1.0 / self.h } else { f64::NAN };
                m.insert(
                    "vertex_distance".into(),
                    json!({"max": num(mx), "mean": num(mean), "max_h": num(mx * inv_h), "mean_h": num(mean * inv_h)}),
                );
            }
            if let Some((a, l)) = &self.loops {
                m.insert(
                    "loops".into(),
                    json!({"axis": a, "loops": l.loops, "closed": l.closed, "closed_frac": num(l.closed_frac), "helix_suspects": l.helix_suspects}),
                );
            }
            if let Some(w) = &self.weld {
                m.insert(
                    "mirror_weld".into(),
                    serde_json::to_value(w).unwrap_or(json!({})),
                );
            }
        }
        v
    }

    /// Human-readable lines.
    pub fn to_text(&self) -> String {
        let q = &self.q;
        let mut s = String::new();
        s += &format!("faces           {}\n", q.faces);
        s += &format!("quad_ratio      {:.4}\n", q.quad_ratio);
        s += &format!("face_sizes      {:?}\n", q.face_sizes);
        s += &format!("irregular       {}\n", q.irregular);
        s += &format!("valence         {:?}\n", q.valence);
        s += &format!("edge_cv         {:.4}\n", q.edge_cv);
        s += &format!("angle_dev_deg   {:.2}\n", q.angle_dev_deg);
        s += &format!("nonmanifold     {}\n", q.nonmanifold_edges);
        s += &format!("boundary_edges  {}\n", q.boundary_edges);
        if let (Some(t), Some(e)) = (self.target, self.count_err) {
            s += &format!("count           {} for {t} ({:+.2}%)\n", q.faces, 100.0 * e);
        }
        if let Some(h) = &self.haus {
            let ih = if self.h > 0.0 { 1.0 / self.h } else { f64::NAN };
            let pd = if self.diag > 0.0 {
                100.0 / self.diag
            } else {
                f64::NAN
            };
            s += &format!(
                "hausdorff       q->in max {:.4e} mean {:.4e}, in->q max {:.4e} mean {:.4e} (units)\n",
                h.q_to_in_max, h.q_to_in_mean, h.in_to_q_max, h.in_to_q_mean
            );
            s += &format!(
                "                symmetric {:.4} h, {:.4} % of diag (h = {:.4e})\n",
                h.symmetric() * ih,
                h.symmetric() * pd,
                self.h
            );
        }
        if let Some((mx, mean)) = self.vdist {
            let ih = if self.h > 0.0 { 1.0 / self.h } else { f64::NAN };
            s += &format!(
                "vertex_distance max {:.4} h, mean {:.4} h\n",
                mx * ih,
                mean * ih
            );
        }
        if let Some((a, l)) = &self.loops {
            s += &format!(
                "loops           axis {a:?}: {} of {} closed ({:.3}), {} helix suspects\n",
                l.closed, l.loops, l.closed_frac, l.helix_suspects
            );
        }
        if let Some(w) = &self.weld {
            s += &format!(
                "mirror_weld     open edges {}, seam poles {}, seam verts {}\n",
                w.open_edges, w.seam_poles, w.seam_verts
            );
        }
        s
    }
}

/// The principal axis (largest variance direction) of the vertices used by `m`, by power
/// iteration on the covariance (deterministic).
pub fn pca_axis(m: &TriMesh) -> V3 {
    let n = m.pos.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in &m.pos {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    let mut cov = [[0.0; 3]; 3];
    for p in &m.pos {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let mut v = [1.0, 0.7, 0.3];
    for _ in 0..200 {
        let w = [
            cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
            cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
            cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2],
        ];
        let l = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if l.is_nan() || l <= 0.0 {
            return [0.0, 0.0, 1.0];
        }
        v = [w[0] / l, w[1] / l, w[2] / l];
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_of_a_long_box() {
        let m = TriMesh {
            pos: vec![
                [0.0, 0.0, -5.0],
                [1.0, 0.0, -5.0],
                [0.0, 1.0, 5.0],
                [1.0, 1.0, 5.0],
            ],
            tris: vec![[0, 1, 2], [1, 3, 2]],
        };
        let a = pca_axis(&m);
        assert!(a[2].abs() > 0.99, "{a:?}");
    }

    #[test]
    fn block_keys() {
        let m = TriMesh {
            pos: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            tris: vec![[0, 1, 2], [0, 2, 3]],
        };
        let mut q = QuadMesh::new();
        q.pos = m.pos.clone();
        q.vflag = vec![0; 4];
        q.push_face(&[0, 1, 2, 3], 0);
        let s = compute(&q, Some(&m), Some(1), Some([0.0, 0.0, 1.0]), None);
        let j = s.to_json();
        assert_eq!(j["faces"], 1);
        assert_eq!(j["count"]["err"], 0.0);
        assert!(j["hausdorff"]["units"]["symmetric"].as_f64().unwrap() < 1e-12);
        assert!(j["loops"]["closed_frac"].is_number());
        assert!(s.to_text().contains("faces"));
    }
}

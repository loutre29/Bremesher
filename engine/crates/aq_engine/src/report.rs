//! The report JSON (DESIGN 9 "Report JSON").
//!
//! Every key of the DESIGN schema is always present. Extra keys: `engine` (M0 or M1),
//! `error`, `warnings`, `h0`, `chains`, `failed_tris`, `count_correction`, `ingest` (the
//! `IngestReport` counters), `params`, `timing_ms.fallback` (wall time of the component
//! phase), and per component `tris`, `area`, `chi`, `closed`, `debris`, `time_ms`, `error`.

use serde_json::{Map, Value, json};

use aq_core::{QuadMesh, Stage};
use aq_prep::IngestReport;

use crate::run::Correction;
use crate::{ABI_VERSION, CompStatus, ENGINE_PATH, Params, Status};

/// One component entry.
pub(crate) struct CompReport<'a> {
    pub id: u32,
    pub status: CompStatus,
    pub target: u32,
    pub debris: bool,
    pub mesh: Option<&'a QuadMesh>,
    pub tris: usize,
    /// Input units.
    pub area: f64,
    pub chi: i64,
    pub closed: bool,
    pub time_ms: f64,
    pub error: Option<String>,
    pub warnings: Vec<String>,
    /// Blocks from the pipeline rung (field, param, layout, ...), merged over the defaults.
    pub extra: Map<String, Value>,
}

/// Everything the report shows.
pub(crate) struct Summary<'a> {
    pub status: Status,
    pub error: String,
    pub ingest: Option<IngestReport>,
    pub ingest_ms: f64,
    /// Wall time of the per-component phase.
    pub comp_ms: f64,
    /// Per-stage times reported by the rungs (summed over components).
    pub stage_ms: Vec<(Stage, f64)>,
    pub total_ms: f64,
    pub warnings: Vec<String>,
    pub target: u32,
    pub result_faces: usize,
    pub h0: f64,
    pub comps: Vec<CompReport<'a>>,
    pub n_failed_tris: usize,
    pub chains: u32,
    pub correction: Correction,
    pub params: Option<Params>,
}

impl Default for Summary<'_> {
    fn default() -> Self {
        Summary {
            status: Status::Ok,
            error: String::new(),
            ingest: None,
            ingest_ms: 0.0,
            comp_ms: 0.0,
            stage_ms: Vec::new(),
            total_ms: 0.0,
            warnings: Vec::new(),
            target: 0,
            result_faces: 0,
            h0: 0.0,
            comps: Vec::new(),
            n_failed_tris: 0,
            chains: 0,
            correction: Correction::default(),
            params: None,
        }
    }
}

/// A finite number, or 0 (JSON has no NaN or infinity).
fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { json!(0) }
}

/// Interior valence 3, 5 and other irregular counts (quad_stats rules).
fn cones(q: Option<&QuadMesh>) -> Value {
    let Some(q) = q else {
        return json!({"v3": 0, "v5": 0, "other": 0});
    };
    let st = aq_eval::quad_stats(q);
    let mut v3 = 0u32;
    let mut v5 = 0u32;
    let mut other = 0u32;
    for (k, &n) in &st.valence {
        match k.as_str() {
            "3" => v3 += n,
            "5" => v5 += n,
            "4" => {}
            k if k.starts_with('b') => {}
            _ => other += n,
        }
    }
    json!({"v3": v3, "v5": v5, "other": other})
}

fn comp_json(c: &CompReport) -> Value {
    let quads = c.mesh.map_or(0, |q| q.n_faces());
    let count_err = if c.target > 0 && c.mesh.is_some() {
        (quads as f64 - c.target as f64) / c.target as f64
    } else {
        0.0
    };
    let mut m = Map::new();
    m.insert("id".into(), json!(c.id));
    m.insert("status".into(), json!(c.status.name()));
    m.insert("rung".into(), json!(c.status.rung()));
    m.insert("target".into(), json!(c.target));
    m.insert("quads".into(), json!(quads));
    m.insert("cones".into(), cones(c.mesh));
    m.insert("field".into(), json!({}));
    m.insert("param".into(), json!({}));
    m.insert(
        "layout".into(),
        json!({"patches": 0, "arcs": 0, "health_mean": 0, "health_max": 0, "seed_rounds": 0}),
    );
    m.insert(
        "quant".into(),
        json!({"sigma": 0, "solves": 0, "cuts": 0, "count_err": num(count_err)}),
    );
    m.insert("patch".into(), json!({}));
    m.insert("post".into(), json!({}));
    let mut warnings: Vec<Value> = c.warnings.iter().map(|w| json!(w)).collect();
    for (k, v) in &c.extra {
        if k == "warnings" {
            if let Value::Array(a) = v {
                warnings.extend(a.iter().cloned());
            }
        } else {
            m.insert(k.clone(), v.clone());
        }
    }
    m.insert("warnings".into(), Value::Array(warnings));
    m.insert("debris".into(), json!(c.debris));
    m.insert("tris".into(), json!(c.tris));
    m.insert("area".into(), num(c.area));
    m.insert("chi".into(), json!(c.chi));
    m.insert("closed".into(), json!(c.closed));
    m.insert("time_ms".into(), num(c.time_ms));
    m.insert("error".into(), json!(c.error.clone().unwrap_or_default()));
    Value::Object(m)
}

fn params_json(p: &Params) -> Value {
    json!({
        "target_quads": p.target_quads,
        "adaptive_size": p.adaptive_size,
        "curvature_align": p.curvature_align,
        "flow_smoothness": p.flow_smoothness,
        "density_strength": p.density_strength,
        "guide_strength": p.guide_strength,
        "hard_edges": p.hard_edges,
        "close_loops": p.close_loops,
        "allow_collapse": p.allow_collapse,
        "ring_min": p.ring_min,
        "max_proxy_tris": p.max_proxy_tris,
        "threads": p.threads,
        "time_limit_s": num(p.time_limit_s as f64),
        "debug_flags": p.debug_flags,
        "seed": p.seed,
    })
}

/// Builds the report.
pub(crate) fn build(s: &Summary) -> Value {
    let mut timing = Map::new();
    for st in [
        Stage::Ingest,
        Stage::Proxy,
        Stage::Curv,
        Stage::Field,
        Stage::Loops,
        Stage::Cut,
        Stage::Param,
        Stage::Layout,
        Stage::Quantize,
        Stage::Patch,
        Stage::Post,
    ] {
        let t: f64 = s.stage_ms.iter().filter(|x| x.0 == st).map(|x| x.1).sum();
        timing.insert(st.name().into(), num(t));
    }
    timing.insert("ingest".into(), num(s.ingest_ms));
    let fb: f64 = s
        .stage_ms
        .iter()
        .filter(|x| x.0 == Stage::Fallback)
        .map(|x| x.1)
        .sum();
    timing.insert(
        "fallback".into(),
        num(if s.stage_ms.is_empty() { s.comp_ms } else { fb }),
    );
    timing.insert("total".into(), num(s.total_ms));
    let comps: Vec<Value> = s.comps.iter().map(comp_json).collect();
    let err = if s.target > 0 && s.result_faces > 0 {
        (s.result_faces as f64 - s.target as f64) / s.target as f64
    } else {
        0.0
    };
    let ingest = match &s.ingest {
        Some(r) => json!({
            "dropped_faces": r.dropped_faces,
            "collapsed_faces": r.collapsed_faces,
            "split_edges": r.split_edges,
            "split_verts": r.split_verts,
            "flipped_faces": r.flipped_faces,
            "dropped_features": r.dropped_features,
            "rerouted_features": r.rerouted_features,
            "components": r.components,
        }),
        None => json!({}),
    };
    json!({
        "abi": ABI_VERSION,
        "engine": ENGINE_PATH,
        "status": s.status.code(),
        "error": s.error,
        "timing_ms": Value::Object(timing),
        "components": comps,
        "count": {"target": s.target, "result": s.result_faces, "err": num(err)},
        "warnings": s.warnings,
        "h0": num(s.h0),
        "chains": s.chains,
        "failed_tris": s.n_failed_tris,
        "count_correction": {
            "ran": s.correction.ran,
            "retried": s.correction.retried,
            "improved": s.correction.improved,
        },
        "ingest": ingest,
        "params": s.params.as_ref().map_or(json!({}), params_json),
    })
}

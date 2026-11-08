//! `autoquad batch`: runs the evaluation suite (DESIGN 10), compares with the QuadriFlow
//! baselines, evaluates the gates and writes `summary.json` and an optional markdown summary.
//!
//! Suite format (`eval/suite.toml`, read with [`crate::toml`]):
//! * `[suite]`: `testdata`, `baselines` (paths relative to the suite file), `seeds`,
//!   `adaptive`, `time_limit_s`, `seeds_required`.
//! * `[gates.<M>]` (M0, M1, M2): global gates, see [`Gates`].
//! * `[[mesh]]`: `name`, `mesh` (file stem, default `name`), `target`, `features`
//!   (`none`, `auto`), `hard_angle`, `sym` (`""`, `x`, `y`, `z`), `upsample`, `perf`,
//!   `loop_axis` (`all`, `x`, `y`, `z`, `pca`, or absent), and per-milestone gates with the
//!   suffix `_m1` / `_m2`: `irregular_<m>` (max), `irregular_exact_<m>`, `loops_min_<m>`,
//!   `helix_max_<m>`; plus `irregular_at_corners`, `feature_cover`, `open_edges_max`,
//!   `seam_poles_max`, and the reference values `irregular_lb`, `irregular_qf`.
//!
//! A mesh passes when its gates hold for seed 0 and for at least `seeds_required` seeds
//! (at most the number of seeds run).

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use aq_core::{QuadMesh, V3};
use aq_engine::{Params, Status};
use aq_eval::{LoopReport, QuadStats, loop_closure, quad_stats};

use crate::flow::{self, Features, Outcome, RemeshOpts};
use crate::stats::pca_axis;

/// Global gates of one milestone (absent key = gate not applied).
#[derive(Clone, Debug, Default)]
pub struct Gates {
    pub quad_ratio: Option<f64>,
    /// false: a component on SF (status sf or debris_sf) fails the gate.
    pub allow_sf: bool,
    pub count_err: Option<f64>,
    /// edge_cv <= min(QuadriFlow value, cap).
    pub edge_cv_cap: Option<f64>,
    pub angle_dev_vs_qf: bool,
    pub vertex_dist_h: Option<f64>,
    pub hausdorff_h: Option<f64>,
    /// Inputs below 50k triangles.
    pub small_time_s: Option<f64>,
    pub perf_time_s: Option<f64>,
    /// Feature coverage tolerance in normalized units (diag 10).
    pub feature_cover_tol: f64,
    /// Apply the per-mesh gates with this suffix ("m1", "m2"), or none.
    pub mesh_suffix: Option<String>,
}

fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}
fn s<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}
fn b(v: &Value, k: &str) -> Option<bool> {
    v.get(k).and_then(Value::as_bool)
}

impl Gates {
    pub fn from_json(v: &Value, milestone: &str) -> Gates {
        Gates {
            quad_ratio: f(v, "quad_ratio"),
            allow_sf: b(v, "allow_sf").unwrap_or(true),
            count_err: f(v, "count_err"),
            edge_cv_cap: f(v, "edge_cv_cap"),
            angle_dev_vs_qf: b(v, "angle_dev_vs_qf").unwrap_or(false),
            vertex_dist_h: f(v, "vertex_dist_h"),
            hausdorff_h: f(v, "hausdorff_h"),
            small_time_s: f(v, "small_time_s"),
            perf_time_s: f(v, "perf_time_s"),
            feature_cover_tol: f(v, "feature_cover_tol").unwrap_or(1e-6),
            mesh_suffix: s(v, "mesh_gates")
                .map(str::to_string)
                .or_else(|| match milestone {
                    "M1" => Some("m1".into()),
                    "M2" => Some("m2".into()),
                    _ => None,
                }),
        }
    }
}

/// One suite entry.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub mesh: String,
    pub target: u32,
    pub features: String,
    pub hard_angle: f64,
    pub sym: String,
    pub upsample: u32,
    pub perf: bool,
    pub loop_axis: Option<String>,
    pub raw: Value,
}

impl Entry {
    fn from_json(v: &Value) -> Result<Entry, String> {
        let name = s(v, "name")
            .ok_or("a [[mesh]] entry has no name")?
            .to_string();
        let target = v
            .get("target")
            .and_then(Value::as_u64)
            .ok_or(format!("{name}: no target"))? as u32;
        Ok(Entry {
            mesh: s(v, "mesh").unwrap_or(&name).to_string(),
            target,
            features: s(v, "features").unwrap_or("none").to_string(),
            hard_angle: f(v, "hard_angle").unwrap_or(30.0),
            sym: s(v, "sym").unwrap_or("").to_string(),
            upsample: v.get("upsample").and_then(Value::as_u64).unwrap_or(0) as u32,
            perf: b(v, "perf").unwrap_or(false),
            loop_axis: s(v, "loop_axis").map(str::to_string),
            raw: v.clone(),
            name,
        })
    }
    fn gate(&self, key: &str, suffix: &Option<String>) -> Option<f64> {
        let sfx = suffix.as_ref()?;
        f(&self.raw, &format!("{key}_{sfx}"))
    }
}

/// A loaded suite.
#[derive(Clone, Debug)]
pub struct Suite {
    pub dir: PathBuf,
    pub testdata: PathBuf,
    pub baselines: PathBuf,
    pub seeds: Vec<u64>,
    pub adaptive: f32,
    pub time_limit_s: f32,
    pub seeds_required: usize,
    pub gates: Value,
    pub entries: Vec<Entry>,
}

pub fn load_suite(path: &Path) -> Result<Suite, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v = crate::toml::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let su = v.get("suite").cloned().unwrap_or(json!({}));
    let rel = |k: &str, d: &str| dir.join(s(&su, k).unwrap_or(d));
    let seeds: Vec<u64> = su
        .get("seeds")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_else(|| vec![0]);
    let entries = v
        .get("mesh")
        .and_then(Value::as_array)
        .ok_or("the suite has no [[mesh]] entry")?
        .iter()
        .map(Entry::from_json)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Suite {
        testdata: rel("testdata", "../testdata"),
        baselines: rel("baselines", "../baselines"),
        seeds,
        adaptive: f(&su, "adaptive").unwrap_or(0.0) as f32,
        time_limit_s: f(&su, "time_limit_s").unwrap_or(60.0) as f32,
        seeds_required: su
            .get("seeds_required")
            .and_then(Value::as_u64)
            .unwrap_or(4) as usize,
        gates: v.get("gates").cloned().unwrap_or(json!({})),
        entries,
        dir,
    })
}

/// One gate result.
#[derive(Clone, Debug)]
pub struct GateResult {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

/// One run (mesh, seed).
#[derive(Clone, Debug)]
pub struct Row {
    pub mesh: String,
    pub seed: u64,
    pub status: i32,
    pub error: String,
    pub faces: u32,
    pub count_err: f64,
    pub stats: QuadStats,
    pub vdist_h: f64,
    pub haus_h: f64,
    pub loops: Option<LoopReport>,
    pub seconds: f64,
    pub comp_status: Vec<String>,
    pub open_edges: Option<u32>,
    pub seam_poles: Option<u32>,
    pub cover: Option<(usize, usize)>,
    pub corners_ok: Option<bool>,
    pub input_tris: usize,
    pub gates: Vec<GateResult>,
}

impl Row {
    pub fn pass(&self) -> bool {
        self.gates.iter().all(|g| g.pass)
    }
}

/// Distance from p to segment ab.
fn seg_dist(p: V3, a: V3, bq: V3) -> f64 {
    use aq_core::math::{dist, dot, lerp, norm2, sub};
    let ab = sub(bq, a);
    let l2 = norm2(ab);
    let t = if l2 > 0.0 {
        (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist(p, lerp(a, bq, t))
}

/// Unique undirected edges of a polygon mesh.
fn quad_edges(q: &QuadMesh) -> Vec<(u32, u32)> {
    let mut e = Vec::new();
    for fi in 0..q.n_faces() {
        let fv = q.face(fi);
        for i in 0..fv.len() {
            let (a, c) = (fv[i], fv[(i + 1) % fv.len()]);
            e.push((a.min(c), a.max(c)));
        }
    }
    e.sort_unstable();
    e.dedup();
    e
}

/// Feature segments covered by quad edges: every sample (5 per segment) within tol of an
/// output edge. Brute force over a uniform grid of the output edges.
pub fn feature_cover(q: &QuadMesh, segs: &[[V3; 2]], tol: f64) -> (usize, usize) {
    let edges = quad_edges(q);
    if segs.is_empty() {
        return (0, 0);
    }
    // Grid over the output edges' bounding boxes.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &q.pos {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max).max(1e-12);
    let n = 32usize;
    let cell = ext / n as f64 + 1e-12;
    let idx = |p: V3, k: usize| (((p[k] - lo[k]) / cell).floor().max(0.0) as usize).min(n - 1);
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); n * n * n];
    for (ei, &(a, c)) in edges.iter().enumerate() {
        let (pa, pc) = (q.pos[a as usize], q.pos[c as usize]);
        let r: Vec<(usize, usize)> = (0..3)
            .map(|k| (idx(pa, k).min(idx(pc, k)), idx(pa, k).max(idx(pc, k))))
            .collect();
        for i in r[0].0..=r[0].1 {
            for j in r[1].0..=r[1].1 {
                for l in r[2].0..=r[2].1 {
                    grid[(i * n + j) * n + l].push(ei as u32);
                }
            }
        }
    }
    let mut covered = 0;
    for sg in segs {
        let ok = (0..5).all(|t| {
            let p = aq_core::math::lerp(sg[0], sg[1], t as f64 / 4.0);
            let (i, j, l) = (idx(p, 0), idx(p, 1), idx(p, 2));
            let mut best = f64::INFINITY;
            for di in i.saturating_sub(1)..=(i + 1).min(n - 1) {
                for dj in j.saturating_sub(1)..=(j + 1).min(n - 1) {
                    for dl in l.saturating_sub(1)..=(l + 1).min(n - 1) {
                        for &ei in &grid[(di * n + dj) * n + dl] {
                            let (a, c) = edges[ei as usize];
                            best = best.min(seg_dist(p, q.pos[a as usize], q.pos[c as usize]));
                        }
                    }
                }
            }
            best <= tol
        });
        if ok {
            covered += 1;
        }
    }
    (covered, segs.len())
}

/// Irregular vertices (quad_stats rules) of a polygon mesh.
pub fn irregular_vertices(q: &QuadMesh) -> Vec<u32> {
    let mut e: Vec<(u32, u32)> = Vec::new();
    for fi in 0..q.n_faces() {
        let fv = q.face(fi);
        for i in 0..fv.len() {
            let (a, c) = (fv[i], fv[(i + 1) % fv.len()]);
            e.push((a.min(c), a.max(c)));
        }
    }
    e.sort_unstable();
    let mut val = vec![0u32; q.pos.len()];
    let mut bnd = vec![false; q.pos.len()];
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j] == e[i] {
            j += 1;
        }
        for v in [e[i].0, e[i].1] {
            val[v as usize] += 1;
            if j - i == 1 {
                bnd[v as usize] = true;
            }
        }
        i = j;
    }
    (0..q.pos.len() as u32)
        .filter(|&v| {
            let k = val[v as usize];
            k > 0
                && if bnd[v as usize] {
                    !(k == 2 || k == 3)
                } else {
                    k != 4
                }
        })
        .collect()
}

/// Points where 3 or more feature segments meet.
fn feature_corners(segs: &[[V3; 2]]) -> Vec<V3> {
    let mut pts: Vec<[u64; 3]> = segs
        .iter()
        .flat_map(|s| s.iter().map(|p| p.map(f64::to_bits)))
        .collect();
    pts.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < pts.len() {
        let mut j = i + 1;
        while j < pts.len() && pts[j] == pts[i] {
            j += 1;
        }
        if j - i >= 3 {
            out.push(pts[i].map(f64::from_bits));
        }
        i = j;
    }
    out
}

/// Baseline stats of `<dir>/<mesh>_qf<target>.obj`, if present.
fn baseline(dir: &Path, mesh: &str, target: u32) -> Option<QuadStats> {
    let p = dir.join(format!("{mesh}_qf{target}.obj"));
    let q = aq_io::read_obj_poly(&p).ok()?;
    Some(quad_stats(&q))
}

/// Runs one (entry, seed) and evaluates its gates.
fn run_one(
    su: &Suite,
    e: &Entry,
    seed: u64,
    out: &Path,
    gates: &Gates,
    qf: Option<&QuadStats>,
) -> Row {
    let mut o = RemeshOpts::new(su.testdata.join(format!("{}.obj", e.mesh)));
    o.output = Some(out.join(format!("{}_s{seed}.obj", e.name)));
    o.report = Some(out.join(format!("{}_s{seed}.json", e.name)));
    o.params = Params {
        target_quads: e.target,
        seed,
        adaptive_size: su.adaptive,
        time_limit_s: su.time_limit_s,
        ..Params::default()
    };
    o.features = if e.features == "auto" {
        Features::Auto
    } else {
        Features::None
    };
    o.hard_angle = e.hard_angle;
    o.sym = e
        .sym
        .chars()
        .filter_map(|c| "xyz".find(c.to_ascii_lowercase()))
        .collect();
    o.upsample = e.upsample;
    let res = flow::run(&o);
    let mut row = Row {
        mesh: e.name.clone(),
        seed,
        status: -1,
        error: String::new(),
        faces: 0,
        count_err: f64::NAN,
        stats: QuadStats::default(),
        vdist_h: f64::NAN,
        haus_h: f64::NAN,
        loops: None,
        seconds: 0.0,
        comp_status: Vec::new(),
        open_edges: None,
        seam_poles: None,
        cover: None,
        corners_ok: None,
        input_tris: 0,
        gates: Vec::new(),
    };
    let oc: Outcome = match res {
        Ok(x) => x,
        Err(err) => {
            row.error = err.clone();
            row.gates.push(GateResult {
                name: "run".into(),
                pass: false,
                detail: err,
            });
            return row;
        }
    };
    row.status = oc.status.code();
    row.error = oc.error.clone();
    row.seconds = oc.seconds;
    row.stats = oc.stats.q.clone();
    row.faces = row.stats.faces;
    row.count_err = oc.stats.count_err.unwrap_or(f64::NAN);
    row.input_tris = oc.input.tris.len();
    let h = oc.stats.h;
    if let Some((mx, _)) = oc.stats.vdist {
        row.vdist_h = mx / h;
    }
    if let Some(hd) = &oc.stats.haus {
        row.haus_h = hd.symmetric() / h;
    }
    row.comp_status = oc.report["components"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| c["status"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if let Some(w) = &oc.weld {
        row.open_edges = Some(w.open_edges);
        row.seam_poles = Some(w.seam_poles);
    }
    if let Some(ax) = &e.loop_axis {
        let axis = match ax.as_str() {
            "all" => None,
            "pca" => Some(pca_axis(&oc.input)),
            other => crate::args::parse_axis(other).ok(),
        };
        row.loops = Some(loop_closure(&oc.mesh, axis));
    }
    let diag = oc.input.bbox_diag();
    let tol_units = gates.feature_cover_tol * diag / aq_engine::host::WORK_DIAG;
    if b(&e.raw, "feature_cover").unwrap_or(false) {
        row.cover = Some(feature_cover(&oc.mesh, &oc.feature_segments, tol_units));
    }
    if b(&e.raw, "irregular_at_corners").unwrap_or(false) {
        let corners = feature_corners(&oc.feature_segments);
        let irr = irregular_vertices(&oc.mesh);
        let ok = irr.iter().all(|&v| {
            corners
                .iter()
                .any(|c| aq_core::math::dist(*c, oc.mesh.pos[v as usize]) <= tol_units)
        });
        row.corners_ok = Some(ok);
    }
    row.gates = evaluate(e, &row, gates, qf);
    row
}

fn gate(v: &mut Vec<GateResult>, name: &str, pass: bool, detail: String) {
    v.push(GateResult {
        name: name.into(),
        pass,
        detail,
    });
}

/// The gates of one row.
fn evaluate(e: &Entry, r: &Row, g: &Gates, qf: Option<&QuadStats>) -> Vec<GateResult> {
    let mut v = Vec::new();
    let ok_status = r.status == Status::Ok.code() || r.status == Status::Degraded.code();
    gate(
        &mut v,
        "success",
        ok_status && r.faces > 0,
        format!("status {}", r.status),
    );
    if let Some(x) = g.quad_ratio {
        gate(
            &mut v,
            "quad_ratio",
            r.stats.quad_ratio >= x,
            format!("{:.4} >= {x}", r.stats.quad_ratio),
        );
    }
    if !g.allow_sf {
        let n = r
            .comp_status
            .iter()
            .filter(|s| s.contains("sf") || *s == "failed")
            .count();
        gate(
            &mut v,
            "no_sf",
            n == 0,
            format!("{n} components on SF or failed"),
        );
    }
    if let Some(x) = g.count_err {
        gate(
            &mut v,
            "count_err",
            r.count_err.abs() <= x,
            format!("|{:+.4}| <= {x}", r.count_err),
        );
    }
    if let Some(cap) = g.edge_cv_cap {
        let lim = qf.map_or(cap, |q| q.edge_cv.min(cap));
        gate(
            &mut v,
            "edge_cv",
            r.stats.edge_cv <= lim,
            format!("{:.4} <= {lim:.4}", r.stats.edge_cv),
        );
    }
    if g.angle_dev_vs_qf
        && let Some(q) = qf
    {
        gate(
            &mut v,
            "angle_dev",
            r.stats.angle_dev_deg <= q.angle_dev_deg,
            format!("{:.2} <= {:.2}", r.stats.angle_dev_deg, q.angle_dev_deg),
        );
    }
    if let Some(x) = g.vertex_dist_h {
        gate(
            &mut v,
            "vertex_dist",
            r.vdist_h <= x,
            format!("{:.3} h <= {x} h", r.vdist_h),
        );
    }
    if let Some(x) = g.hausdorff_h {
        gate(
            &mut v,
            "hausdorff",
            r.haus_h <= x,
            format!("{:.3} h <= {x} h", r.haus_h),
        );
    }
    if e.perf {
        if let Some(x) = g.perf_time_s {
            gate(
                &mut v,
                "time",
                r.seconds < x,
                format!("{:.2} s < {x} s", r.seconds),
            );
        }
    } else if r.input_tris < 50_000
        && let Some(x) = g.small_time_s
    {
        gate(
            &mut v,
            "time",
            r.seconds < x,
            format!("{:.2} s < {x} s", r.seconds),
        );
    }
    let sfx = &g.mesh_suffix;
    if let Some(x) = e.gate("irregular", sfx) {
        gate(
            &mut v,
            "irregular",
            (r.stats.irregular as f64) <= x,
            format!("{} <= {x}", r.stats.irregular),
        );
    }
    if let Some(x) = e.gate("irregular_exact", sfx) {
        gate(
            &mut v,
            "irregular_exact",
            (r.stats.irregular as f64) == x,
            format!("{} == {x}", r.stats.irregular),
        );
    }
    if let Some(x) = e.gate("loops_min", sfx) {
        let fr = r.loops.map_or(0.0, |l| l.closed_frac);
        gate(&mut v, "loops", fr >= x, format!("closed {fr:.3} >= {x}"));
    }
    if let Some(x) = e.gate("helix_max", sfx) {
        let n = r.loops.map_or(0, |l| l.helix_suspects);
        gate(&mut v, "helix", (n as f64) <= x, format!("{n} <= {x}"));
    }
    if sfx.is_some()
        && let Some(ok) = r.corners_ok
    {
        gate(&mut v, "irregular_at_corners", ok, format!("{ok}"));
    }
    if let Some((c, n)) = r.cover {
        gate(
            &mut v,
            "feature_cover",
            c == n,
            format!("{c} of {n} feature edges covered"),
        );
    }
    if let Some(x) = f(&e.raw, "open_edges_max") {
        let n = r.open_edges.unwrap_or(u32::MAX);
        gate(&mut v, "open_edges", (n as f64) <= x, format!("{n} <= {x}"));
    }
    if let Some(x) = f(&e.raw, "seam_poles_max") {
        let n = r.seam_poles.unwrap_or(u32::MAX);
        gate(&mut v, "seam_poles", (n as f64) <= x, format!("{n} <= {x}"));
    }
    v
}

/// Batch options.
#[derive(Clone, Debug)]
pub struct BatchOpts {
    pub suite: PathBuf,
    pub out: PathBuf,
    pub seeds: Option<Vec<u64>>,
    pub compare: Option<PathBuf>,
    pub markdown: Option<PathBuf>,
    pub milestone: String,
    /// Only these entries (by name), when not empty.
    pub only: Vec<String>,
}

/// Result of a batch.
#[derive(Clone, Debug)]
pub struct BatchResult {
    pub rows: Vec<Row>,
    /// Per entry: (name, seed-0 pass, seeds passed, seeds run, mesh pass).
    pub verdicts: Vec<(String, bool, usize, usize, bool)>,
    pub summary: Value,
    /// Seeds that must pass (besides the first seed), at most the number of seeds run.
    pub required: usize,
}

pub fn run_batch(o: &BatchOpts) -> Result<BatchResult, String> {
    let su = load_suite(&o.suite)?;
    let seeds = o.seeds.clone().unwrap_or_else(|| su.seeds.clone());
    let gv = su
        .gates
        .get(&o.milestone)
        .cloned()
        .ok_or(format!("the suite has no [gates.{}]", o.milestone))?;
    let gates = Gates::from_json(&gv, &o.milestone);
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    // Without --compare, the suite's baselines folder is used when it exists.
    let compare = o
        .compare
        .clone()
        .or_else(|| su.baselines.is_dir().then(|| su.baselines.clone()));
    let mut rows = Vec::new();
    let mut verdicts = Vec::new();
    let mut entries_json = Vec::new();
    for e in &su.entries {
        if !o.only.is_empty() && !o.only.contains(&e.name) {
            continue;
        }
        let qf = compare
            .as_ref()
            .and_then(|d| baseline(d, &e.mesh, e.target));
        let mut er: Vec<Row> = Vec::new();
        for &seed in &seeds {
            let r = run_one(&su, e, seed, &o.out, &gates, qf.as_ref());
            eprintln!(
                "{:24} seed {seed}: status {} faces {} irregular {} {:.1} s {}",
                e.name,
                r.status,
                r.faces,
                r.stats.irregular,
                r.seconds,
                if r.pass() { "PASS" } else { "FAIL" }
            );
            er.push(r);
        }
        let seed0 = er
            .iter()
            .find(|r| r.seed == seeds.first().copied().unwrap_or(0))
            .is_some_and(Row::pass);
        let passed = er.iter().filter(|r| r.pass()).count();
        let need = su.seeds_required.min(er.len());
        let ok = seed0 && passed >= need;
        verdicts.push((e.name.clone(), seed0, passed, er.len(), ok));
        entries_json.push(json!({
            "name": e.name,
            "target": e.target,
            "qf": qf.as_ref().map(|q| json!({"irregular": q.irregular, "edge_cv": q.edge_cv, "angle_dev_deg": q.angle_dev_deg, "faces": q.faces})),
            "irregular_lb": e.raw.get("irregular_lb"),
            "irregular_qf": e.raw.get("irregular_qf"),
            "seed0_pass": seed0, "seeds_passed": passed, "seeds_run": er.len(), "pass": ok,
            "runs": er.iter().map(row_json).collect::<Vec<_>>(),
        }));
        rows.extend(er);
    }
    let all_ok = verdicts.iter().all(|v| v.4);
    let summary = json!({
        "suite": o.suite.display().to_string(),
        "suite_dir": su.dir.display().to_string(),
        "testdata": su.testdata.display().to_string(),
        "baselines": compare.as_ref().map(|d| d.display().to_string()),
        "milestone": o.milestone,
        "engine": aq_engine::ENGINE_PATH,
        "seeds": seeds,
        "pass": all_ok,
        "meshes_passed": verdicts.iter().filter(|v| v.4).count(),
        "meshes": entries_json,
    });
    flow::write_json(&o.out.join("summary.json"), &summary)?;
    let required = su.seeds_required.min(seeds.len());
    let res = BatchResult {
        rows,
        verdicts,
        summary,
        required,
    };
    if let Some(md) = &o.markdown {
        let text = markdown(&res, o, &seeds);
        if let Some(d) = md.parent()
            && !d.as_os_str().is_empty()
        {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::write(md, text).map_err(|e| format!("{}: {e}", md.display()))?;
    }
    Ok(res)
}

fn nz(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn row_json(r: &Row) -> Value {
    json!({
        "seed": r.seed, "status": r.status, "error": r.error, "faces": r.faces,
        "count_err": nz(r.count_err), "irregular": r.stats.irregular, "quad_ratio": nz(r.stats.quad_ratio),
        "edge_cv": nz(r.stats.edge_cv), "angle_dev_deg": nz(r.stats.angle_dev_deg),
        "vertex_dist_h": nz(r.vdist_h), "hausdorff_h": nz(r.haus_h),
        "loops": r.loops.map(|l| json!({"loops": l.loops, "closed": l.closed, "closed_frac": l.closed_frac, "helix_suspects": l.helix_suspects})),
        "seconds": r.seconds, "comp_status": r.comp_status,
        "open_edges": r.open_edges, "seam_poles": r.seam_poles,
        "feature_cover": r.cover.map(|(c, n)| json!([c, n])), "irregular_at_corners": r.corners_ok,
        "pass": r.pass(),
        "gates": r.gates.iter().map(|g| json!({"name": g.name, "pass": g.pass, "detail": g.detail})).collect::<Vec<_>>(),
    })
}

fn fmt(x: f64, d: usize) -> String {
    if x.is_finite() {
        format!("{x:.d$}")
    } else {
        "-".into()
    }
}

/// The markdown summary: every metric against QuadriFlow and against the gate.
fn markdown(res: &BatchResult, o: &BatchOpts, seeds: &[u64]) -> String {
    let mut m = String::new();
    m += &format!("# AutoQuad batch summary (gates {})\n\n", o.milestone);
    m += &format!(
        "Suite `{}`, engine path {}, seeds {:?}. A mesh passes when its gates hold for seed {} and for at least {} of the seeds run.\n\n",
        o.suite.display(),
        aq_engine::ENGINE_PATH,
        seeds,
        seeds.first().copied().unwrap_or(0),
        res.required
    );
    m += "| mesh | status | faces | count err | irregular (QF) | edge CV (QF) | angle dev (QF) | vertex dist / h | Hausdorff / h | loops closed | time s | components | seeds passed | verdict |\n";
    m += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
    let meshes = res.summary["meshes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (i, v) in res.verdicts.iter().enumerate() {
        let Some(r) = res.rows.iter().find(|r| r.mesh == v.0) else {
            continue;
        };
        let qf = meshes
            .get(i)
            .map(|x| x["qf"].clone())
            .unwrap_or(Value::Null);
        let q = |k: &str, d: usize| {
            qf.get(k)
                .and_then(Value::as_f64)
                .map_or("-".to_string(), |x| fmt(x, d))
        };
        let loops = r.loops.map_or("-".to_string(), |l| {
            format!("{} / {} ({:.2})", l.closed, l.loops, l.closed_frac)
        });
        let mut comps: Vec<(String, usize)> = Vec::new();
        for c in &r.comp_status {
            match comps.iter_mut().find(|x| &x.0 == c) {
                Some(x) => x.1 += 1,
                None => comps.push((c.clone(), 1)),
            }
        }
        let comps: Vec<String> = comps.iter().map(|(c, n)| format!("{n} {c}")).collect();
        m += &format!(
            "| {} | {} | {} | {}% | {} ({}) | {} ({}) | {} ({}) | {} | {} | {} | {} | {} | {} of {} | {} |\n",
            v.0,
            r.status,
            r.faces,
            fmt(100.0 * r.count_err, 1),
            r.stats.irregular,
            q("irregular", 0),
            fmt(r.stats.edge_cv, 3),
            q("edge_cv", 3),
            fmt(r.stats.angle_dev_deg, 2),
            q("angle_dev_deg", 2),
            fmt(r.vdist_h, 3),
            fmt(r.haus_h, 3),
            loops,
            fmt(r.seconds, 2),
            comps.join(", "),
            v.2,
            v.3,
            if v.4 { "PASS" } else { "FAIL" }
        );
    }
    m += "\n## Gates (first seed)\n\n";
    for v in &res.verdicts {
        let Some(r) = res.rows.iter().find(|r| r.mesh == v.0) else {
            continue;
        };
        let fails: Vec<String> = r
            .gates
            .iter()
            .filter(|g| !g.pass)
            .map(|g| format!("{} ({})", g.name, g.detail))
            .collect();
        let passes: Vec<String> = r
            .gates
            .iter()
            .filter(|g| g.pass)
            .map(|g| g.name.clone())
            .collect();
        m += &format!("- **{}**: ", v.0);
        if fails.is_empty() {
            m += "all gates hold";
        } else {
            m += &format!("fails {}", fails.join("; "));
        }
        if !passes.is_empty() {
            m += &format!(". Holds: {}", passes.join(", "));
        }
        m += ".\n";
    }
    let n_ok = res.verdicts.iter().filter(|v| v.4).count();
    m += &format!(
        "\n**{n_ok} of {} meshes pass the {} gates.**\n",
        res.verdicts.len(),
        o.milestone
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_grid(n: u32) -> QuadMesh {
        let mut q = QuadMesh::new();
        for j in 0..=n {
            for i in 0..=n {
                q.pos.push([i as f64, j as f64, 0.0]);
            }
        }
        q.vflag = vec![0; q.pos.len()];
        let id = |i: u32, j: u32| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                q.push_face(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], 0);
            }
        }
        q
    }

    #[test]
    fn cover_and_irregular() {
        let q = square_grid(4);
        // The grid line y = 2 is covered, a segment between grid lines is not.
        let segs = vec![
            [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0]],
            [[0.5, 0.5, 0.0], [3.5, 0.5, 0.0]],
        ];
        assert_eq!(feature_cover(&q, &segs, 1e-9), (1, 2));
        // A regular grid: only the 4 corners (boundary valence 2) are regular too.
        assert!(irregular_vertices(&q).is_empty());
        let c = feature_corners(&[
            [[0.0; 3], [1.0, 0.0, 0.0]],
            [[0.0; 3], [0.0, 1.0, 0.0]],
            [[0.0; 3], [0.0, 0.0, 1.0]],
        ]);
        assert_eq!(c, vec![[0.0; 3]]);
    }

    #[test]
    fn gates_from_json() {
        let g = Gates::from_json(
            &json!({"quad_ratio": 1.0, "allow_sf": false, "count_err": 0.1}),
            "M1",
        );
        assert_eq!(g.quad_ratio, Some(1.0));
        assert!(!g.allow_sf);
        assert_eq!(g.mesh_suffix.as_deref(), Some("m1"));
        let g0 = Gates::from_json(&json!({}), "M0");
        assert!(g0.allow_sf && g0.mesh_suffix.is_none());
    }
}

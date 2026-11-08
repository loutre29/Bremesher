//! The engine call (DESIGN 7.1): ingest, components, budgets, ladder, merge, report.

use std::borrow::Cow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use aq_core::{AqError, Ctl, ErrKind, FeatureKind, Projector, QuadMesh, Stage};
use aq_fallback::{SfParams, sf_batch, sf_quadrangulate};
use aq_prep::{
    ComponentMesh, CornerRule, Ingested, apply_hard_mode, build_feature_set, ingest,
    split_components,
};

use crate::budget::{self, CompPlan};
use crate::merge::{self, euler_char};
use crate::pipeline::{self, Rung, RungInput};
use crate::report::{self, CompReport, Summary};
use crate::{CompStatus, Input, MAX_TARGET_QUADS, Output, Params, Status, debug};

/// SF relaxation iterations (DESIGN 6.14 step 4).
const SF_RELAX_ITERS: u32 = 5;
/// R1 is entered only while the component used less than this fraction of T_c (7.2).
const R1_TIME_FRAC: f64 = 0.4;
/// Count correction threshold (7.1 step 8, 13).
const COUNT_CORRECTION: f64 = 0.08;
/// Count correction needs at least this much time left.
const COUNT_CORRECTION_MIN_TIME: Duration = Duration::from_secs(1);
/// Thread cap of the per-call pool.
const MAX_THREADS: u32 = 1024;

/// Text of a caught panic payload.
pub(crate) fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Test hook: a deliberate panic (debug flags `PANIC_*`).
#[allow(clippy::panic)]
fn injected_panic(what: &str) -> ! {
    std::panic::panic_any(format!("injected panic ({what})"))
}

pub(crate) fn remesh(input: Input, p: &Params, ctl: &Ctl) -> Output {
    let t0 = Instant::now();
    let r = catch_unwind(AssertUnwindSafe(|| run_checked(input, p, ctl, t0)));
    let out = match r {
        Ok(o) => o,
        Err(payload) => {
            let msg = format!("panic: {}", panic_text(payload.as_ref()));
            empty_output(Status::Panic, msg, t0, Vec::new())
        }
    };
    ctl.progress(Stage::Engine, 1.0);
    out
}

/// An output with no mesh (statuses 3, 4, 5, and failures before the components exist).
fn empty_output(status: Status, error: String, t0: Instant, warnings: Vec<String>) -> Output {
    empty_with_failed(status, error, t0, warnings, Vec::new())
}

fn empty_with_failed(
    status: Status,
    error: String,
    t0: Instant,
    warnings: Vec<String>,
    failed: Vec<u32>,
) -> Output {
    let s = Summary {
        status,
        error: error.clone(),
        total_ms: ms(t0.elapsed()),
        warnings,
        n_failed_tris: failed.len(),
        ..Summary::default()
    };
    Output {
        status,
        mesh: QuadMesh::new(),
        comp_status: Vec::new(),
        failed_tris: failed,
        report: report::build(&s),
        error,
    }
}

/// See [`crate::error_report`].
pub(crate) fn error_report(status: Status, msg: &str) -> serde_json::Value {
    report::build(&Summary {
        status,
        error: msg.to_string(),
        ..Summary::default()
    })
}

/// Status 3 (7.2: a cancel returns at once with an empty output).
fn cancelled(t0: Instant, warnings: Vec<String>) -> Output {
    empty_output(
        Status::Cancelled,
        "cancelled by caller".into(),
        t0,
        warnings,
    )
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Checks and clamps the parameters (DESIGN 8.1 ranges). `Err` is `Status::InvalidInput`.
fn checked_params(p: &Params, warnings: &mut Vec<String>) -> Result<Params, String> {
    let mut q = p.clone();
    if q.target_quads == 0 || q.target_quads > MAX_TARGET_QUADS {
        return Err(format!(
            "target_quads = {} (must be in 1..={MAX_TARGET_QUADS})",
            q.target_quads
        ));
    }
    if q.time_limit_s.is_nan() || q.time_limit_s <= 0.0 {
        return Err(format!(
            "time_limit_s = {} (must be positive)",
            q.time_limit_s
        ));
    }
    let mut clamp = |name: &str, v: &mut f32, lo: f32, hi: f32| -> Result<(), String> {
        if !v.is_finite() {
            return Err(format!("{name} = {v} is not finite"));
        }
        if *v < lo || *v > hi {
            let c = v.clamp(lo, hi);
            warnings.push(format!("{name} = {v} clamped to {c} (range [{lo}, {hi}])"));
            *v = c;
        }
        Ok(())
    };
    clamp("adaptive_size", &mut q.adaptive_size, 0.0, 1.0)?;
    clamp("curvature_align", &mut q.curvature_align, 0.0, 4.0)?;
    clamp("flow_smoothness", &mut q.flow_smoothness, 0.5, 3.0)?;
    clamp("density_strength", &mut q.density_strength, 0.0, 2.0)?;
    clamp("guide_strength", &mut q.guide_strength, 0.0, 4.0)?;
    if q.hard_edges > 2 {
        warnings.push(format!("hard_edges = {} clamped to 2", q.hard_edges));
        q.hard_edges = 2;
    }
    if q.threads > MAX_THREADS {
        warnings.push(format!("threads = {} clamped to {MAX_THREADS}", q.threads));
        q.threads = MAX_THREADS;
    }
    if let Some(s) = q.stop_after
        && s > 11
    {
        return Err(format!("stop_after = S{s} (must be S0..S11)"));
    }
    Ok(q)
}

fn run_checked(input: Input, p: &Params, ctl: &Ctl, t0: Instant) -> Output {
    ctl.progress(Stage::Engine, 0.0);
    let mut warnings = Vec::new();
    let p = match checked_params(p, &mut warnings) {
        Ok(p) => p,
        Err(e) => return empty_output(Status::InvalidInput, e, t0, warnings),
    };
    if ctl.is_cancelled() {
        return empty_output(
            Status::Cancelled,
            "cancelled by caller".into(),
            t0,
            warnings,
        );
    }
    // A limit too large for a Duration or an Instant means no limit (never a panic).
    let deadline = Duration::try_from_secs_f64(p.time_limit_s as f64)
        .ok()
        .and_then(|d| t0.checked_add(d))
        .unwrap_or(ctl.deadline());
    let ctl = ctl.sub(deadline);
    if p.debug_flags & debug::PANIC_TOP != 0 {
        injected_panic("top level");
    }
    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(p.threads as usize)
        .build()
    {
        Ok(pool) => pool,
        Err(e) => {
            return empty_output(Status::Failed, format!("thread pool: {e}"), t0, warnings);
        }
    };
    pool.install(|| body(input, &p, &ctl, t0, warnings))
}

/// Status, error and failed triangles for an error before the components exist.
fn early_failure(e: &AqError, n_tris: usize, t0: Instant, warnings: Vec<String>) -> Output {
    match e.kind {
        ErrKind::InvalidInput => empty_output(Status::InvalidInput, e.to_string(), t0, warnings),
        ErrKind::Cancelled => empty_output(Status::Cancelled, e.to_string(), t0, warnings),
        _ => {
            let all: Vec<u32> = (0..n_tris as u32).collect();
            empty_with_failed(Status::Failed, e.to_string(), t0, warnings, all)
        }
    }
}

/// Per S0 component (in `Topology::face_comp` order): its Euler characteristic and its
/// lowest face.
struct S0Comps {
    chi: Vec<i64>,
    first_face: Vec<u32>,
}

/// S0 invariants (7.3): manifold, oriented, finite, indices valid, chi per component.
///
/// The chi of every component is `V - E + F` over the ingested topology (the same count as
/// `Topology::euler_char`, done in one pass so that many components stay linear), and must
/// be that of a connected orientable surface: `chi = 2 - 2g - b` with `g >= 0`.
fn check_s0(ing: &Ingested) -> Result<S0Comps, AqError> {
    ing.mesh.validate().map_err(|e| e.at(Stage::Ingest))?;
    let t = &ing.topo;
    t.validate(&ing.mesh).map_err(|e| e.at(Stage::Ingest))?;
    if ing.face_src.len() != ing.mesh.tris.len() {
        return Err(AqError::invariant("S0: face_src size").at(Stage::Ingest));
    }
    let nc = t.n_comp as usize;
    let bad = || AqError::invariant("S0: face_comp out of range").at(Stage::Ingest);
    let mut chi = vec![0i64; nc];
    let mut first_face = vec![aq_core::NONE; nc];
    for (f, &c) in t.face_comp.iter().enumerate() {
        let x = chi.get_mut(c as usize).ok_or_else(bad)?;
        *x += 1;
        for k in 0..3 {
            if t.is_canonical((3 * f + k) as u32) {
                *x -= 1;
            }
        }
        if first_face[c as usize] == aq_core::NONE {
            first_face[c as usize] = f as u32;
        }
    }
    for &h in &t.vert_he {
        if h != aq_core::NONE {
            let c = *t.face_comp.get((h / 3) as usize).ok_or_else(bad)?;
            *chi.get_mut(c as usize).ok_or_else(bad)? += 1;
        }
    }
    let mut loops = vec![0i64; nc];
    for l in &t.boundary_loops {
        if let Some(&h) = l.first() {
            let c = *t.face_comp.get((h / 3) as usize).ok_or_else(bad)?;
            *loops.get_mut(c as usize).ok_or_else(bad)? += 1;
        }
    }
    for c in 0..nc {
        let two_g = 2 - loops[c] - chi[c];
        if first_face[c] == aq_core::NONE || two_g < 0 || two_g % 2 != 0 {
            return Err(AqError::invariant(format!(
                "S0: component {c} has chi {} with {} boundary loops",
                chi[c], loops[c]
            ))
            .at(Stage::Ingest));
        }
    }
    Ok(S0Comps { chi, first_face })
}

/// Drops the split components whose chi differs from S0's (7.3). Components are matched in
/// order by their first face, since `split_components` keeps the `face_comp` order and only
/// skips components. A dropped component's triangles become failed triangles.
fn match_s0_chi(
    comps: &mut Vec<ComponentMesh>,
    s0: &S0Comps,
    ing: &Ingested,
    warnings: &mut Vec<String>,
) {
    let mut j = 0usize;
    let mut bad = 0usize;
    comps.retain(|c| {
        let src0 = c.face_src.first().copied();
        while j < s0.first_face.len()
            && ing.face_src.get(s0.first_face[j] as usize).copied() != src0
        {
            j += 1;
        }
        let ok = j < s0.chi.len() && s0.chi[j] == c.chi;
        if ok {
            j += 1;
        } else {
            bad += 1;
        }
        ok
    });
    if bad > 0 {
        warnings.push(format!(
            "{bad} components dropped: their chi differs from the ingested one"
        ));
    }
}

fn body(input: Input, p: &Params, ctl: &Ctl, t0: Instant, mut warnings: Vec<String>) -> Output {
    let n_in_verts = input.mesh.pos.len();
    let n_in_tris = input.mesh.tris.len();
    if let Some(d) = input.density
        && d.len() != n_in_verts
    {
        let msg = format!("density has {} entries for {n_in_verts} vertices", d.len());
        return empty_output(Status::InvalidInput, msg, t0, warnings);
    }
    if p.adaptive_size > 0.0 || input.density.is_some() {
        warnings.push(if pipeline::AVAILABLE {
            "adaptive_size and the density map are accepted but ignored in M1 (uniform sizing)"
                .into()
        } else {
            "adaptive_size and the density map are accepted but not used on the M0 path (SF)".into()
        });
    }
    // S0.
    let t_ing = Instant::now();
    let ing = match ingest(input.mesh, input.feat_pairs, input.feat_kind, ctl) {
        Ok(i) => i,
        Err(e) => return early_failure(&e, n_in_tris, t0, warnings),
    };
    let s0 = match check_s0(&ing) {
        Ok(s) => s,
        Err(e) => return early_failure(&e, n_in_tris, t0, warnings),
    };
    let ingest_ms = ms(t_ing.elapsed());
    let dump = DumpCtx::new(p, &mut warnings);
    if let Some(d) = &dump {
        d.tri("ingest.obj", &ing.mesh, &mut warnings);
    }
    if let Some(s) = p.stop_after {
        if s <= 1 {
            warnings.push(format!(
                "stopped after S{s}: the output is the ingested triangle mesh"
            ));
            let mesh = aq_eval::poly_from_tris(&ing.mesh);
            let summ = Summary {
                status: Status::Ok,
                ingest: Some(ing.report.clone()),
                ingest_ms,
                total_ms: ms(t0.elapsed()),
                warnings,
                target: p.target_quads,
                result_faces: mesh.n_faces(),
                params: Some(p.clone()),
                ..Summary::default()
            };
            return Output {
                status: Status::Ok,
                mesh,
                comp_status: Vec::new(),
                failed_tris: Vec::new(),
                report: report::build(&summ),
                error: String::new(),
            };
        }
        warnings.push(format!("stop_after S{s} has no effect on the M0 path"));
    }
    // S1: components, scale, budgets.
    let mut comps = split_components(&ing);
    match_s0_chi(&mut comps, &s0, &ing, &mut warnings);
    drop(s0);
    let mut failed_tris: Vec<u32> = Vec::new();
    {
        let mut covered = vec![false; n_in_tris];
        for c in &comps {
            for &f in &c.face_src {
                if let Some(x) = covered.get_mut(f as usize) {
                    *x = true;
                }
            }
        }
        for &f in &ing.face_src {
            if covered.get(f as usize) == Some(&false) {
                failed_tris.push(f);
            }
        }
        if !failed_tris.is_empty() {
            warnings.push(format!(
                "{} triangles in components that could not be split off",
                failed_tris.len()
            ));
        }
    }
    let ingest_report = ing.report.clone();
    drop(ing);
    let a_total: f64 = comps.iter().map(|c| c.area).sum();
    let h0 = (a_total / p.target_quads as f64).sqrt();
    if comps.is_empty() || !(h0.is_finite() && h0 > 0.0) {
        let mut all: Vec<u32> = failed_tris;
        for c in &comps {
            all.extend_from_slice(&c.face_src);
        }
        all.sort_unstable();
        all.dedup();
        let msg = format!("no usable component (total area {a_total})");
        return empty_with_failed(Status::Failed, msg, t0, warnings, all);
    }
    let s = 1.0 / h0;
    for c in comps.iter_mut() {
        for q in c.mesh.pos.iter_mut() {
            *q = [q[0] * s, q[1] * s, q[2] * s];
        }
        c.area *= s * s;
    }
    let areas: Vec<f64> = comps.iter().map(|c| c.area).collect();
    let closed: Vec<bool> = comps.iter().map(|c| c.closed).collect();
    // WP-20: with several components, the thickness cap is planned globally: thin components
    // get their quads from the others (shares by quad demand instead of area) and every rung
    // uses the same cap floor. A single component plans inside its rung.
    let (plan_areas, thick_floor) = budget::thickness_shares(&comps, &areas, p, ctl, &mut warnings);
    let plan = budget::plan_targets(&plan_areas, &closed, p.target_quads);
    let proxy_tris = budget::proxy_shares(&plan, p.max_proxy_tris);
    let shares = budget::time_shares(&plan, ctl.remaining());
    // S2..S11 per component, in parallel.
    let t_comp = Instant::now();
    let jobs = JobCtx {
        p,
        density: input.density,
        ctl,
        thick_floor,
    };
    let mut results: Vec<JobResult> = comps
        .par_iter()
        .enumerate()
        .map(|(i, c)| jobs.run(i as u32, c, plan[i], proxy_tris[i], shares[i]))
        .collect();
    if ctl.is_cancelled() || results.iter().any(|r| r.cancelled) {
        return empty_output(
            Status::Cancelled,
            "cancelled by caller".into(),
            t0,
            warnings,
        );
    }
    // Count correction (7.1 step 8), once.
    let corr = count_correction(&jobs, &comps, &plan, &mut results);
    if ctl.is_cancelled() {
        return cancelled(t0, warnings);
    }
    let comp_ms = ms(t_comp.elapsed());
    // Merge.
    let mut comp_status = Vec::with_capacity(results.len());
    let mut parts: Vec<(u32, &QuadMesh)> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut degraded = false;
    for (i, r) in results.iter().enumerate() {
        comp_status.push(r.status);
        degraded |= r.degraded;
        match &r.mesh {
            Some(q) => parts.push((i as u32, q)),
            None => {
                failed_tris.extend_from_slice(&comps[i].face_src);
                errors.push(format!(
                    "component {i}: {}",
                    r.error.as_deref().unwrap_or("failed")
                ));
            }
        }
        if let (Some(d), Some(q)) = (&dump, &r.mesh) {
            d.quad(&format!("comp_{i}.obj"), q, &mut warnings);
            if ctl.is_cancelled() {
                return cancelled(t0, warnings);
            }
        }
    }
    if ctl.is_cancelled() {
        return cancelled(t0, warnings);
    }
    failed_tris.sort_unstable();
    failed_tris.dedup();
    let (mesh, mrep) = match merge::merge(&parts, h0) {
        Ok(x) => x,
        Err(e) => {
            let all: Vec<u32> = comps
                .iter()
                .flat_map(|c| c.face_src.iter().copied())
                .collect();
            return empty_with_failed(Status::Failed, e.to_string(), t0, warnings, all);
        }
    };
    if mrep.chain_overflow > 0 {
        warnings.push(format!(
            "{} vertices lost their chain id (more than 2^20 - 1 chains)",
            mrep.chain_overflow
        ));
    }
    if ctl.is_cancelled() {
        return cancelled(t0, warnings);
    }
    if let Some(d) = &dump {
        d.quad("result.obj", &mesh, &mut warnings);
    }
    if ctl.is_cancelled() {
        return cancelled(t0, warnings);
    }
    let status = if parts.is_empty() {
        Status::Failed
    } else if degraded || !failed_tris.is_empty() {
        Status::Degraded
    } else {
        Status::Ok
    };
    let error = errors.join("; ");
    let comp_reports: Vec<CompReport> = results
        .iter()
        .enumerate()
        .map(|(i, r)| CompReport {
            id: i as u32,
            status: r.status,
            target: plan[i].target,
            debris: plan[i].debris,
            mesh: r.mesh.as_ref(),
            tris: comps[i].mesh.tris.len(),
            area: comps[i].area * h0 * h0,
            chi: comps[i].chi,
            closed: comps[i].closed,
            time_ms: r.ms,
            error: r.error.clone(),
            warnings: r.warnings.clone(),
            extra: r.extra.clone(),
        })
        .collect();
    let mut stage_ms: Vec<(Stage, f64)> = Vec::new();
    for r in &results {
        stage_ms.extend_from_slice(&r.stage_ms);
    }
    if ctl.is_cancelled() {
        return cancelled(t0, warnings);
    }
    let mut summ = Summary {
        status,
        error: error.clone(),
        ingest: Some(ingest_report),
        ingest_ms,
        comp_ms,
        stage_ms,
        total_ms: ms(t0.elapsed()),
        warnings,
        target: p.target_quads,
        result_faces: mesh.n_faces(),
        h0,
        comps: comp_reports,
        n_failed_tris: failed_tris.len(),
        chains: mrep.chains,
        correction: corr,
        params: Some(p.clone()),
    };
    let report = report::build(&summ);
    // Polled after the report too: its per-component quad statistics are the last long step.
    if ctl.is_cancelled() {
        return cancelled(t0, std::mem::take(&mut summ.warnings));
    }
    Output {
        status,
        mesh,
        comp_status,
        failed_tris,
        report,
        error,
    }
}

/// Result of one component job.
#[derive(Clone, Debug)]
struct JobResult {
    status: CompStatus,
    /// Quad units, component-local chain ids.
    mesh: Option<QuadMesh>,
    /// The ladder went below R0 after a failure, or the rung flagged its result.
    degraded: bool,
    cancelled: bool,
    error: Option<String>,
    warnings: Vec<String>,
    ms: f64,
    extra: serde_json::Map<String, serde_json::Value>,
    stage_ms: Vec<(Stage, f64)>,
    /// Proxy and layout of a full-path result, for the count correction (WP-16 round 3).
    state: Option<std::sync::Arc<pipeline::Requant>>,
}

struct JobCtx<'a> {
    p: &'a Params,
    density: Option<&'a [f32]>,
    ctl: &'a Ctl<'a>,
    /// WP-20: global floor of the thickness caps (quad units), see `budget::thickness_shares`.
    thick_floor: Option<f64>,
}

/// `Ok(value)`, `Err(error)`, or a caught panic turned into an `Internal` error.
fn guarded<T>(f: impl FnOnce() -> Result<T, AqError>) -> Result<T, AqError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => Err(AqError::new(
            Stage::Engine,
            ErrKind::Internal,
            format!("panic: {}", panic_text(payload.as_ref())),
        )),
    }
}

/// Verifies a component result (7.3 S10/S11): valid CSR, manifold and oriented, all quads,
/// at least one face, and the component's Euler characteristic.
fn verify(q: &QuadMesh, chi: i64) -> Result<(), AqError> {
    q.validate().map_err(|e| e.at(Stage::Engine))?;
    if q.n_faces() == 0 {
        return Err(AqError::invariant("result has no face").at(Stage::Engine));
    }
    if !q.is_all_quads() {
        return Err(AqError::invariant("result is not all quads").at(Stage::Engine));
    }
    let x = euler_char(q);
    if x != chi {
        return Err(
            AqError::invariant(format!("result chi {x} != component chi {chi}")).at(Stage::Engine),
        );
    }
    Ok(())
}

fn stop_kind(e: &AqError) -> bool {
    e.kind == ErrKind::Cancelled
}

/// The component's feature set as SF uses it: `hard_edges` applied (0 drops Hard, 1 makes it
/// soft), and soft guides dropped (SF has no field to align, DESIGN 11.1 M1 maps guides to soft
/// field alignment only). Boundary, Symmetry, and crisp Hard and GuideHard chains stay.
fn sf_component(c: &ComponentMesh, hard_mode: u8) -> Result<Cow<'_, ComponentMesh>, AqError> {
    let mut kinds = c.feats.he_kind.clone();
    apply_hard_mode(&mut kinds, hard_mode);
    for k in kinds.iter_mut() {
        if *k == FeatureKind::Guide {
            *k = FeatureKind::None;
        }
    }
    if kinds == c.feats.he_kind {
        return Ok(Cow::Borrowed(c));
    }
    let zero = |_: u32| 0.0;
    let rule = CornerRule {
        corner_deg: 50.0,
        sym_corner_deg: 70.0,
        window: &zero,
        fixed: None,
    };
    let feats =
        build_feature_set(&c.mesh, &c.topo, kinds, &rule).map_err(|e| e.at(Stage::Engine))?;
    let mut c2 = c.clone();
    c2.feats = feats;
    Ok(Cow::Owned(c2))
}

impl JobCtx<'_> {
    fn sf(
        &self,
        c: &ComponentMesh,
        target: u32,
        debris: bool,
        inject: bool,
    ) -> Result<QuadMesh, AqError> {
        if inject {
            injected_panic("component job");
        }
        let c2 = sf_component(c, self.p.hard_edges)?;
        let sp = SfParams {
            target: target.max(1),
            relax_iters: SF_RELAX_ITERS,
        };
        let q = if debris {
            let mut v = sf_batch(&[c2.as_ref()], &[sp], self.ctl)?;
            v.pop()
                .ok_or_else(|| AqError::internal("sf_batch returned nothing").at(Stage::Fallback))?
        } else {
            let (fr, bvh) = c2.source_geometry()?;
            let proj = Projector::new(&c2.mesh, &c2.topo, &fr, &bvh);
            sf_quadrangulate(&c2, &proj, &sp, self.ctl)?
        };
        verify(&q, c.chi)?;
        Ok(q)
    }

    /// The ladder for one component (DESIGN 7.2).
    fn run(
        &self,
        id: u32,
        c: &ComponentMesh,
        plan: CompPlan,
        proxy_tris: u32,
        share: Duration,
    ) -> JobResult {
        let t = Instant::now();
        let mut jr = JobResult {
            status: CompStatus::Failed,
            mesh: None,
            degraded: false,
            cancelled: false,
            error: None,
            warnings: Vec::new(),
            ms: 0.0,
            extra: serde_json::Map::new(),
            stage_ms: Vec::new(),
            state: None,
        };
        let mut inject = id == 0 && self.p.debug_flags & debug::PANIC_COMPONENT != 0;
        let force_sf = self.p.debug_flags & debug::FORCE_SF != 0;
        if pipeline::AVAILABLE && !plan.debris && !force_sf {
            let sub_deadline = t.checked_add(share).unwrap_or(self.ctl.deadline());
            let sub = self.ctl.sub(sub_deadline);
            let force_r1 = self.p.debug_flags & debug::FORCE_R1 != 0;
            let mut rungs: Vec<Rung> = Vec::new();
            if !force_r1 {
                rungs.push(Rung::R0);
            }
            rungs.push(Rung::R1);
            for rung in rungs {
                if rung == Rung::R1
                    && !force_r1
                    && t.elapsed().as_secs_f64() >= R1_TIME_FRAC * share.as_secs_f64()
                {
                    jr.warnings
                        .push("R1 skipped: R0 used 40% of the time share".into());
                    break;
                }
                let inp = RungInput {
                    comp: c,
                    comp_id: id,
                    target: plan.target,
                    max_proxy_tris: proxy_tris,
                    rung,
                    params: self.p,
                    density: self.density,
                    thick_floor: self.thick_floor,
                };
                let take_inject = std::mem::take(&mut inject);
                let mut trace = serde_json::Map::new();
                let r = guarded(|| {
                    if take_inject {
                        injected_panic("component rung");
                    }
                    let out = pipeline::run_rung_traced(&inp, &sub, &mut trace)?;
                    verify(&out.mesh, c.chi)?;
                    Ok(out)
                });
                match r {
                    Ok(out) => {
                        jr.status = if rung == Rung::R0 {
                            CompStatus::Full
                        } else {
                            CompStatus::Retried
                        };
                        jr.degraded = out.degraded || rung == Rung::R1;
                        jr.mesh = Some(out.mesh);
                        jr.extra.extend(out.report);
                        jr.stage_ms = out.timing_ms;
                        jr.state = out.state.map(std::sync::Arc::from);
                        jr.ms = ms(t.elapsed());
                        return jr;
                    }
                    Err(e) if stop_kind(&e) => {
                        jr.cancelled = true;
                        jr.error = Some(e.to_string());
                        return jr;
                    }
                    Err(e) => {
                        jr.warnings.push(format!("{rung:?} failed: {e}"));
                        trace.insert("error".into(), serde_json::Value::String(e.to_string()));
                        jr.extra
                            .insert(format!("{rung:?}_trace"), serde_json::Value::Object(trace));
                    }
                }
            }
            jr.degraded = true;
        }
        let r = guarded(|| self.sf(c, plan.target, plan.debris, inject));
        match r {
            Ok(q) => {
                jr.status = if plan.debris {
                    CompStatus::DebrisSf
                } else {
                    CompStatus::Sf
                };
                jr.mesh = Some(q);
            }
            Err(e) if stop_kind(&e) => {
                jr.cancelled = true;
                jr.error = Some(e.to_string());
            }
            Err(e) => {
                jr.status = CompStatus::Failed;
                jr.error = Some(e.to_string());
            }
        }
        jr.ms = ms(t.elapsed());
        jr
    }
}

/// What the count correction did (7.1 step 8).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Correction {
    pub ran: bool,
    pub retried: u32,
    pub improved: u32,
}

/// If the total count is off by more than 8% and time remains, re-runs SF once on the
/// components with a count error above 8% (largest absolute error first), with the target
/// scaled by target / count, and keeps a result when it is closer to the component target.
/// At M0 every component is on SF, so this re-runs SF (WP-16 extends it to S9..S11).
fn count_correction(
    jobs: &JobCtx,
    comps: &[ComponentMesh],
    plan: &[CompPlan],
    results: &mut [JobResult],
) -> Correction {
    let mut corr = Correction::default();
    let total: usize = results
        .iter()
        .filter_map(|r| r.mesh.as_ref())
        .map(|q| q.n_faces())
        .sum();
    let target = jobs.p.target_quads as f64;
    if total == 0 || ((total as f64 - target) / target).abs() <= COUNT_CORRECTION {
        return corr;
    }
    if jobs.ctl.remaining() < COUNT_CORRECTION_MIN_TIME {
        return corr;
    }
    corr.ran = true;
    let mut cand: Vec<(usize, u32)> = Vec::new();
    for (i, r) in results.iter().enumerate() {
        let Some(q) = &r.mesh else { continue };
        if r.status != CompStatus::Sf {
            continue;
        }
        let n = q.n_faces() as f64;
        let t = plan[i].target as f64;
        if n > 0.0 && ((n - t) / t).abs() > COUNT_CORRECTION {
            let nt = (t * t / n).round().clamp(1.0, MAX_TARGET_QUADS as f64) as u32;
            cand.push((i, nt));
        }
    }
    cand.sort_by(|a, b| {
        let e = |&(i, _): &(usize, u32)| {
            let n = results[i].mesh.as_ref().map_or(0, |q| q.n_faces()) as i64;
            (n - plan[i].target as i64).abs()
        };
        e(b).cmp(&e(a)).then(a.0.cmp(&b.0))
    });
    corr.retried = cand.len() as u32;
    let redo: Vec<(usize, Result<QuadMesh, AqError>)> = cand
        .par_iter()
        .map(|&(i, nt)| (i, guarded(|| jobs.sf(&comps[i], nt, false, false))))
        .collect();
    for (i, r) in redo {
        let Ok(q) = r else { continue };
        let t = plan[i].target as i64;
        let old = results[i]
            .mesh
            .as_ref()
            .map_or(i64::MAX, |m| (m.n_faces() as i64 - t).abs());
        if (q.n_faces() as i64 - t).abs() < old {
            results[i].mesh = Some(q);
            results[i].warnings.push(format!(
                "count correction: SF re-run with target {}",
                cand_target(&cand, i)
            ));
            corr.improved += 1;
        }
    }
    full_path_correction(jobs, plan, results, &mut corr);
    corr
}

/// A requantized component: its quad mesh and its new stage blocks.
type Requantized = (QuadMesh, serde_json::Map<String, serde_json::Value>);

/// Count correction on the full-path components (DESIGN 7.1 step 8: "re-run S9 to S11";
/// WP-16 round 3). When the total is still off by more than 8%, the components whose own
/// count is within 8% of their target absorb the error of the others (thin parts held up by
/// the ring_min floor, debris on SF): their kept layouts are quantized again with the targets
/// scaled by (target - fixed) / (their count), in parallel. The new results are kept when the
/// total error gets smaller.
fn full_path_correction(
    jobs: &JobCtx,
    plan: &[CompPlan],
    results: &mut [JobResult],
    corr: &mut Correction,
) {
    let target = jobs.p.target_quads as f64;
    let count = |r: &JobResult| r.mesh.as_ref().map_or(0, |q| q.n_faces()) as f64;
    let total: f64 = results.iter().map(count).sum();
    if total <= 0.0 || ((total - target) / target).abs() <= COUNT_CORRECTION {
        return;
    }
    if jobs.ctl.remaining() < COUNT_CORRECTION_MIN_TIME {
        return;
    }
    let adjustable: Vec<usize> = (0..results.len())
        .filter(|&i| {
            let r = &results[i];
            let t = plan[i].target as f64;
            r.state.is_some()
                && r.mesh.is_some()
                && t > 0.0
                && ((count(r) - t) / t).abs() <= COUNT_CORRECTION
        })
        .collect();
    let adj_sum: f64 = adjustable.iter().map(|&i| count(&results[i])).sum();
    if adj_sum <= 0.0 {
        return;
    }
    let fixed = total - adj_sum;
    let scale = ((target - fixed) / adj_sum).clamp(0.5, 2.0);
    if (scale - 1.0).abs() < 0.01 {
        return;
    }
    corr.ran = true;
    corr.retried += adjustable.len() as u32;
    let redo: Vec<(usize, Result<Requantized, AqError>)> = adjustable
        .par_iter()
        .map(|&i| {
            let st = results[i].state.as_deref();
            let n_new = count(&results[i]) * scale;
            (
                i,
                guarded(|| match st {
                    Some(st) => pipeline::requantize(st, n_new, jobs.p, jobs.ctl),
                    None => Err(AqError::invariant("no kept layout").at(Stage::Engine)),
                }),
            )
        })
        .collect();
    let mut new_total = fixed;
    let mut fresh: Vec<(usize, QuadMesh, serde_json::Map<String, serde_json::Value>)> = Vec::new();
    for (i, r) in redo {
        match r {
            Ok((q, rep)) if q.n_faces() > 0 => {
                new_total += q.n_faces() as f64;
                fresh.push((i, q, rep));
            }
            _ => new_total += count(&results[i]),
        }
    }
    if (new_total - target).abs() >= (total - target).abs() {
        return;
    }
    for (i, q, rep) in fresh {
        let r = &mut results[i];
        r.warnings.push(format!(
            "count correction: S9 to S11 re-run with target {:.0} (was {})",
            count(r) * scale,
            plan[i].target
        ));
        r.mesh = Some(q);
        for (k, v) in rep {
            if k == "warnings" {
                continue;
            }
            r.extra.insert(k, v);
        }
        corr.improved += 1;
    }
}

fn cand_target(cand: &[(usize, u32)], i: usize) -> u32 {
    cand.iter().find(|c| c.0 == i).map_or(0, |c| c.1)
}

/// Stage dumps (debug flag bit 0 with a dump directory).
struct DumpCtx {
    dir: aq_io::DumpDir,
}

impl DumpCtx {
    fn new(p: &Params, warnings: &mut Vec<String>) -> Option<DumpCtx> {
        if p.debug_flags & debug::DUMP == 0 {
            return None;
        }
        let dir = p.dump_dir.as_ref()?;
        if let Err(e) = std::fs::create_dir_all(dir) {
            warnings.push(format!("dump dir {}: {e}", dir.display()));
            return None;
        }
        Some(DumpCtx {
            dir: aq_io::DumpDir::new(dir.clone()),
        })
    }
    fn tri(&self, name: &str, m: &aq_core::TriMesh, warnings: &mut Vec<String>) {
        if let Err(e) = aq_io::write_obj_tri(&self.dir.path(name), m) {
            warnings.push(format!("dump {name}: {e}"));
        }
    }
    fn quad(&self, name: &str, q: &QuadMesh, warnings: &mut Vec<String>) {
        if let Err(e) = self.dir.quads(name, q) {
            warnings.push(format!("dump {name}: {e}"));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn params_validation() {
        let mut w = Vec::new();
        let d = Params::default();
        assert!(checked_params(&d, &mut w).is_ok());
        assert!(w.is_empty());
        let bad = Params {
            target_quads: 0,
            ..d.clone()
        };
        assert!(checked_params(&bad, &mut w).is_err());
        let bad = Params {
            time_limit_s: f32::NAN,
            ..d.clone()
        };
        assert!(checked_params(&bad, &mut w).is_err());
        let bad = Params {
            time_limit_s: 0.0,
            ..d.clone()
        };
        assert!(checked_params(&bad, &mut w).is_err());
        let bad = Params {
            curvature_align: f32::INFINITY,
            ..d.clone()
        };
        assert!(checked_params(&bad, &mut w).is_err());
        let ok = Params {
            time_limit_s: f32::INFINITY,
            ..d.clone()
        };
        assert!(checked_params(&ok, &mut w).is_ok());
        let clamped = Params {
            adaptive_size: 3.0,
            hard_edges: 9,
            ..d.clone()
        };
        let q = checked_params(&clamped, &mut w).unwrap();
        assert_eq!(q.adaptive_size, 1.0);
        assert_eq!(q.hard_edges, 2);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn panic_text_variants() {
        let r = catch_unwind(|| injected_panic("x"));
        let e = r.err().unwrap();
        assert_eq!(panic_text(e.as_ref()), "injected panic (x)");
        let g: Result<(), AqError> = guarded(|| injected_panic("y"));
        let e = g.unwrap_err();
        assert_eq!(e.kind, ErrKind::Internal);
        assert!(e.msg.contains("injected panic (y)"));
    }
}

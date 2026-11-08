//! The full per-component pipeline, rungs R0 and R1 (S2 to S11, DESIGN 7.1 step 6, 7.2, 7.3).
//!
//! **Owned by WP-16** (DESIGN 11.2). [`run_rung`] runs S2 to S11 on one component and checks
//! the stage invariants of 7.3 between the stages. It returns typed errors and never retries
//! an upstream stage: the engine (`run.rs`) owns the ladder (R1 re-run, SF, QuadriFlow) and
//! the panic catch around this call.
//!
//! Units. `RungInput::comp` arrives in global quad units (h0 = 1). The rung rescales the
//! component once more by `sqrt(N_c / A_c)`, so that the component's own quad size is exactly
//! 1 inside the stages (their absolute parameters, such as the loop stride of 2 h or the
//! cancellation distance of 1.2 h, assume h = 1). The result is scaled back to global quad
//! units before it is returned. See CHANGES.md, WP-16.

use std::time::Instant;

use serde_json::{Map, Value, json};

use aq_core::{
    AqError, Bvh, Ctl, CutGraph, ErrKind, FeatureKind, LoopCand, NONE, Projector, QuadMesh, Stage,
};
use aq_prep::{ComponentMesh, Proxy, ProxyParams};

use crate::Params;

/// S11 warning threshold on quads with a non-positive corner: this fraction of the faces...
const FLIP_FRAC: f64 = 0.005;
/// ...or at least this many.
const FLIP_MIN: f64 = 2.0;
/// S11 warning threshold on a quad vertex's distance to the proxy, in h.
const VDIST_MAX: f64 = 0.5;

/// True once WP-16 has wired S2 to S11 (M1).
pub const AVAILABLE: bool = true;

/// Which rung runs (DESIGN 7.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rung {
    /// R0: the user's parameters.
    R0,
    /// R1: curvature_align x0.5, adaptive 0, cancel_dist x1.5, loops limited to rings,
    /// allow_collapse kept, h0 jitter x0.95.
    R1,
}

/// Everything a rung needs for one component.
pub struct RungInput<'a> {
    /// The component in quad units (h0 = 1).
    pub comp: &'a ComponentMesh,
    /// Component id (index in `split_components` order).
    pub comp_id: u32,
    /// Quad target N_c of the component.
    pub target: u32,
    /// The component's share of `max_proxy_tris`.
    pub max_proxy_tris: u32,
    pub rung: Rung,
    /// The (validated, clamped) user parameters.
    pub params: &'a Params,
    /// Per **input** vertex density multipliers (`comp.vert_src` maps into it), or None.
    pub density: Option<&'a [f32]>,
    /// WP-20: the floor of the thickness caps planned across components by the engine, in
    /// global quad units (h0 = 1), or None. The rung uses it as a lower bound of the floor it
    /// solves from its own budget.
    pub thick_floor: Option<f64>,
}

/// What a successful rung returns.
pub struct RungOutput {
    /// Quad mesh of the component in quad units.
    pub mesh: QuadMesh,
    /// Set when the result is usable but flagged (7.2: layout health above 0.1 on more than 5%
    /// of the patches).
    pub degraded: bool,
    /// Per-stage report blocks, merged into the component entry of the report JSON
    /// (keys "field", "param", "layout", "quant", "patch", "post", "cones", "warnings").
    pub report: serde_json::Map<String, serde_json::Value>,
    /// Milliseconds per stage, added to the report `timing_ms` (keys: `Stage::name`).
    pub timing_ms: Vec<(Stage, f64)>,
    /// Proxy and layout kept for the count correction (WP-16 round 3).
    pub state: Option<Box<Requant>>,
}

/// Lambda numerator per unit of the user's curvature_align (DESIGN 13 says 1.0). Tuned by
/// WP-16 round 2 to 0.5: on the suite it leaves the cone count unchanged on every mesh except
/// noisy_blob (24 to 14 cones) and retopo_a (46 to 44), where the stronger alignment followed
/// the noise (CHANGES.md, WP-16 round 2).
const ALIGN_NUMERATOR: f64 = 0.5;
/// R1 changes (DESIGN 7.2).
const R1_ALIGN: f64 = 0.5;
const R1_CANCEL: f64 = 1.5;
/// "h0 jitter x0.95": in R1 the proxy edge length is 0.95 h/3, which changes every discrete
/// decision downstream while keeping h = 1 for the stage parameters (CHANGES.md, WP-16).
const R1_JITTER: f64 = 0.95;
/// S8 seeded re-runs (DESIGN 13 says 3). WP-23: with fewer cones, separatrices cut fewer
/// annular regions, and retopo_b needed 6 seed rounds after the far cancellation; each round
/// only runs when the previous one left invalid patches (CHANGES.md, WP-23).
const LAYOUT_ROUNDS: u32 = 8;
/// Proxy edge length in h (DESIGN 13).
const PROXY_LEN: f64 = 1.0 / 3.0;
/// S2 sliver invariant: faces with a minimum angle below 1 degree (7.3).
const SLIVER_DEG: f64 = 1.0;
const SLIVER_FRAC: f64 = 1e-3;
/// S3 and S7 count invariants (7.3).
const COUNT_TOL: f64 = 1e-6;
/// 7.2: a layout whose health fraction above 0.1 exceeds 5% of the patches is Degraded.
const HEALTH_FRAC: f64 = 0.05;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn inv(stage: Stage, msg: impl Into<String>) -> AqError {
    AqError::invariant(msg).at(stage)
}

/// Minimum corner angle of face f, in degrees.
fn min_angle_deg(m: &aq_core::TriMesh, f: u32) -> f64 {
    use aq_core::math::{dot, norm, sub};
    let [a, b, c] = m.tri_pos(f);
    let ang = |p: [f64; 3], q: [f64; 3], r: [f64; 3]| {
        let u = sub(q, p);
        let v = sub(r, p);
        let (nu, nv) = (norm(u), norm(v));
        if !(nu > 0.0 && nv > 0.0) {
            return 0.0;
        }
        (dot(u, v) / (nu * nv)).clamp(-1.0, 1.0).acos()
    };
    ang(a, b, c)
        .min(ang(b, c, a))
        .min(ang(c, a, b))
        .to_degrees()
}

/// WP-20: the M2 thickness cap in S3 (`SizingParams::thickness`).
pub const THICKNESS: bool = true;

/// S2 proxy refinement (WP-16 round 2): rebuild when faces with h below this...
const REFINE_H: f64 = 0.7;
/// ...cover more than this fraction of the area.
const REFINE_AREA: f64 = 0.01;
/// Smallest h the refined proxy follows (DESIGN 13 g clamp 1/4).
const REFINE_MIN_H: f64 = 0.25;

/// S2: proxy, feature isolation, chain classes and validation.
fn make_proxy(
    comp: &ComponentMesh,
    proj: &Projector,
    len: &dyn Fn([f64; 3]) -> f64,
    pp: &ProxyParams,
    ctl: &Ctl,
) -> Result<Proxy, AqError> {
    let mut px = aq_prep::build_proxy(comp, proj, len, pp, ctl)?;
    aq_prep::isolate_features_with(&mut px, Some(proj))?;
    let ones = vec![1.0; px.mesh.pos.len()];
    aq_prep::classify_chains(&mut px, &ones, pp);
    px.validate()?;
    Ok(px)
}

/// V - E + F of the proxy.
fn tri_chi(px: &Proxy) -> i64 {
    let nf = px.mesh.tris.len() as i64;
    let e = (0..3 * px.mesh.tris.len() as u32)
        .filter(|&h| px.topo.is_canonical(h))
        .count() as i64;
    let v = px.topo.vert_he.iter().filter(|&&h| h != NONE).count() as i64;
    v - e + nf
}

/// Stage dumps of one rung (debug bit 0 with a dump directory), in local units.
struct Dumps {
    dir: Option<aq_io::DumpDir>,
    warnings: Vec<String>,
}

impl Dumps {
    fn new(inp: &RungInput) -> Dumps {
        let p = inp.params;
        let mut warnings = Vec::new();
        let mut dir = None;
        if p.debug_flags & crate::debug::DUMP != 0
            && let Some(d) = &p.dump_dir
        {
            let sub = d.join(format!("comp_{}_{:?}", inp.comp_id, inp.rung));
            match std::fs::create_dir_all(&sub) {
                Ok(()) => dir = Some(aq_io::DumpDir::new(sub)),
                Err(e) => warnings.push(format!("dump dir: {e}")),
            }
        }
        Dumps { dir, warnings }
    }
    fn run(
        &mut self,
        name: &str,
        f: impl FnOnce(&aq_io::DumpDir) -> Result<std::path::PathBuf, AqError>,
    ) {
        if let Some(d) = &self.dir
            && let Err(e) = f(d)
        {
            self.warnings.push(format!("dump {name}: {e}"));
        }
    }
}

/// Runs one rung on one component (S2 to S11).
pub fn run_rung(inp: &RungInput, ctl: &Ctl) -> Result<RungOutput, AqError> {
    run_rung_traced(inp, ctl, &mut Map::new())
}

/// [`run_rung`] that writes each stage's report block into `report` as soon as the stage
/// finishes, so a failed rung still leaves the blocks of the stages that ran (the engine
/// stores them in the component report for diagnosis).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN counts must fail the invariants
pub fn run_rung_traced(
    inp: &RungInput,
    ctl: &Ctl,
    report: &mut Map<String, Value>,
) -> Result<RungOutput, AqError> {
    ctl.check()?;
    let p = inp.params;
    let r1 = inp.rung == Rung::R1;
    let n_c = inp.target.max(1) as f64;
    let mut warnings: Vec<String> = Vec::new();
    let mut timing: Vec<(Stage, f64)> = Vec::new();
    let mut dumps = Dumps::new(inp);
    // Local scale: the component's own h becomes 1.
    let a_c = inp.comp.area;
    if !(a_c.is_finite() && a_c > 0.0) {
        return Err(AqError::new(
            Stage::Engine,
            ErrKind::InvalidInput,
            format!("component area {a_c}"),
        ));
    }
    let s_loc = (n_c / a_c).sqrt();
    let comp = inp.comp.scaled(s_loc);
    let seed = p.seed ^ (inp.comp_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

    // S2 proxy, then S3 curvature and sizing (M1: uniform sizing, adaptive 0, no density).
    // WP-16 round 2: when the tube cap makes h smaller than the uniform proxy can carry
    // (fingers, strands, spike tips: h down to 0.2 with proxy edges of 1/3), the proxy is
    // rebuilt once with edges of h/3 from the first size field, and S3 runs again on it
    // (DESIGN 13 "proxy len h/3" read with the local h; CHANGES.md, WP-16 round 2), on R0 only.
    let t = Instant::now();
    ctl.progress(Stage::Proxy, 0.0);
    let (src_fr, src_bvh) = comp.source_geometry()?;
    let proj = Projector::new(&comp.mesh, &comp.topo, &src_fr, &src_bvh);
    let pp = ProxyParams {
        max_tris: inp.max_proxy_tris,
        hard_mode: p.hard_edges,
        ..ProxyParams::default()
    };
    let jit = if r1 { R1_JITTER } else { 1.0 };
    let len = PROXY_LEN * jit;
    let mut px = make_proxy(&comp, &proj, &|_| len, &pp, ctl)?;
    timing.push((Stage::Proxy, ms(t)));
    let t = Instant::now();
    let cp = aq_curv::CurvParams {
        scale: p.flow_smoothness as f64,
        ..Default::default()
    };
    // WP-20: thickness cap on (DESIGN 6.5 M2), with the engine's global floor when planned.
    let sp = aq_curv::SizingParams {
        ring_min: p.ring_min,
        thickness: THICKNESS,
        thick_floor: inp.thick_floor.map(|f| f * s_loc),
        ..aq_curv::SizingParams::new(n_c)
    };
    let (mut curv, mut size, mut s3) = {
        let cview = aq_curv::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        aq_curv::curvature_and_sizing(&cview, None, &cp, &sp, ctl)?
    };
    timing.push((Stage::Curv, ms(t)));
    let small_area: f64 = (0..px.mesh.tris.len())
        .filter(|&f| size.h_face[f] < REFINE_H)
        .map(|f| px.frames.area[f])
        .sum();
    let area_px: f64 = px.frames.area.iter().sum();
    let mut refined = false;
    // R1 keeps the uniform proxy: a second refined build costs 4 s on basemesh_female, and the
    // retry differs from R0 by its field and proxy jitter already.
    if !r1 && small_area > REFINE_AREA * area_px {
        ctl.check()?;
        let t = Instant::now();
        let bvh1 = Bvh::try_build(&px.mesh).map_err(|e| e.at(Stage::Proxy))?;
        let hf = size.h_face.clone();
        let len_at = |q: [f64; 3]| {
            let (sp, _, _) = bvh1.closest(q);
            let h = hf.get(sp.face as usize).copied().unwrap_or(1.0);
            PROXY_LEN * jit * h.clamp(REFINE_MIN_H, 1.0)
        };
        let px2 = make_proxy(&comp, &proj, &len_at, &pp, ctl)?;
        drop(bvh1);
        px = px2;
        timing.push((Stage::Proxy, ms(t)));
        let t = Instant::now();
        let cview = aq_curv::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        (curv, size, s3) = aq_curv::curvature_and_sizing(&cview, None, &cp, &sp, ctl)?;
        timing.push((Stage::Curv, ms(t)));
        refined = true;
    }
    let chi = tri_chi(&px);
    if chi != comp.chi || px.topo.n_comp != 1 {
        return Err(inv(
            Stage::Proxy,
            format!(
                "S2: proxy chi {chi} ({} comps) != {}",
                px.topo.n_comp, comp.chi
            ),
        ));
    }
    let nf = px.mesh.tris.len();
    let slivers = (0..nf as u32)
        .filter(|&f| min_angle_deg(&px.mesh, f) < SLIVER_DEG)
        .count();
    if slivers as f64 > SLIVER_FRAC * nf as f64 {
        // WP-03 follow-up item 1: narrow strips between chains force such faces; warn only.
        warnings.push(format!("S2: {slivers} of {nf} proxy faces below 1 degree"));
    }
    dumps.run("proxy", |d| d.proxy(&px.mesh, Some(&px.feats)));
    curv.validate(nf, px.mesh.pos.len())
        .map_err(|e| e.at(Stage::Curv))?;
    size.validate(&px.frames, px.mesh.pos.len())
        .map_err(|e| e.at(Stage::Curv))?;
    let npred = size.predicted_count(&px.frames);
    if !((npred - n_c).abs() <= COUNT_TOL * n_c)
        || size
            .h_face
            .iter()
            .chain(&size.h_vert)
            .any(|h| !(h.is_finite() && *h > 0.0))
    {
        return Err(inv(
            Stage::Curv,
            format!("S3: sum A/h^2 = {npred} for N_c = {n_c}"),
        ));
    }
    if s3.sizing.cap_infeasible {
        warnings.push("S3: the tube cap alone needs more quads than the component target".into());
    }
    report.insert("proxy".into(), json!({ "faces": nf, "refined": refined, "small_h_area_frac": small_area / area_px.max(1e-300) }));
    {
        let sz = &s3.sizing;
        let (hmin, hmax) = size.h_vert.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &h| (a.min(h), b.max(h)));
        report.insert("sizing".into(), json!({
            "tube_verts": sz.tube_verts, "capped_verts": sz.capped_verts, "graded_verts": sz.graded_verts,
            "cap_infeasible": sz.cap_infeasible, "thick_verts": sz.thick_verts, "thick_capped": sz.thick_capped,
            "thick_floor": sz.thick_floor, "thick_share": sz.thick_share, "thick_ms": sz.thick_ms,
            "h_min": hmin, "h_max": hmax,
        }));
    }
    ctl.check()?;

    // S4 cross field.
    let t = Instant::now();
    let mut fp = aq_field::FieldParams {
        align: ALIGN_NUMERATOR * p.curvature_align as f64 * if r1 { R1_ALIGN } else { 1.0 },
        guide_w: p.guide_strength as f64,
        hard_mode: p.hard_edges,
        cancel_dist: 1.2 * if r1 { R1_CANCEL } else { 1.0 },
        seed,
        ..Default::default()
    };
    // WP-23: when S5 to S8 fail after the WP-23 edits (far cancellation, merges of -1 pairs
    // and into boundary vertices) changed the field, S4 runs again without them and S5 to S8
    // follow once more, before the rung gives up.
    let mut s4 = s4_field(&px, &mut curv, &size, &fp, &cp, &sp, ctl)?;
    timing.push((Stage::Field, ms(t)));
    let mut trouble = Vec::new();
    let mut s58 = s5_s8(&px, &s4.field, &curv, &size, p, r1, seed, n_c, comp.chi, ctl, &mut timing, report, &mut dumps, &mut trouble);
    let edits = wp23_edits(&s4.frep);
    let retry = |e: &AqError| !matches!(e.kind, ErrKind::Cancelled | ErrKind::Timeout | ErrKind::Budget);
    // M2 integration: first only the WP-23 edits next to the S7 folds and bad cones of the
    // failed run are left out (the rest of the field is kept); then all of them.
    if let Err(e) = &s58
        && edits > 0
        && retry(e)
        && !trouble.is_empty()
    {
        ctl.check()?;
        let t = Instant::now();
        let fp2 = aq_field::FieldParams { keep_out: std::mem::take(&mut trouble), ..fp.clone() };
        let s4b = s4_field(&px, &mut curv, &size, &fp2, &cp, &sp, ctl)?;
        timing.push((Stage::Field, ms(t)));
        let edits2 = wp23_edits(&s4b.frep);
        if edits2 > 0 && edits2 < edits {
            warnings.push(format!(
                "S4: {} of {edits} WP-23 edits left out next to the S7 folds after \"{e}\"; S4 to S8 re-run",
                edits - edits2
            ));
            s4 = s4b;
            s58 = s5_s8(&px, &s4.field, &curv, &size, p, r1, seed, n_c, comp.chi, ctl, &mut timing, report, &mut dumps, &mut trouble);
        }
    }
    if let Err(e) = &s58
        && wp23_edits(&s4.frep) > 0
        && retry(e)
    {
        warnings.push(format!(
            "S4: {} WP-23 edits dropped after \"{e}\"; S4 to S8 re-run without the far cancellation and the merges",
            wp23_edits(&s4.frep)
        ));
        ctl.check()?;
        let t = Instant::now();
        fp.far_cancel = 0.0;
        fp.merge_neg = 0.0;
        fp.boundary_merge = 0.0;
        s4 = s4_field(&px, &mut curv, &size, &fp, &cp, &sp, ctl)?;
        timing.push((Stage::Field, ms(t)));
        s58 = s5_s8(&px, &s4.field, &curv, &size, p, r1, seed, n_c, comp.chi, ctl, &mut timing, report, &mut dumps, &mut trouble);
    }
    let S4 { field, frep, s_ref, rerun } = s4;
    if rerun {
        warnings.push(format!(
            "S4: pole sanity cap: {} cones against a smooth reference of {s_ref}; field re-run with curvature scale x{CAP_SCALE} and align x{CAP_ALIGN}",
            frep.cones_final
        ));
    }
    report.insert(
        "field".into(),
        json!({
            "cones_initial": frep.cones_initial, "cones_final": frep.cones_final, "split": frep.split,
            "cancelled": frep.cancelled, "far_cancelled": frep.far_cancelled, "corner_merged": frep.corner_merged,
            "boundary_merged": frep.boundary_merged, "merged_neg": frep.merged_neg,
            "reverted": frep.reverted, "pushed": frep.pushed,
            "max_abs_delta": frep.max_abs_delta, "eigen_path": frep.eigen_path,
            "s_ref": s_ref, "excess_poles": frep.excess_poles, "cap_rerun": rerun,
            "proxy_faces": nf, "chains": px.feats.chains.len(),
        }),
    );
    dumps.run("field", |d| {
        d.field(&px.mesh, &px.frames, &field.theta, 0.0)
    });
    dumps.run("cones", |d| d.cones(&px.mesh.pos, &field.cones()));
    let S58 { lay, degraded } = s58?;
    if degraded {
        warnings.push(format!(
            "S8: layout health above 0.1 on {:.1}% of the patches",
            100.0 * lay.health.frac_above
        ));
    }
    ctl.check()?;

    // S9 to S11, then back to global quad units.
    let state = Requant {
        px,
        lay,
        chi: comp.chi,
        s_loc,
        n_c,
    };
    let mut q = s9_s11(
        &state,
        n_c,
        p,
        ctl,
        report,
        &mut timing,
        &mut warnings,
        Some(&mut dumps),
    )?;
    let back = 1.0 / s_loc;
    for v in q.pos.iter_mut() {
        *v = [v[0] * back, v[1] * back, v[2] * back];
    }
    warnings.append(&mut dumps.warnings);
    report.insert(
        "warnings".into(),
        Value::Array(warnings.into_iter().map(Value::String).collect()),
    );
    Ok(RungOutput {
        mesh: q,
        degraded,
        report: std::mem::take(report),
        timing_ms: timing,
        state: Some(Box::new(state)),
    })
}

/// What a successful rung keeps for the count correction (DESIGN 7.1 step 8): the proxy and
/// the layout in local units, so that S9 to S11 can run again with another target.
pub struct Requant {
    px: Proxy,
    lay: aq_core::Layout,
    chi: i64,
    s_loc: f64,
    /// The target the rung ran with.
    pub n_c: f64,
}

impl std::fmt::Debug for Requant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Requant {{ proxy faces {}, patches {}, n_c {} }}",
            self.px.mesh.tris.len(),
            self.lay.tmesh.patches.len(),
            self.n_c
        )
    }
}

/// Count correction (DESIGN 7.1 step 8, WP-16 round 3): S9 to S11 again on a kept layout with
/// target `n_new`. Returns the quad mesh in global quad units and the new stage blocks
/// ("quant", "patch", "post", "warnings").
pub fn requantize(
    st: &Requant,
    n_new: f64,
    p: &Params,
    ctl: &Ctl,
) -> Result<(QuadMesh, Map<String, Value>), AqError> {
    let mut report = Map::new();
    let mut timing = Vec::new();
    let mut warnings = Vec::new();
    let mut q = s9_s11(
        st,
        n_new.max(1.0),
        p,
        ctl,
        &mut report,
        &mut timing,
        &mut warnings,
        None,
    )?;
    let back = 1.0 / st.s_loc;
    for v in q.pos.iter_mut() {
        *v = [v[0] * back, v[1] * back, v[2] * back];
    }
    report.insert(
        "warnings".into(),
        Value::Array(warnings.into_iter().map(Value::String).collect()),
    );
    Ok((q, report))
}

/// S9 quantization, S10 patches, S11 relaxation and cleanup on a layout (local units).
#[allow(clippy::too_many_arguments)]
fn s9_s11(
    st: &Requant,
    n_c: f64,
    p: &Params,
    ctl: &Ctl,
    report: &mut Map<String, Value>,
    timing: &mut Vec<(Stage, f64)>,
    warnings: &mut Vec<String>,
    mut dumps: Option<&mut Dumps>,
) -> Result<QuadMesh, AqError> {
    let (px, lay) = (&st.px, &st.lay);
    // S9 quantization.
    let t = Instant::now();
    let qp = aq_quantize::QuantParams {
        allow_collapse: p.allow_collapse != 0,
        ring_min: p.ring_min,
        ..Default::default()
    };
    let quant = aq_quantize::quantize(&lay.tmesh, n_c, &qp, ctl)?;
    quant
        .comb
        .validate(&lay.tmesh)
        .map_err(|e| e.at(Stage::Quantize))?;
    for (a, (&n, &lb)) in quant.n.iter().zip(&quant.lb).enumerate() {
        if n < lb {
            return Err(inv(
                Stage::Quantize,
                format!("S9: arc {a} n {n} below lb {lb}"),
            ));
        }
    }
    let qerr = (quant.count as f64 - n_c) / n_c;
    report.insert(
        "quant".into(),
        json!({
            "sigma": quant.sigma, "solves": quant.solves, "cuts": quant.cuts, "count": quant.count,
            "count_err": qerr, "zero_arcs": quant.n.iter().filter(|&&n| n == 0).count(),
        }),
    );
    for w in &quant.warnings {
        warnings.push(format!("S9: {w}"));
    }
    timing.push((Stage::Quantize, ms(t)));
    if let Some(d) = dumps.as_deref_mut() {
        d.run("quant", |d| {
            d.quant(&lay.tmesh, Some(&quant.n), Some(&quant.comb))
        });
    }
    ctl.check()?;

    // S10 patches.
    let t = Instant::now();
    let chain_kind: Vec<FeatureKind> = px.feats.chains.iter().map(|c| c.kind).collect();
    let (mut q, prep) = aq_patch::mesh_patches_with(
        lay,
        &quant.comb,
        &chain_kind,
        &aq_patch::PatchParams::default(),
        ctl,
    )?;
    check_quads(&q, st.chi, Stage::Patch)?;
    report.insert(
        "patch".into(),
        json!({
            "field_aligned": prep.field_aligned, "barrier": prep.barrier, "floater": prep.floater,
            "remeshed": prep.remeshed, "zero_patches": prep.zero_patches, "split_edges": prep.split_edges,
            "bad_quads": prep.bad_quads,
        }),
    );
    timing.push((Stage::Patch, ms(t)));
    if let Some(d) = dumps {
        d.run("patches", |d| d.patches(&q, &quant.comb.quad_patch));
    }
    ctl.check()?;

    // S11 relaxation and cleanup.
    let t = Instant::now();
    let bvh = Bvh::try_build(&px.mesh).map_err(|e| e.at(Stage::Post))?;
    let post_px = aq_post::PostProxy::new(&px.mesh, &px.topo, &px.frames, &px.feats, &bvh);
    let n_chains = px.feats.chains.len() as u32;
    let chain_of_vert: Vec<u32> = q
        .vflag
        .iter()
        .map(|&f| {
            let c = f & aq_post::VF_CHAIN_MASK;
            if c == 0 || c > n_chains { NONE } else { c - 1 }
        })
        .collect();
    let before = q.clone();
    let rr = aq_post::relax(
        &mut q,
        &post_px,
        &chain_of_vert,
        &aq_post::PostParams::default(),
        ctl,
    )?;
    let cr = aq_post::cleanup_with(
        &mut q,
        &aq_post::CleanupParams {
            h: 1.0,
            ..Default::default()
        },
    );
    if let Err(e) = check_quads(&q, st.chi, Stage::Post) {
        warnings.push(format!("S11 result rejected, S10 mesh kept: {e}"));
        q = before;
    }
    // WP-16 round 2: thin fins, rims and spike tips narrower than h fall between the quad
    // vertices (Hausdorff above 1 h); the nearest free vertex is pulled onto such proxy points.
    // A second pass without smoothing catches the points the smoothing pulled away from.
    let attracted = attract_far(
        &mut q,
        &px.mesh,
        &px.topo.is_bnd_vert,
        &bvh,
        ATTRACT_TAU,
        true,
    )? + attract_far(
        &mut q,
        &px.mesh,
        &px.topo.is_bnd_vert,
        &bvh,
        ATTRACT_TAU,
        false,
    )?;
    let cerr = (q.n_faces() as f64 - n_c) / n_c;
    let flipped = flipped_quads(&q, &px.mesh, &bvh);
    // Vertex distance to the proxy in local h units (h = 1 here).
    let dists: Vec<f64> = q.pos.iter().map(|&p| bvh.closest(p).2).collect();
    let vd_mean = dists.iter().sum::<f64>() / dists.len().max(1) as f64;
    let vd_max = dists.iter().copied().fold(0.0, f64::max);
    report.insert(
        "post".into(),
        json!({
            "iters": rr.iters, "moved": rr.moved, "rejected_normal": rr.rejected_normal,
            "projection_failed": rr.projection_failed, "chain_off": rr.chain_off, "max_disp": rr.max_disp,
            "doublets": cr.doublets, "diag_collapses": cr.diag_collapses, "faces_before": cr.faces_before,
            "faces_after": cr.faces_after, "count_err": cerr,
            "flipped_quads": flipped, "vdist_mean": vd_mean, "vdist_max": vd_max, "attracted": attracted,
        }),
    );
    timing.push((Stage::Post, ms(t)));
    // 7.3 S10/S11 corner Jacobians (WP-16, CHANGES.md): inverted corners left after the
    // relaxation are counted and reported as a warning, not an error. Failing the component
    // was tried: SF does worse on the parts that cause them (thin parts below h).
    let flip_max = (FLIP_FRAC * q.n_faces() as f64).max(FLIP_MIN);
    if f64::from(flipped) > flip_max {
        warnings.push(format!("S11: {flipped} quads with a non-positive corner"));
    }
    if vd_max > VDIST_MAX {
        warnings.push(format!("S11: a vertex is {vd_max:.2} h off the proxy"));
    }
    Ok(q)
}

/// S11 attraction (WP-16 round 2): proxy points farther than this many h from the quad mesh...
const ATTRACT_TAU: f64 = 0.6;
/// ...pull a vertex when its quads keep every corner angle above this (degrees).
const ATTRACT_MIN_DEG: f64 = 12.0;
/// At most this many attraction rounds (the distances are recomputed each round).
const ATTRACT_ROUNDS: usize = 3;

/// Corner angles (degrees) of quad f, and whether every corner turns positively about the
/// proxy normal at the quad centroid.
fn quad_ok(q: &QuadMesh, f: usize, px: &aq_core::TriMesh, bvh: &Bvh) -> bool {
    use aq_core::math::{add, cross, dot, norm, scale, sub};
    let fv = q.face(f);
    let k = fv.len();
    let c = fv.iter().fold([0.0; 3], |s, &v| add(s, q.pos[v as usize]));
    let (sp, _, _) = bvh.closest(scale(c, 1.0 / k as f64));
    let nrm = px.face_normal(sp.face).unwrap_or([0.0; 3]);
    let cmin = ATTRACT_MIN_DEG.to_radians().cos();
    (0..k).all(|i| {
        let v = q.pos[fv[i] as usize];
        let a = sub(q.pos[fv[(i + 1) % k] as usize], v);
        let b = sub(q.pos[fv[(i + k - 1) % k] as usize], v);
        let (na, nb) = (norm(a), norm(b));
        na > 0.0 && nb > 0.0 && dot(cross(a, b), nrm) > 0.0 && dot(a, b) / (na * nb) < cmin
    })
}

/// Pulls the nearest free quad vertex (no chain, corner or boundary flag) onto each interior
/// proxy vertex (the boundary is the boundary vertices' job: pulling the row next to it out
/// gave a sawtooth hem on retopo_a) farther than `tau` h from the quad mesh, farthest first, when every quad around the
/// vertex stays valid (positive corners about the proxy normal, corner angles above
/// ATTRACT_MIN_DEG). The proxy vertex lies on the surface, so the moved vertex does too.
/// With `smooth`, the rings around the moved vertices are smoothed afterwards. Returns the
/// number of moved vertices.
fn attract_far(
    q: &mut QuadMesh,
    px: &aq_core::TriMesh,
    px_bnd: &[bool],
    bvh: &Bvh,
    tau: f64,
    smooth: bool,
) -> Result<u32, AqError> {
    let nfq = q.n_faces();
    let mut vf: Vec<Vec<u32>> = vec![Vec::new(); q.pos.len()];
    for f in 0..nfq {
        for &v in q.face(f) {
            vf[v as usize].push(f as u32);
        }
    }
    let free =
        |fl: u32| fl & (aq_post::VF_CHAIN_MASK | aq_post::VF_CORNER | aq_post::VF_BOUNDARY) == 0;
    let mut moved = vec![false; q.pos.len()];
    let mut n_moved = 0u32;
    for _ in 0..ATTRACT_ROUNDS {
        let tm = aq_eval::fan_triangulate(q);
        let tri_face: Vec<u32> = (0..nfq)
            .flat_map(|f| std::iter::repeat_n(f as u32, q.face(f).len().saturating_sub(2)))
            .collect();
        let qb = Bvh::try_build(&tm).map_err(|e| e.at(Stage::Post))?;
        let mut far: Vec<(f64, u32)> = px
            .pos
            .iter()
            .enumerate()
            .filter(|&(i, _)| !px_bnd.get(i).copied().unwrap_or(true))
            .filter_map(|(i, &p)| {
                let d = qb.closest(p).2;
                (d > tau).then_some((d, i as u32))
            })
            .collect();
        if far.is_empty() {
            break;
        }
        far.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut round_moves = 0;
        let mut locked = vec![false; q.pos.len()];
        for &(_, pi) in &far {
            let p = px.pos[pi as usize];
            let (sp, _, d) = qb.closest(p);
            if d <= tau {
                continue;
            }
            let Some(&f) = tri_face.get(sp.face as usize) else {
                continue;
            };
            // Free vertices of the closest quad, nearest first.
            let mut cand: Vec<(f64, u32)> = q
                .face(f as usize)
                .iter()
                .filter(|&&v| {
                    free(q.vflag[v as usize]) && !moved[v as usize] && !locked[v as usize]
                })
                .map(|&v| (aq_core::math::dist(q.pos[v as usize], p), v))
                .collect();
            cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (_, v) in cand {
                let old = q.pos[v as usize];
                q.pos[v as usize] = p;
                if vf[v as usize]
                    .iter()
                    .all(|&g| quad_ok(q, g as usize, px, bvh))
                {
                    moved[v as usize] = true;
                    // The quads around v changed: their vertices wait for the next round.
                    for &g in &vf[v as usize] {
                        for &w in q.face(g as usize) {
                            locked[w as usize] = true;
                        }
                    }
                    n_moved += 1;
                    round_moves += 1;
                    break;
                }
                q.pos[v as usize] = old;
            }
        }
        if round_moves == 0 {
            break;
        }
    }
    if smooth && n_moved > 0 {
        smooth_around(q, &vf, &moved, px, bvh, &free);
    }
    Ok(n_moved)
}

/// Rings around the attracted vertices that are smoothed afterwards.
const ATTRACT_RINGS: usize = 3;
/// Smoothing sweeps over those rings.
const ATTRACT_SWEEPS: usize = 8;

/// Spreads the stretch left by attraction: the free, unmoved vertices within ATTRACT_RINGS
/// rings of a moved vertex go to the projected mean of their quad neighbours, a step at a
/// time, when every quad around them stays valid.
fn smooth_around(
    q: &mut QuadMesh,
    vf: &[Vec<u32>],
    moved: &[bool],
    px: &aq_core::TriMesh,
    bvh: &Bvh,
    free: &dyn Fn(u32) -> bool,
) {
    use aq_core::math::{add, scale};
    let nv = q.pos.len();
    // Edge neighbours of each vertex.
    let mut nb: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        let k = fv.len();
        for i in 0..k {
            let (a, b) = (fv[i], fv[(i + 1) % k]);
            if !nb[a as usize].contains(&b) {
                nb[a as usize].push(b);
            }
            if !nb[b as usize].contains(&a) {
                nb[b as usize].push(a);
            }
        }
    }
    let mut ring: Vec<u32> = (0..nv as u32).filter(|&v| moved[v as usize]).collect();
    let mut seen = moved.to_vec();
    let mut active: Vec<u32> = Vec::new();
    for _ in 0..ATTRACT_RINGS {
        let mut next = Vec::new();
        for &v in &ring {
            for &w in &nb[v as usize] {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    next.push(w);
                    if free(q.vflag[w as usize]) {
                        active.push(w);
                    }
                }
            }
        }
        ring = next;
    }
    active.sort_unstable();
    for _ in 0..ATTRACT_SWEEPS {
        for &v in &active {
            let n = &nb[v as usize];
            if n.is_empty() {
                continue;
            }
            let c = n.iter().fold([0.0; 3], |s, &w| add(s, q.pos[w as usize]));
            let target = bvh.closest(scale(c, 1.0 / n.len() as f64)).1;
            let old = q.pos[v as usize];
            q.pos[v as usize] = target;
            if !vf[v as usize]
                .iter()
                .all(|&g| quad_ok(q, g as usize, px, bvh))
            {
                q.pos[v as usize] = old;
            }
        }
    }
}

/// Quads with a corner whose Jacobian is not positive against the proxy normal at the
/// quad centroid's closest proxy point (WP-16 replacement for the hard S10 corner check).
fn flipped_quads(q: &QuadMesh, px: &aq_core::TriMesh, bvh: &Bvh) -> u32 {
    use aq_core::math::{add, cross, dot, scale, sub};
    let mut n = 0;
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        let k = fv.len();
        let c = fv.iter().fold([0.0; 3], |s, &v| add(s, q.pos[v as usize]));
        let (sp, _, _) = bvh.closest(scale(c, 1.0 / k as f64));
        let nrm = px.face_normal(sp.face).unwrap_or([0.0; 3]);
        let bad = (0..k).any(|i| {
            let v = q.pos[fv[i] as usize];
            let a = sub(q.pos[fv[(i + 1) % k] as usize], v);
            let b = sub(q.pos[fv[(i + k - 1) % k] as usize], v);
            dot(cross(a, b), nrm) <= 0.0
        });
        if bad {
            n += 1;
        }
    }
    n
}

/// What [`s5_s8`] returns.
struct S58 {
    lay: aq_core::Layout,
    /// 7.2: the layout health above 0.1 on more than 5% of the patches.
    degraded: bool,
}

/// S5 to S8 on a field: loops, cut, seamless parametrization (with its report block) and the
/// layout (with its report block), with the 7.3 checks (WP-23 moved them here from
/// `run_rung_traced`, unchanged, so that they can run again on a second field).
#[allow(clippy::too_many_arguments)]
fn s5_s8(
    px: &Proxy,
    field: &aq_core::CrossField,
    curv: &aq_core::CurvField,
    size: &aq_core::SizeField,
    p: &Params,
    r1: bool,
    seed: u64,
    n_c: f64,
    chi: i64,
    ctl: &Ctl,
    timing: &mut Vec<(Stage, f64)>,
    report: &mut Map<String, Value>,
    dumps: &mut Dumps,
    trouble: &mut Vec<[f64; 3]>,
) -> Result<S58, AqError> {
    let nf = px.mesh.tris.len();
    // S5 loops, S6 cut, S7 seamless parametrization.
    let S57 { loops, prm, pdiag } = s5_s7(px, field, curv, size, p, r1, seed, n_c, ctl, timing)?;
    // Interior cones S8 cannot start from (wrong uv winding, or a fold left within 1.5 h):
    // reported for diagnosis (WP-16 round 3).
    let bad = bad_cones(px, field, &prm);
    // M2 integration: where S8 may fail (folded or degenerate uv faces, bad cones), for the
    // S4 re-run that leaves the WP-23 edits there out.
    trouble.clear();
    trouble.extend(uv_trouble_points(px, &prm));
    trouble.extend(bad.iter().map(|&v| px.mesh.pos[v as usize]));
    let pr = &prm.report;
    report.insert(
        "param".into(),
        json!({
            "n_free": pr.n_free, "rows": pr.rows, "redundant": pr.redundant, "inconsistent": pr.inconsistent,
            "penalty_rows": pr.penalty_rows, "loops": loops.len(), "loops_exact": pr.loops_exact,
            "loops_dropped": pr.loops_dropped, "loop_set": format!("{:?}", pdiag.loop_set),
            "e0": pr.e0, "e_final": pr.e_final, "flips_initial": pr.flips_initial,
            "flips_after_stiff": pr.flips_after_stiff, "folds_after_repair": pr.folds_after_repair,
            "n_pred": pr.n_pred, "scale": pr.scale, "repaired": pdiag.repair.repaired,
            "skipped_cone": pdiag.repair.skipped_cone, "untangled": pdiag.repair.untangled,
            "bad_cones": bad.len(),
        }),
    );
    dumps.run("param", |d| d.param(&px.mesh, &prm));
    ctl.check()?;

    // S8 layout.
    report.insert(
        "timing_ms_before_s8".into(),
        json!(
            timing
                .iter()
                .map(|(st, t)| format!("{}:{:.0}", st.name(), t))
                .collect::<Vec<_>>()
        ),
    );
    let t = Instant::now();
    let yview = aq_layout::ProxyView {
        mesh: &px.mesh,
        topo: &px.topo,
        feats: &px.feats,
    };
    let lp = aq_layout::LayoutParams { max_rounds: LAYOUT_ROUNDS, ..Default::default() };
    let (lay, lstats) = aq_layout::build_layout_with(&yview, &prm, &loops, &[], &lp, ctl)?;
    lay.validate(nf).map_err(|e| e.at(Stage::Layout))?;
    if lay.tmesh.chi as i64 != chi {
        return Err(inv(
            Stage::Layout,
            format!("S8: T-mesh chi {} != {}", lay.tmesh.chi, chi),
        ));
    }
    let h = &lay.health;
    let degraded = h.frac_above > HEALTH_FRAC;
    report.insert(
        "layout".into(),
        json!({
            "patches": lay.tmesh.patches.len(), "arcs": lay.tmesh.arcs.len(), "nodes": lay.tmesh.nodes.len(),
            "health_mean": h.mean, "health_max": h.max, "health_frac": h.frac_above,
            "seed_rounds": lstats.runs.saturating_sub(1), "seeds": lstats.seeds.len(),
            "loops": lstats.n_loops, "traces": lstats.n_traces, "wfaces": lay.wmesh.tris.len(),
        }),
    );
    timing.push((Stage::Layout, ms(t)));
    dumps.run("layout", |d| d.layout(&lay));
    Ok(S58 { lay, degraded })
}

/// WP-23 pole sanity cap (DESIGN 6.6 M2): the re-run multiplies the curvature scale by this...
const CAP_SCALE: f64 = 1.5;
/// ...and the alignment by this.
const CAP_ALIGN: f64 = 0.7;

/// What [`s4_field`] returns.
struct S4 {
    field: aq_core::CrossField,
    frep: aq_field::FieldReport,
    /// Cones of the pure smooth run (alignment 0), the cap reference S_ref.
    s_ref: u32,
    /// True when the cap fired and the re-run was kept.
    rerun: bool,
}

/// S4 with the M2 pole sanity cap (DESIGN 6.6, WP-23). The field and its smooth reference
/// (the same S4 with alignment 0) are solved side by side. When the field keeps more than
/// `max(1.5 S_ref, S_ref + 8)` cones, `excess_poles` is set and S3 (curvature only, the size
/// field is kept) and S4 run again with curvature scale x1.5 and align x0.7; the re-run is
/// kept when it has fewer cones, and its curvature replaces `curv`.
fn s4_field(
    px: &Proxy,
    curv: &mut aq_core::CurvField,
    size: &aq_core::SizeField,
    fp: &aq_field::FieldParams,
    cp: &aq_curv::CurvParams,
    sp: &aq_curv::SizingParams,
    ctl: &Ctl,
) -> Result<S4, AqError> {
    let fview = aq_field::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
    let smooth = aq_field::FieldParams { align: 0.0, ..fp.clone() };
    let c0: &aq_core::CurvField = curv;
    let (main, refr) = rayon::join(
        || aq_field::solve_field(&fview, c0, size, fp, ctl),
        || aq_field::solve_field(&fview, c0, size, &smooth, ctl),
    );
    let (mut field, mut frep) = main?;
    let s_ref = refr?.1.cones_final;
    frep.excess_poles = aq_field::excess_poles(frep.cones_final, s_ref);
    let mut rerun = false;
    if frep.excess_poles && fp.align > 0.0 {
        ctl.check()?;
        let cp2 = aq_curv::CurvParams { scale: cp.scale * CAP_SCALE, ..*cp };
        let cview = aq_curv::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        // M2 integration: only the curvature is measured again; the S3 size field, with its
        // thickness cap and floor, is kept. `curvature_and_sizing` with the thickness-aware
        // `sp` would also cast the shape diameter rays and size again, both thrown away. As in
        // S3, pass 1 measures with h = 1 and a second pass (adaptive) with the kept h.
        let c2 = if sp.adaptive > 0.0 {
            aq_curv::curvature(&cview, &size.h_vert, &cp2, ctl)?
        } else {
            let h1 = vec![1.0; px.mesh.pos.len()];
            aq_curv::curvature(&cview, &h1, &cp2, ctl)?
        };
        let fp2 = aq_field::FieldParams { align: fp.align * CAP_ALIGN, ..fp.clone() };
        let (f2, mut r2) = aq_field::solve_field(&fview, &c2, size, &fp2, ctl)?;
        if r2.cones_final < frep.cones_final {
            r2.excess_poles = true;
            field = f2;
            frep = r2;
            *curv = c2;
            rerun = true;
        }
    }
    field
        .validate(&px.mesh, &px.topo, &px.frames)
        .map_err(|e| e.at(Stage::Field))?;
    check_field_index(px, &field)?;
    Ok(S4 { field, frep, s_ref, rerun })
}

/// 7.3 after S4: interior non-corner cones have |index4| <= 1, or -2 for a merged pair of -1
/// cones (WP-23, `aq_field::interior_index_ok`).
fn check_field_index(px: &Proxy, field: &aq_core::CrossField) -> Result<(), AqError> {
    for v in 0..px.mesh.pos.len() {
        if !px.topo.is_bnd_vert[v] && !px.feats.vert_corner[v] && !aq_field::interior_index_ok(field.index4[v]) {
            return Err(inv(
                Stage::Field,
                format!("S4: interior vertex {v} has index4 {}", field.index4[v]),
            ));
        }
    }
    Ok(())
}

/// Results of S5 to S7.
struct S57 {
    loops: Vec<LoopCand>,
    prm: aq_core::SeamlessParam,
    pdiag: aq_param::ParamDiag,
}

/// S5 loops, S6 cut and S7 seamless parametrization with their 7.3 checks.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN counts must fail the invariants
fn s5_s7(
    px: &Proxy,
    field: &aq_core::CrossField,
    curv: &aq_core::CurvField,
    size: &aq_core::SizeField,
    p: &Params,
    r1: bool,
    seed: u64,
    n_c: f64,
    ctl: &Ctl,
    timing: &mut Vec<(Stage, f64)>,
) -> Result<S57, AqError> {
    // S5 loops.
    let t = Instant::now();
    let mut loops: Vec<LoopCand> = Vec::new();
    if p.close_loops != 0 {
        let lview = aq_loops::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        let lp = aq_loops::LoopParams {
            seed,
            ..Default::default()
        };
        loops = aq_loops::find_loops(&lview, field, curv, size, &lp, ctl)?;
        if r1 {
            loops.retain(|l| l.ring);
        }
        for l in &loops {
            l.validate(&px.mesh, &px.topo)
                .map_err(|e| e.at(Stage::Loops))?;
        }
    }
    timing.push((Stage::Loops, ms(t)));
    ctl.check()?;

    // S6 cut.
    let t = Instant::now();
    let kview = aq_cut::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
    let cut: CutGraph = aq_cut::cut_and_comb(&kview, field, seed)?;
    cut.validate(&px.mesh, &px.topo, Some(field))
        .map_err(|e| e.at(Stage::Cut))?;
    let cut_chi = aq_cut::cut_mesh_euler(&kview, &cut.is_cut);
    if cut_chi.iter().any(|&x| x != 1) {
        return Err(inv(Stage::Cut, format!("S6: cut chi {cut_chi:?}")));
    }
    timing.push((Stage::Cut, ms(t)));
    ctl.check()?;

    // S7 seamless parametrization.
    let t = Instant::now();
    let pview = aq_param::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
    let prp = aq_param::ParamParams {
        loops: p.close_loops != 0,
        n_target: n_c,
        ..Default::default()
    };
    let (prm, pdiag) = aq_param::seamless_detailed(&pview, field, &cut, size, &loops, &prp, ctl)?;
    prm.validate(&px.mesh, &px.topo)
        .map_err(|e| e.at(Stage::Param))?;
    if !((prm.report.n_pred - n_c).abs() <= COUNT_TOL * n_c) {
        return Err(inv(
            Stage::Param,
            format!("S7: N_pred {} for N_c {n_c}", prm.report.n_pred),
        ));
    }
    timing.push((Stage::Param, ms(t)));
    ctl.check()?;
    Ok(S57 { loops, prm, pdiag })
}

/// Signed uv angle at corner k of face f of the param.
fn uv_corner_angle(prm: &aq_core::SeamlessParam, f: usize, k: usize) -> f64 {
    let a = prm.uv[f][k];
    let b = prm.uv[f][(k + 1) % 3];
    let c = prm.uv[f][(k + 2) % 3];
    let (u, w) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
    (u[0] * w[1] - u[1] * w[0]).atan2(u[0] * w[0] + u[1] * w[1])
}

/// Centroids of the faces S7 left folded.
fn fold_points(px: &Proxy, prm: &aq_core::SeamlessParam) -> Vec<[f64; 3]> {
    prm.folded
        .iter()
        .enumerate()
        .filter(|&(_, &fd)| fd)
        .map(|(f, _)| {
            let [a, b, c] = px.mesh.tri_pos(f as u32);
            [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ]
        })
        .collect()
}

/// Centroids of the faces S8 cannot use: left folded by S7, or with a uv orientation that is
/// not strictly positive (the `aq_layout` emitter check).
fn uv_trouble_points(px: &Proxy, prm: &aq_core::SeamlessParam) -> Vec<[f64; 3]> {
    (0..px.mesh.tris.len())
        .filter(|&f| {
            let u = prm.uv[f];
            prm.folded.get(f).copied().unwrap_or(false) || aq_core::exact::orient2d(u[0], u[1], u[2]) <= 0.0
        })
        .map(|f| {
            let [a, b, c] = px.mesh.tri_pos(f as u32);
            [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0]
        })
        .collect()
}

/// Number of WP-23 edits (far cancellations, -1 pair merges, boundary merges) in a field.
fn wp23_edits(r: &aq_field::FieldReport) -> u32 {
    r.far_cancelled + r.merged_neg + r.boundary_merged
}

/// A cone within this distance (in h, Euclidean) of a face left folded by S7 counts as bad.
const BAD_CONE_FOLD_DIST: f64 = 1.5;

/// Interior non-corner cones that S8 cannot start from: the uv angle sum of the fan is not
/// 2 pi - index4 pi/2, or a face left folded by S7 is within BAD_CONE_FOLD_DIST h (the folds
/// the repair cannot undo sit next to a cone whose neighbourhood winds wrong). Sorted.
fn bad_cones(px: &Proxy, field: &aq_core::CrossField, prm: &aq_core::SeamlessParam) -> Vec<u32> {
    let t = &px.topo;
    let fold_pts = fold_points(px, prm);
    (0..px.mesh.pos.len() as u32)
        .filter(|&v| {
            let i4 = field.index4[v as usize];
            if i4 == 0 || t.is_bnd_vert[v as usize] || px.feats.vert_corner[v as usize] {
                return false;
            }
            let mut sum = 0.0;
            let mut folded = false;
            for h in t.fan_out(v) {
                let f = (h / 3) as usize;
                sum += uv_corner_angle(prm, f, (h % 3) as usize);
                folded |= prm.folded.get(f).copied().unwrap_or(false);
            }
            let want = std::f64::consts::TAU - f64::from(i4) * std::f64::consts::FRAC_PI_2;
            let pv = px.mesh.pos[v as usize];
            let near_fold = fold_pts
                .iter()
                .any(|&q| aq_core::math::dist(pv, q) < BAD_CONE_FOLD_DIST);
            folded || near_fold || (sum - want).abs() > 0.1
        })
        .collect()
}

/// 7.3 after S10 and S11: valid CSR, manifold and oriented, all quads, chi.
fn check_quads(q: &QuadMesh, chi: i64, stage: Stage) -> Result<(), AqError> {
    q.validate().map_err(|e| e.at(stage))?;
    if q.n_faces() == 0 || !q.is_all_quads() {
        return Err(inv(stage, "quad mesh empty or not all quads"));
    }
    let x = crate::merge::euler_char(q);
    if x != chi {
        return Err(inv(stage, format!("quad mesh chi {x} != {chi}")));
    }
    Ok(())
}

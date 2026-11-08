//! # aq_ffi: the C ABI, `autoquad_engine.dll` (DESIGN 8)
//!
//! The `#[repr(C)]` structs mirror `include/autoquad_engine.h` (DESIGN 8.1) field by field;
//! `tests/abi.rs` pins every size and offset. Rules (DESIGN 8.2):
//!
//! * **Ownership:** all output memory is allocated here and freed only by [`aq_free`]. Input
//!   pointers are borrowed for the duration of the call and copied, never retained.
//! * **Errors:** [`aq_remesh`] returns `(*out)->status`; `*out` is always set, except when `out`
//!   itself is NULL, which returns 4. No global state: the error text is in the result and the
//!   cancel flag in `AqControl`, so concurrent calls are safe.
//! * **catch_unwind** wraps every exported function; a panic gives status 5 with its message.
//! * **Progress:** the engine stores `progress_bits` (f32 bits) and `stage` with relaxed atomic
//!   stores through the caller's `AqControl`, and polls `cancel` the same way.
//! * The call blocks; the engine builds its own rayon pool per call.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI32, AtomicU32};

use aq_core::{Ctl, CtlState, Stage, TriMesh, V3};
use aq_engine::{ABI_VERSION, Input, Params, Status};

/// `AQ_ABI_VERSION`.
pub const AQ_ABI_VERSION: u32 = ABI_VERSION;

/// Status codes of `AqResult.status`.
pub const AQ_OK: i32 = 0;
pub const AQ_DEGRADED: i32 = 1;
pub const AQ_FAILED: i32 = 2;
pub const AQ_CANCELLED: i32 = 3;
pub const AQ_INVALID_INPUT: i32 = 4;
pub const AQ_PANIC: i32 = 5;

/// Engine parameters (DESIGN 8.1 `AqParams`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AqParams {
    /// `sizeof(AqParams)`; a mismatch gives status 4.
    pub struct_size: u32,
    pub target_quads: u32,
    pub adaptive_size: f32,
    pub curvature_align: f32,
    pub flow_smoothness: f32,
    pub density_strength: f32,
    pub guide_strength: f32,
    pub hard_edges: u8,
    pub close_loops: u8,
    pub allow_collapse: u8,
    pub reserved0: u8,
    pub ring_min: u32,
    pub max_proxy_tris: u32,
    pub threads: u32,
    pub time_limit_s: f32,
    pub debug_flags: u32,
    pub seed: u64,
    /// NULL or a UTF-8 path.
    pub dump_dir: *const c_char,
}

/// Input mesh (DESIGN 8.1 `AqMeshIn`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AqMeshIn {
    pub n_verts: u32,
    /// 3 * n_verts doubles.
    pub pos: *const f64,
    pub n_tris: u32,
    /// 3 * n_tris indices, CCW.
    pub tris: *const u32,
    pub n_feat: u32,
    /// 2 * n_feat vertex indices.
    pub feat: *const u32,
    /// n_feat kinds: 1 Hard, 2 Guide, 3 GuideHard, 5 Symmetry.
    pub feat_kind: *const u8,
    /// n_verts multipliers or NULL.
    pub density: *const f32,
}

/// Control block, allocated and polled by the caller (DESIGN 8.1 `AqControl`).
#[repr(C)]
#[derive(Debug, Default)]
pub struct AqControl {
    /// The caller sets 1 to cancel.
    pub cancel: AtomicI32,
    /// f32 bits in [0, 1], written by the engine.
    pub progress_bits: AtomicU32,
    /// `Stage` value, written by the engine.
    pub stage: AtomicU32,
}

/// Result (DESIGN 8.1 `AqResult`). Allocated by [`aq_remesh`], freed by [`aq_free`].
#[repr(C)]
#[derive(Debug)]
pub struct AqResult {
    pub status: i32,
    pub n_verts: u32,
    /// 3 * n_verts doubles.
    pub pos: *mut f64,
    pub n_faces: u32,
    /// n_faces + 1 offsets.
    pub face_offsets: *mut u32,
    pub face_verts: *mut u32,
    /// n_verts flags (DESIGN 8.4).
    pub vflag: *mut u32,
    /// n_faces component ids.
    pub face_comp: *mut u32,
    pub n_comp: u32,
    /// n_comp statuses: 0 full, 1 retried, 2 SF, 3 failed (QuadriFlow), 4 debris SF.
    pub comp_status: *mut u8,
    pub n_failed_tris: u32,
    pub failed_tris: *mut u32,
    /// UTF-8, NUL-terminated, never NULL.
    pub report_json: *mut c_char,
    /// UTF-8 message, "" when status 0, never NULL.
    pub error: *mut c_char,
}

/// The allocation behind an `AqResult*`: the public struct first (offset 0, `repr(C)`), then
/// the buffer lengths `aq_free` needs. Callers only see the first field.
#[repr(C)]
struct ResultAlloc {
    r: AqResult,
    len_pos: usize,
    len_face_offsets: usize,
    len_face_verts: usize,
    len_vflag: usize,
    len_face_comp: usize,
    len_comp_status: usize,
    len_failed_tris: usize,
}

/// A boxed slice turned into a raw pointer, NULL when empty.
fn into_raw<T>(v: Vec<T>) -> (*mut T, usize) {
    if v.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let b = v.into_boxed_slice();
    let n = b.len();
    (Box::into_raw(b) as *mut T, n)
}

/// Frees a pointer made by [`into_raw`].
///
/// # Safety
/// `p` must be NULL or come from `into_raw` with length `n`, and not be freed yet.
unsafe fn free_raw<T>(p: *mut T, n: usize) {
    if !p.is_null() {
        // SAFETY: per the contract, (p, n) is a leaked Box<[T]>.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n)) });
    }
}

/// A C string (interior NUL bytes replaced by spaces).
fn c_string(s: &str) -> *mut c_char {
    let clean: String = s.chars().map(|c| if c == '\0' { ' ' } else { c }).collect();
    CString::new(clean).unwrap_or_default().into_raw()
}

/// Builds the result allocation from an engine output.
fn make_result(o: aq_engine::Output) -> *mut AqResult {
    let report = o.report_json();
    let q = o.mesh;
    let n_verts = q.pos.len() as u32;
    let n_faces = q.n_faces() as u32;
    let mut pos = Vec::new();
    if pos.try_reserve_exact(3 * q.pos.len()).is_err() {
        return make_error(AQ_FAILED, "out of memory while copying the result");
    }
    for p in &q.pos {
        pos.extend_from_slice(p);
    }
    let face_off = if q.face_off.is_empty() {
        vec![0]
    } else {
        q.face_off
    };
    let comp_status: Vec<u8> = o.comp_status.iter().map(|&s| s as u8).collect();
    let n_comp = comp_status.len() as u32;
    let n_failed = o.failed_tris.len() as u32;
    let (pos, len_pos) = into_raw(pos);
    let (face_offsets, len_face_offsets) = into_raw(face_off);
    let (face_verts, len_face_verts) = into_raw(q.face_verts);
    let (vflag, len_vflag) = into_raw(q.vflag);
    let (face_comp, len_face_comp) = into_raw(q.face_comp);
    let (comp_status, len_comp_status) = into_raw(comp_status);
    let (failed_tris, len_failed_tris) = into_raw(o.failed_tris);
    let a = Box::new(ResultAlloc {
        r: AqResult {
            status: o.status.code(),
            n_verts,
            pos,
            n_faces,
            face_offsets,
            face_verts,
            vflag,
            face_comp,
            n_comp,
            comp_status,
            n_failed_tris: n_failed,
            failed_tris,
            report_json: c_string(&report),
            error: c_string(&o.error),
        },
        len_pos,
        len_face_offsets,
        len_face_verts,
        len_vflag,
        len_face_comp,
        len_comp_status,
        len_failed_tris,
    });
    Box::into_raw(a) as *mut AqResult
}

/// A result with no mesh, the given status and message.
fn make_error(status: i32, msg: &str) -> *mut AqResult {
    // The engine's report builder, so every DESIGN 9 key is present (zero timings).
    let st = Status::from_code(status).unwrap_or(Status::Failed);
    let mut report = aq_engine::error_report(st, msg);
    if let Some(o) = report.as_object_mut() {
        o.insert("status".into(), serde_json::json!(status));
    }
    let report = report.to_string();
    let (face_offsets, len_face_offsets) = into_raw(vec![0u32]);
    let a = Box::new(ResultAlloc {
        r: AqResult {
            status,
            n_verts: 0,
            pos: std::ptr::null_mut(),
            n_faces: 0,
            face_offsets,
            face_verts: std::ptr::null_mut(),
            vflag: std::ptr::null_mut(),
            face_comp: std::ptr::null_mut(),
            n_comp: 0,
            comp_status: std::ptr::null_mut(),
            n_failed_tris: 0,
            failed_tris: std::ptr::null_mut(),
            report_json: c_string(&report),
            error: c_string(msg),
        },
        len_pos: 0,
        len_face_offsets,
        len_face_verts: 0,
        len_vflag: 0,
        len_face_comp: 0,
        len_comp_status: 0,
        len_failed_tris: 0,
    });
    Box::into_raw(a) as *mut AqResult
}

/// The defaults of DESIGN 8.1.
pub fn default_params() -> AqParams {
    let d = Params::default();
    AqParams {
        struct_size: std::mem::size_of::<AqParams>() as u32,
        target_quads: d.target_quads,
        adaptive_size: d.adaptive_size,
        curvature_align: d.curvature_align,
        flow_smoothness: d.flow_smoothness,
        density_strength: d.density_strength,
        guide_strength: d.guide_strength,
        hard_edges: d.hard_edges,
        close_loops: d.close_loops,
        allow_collapse: d.allow_collapse,
        reserved0: 0,
        ring_min: d.ring_min,
        max_proxy_tris: d.max_proxy_tris,
        threads: d.threads,
        time_limit_s: d.time_limit_s,
        debug_flags: d.debug_flags,
        seed: d.seed,
        dump_dir: std::ptr::null(),
    }
}

/// `AQ_ABI_VERSION` (1).
#[unsafe(no_mangle)]
pub extern "C" fn aq_abi_version() -> u32 {
    AQ_ABI_VERSION
}

/// Writes the defaults into `*p` (with `struct_size` set). NULL-safe.
///
/// # Safety
/// `p` must be NULL or point to writable memory of `sizeof(AqParams)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aq_params_default(p: *mut AqParams) {
    if p.is_null() || !(p as usize).is_multiple_of(std::mem::align_of::<AqParams>()) {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: p is non-null, aligned and writable per the contract.
        unsafe { p.write(default_params()) };
    }));
}

/// `true` when `p` is non-null and aligned for T.
fn ok_ptr<T>(p: *const T) -> bool {
    !p.is_null() && (p as usize).is_multiple_of(std::mem::align_of::<T>())
}

/// A borrowed slice of `n` elements, or `Err` when the pointer is NULL (with n > 0) or
/// misaligned. n = 0 gives an empty slice whatever the pointer.
///
/// # Safety
/// When `p` is non-null and n > 0, `p` must point to n readable elements.
unsafe fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Result<&'a [T], String> {
    if n == 0 {
        return Ok(&[]);
    }
    if !ok_ptr(p) {
        return Err(format!("{what} is NULL or misaligned with {n} elements"));
    }
    // SAFETY: non-null, aligned, n readable elements per the contract.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// Converts the C parameters. `Err` is status 4.
///
/// # Safety
/// `p` must be non-null and point to at least `struct_size` readable bytes; when the size
/// matches, it is a full `AqParams`.
unsafe fn read_params(p: *const AqParams) -> Result<Params, String> {
    if !ok_ptr(p) {
        return Err("params is NULL or misaligned".into());
    }
    // SAFETY: the first field is a u32 and p is aligned and readable.
    let size = unsafe { std::ptr::read(p as *const u32) };
    if size as usize != std::mem::size_of::<AqParams>() {
        return Err(format!(
            "AqParams.struct_size = {size}, expected {} (ABI {ABI_VERSION})",
            std::mem::size_of::<AqParams>()
        ));
    }
    // SAFETY: the size matches, so p is a full AqParams.
    let c = unsafe { std::ptr::read(p) };
    let dump_dir = if c.dump_dir.is_null() {
        None
    } else {
        // SAFETY: a non-null dump_dir is a NUL-terminated string per the header.
        let s = unsafe { CStr::from_ptr(c.dump_dir) };
        Some(PathBuf::from(
            s.to_str()
                .map_err(|_| "dump_dir is not UTF-8".to_string())?,
        ))
    };
    Ok(Params {
        target_quads: c.target_quads,
        adaptive_size: c.adaptive_size,
        curvature_align: c.curvature_align,
        flow_smoothness: c.flow_smoothness,
        density_strength: c.density_strength,
        guide_strength: c.guide_strength,
        hard_edges: c.hard_edges,
        close_loops: c.close_loops,
        allow_collapse: c.allow_collapse,
        ring_min: c.ring_min,
        max_proxy_tris: c.max_proxy_tris,
        threads: c.threads,
        time_limit_s: c.time_limit_s,
        debug_flags: c.debug_flags,
        seed: c.seed,
        dump_dir,
        stop_after: None,
    })
}

/// The copied input arrays.
struct OwnedInput {
    mesh: TriMesh,
    pairs: Vec<[u32; 2]>,
    kinds: Vec<u8>,
    density: Option<Vec<f32>>,
}

/// Copies the borrowed input. `Err((status, message))`.
///
/// # Safety
/// `m` must be non-null and point to an `AqMeshIn` whose pointers hold the stated counts.
unsafe fn read_input(m: *const AqMeshIn) -> Result<OwnedInput, (i32, String)> {
    let bad = |s: String| (AQ_INVALID_INPUT, s);
    if !ok_ptr(m) {
        return Err(bad("mesh is NULL or misaligned".into()));
    }
    // SAFETY: non-null and aligned per the check; readable per the contract.
    let m = unsafe { std::ptr::read(m) };
    let nv = m.n_verts as usize;
    let nt = m.n_tris as usize;
    let nfe = m.n_feat as usize;
    // SAFETY: each pointer holds the stated count per the contract.
    let pos = unsafe { slice(m.pos, 3 * nv, "pos") }.map_err(bad)?;
    let tris = unsafe { slice(m.tris, 3 * nt, "tris") }.map_err(bad)?;
    let feat = unsafe { slice(m.feat, 2 * nfe, "feat") }.map_err(bad)?;
    let kind = unsafe { slice(m.feat_kind, nfe, "feat_kind") }.map_err(bad)?;
    let density = if m.density.is_null() {
        None
    } else {
        Some(unsafe { slice(m.density, nv, "density") }.map_err(bad)?)
    };
    let oom = |what: &str| (AQ_FAILED, format!("out of memory while copying {what}"));
    let mut vpos: Vec<V3> = Vec::new();
    vpos.try_reserve_exact(nv).map_err(|_| oom("pos"))?;
    vpos.extend(pos.as_chunks::<3>().0.iter().copied());
    let mut vtris: Vec<[u32; 3]> = Vec::new();
    vtris.try_reserve_exact(nt).map_err(|_| oom("tris"))?;
    vtris.extend(tris.as_chunks::<3>().0.iter().copied());
    let mut pairs: Vec<[u32; 2]> = Vec::new();
    pairs.try_reserve_exact(nfe).map_err(|_| oom("feat"))?;
    pairs.extend(feat.as_chunks::<2>().0.iter().copied());
    let mut kinds: Vec<u8> = Vec::new();
    kinds.try_reserve_exact(nfe).map_err(|_| oom("feat_kind"))?;
    kinds.extend_from_slice(kind);
    let density = match density {
        Some(d) => {
            let mut v: Vec<f32> = Vec::new();
            v.try_reserve_exact(nv).map_err(|_| oom("density"))?;
            v.extend_from_slice(d);
            Some(v)
        }
        None => None,
    };
    Ok(OwnedInput {
        mesh: TriMesh {
            pos: vpos,
            tris: vtris,
        },
        pairs,
        kinds,
        density,
    })
}

/// The body of [`aq_remesh`] (inside catch_unwind).
///
/// # Safety
/// As for [`aq_remesh`].
unsafe fn remesh_impl(
    m: *const AqMeshIn,
    p: *const AqParams,
    ctl: *mut AqControl,
) -> *mut AqResult {
    // Control block: the caller's atomics, or local ones when ctl is NULL or misaligned.
    let local = CtlState::new();
    let far = local.unbounded().deadline();
    let c: Ctl = if ok_ptr(ctl as *const AqControl) {
        // SAFETY: non-null, aligned, valid for the whole call per the contract; the fields
        // are atomics (the C side declares them volatile).
        let r: &AqControl = unsafe { &*ctl };
        Ctl::new(&r.cancel, &r.progress_bits, &r.stage, far)
    } else {
        local.ctl(far)
    };
    c.progress(Stage::Ffi, 0.0);
    // SAFETY: forwarded contract.
    let params = match unsafe { read_params(p) } {
        Ok(x) => x,
        Err(e) => return make_error(AQ_INVALID_INPUT, &e),
    };
    // SAFETY: forwarded contract.
    let inp = match unsafe { read_input(m) } {
        Ok(x) => x,
        Err((s, e)) => return make_error(s, &e),
    };
    let input = Input {
        mesh: inp.mesh,
        feat_pairs: &inp.pairs,
        feat_kind: &inp.kinds,
        density: inp.density.as_deref(),
    };
    let out = aq_engine::remesh(input, &params, &c);
    let r = make_result(out);
    c.progress(Stage::Engine, 1.0);
    r
}

/// Runs the engine. Blocks until done. Returns `(*out)->status`.
///
/// * `in`, `p`: borrowed for the call. `p->struct_size` must equal `sizeof(AqParams)`.
/// * `ctl`: nullable; polled for `cancel`, written with `progress_bits` and `stage`.
/// * `out`: receives a result to free with [`aq_free`]; when `out` is NULL nothing is
///   allocated and 4 is returned.
///
/// # Safety
/// The pointers must be NULL or valid as described in the header (DESIGN 8.1).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aq_remesh(
    m: *const AqMeshIn,
    p: *const AqParams,
    ctl: *mut AqControl,
    out: *mut *mut AqResult,
) -> i32 {
    if !ok_ptr(out as *const *mut AqResult) {
        return AQ_INVALID_INPUT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: forwarded contract.
        unsafe { remesh_impl(m, p, ctl) }
    }));
    let res = match r {
        Ok(res) => res,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            // Building the error result allocates; guard it too.
            match catch_unwind(|| make_error(AQ_PANIC, &format!("panic: {msg}"))) {
                Ok(x) => x,
                Err(_) => std::ptr::null_mut(),
            }
        }
    };
    // SAFETY: out is non-null and aligned per the check above.
    unsafe { out.write(res) };
    if res.is_null() {
        return AQ_PANIC;
    }
    // SAFETY: res is a live allocation from make_result / make_error.
    unsafe { (*res).status }
}

/// Frees a result from [`aq_remesh`]. NULL-safe. Every buffer of the result is released;
/// the pointer must not be used afterwards.
///
/// # Safety
/// `r` must be NULL or a pointer returned through `aq_remesh`'s `out` that was not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aq_free(r: *mut AqResult) {
    if r.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: r comes from Box::into_raw(Box<ResultAlloc>) per the contract.
        let a = unsafe { Box::from_raw(r as *mut ResultAlloc) };
        // SAFETY: each pointer was made by into_raw with the recorded length.
        unsafe {
            free_raw(a.r.pos, a.len_pos);
            free_raw(a.r.face_offsets, a.len_face_offsets);
            free_raw(a.r.face_verts, a.len_face_verts);
            free_raw(a.r.vflag, a.len_vflag);
            free_raw(a.r.face_comp, a.len_face_comp);
            free_raw(a.r.comp_status, a.len_comp_status);
            free_raw(a.r.failed_tris, a.len_failed_tris);
            if !a.r.report_json.is_null() {
                drop(CString::from_raw(a.r.report_json));
            }
            if !a.r.error.is_null() {
                drop(CString::from_raw(a.r.error));
            }
        }
    }));
}

/// The stage enum value as the engine writes it to `AqControl.stage` (for callers in Rust).
pub fn stage_name(v: u32) -> &'static str {
    Stage::from_u32(v).map_or("unknown", |s| s.name())
}

/// Maps an engine status to its code (for callers in Rust).
pub fn status_code(s: Status) -> i32 {
    s.code()
}

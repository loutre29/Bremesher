//! Calls the exported C functions the way a C caller would and copies the result out.
#![allow(dead_code)]

use aq_core::{QuadMesh, TriMesh};
use autoquad_engine::{
    AqControl, AqMeshIn, AqParams, AqResult, aq_free, aq_params_default, aq_remesh,
};

/// A copied `AqResult`.
#[derive(Clone, Debug, PartialEq)]
pub struct Res {
    pub ret: i32,
    pub status: i32,
    pub mesh: QuadMesh,
    pub comp_status: Vec<u8>,
    pub failed_tris: Vec<u32>,
    pub report: String,
    pub error: String,
}

pub fn defaults() -> AqParams {
    let mut p = std::mem::MaybeUninit::<AqParams>::uninit();
    unsafe {
        aq_params_default(p.as_mut_ptr());
        p.assume_init()
    }
}

pub fn params(target: u32) -> AqParams {
    AqParams {
        target_quads: target,
        threads: 2,
        ..defaults()
    }
}

/// Raw input arrays (kept alive while the AqMeshIn borrows them).
pub struct RawIn {
    pub pos: Vec<f64>,
    pub tris: Vec<u32>,
    pub feat: Vec<u32>,
    pub kind: Vec<u8>,
    pub density: Option<Vec<f32>>,
}

impl RawIn {
    pub fn new(m: &TriMesh) -> RawIn {
        RawIn {
            pos: m.pos.iter().flat_map(|p| p.iter().copied()).collect(),
            tris: m.tris.iter().flat_map(|t| t.iter().copied()).collect(),
            feat: Vec::new(),
            kind: Vec::new(),
            density: None,
        }
    }
    pub fn with_feats(mut self, pairs: &[[u32; 2]], kind: &[u8]) -> RawIn {
        self.feat = pairs.iter().flat_map(|p| p.iter().copied()).collect();
        self.kind = kind.to_vec();
        self
    }
    pub fn mesh_in(&self) -> AqMeshIn {
        AqMeshIn {
            n_verts: (self.pos.len() / 3) as u32,
            pos: self.pos.as_ptr(),
            n_tris: (self.tris.len() / 3) as u32,
            tris: self.tris.as_ptr(),
            n_feat: self.kind.len() as u32,
            feat: self.feat.as_ptr(),
            feat_kind: self.kind.as_ptr(),
            density: self
                .density
                .as_ref()
                .map_or(std::ptr::null(), |d| d.as_ptr()),
        }
    }
}

/// Copies a result and frees it.
pub unsafe fn take(ret: i32, r: *mut AqResult) -> Res {
    assert!(!r.is_null());
    let a = unsafe { &*r };
    let copy = |p: *const u32, n: usize| -> Vec<u32> {
        if n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
        }
    };
    let nv = a.n_verts as usize;
    let nf = a.n_faces as usize;
    let pos: Vec<[f64; 3]> = if nv == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(a.pos, 3 * nv) }
            .as_chunks::<3>()
            .0
            .to_vec()
    };
    assert!(!a.face_offsets.is_null(), "face_offsets is never NULL");
    let face_off = copy(a.face_offsets, nf + 1);
    let face_verts = copy(a.face_verts, *face_off.last().unwrap() as usize);
    let vflag = copy(a.vflag, nv);
    let face_comp = copy(a.face_comp, nf);
    let comp_status = if a.n_comp == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(a.comp_status, a.n_comp as usize) }.to_vec()
    };
    let failed_tris = copy(a.failed_tris, a.n_failed_tris as usize);
    assert!(!a.report_json.is_null() && !a.error.is_null());
    let report = unsafe { std::ffi::CStr::from_ptr(a.report_json) }
        .to_str()
        .unwrap()
        .to_string();
    let error = unsafe { std::ffi::CStr::from_ptr(a.error) }
        .to_str()
        .unwrap()
        .to_string();
    let res = Res {
        ret,
        status: a.status,
        mesh: QuadMesh {
            pos,
            face_off,
            face_verts,
            vflag,
            face_comp,
        },
        comp_status,
        failed_tris,
        report,
        error,
    };
    unsafe { aq_free(r) };
    res
}

/// One `aq_remesh` call.
pub fn call_raw(inp: &RawIn, p: &AqParams, ctl: Option<&AqControl>) -> Res {
    let m = inp.mesh_in();
    let mut out: *mut AqResult = std::ptr::null_mut();
    let c = ctl.map_or(std::ptr::null_mut(), |c| {
        c as *const AqControl as *mut AqControl
    });
    let ret = unsafe { aq_remesh(&m, p, c, &mut out) };
    let r = unsafe { take(ret, out) };
    assert_eq!(r.ret, r.status, "return value equals status");
    r
}

pub fn call(m: &TriMesh, p: &AqParams) -> Res {
    call_raw(&RawIn::new(m), p, None)
}

/// Checks a status 0 or 1 result: valid, all quads, report JSON parses with the status.
pub fn check_ok(r: &Res) {
    assert!(
        r.status == 0 || r.status == 1,
        "status {} ({})",
        r.status,
        r.error
    );
    r.mesh.validate().unwrap();
    assert!(r.mesh.is_all_quads());
    assert!(r.mesh.n_faces() > 0);
    for &c in &r.mesh.face_comp {
        assert!((c as usize) < r.comp_status.len());
    }
    let j: serde_json::Value = serde_json::from_str(&r.report).unwrap();
    assert_eq!(j["status"], r.status);
    assert_eq!(j["abi"], 1);
}

/// Scales about the origin to a bounding box diagonal of 10 (the add-on's WORK_DIAG).
pub fn normalized(mut m: TriMesh) -> TriMesh {
    aq_engine::host::normalize(&mut m);
    m
}

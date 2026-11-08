//! Shared helpers of the aq_param tests.
#![allow(dead_code)]

use aq_core::param::FAMILY_NONE;
use aq_core::{
    CrossField, CtlState, CutGraph, LoopCand, NONE, SeamlessParam, SizeField, Transition,
};
use aq_param::{ParamDiag, ParamParams, ProxyRef, loop_holonomy, seamless_detailed};
use aq_testkit::{ParamCase, Surface};

pub fn px(s: &Surface) -> ProxyRef<'_> {
    ProxyRef::new(&s.mesh, &s.topo, &s.frames, &s.feats)
}

/// Runs S7 with an unbounded ctl.
pub fn run(
    s: &Surface,
    f: &CrossField,
    cut: &CutGraph,
    size: &SizeField,
    loops: &[LoopCand],
    p: &ParamParams,
) -> (SeamlessParam, ParamDiag) {
    let st = CtlState::new();
    let ctl = st.unbounded();
    match seamless_detailed(&px(s), f, cut, size, loops, p, &ctl) {
        Ok(r) => r,
        Err(e) => panic!("seamless failed: {e}"),
    }
}

/// Runs S7 on a testkit case (its field, cut, size and loops).
pub fn run_case(c: &ParamCase, p: &ParamParams) -> (SeamlessParam, ParamDiag) {
    run(&c.surface, &c.field, &c.cut, &c.size, &c.loops, p)
}

/// Params with the target count equal to the size field's prediction.
pub fn params() -> ParamParams {
    ParamParams::default()
}

/// Every exactness statement of DESIGN 6.9 "Exactness after the snap", checked independently
/// of `SeamlessParam::validate`: both endpoints of every cut halfedge satisfy its transition
/// bitwise, chain iso coordinates are bitwise equal, every exact loop closes bitwise.
pub fn check_exact(s: &Surface, cut: &CutGraph, sp: &SeamlessParam, loops: &[LoopCand]) {
    let (m, t) = (&s.mesh, &s.topo);
    sp.validate(m, t).unwrap();
    let nh = 3 * m.tris.len();
    let mut n_cut = 0;
    for h in 0..nh as u32 {
        let tw = t.twin[h as usize];
        if tw == NONE {
            assert!(sp.trans[h as usize].is_identity());
            continue;
        }
        let tr = sp.trans[h as usize];
        if !cut.is_cut[h as usize] {
            assert!(tr.is_identity(), "non-cut he {h} has {tr:?}");
        } else {
            n_cut += 1;
        }
        let (a_f, b_f) = sp.he_uv(h);
        let (b_g, a_g) = sp.he_uv(tw);
        assert!(
            tr.apply(a_f) == a_g && tr.apply(b_f) == b_g,
            "he {h}: transition not exact"
        );
        assert_eq!(sp.trans[tw as usize], tr.inverse());
    }
    let _ = n_cut;
    for h in 0..nh as u32 {
        let fam = sp.he_family[h as usize];
        if fam == FAMILY_NONE {
            continue;
        }
        let (a, b) = sp.he_uv(h);
        let q = if fam == 0 { 1 } else { 0 };
        assert!(a[q] == b[q], "chain he {h} not iso exact");
    }
    for &li in &sp.loops_exact {
        let l = &loops[li as usize];
        let c = loop_holonomy(&sp.trans, &l.hes);
        assert_eq!(c.r, 0, "exact loop {li} rotates");
        let k =
            aq_core::cross::raw_to_combed_branch(l.branch, cut.comb_offset[l.start_face as usize]);
        let perp = if k.is_multiple_of(2) { c.t[1] } else { c.t[0] };
        assert!(perp == 0.0, "exact loop {li} drifts by {perp}");
    }
    for c in sp.uv.iter().flatten().flatten() {
        assert!(aq_core::tol::is_dyadic(*c) && c.abs() < aq_core::tol::UV_BOUND);
    }
}

/// Transversal drift of loop l under the param.
pub fn loop_drift(cut: &CutGraph, trans: &[Transition], l: &LoopCand) -> f64 {
    let c = loop_holonomy(trans, &l.hes);
    let k = aq_core::cross::raw_to_combed_branch(l.branch, cut.comb_offset[l.start_face as usize]);
    if k.is_multiple_of(2) { c.t[1] } else { c.t[0] }
}

pub fn n_folded(sp: &SeamlessParam) -> usize {
    sp.folded.iter().filter(|&&b| b).count()
}

/// |grad u| and |grad v| per face.
pub fn grad_norms(s: &Surface, uv: &[[[f64; 2]; 3]]) -> Vec<(f64, f64)> {
    (0..s.n_faces())
        .map(|f| {
            let j = aq_core::cross::jacobian(&s.frames, &s.mesh, f as u32, &uv[f]);
            (j[0][0].hypot(j[0][1]), j[1][0].hypot(j[1][1]))
        })
        .collect()
}

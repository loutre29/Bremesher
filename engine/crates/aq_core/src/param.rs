//! Seamless parametrization contract (DESIGN 5 `aq_core::param`, 6.9, C10, C14).

use crate::cross::Transition;
use crate::error::{AqError, ensure};
use crate::mesh::{NONE, TriMesh, V2};
use crate::tol;
use crate::topo::Topology;

/// S7 statistics (DESIGN 6.9).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamReport {
    pub n_free: u32,
    pub rows: u32,
    pub redundant: u32,
    pub inconsistent: u32,
    pub penalty_rows: u32,
    pub loops_exact: u32,
    pub loops_dropped: u32,
    pub e0: f64,
    pub e_final: f64,
    pub flips_initial: u32,
    pub flips_after_stiff: u32,
    pub folds_after_repair: u32,
    pub n_pred: f64,
    pub scale: f64,
}

impl ParamReport {
    pub fn validate(&self) -> Result<(), AqError> {
        ensure(
            self.e0.is_finite() && self.e_final.is_finite() && self.n_pred.is_finite() && self.scale.is_finite(),
            || "ParamReport has non-finite values".into(),
        )
    }
}

/// Family code of a halfedge without a family.
pub const FAMILY_NONE: u8 = 255;

/// S7 output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeamlessParam {
    /// Per face corner, dyadic multiples of 2^-24, quad units.
    pub uv: Vec<[V2; 3]>,
    /// Per halfedge (identity off the cut and on the boundary).
    pub trans: Vec<Transition>,
    pub is_cut: Vec<bool>,
    /// (vertex, index4).
    pub cones: Vec<(u32, i8)>,
    /// Per halfedge on chains: 0 U, 1 V, 255 none.
    pub he_family: Vec<u8>,
    /// Indices of the LoopCand whose closure was eliminated exactly.
    pub loops_exact: Vec<u32>,
    /// Per face after repair.
    pub folded: Vec<bool>,
    pub report: ParamReport,
}

impl SeamlessParam {
    /// uv of the tail and head corners of halfedge h, in the chart of face(h).
    pub fn he_uv(&self, h: u32) -> (V2, V2) {
        let f = (h / 3) as usize;
        let k = (h % 3) as usize;
        (self.uv[f][k], self.uv[f][(k + 1) % 3])
    }

    /// Checks sizes; every uv dyadic (2^-24) with |uv| < 2^27; transitions identity off the cut
    /// and on the boundary; cut flags symmetric; `trans[twin] == trans[he].inverse()`; both
    /// endpoints of every interior halfedge satisfy the transition bitwise (f64 `==`); chain
    /// iso lines exact (U: equal v, V: equal u, per `he_family`); cones in range and nonzero.
    pub fn validate(&self, m: &TriMesh, t: &Topology) -> Result<(), AqError> {
        let nf = m.tris.len();
        let nh = 3 * nf;
        ensure(
            self.uv.len() == nf
                && self.trans.len() == nh
                && self.is_cut.len() == nh
                && self.he_family.len() == nh
                && self.folded.len() == nf,
            || "SeamlessParam sizes".into(),
        )?;
        for (f, c) in self.uv.iter().enumerate() {
            for a in c {
                for &x in a {
                    if !x.is_finite() || x.abs() >= tol::UV_BOUND {
                        return Err(AqError::numerical(format!("uv of face {f} out of bound: {x}")));
                    }
                    ensure(tol::is_dyadic(x), || format!("uv of face {f} not dyadic: {x}"))?;
                }
            }
        }
        for h in 0..nh {
            let tr = self.trans[h];
            tr.validate()?;
            let tw = t.twin[h];
            if tw == NONE || !self.is_cut[h] {
                ensure(tr.is_identity(), || format!("non-identity transition off the cut at {h}"))?;
            }
            if tw == NONE {
                ensure(!self.is_cut[h], || format!("boundary he {h} is cut"))?;
                continue;
            }
            let tw = tw as usize;
            ensure(self.is_cut[tw] == self.is_cut[h], || format!("cut flag asymmetric at {h}"))?;
            ensure(self.trans[tw] == tr.inverse(), || format!("trans[twin] != inverse at {h}"))?;
            // Endpoints: tail(h) = head(twin), head(h) = tail(twin).
            let (a_f, b_f) = self.he_uv(h as u32);
            let (a_g, b_g) = self.he_uv(tw as u32);
            ensure(tr.apply(a_f) == b_g && tr.apply(b_f) == a_g, || {
                format!("transition not exact across he {h}: {a_f:?},{b_f:?} -> {b_g:?},{a_g:?} with {tr:?}")
            })?;
        }
        for h in 0..nh {
            let fam = self.he_family[h];
            ensure(fam == 0 || fam == 1 || fam == FAMILY_NONE, || format!("he_family[{h}] = {fam}"))?;
            if fam == FAMILY_NONE {
                continue;
            }
            let (a, b) = self.he_uv(h as u32);
            // U edge: v constant; V edge: u constant.
            let c = if fam == 0 { 1 } else { 0 };
            ensure(a[c] == b[c], || format!("chain he {h} (family {fam}) not iso exact"))?;
        }
        for &(v, i) in &self.cones {
            ensure((v as usize) < m.pos.len() && i != 0, || format!("bad cone ({v}, {i})"))?;
        }
        self.report.validate()
    }
}

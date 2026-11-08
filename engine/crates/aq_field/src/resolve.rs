//! DESIGN 6.6 step 7: fixed-matching re-solve.
//!
//! With p fixed, the energy
//! `E = sum_e w_e r_e^2 + sum_f (a_c (theta_f - phihat_f)^2 + a_g (theta_f - psihat_f)^2)`,
//! `r_e = theta_g - theta_f - kappa_e - p_e pi/2` (lifted by the multiple of 2pi that makes
//! the current residual equal the current delta), is minimized over the free faces. Hard faces
//! sit at the branch of their constraint edge nearest their current theta, phihat and psihat
//! are the branches of the curvature and guide targets nearest the current theta. A component
//! with no hard face and no alignment pins its lowest face.
//!
//! The matrix depends only on the weights and on which faces are fixed, so one factorization
//! serves every re-solve of an S4 run (DESIGN 6.6 complexity: the pattern is reused).

use aq_core::alloc::try_vec;
use aq_core::cross::delta_edge;
use aq_core::{AqError, CrossField, Ctl, CurvField, NONE, SizeField, Stage, delta_from_p, index4_all};
use aq_linalg::{Chol, Ordering, Triplets};

use crate::setup::{Setup, snap};
use crate::{FieldParams, ProxyRef, check_field_sizes, check_inputs, inv};

pub(crate) struct Resolver {
    /// Per face: unknown index, or NONE for a fixed (hard or pinned) face.
    idx: Vec<u32>,
    n: usize,
    chol: Option<Chol>,
}

impl Resolver {
    pub fn new(px: &ProxyRef, st: &Setup, _f: &CrossField, ctl: Option<&Ctl>) -> Result<Self, AqError> {
        let t = px.topo;
        let nf = px.mesh.tris.len();
        let nc = t.n_comp as usize;
        let mut anchored = vec![false; nc];
        for f in 0..nf {
            if st.hard[f] || st.a_tot(f) > 0.0 {
                anchored[t.face_comp[f] as usize] = true;
            }
        }
        let mut pinned_comp = vec![false; nc];
        let mut idx = try_vec(nf, NONE)?;
        let mut n = 0usize;
        for f in 0..nf {
            let c = t.face_comp[f] as usize;
            if st.hard[f] {
                continue;
            }
            if !anchored[c] && !pinned_comp[c] {
                pinned_comp[c] = true; // lowest face of the component stays where it is
                continue;
            }
            idx[f] = n as u32;
            n += 1;
        }
        let mut rs = Resolver { idx, n, chol: None };
        if n > 0 {
            let mut tr = Triplets::with_capacity(n, 3 * n).map_err(|e| e.at(Stage::Field))?;
            for h in 0..3 * nf {
                let tw = t.twin[h];
                if tw == NONE || (tw as usize) < h {
                    continue;
                }
                let w = st.w[h];
                match (rs.idx[h / 3], rs.idx[tw as usize / 3]) {
                    (NONE, NONE) => {}
                    (fi, NONE) => tr.push(fi as usize, fi as usize, w),
                    (NONE, gi) => tr.push(gi as usize, gi as usize, w),
                    (fi, gi) => {
                        tr.push(fi as usize, fi as usize, w);
                        tr.push(gi as usize, gi as usize, w);
                        tr.push(gi as usize, fi as usize, -w);
                    }
                }
            }
            for f in 0..nf {
                if rs.idx[f] != NONE {
                    tr.push(rs.idx[f] as usize, rs.idx[f] as usize, st.a_tot(f));
                }
            }
            let a = tr.to_csc().map_err(|e| e.at(Stage::Field))?;
            let check = || ctl.map_or(Ok(()), |c| c.check());
            let sym = Chol::analyze_ctl(&a, Ordering::Auto, ctl).map_err(|e| e.at(Stage::Field))?;
            check()?;
            rs.chol = Some(Chol::factor(&sym, &a).map_err(|e| e.at(Stage::Field))?);
            check()?;
        }
        Ok(rs)
    }

    /// One re-solve: theta updated, delta recomputed, index4 asserted unchanged.
    pub fn solve(&mut self, px: &ProxyRef, st: &Setup, f: &mut CrossField) -> Result<(), AqError> {
        let t = px.topo;
        let fr = px.frames;
        let nf = px.mesh.tris.len();
        for fc in 0..nf {
            if st.hard[fc] {
                f.theta[fc] = snap(f.theta[fc], st.edge_angle[fc]);
            }
        }
        if let Some(ch) = &self.chol {
            let mut b = try_vec(self.n, 0.0f64)?;
            for h in 0..3 * nf {
                let tw = t.twin[h];
                if tw == NONE || (tw as usize) < h {
                    continue;
                }
                let (fa, ga) = (h / 3, tw as usize / 3);
                let w = st.w[h];
                let (tf, tg) = (f.theta[fa], f.theta[ga]);
                let r0 = delta_edge(tf, tg, fr.kappa[h], f.p[h]);
                let c = tg - tf - r0; // lifted kappa + p pi/2
                match (self.idx[fa], self.idx[ga]) {
                    (NONE, NONE) => {}
                    (fi, NONE) => b[fi as usize] += w * (tg - c),
                    (NONE, gi) => b[gi as usize] += w * (tf + c),
                    (fi, gi) => {
                        b[fi as usize] -= w * c;
                        b[gi as usize] += w * c;
                    }
                }
            }
            for fc in 0..nf {
                let i = self.idx[fc];
                if i == NONE {
                    continue;
                }
                let th = f.theta[fc];
                b[i as usize] += st.a_c[fc] * snap(th, st.phi[fc]) + st.a_g[fc] * snap(th, st.psi_g[fc]);
            }
            ch.try_solve(&mut b).map_err(|e| e.at(Stage::Field))?;
            if b.iter().any(|x| !x.is_finite()) {
                return Err(AqError::numerical("fixed-matching re-solve produced a non-finite theta").at(Stage::Field));
            }
            for fc in 0..nf {
                let i = self.idx[fc];
                if i != NONE {
                    f.theta[fc] = b[i as usize];
                }
            }
        }
        f.delta = delta_from_p(px.mesh, t, fr, &f.theta, &f.p);
        let idx = index4_all(px.mesh, t, fr, &f.delta).map_err(|e| e.at(Stage::Field))?;
        if idx != f.index4 {
            let v = idx.iter().zip(&f.index4).position(|(a, b)| a != b).unwrap_or(0);
            return Err(inv(format!(
                "fixed-matching re-solve changed index4 at vertex {v} ({} -> {})",
                f.index4.get(v).copied().unwrap_or(0),
                idx.get(v).copied().unwrap_or(0)
            )));
        }
        Ok(())
    }
}

/// The step 7 energy of the current field (delta recomputed from theta and p).
pub(crate) fn energy(px: &ProxyRef, st: &Setup, f: &CrossField) -> f64 {
    let t = px.topo;
    let nf = px.mesh.tris.len();
    let mut e = 0.0;
    for h in 0..3 * nf {
        let tw = t.twin[h];
        if tw == NONE || (tw as usize) < h {
            continue;
        }
        let d = delta_edge(f.theta[h / 3], f.theta[tw as usize / 3], px.frames.kappa[h], f.p[h]);
        e += st.w[h] * d * d;
    }
    for fc in 0..nf {
        if st.hard[fc] {
            continue;
        }
        let th = f.theta[fc];
        let dc = th - snap(th, st.phi[fc]);
        let dg = th - snap(th, st.psi_g[fc]);
        e += st.a_c[fc] * dc * dc + st.a_g[fc] * dg * dg;
    }
    e
}

/// The step 7 energy per face: its alignment terms plus half of the smoothness term of each of
/// its interior edges (the sum over faces is [`energy`]).
pub(crate) fn face_energies(px: &ProxyRef, st: &Setup, f: &CrossField) -> Vec<f64> {
    let t = px.topo;
    let nf = px.mesh.tris.len();
    let mut e = vec![0.0; nf];
    for h in 0..3 * nf {
        let tw = t.twin[h];
        if tw == NONE || (tw as usize) < h {
            continue;
        }
        let d = delta_edge(f.theta[h / 3], f.theta[tw as usize / 3], px.frames.kappa[h], f.p[h]);
        let x = 0.5 * st.w[h] * d * d;
        e[h / 3] += x;
        e[tw as usize / 3] += x;
    }
    for fc in 0..nf {
        if st.hard[fc] {
            continue;
        }
        let th = f.theta[fc];
        let dc = th - snap(th, st.phi[fc]);
        let dg = th - snap(th, st.psi_g[fc]);
        e[fc] += st.a_c[fc] * dc * dc + st.a_g[fc] * dg * dg;
    }
    e
}

/// The fixed-matching energy of DESIGN 6.6 step 7 for field `f` (hard faces excluded from the
/// alignment sum, delta recomputed from theta and p).
pub fn field_energy(
    px: &ProxyRef,
    c: &CurvField,
    s: &SizeField,
    f: &CrossField,
    p: &FieldParams,
) -> Result<f64, AqError> {
    check_inputs(px, c, s, p)?;
    check_field_sizes(px, f)?;
    let st = Setup::new(px, c, s, p, None)?;
    Ok(energy(px, &st, f))
}

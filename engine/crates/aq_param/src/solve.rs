//! Energy assembly and solve (DESIGN 6.9 "Energy", "Stiffening").
//!
//! `E = sum_f w_f A_f (|grad u - d'_f/h_f|^2 + |grad v - n x d'_f/h_f|^2)`. The corner values
//! are integer expressions over the free variables, so every face contributes
//! `w_f A_f G_f^T G_f` with `G_f` the 4 x n_loc map from its local free variables to
//! `(grad u, grad v)` in the face frame. The pattern is built once per [`System`]; a new set of
//! weights only rewrites the values, so stiffening refactors with the same symbolic analysis.

use aq_core::cross::{det2, jacobian};
use aq_core::{AqError, Ctl, ErrKind, Frames, TriMesh, V2};
use aq_linalg::{Chol, Csc, Ordering, Symbolic, Triplets};

use crate::system::System;

/// Barycentric gradients of the 3 corners of face f in its frame (C4, consistent with
/// `aq_core::cross::jacobian`: grad = sum_k value_k * g[k]).
pub(crate) fn bary_grads(fr: &Frames, m: &TriMesh, f: u32) -> [V2; 3] {
    let [_, q1, q2] = fr.face_coords(m, f);
    let dq = q1[0] * q2[1] - q1[1] * q2[0];
    let inv = 1.0 / dq;
    let g1 = [q2[1] * inv, -q2[0] * inv];
    let g2 = [-q1[1] * inv, q1[0] * inv];
    [[-(g1[0] + g2[0]), -(g1[1] + g2[1])], g1, g2]
}

/// Target `(a, b)` = `(d'/h, n x d'/h)` in the face frame, flattened as
/// `[a.x, a.y, b.x, b.y]`.
pub(crate) fn face_target(theta_combed: f64, h: f64) -> [f64; 4] {
    let (s, c) = theta_combed.sin_cos();
    let ih = 1.0 / h;
    [c * ih, s * ih, -s * ih, c * ih]
}

/// Per-face local operators and the matrix pattern of one system.
pub(crate) struct Assembly {
    pub n: usize,
    /// CSR over faces: local free variables (sorted, unique).
    face_off: Vec<u32>,
    face_vars: Vec<u32>,
    /// Per face, 4 x n_loc row-major (rows grad u x, grad u y, grad v x, grad v y), aligned with
    /// `face_vars` times 4.
    face_g: Vec<f64>,
    pub target: Vec<[f64; 4]>,
    pub area: Vec<f64>,
    /// Per face, CSC value index of each local pair (a <= b), row-major upper triangle.
    map_off: Vec<u32>,
    map: Vec<u32>,
    /// Penalty rows: (free indices, coefficients, value indices of the pairs a <= b).
    pen: Vec<(Vec<u32>, Vec<f64>, Vec<u32>)>,
    pub csc: Csc,
}

fn val_index(csc: &Csc, i: usize, j: usize) -> Result<u32, AqError> {
    let (r, c) = if i >= j { (i, j) } else { (j, i) };
    let cp = csc.col_ptr();
    let rows = &csc.row_idx()[cp[c]..cp[c + 1]];
    rows.binary_search(&r)
        .map(|k| (cp[c] + k) as u32)
        .map_err(|_| {
            AqError::internal(format!(
                "aq_param: entry ({r}, {c}) missing from the pattern"
            ))
        })
}

impl Assembly {
    pub fn new(
        sys: &System,
        m: &TriMesh,
        fr: &Frames,
        theta_combed: &[f64],
        h_face: &[f64],
        ctl: &Ctl,
    ) -> Result<Assembly, AqError> {
        let nf = m.tris.len();
        let n = sys.n_free;
        let mut face_off: Vec<u32> = aq_core::alloc::try_with_capacity(nf + 1)?;
        let mut face_vars: Vec<u32> = aq_core::alloc::try_with_capacity(6 * nf)?;
        let mut face_g: Vec<f64> = aq_core::alloc::try_with_capacity(24 * nf)?;
        let mut target = aq_core::alloc::try_with_capacity(nf)?;
        let mut area = aq_core::alloc::try_with_capacity(nf)?;
        face_off.push(0);
        let mut loc: Vec<u32> = Vec::new();
        let mut n_pairs = 0usize;
        for f in 0..nf {
            if f % 16384 == 0 {
                ctl.check()?;
            }
            loc.clear();
            for k in 0..3 {
                for q in 0..2 {
                    loc.extend_from_slice(sys.slot(2 * (3 * f + k) + q).0);
                }
            }
            loc.sort_unstable();
            loc.dedup();
            let nl = loc.len();
            let gb = bary_grads(fr, m, f as u32);
            let base = face_g.len();
            face_g.resize(base + 4 * nl, 0.0);
            for k in 0..3 {
                for q in 0..2 {
                    let (ix, cs) = sys.slot(2 * (3 * f + k) + q);
                    for (&i, &c) in ix.iter().zip(cs) {
                        let Ok(a) = loc.binary_search(&i) else {
                            return Err(AqError::internal("aq_param: local variable lookup"));
                        };
                        for d in 0..2 {
                            face_g[base + (2 * q + d) * nl + a] += c as f64 * gb[k][d];
                        }
                    }
                }
            }
            face_vars.extend_from_slice(&loc);
            face_off.push(face_vars.len() as u32);
            let h = h_face[f];
            if !(h.is_finite() && h > 0.0) {
                return Err(AqError::invalid_input(format!("h_face[{f}] = {h}")));
            }
            target.push(face_target(theta_combed[f], h));
            area.push(fr.area[f]);
            n_pairs += nl * (nl + 1) / 2;
        }
        let pen_pairs: usize = sys.pen.iter().map(|r| r.len() * (r.len() + 1) / 2).sum();
        let mut tr = Triplets::with_capacity(n, n_pairs + pen_pairs)?;
        for f in 0..nf {
            let lv = &face_vars[face_off[f] as usize..face_off[f + 1] as usize];
            for a in 0..lv.len() {
                for b in a..lv.len() {
                    tr.push(lv[b] as usize, lv[a] as usize, 0.0);
                }
            }
        }
        let mut pen = Vec::with_capacity(sys.pen.len());
        for r in &sys.pen {
            let mut idx: Vec<u32> = r.iter().map(|e| e.0).collect();
            let mut co: Vec<f64> = r.iter().map(|e| e.1 as f64).collect();
            // Sort by index (rows are sorted by variable id, which is the free order).
            let mut perm: Vec<usize> = (0..idx.len()).collect();
            perm.sort_by_key(|&k| idx[k]);
            idx = perm.iter().map(|&k| idx[k]).collect();
            co = perm.iter().map(|&k| co[k]).collect();
            for a in 0..idx.len() {
                for b in a..idx.len() {
                    tr.push(idx[b] as usize, idx[a] as usize, 0.0);
                }
            }
            pen.push((idx, co, Vec::new()));
        }
        let csc = tr.to_csc()?;
        drop(tr);
        let mut map_off: Vec<u32> = aq_core::alloc::try_with_capacity(nf + 1)?;
        let mut map: Vec<u32> = aq_core::alloc::try_with_capacity(n_pairs)?;
        map_off.push(0);
        for f in 0..nf {
            let lv = &face_vars[face_off[f] as usize..face_off[f + 1] as usize];
            for a in 0..lv.len() {
                for b in a..lv.len() {
                    map.push(val_index(&csc, lv[b] as usize, lv[a] as usize)?);
                }
            }
            map_off.push(map.len() as u32);
        }
        for p in pen.iter_mut() {
            let mut mm = Vec::new();
            for a in 0..p.0.len() {
                for b in a..p.0.len() {
                    mm.push(val_index(&csc, p.0[b] as usize, p.0[a] as usize)?);
                }
            }
            p.2 = mm;
        }
        Ok(Assembly {
            n,
            face_off,
            face_vars,
            face_g,
            target,
            area,
            map_off,
            map,
            pen,
            csc,
        })
    }

    #[inline]
    fn face(&self, f: usize) -> (&[u32], &[f64]) {
        let (a, b) = (self.face_off[f] as usize, self.face_off[f + 1] as usize);
        (&self.face_vars[a..b], &self.face_g[4 * a..4 * b])
    }

    /// Writes the matrix values for weights w (penalty weight `penalty_w * mean diag` of the
    /// energy part) and returns the right-hand side.
    pub fn assemble(&mut self, w: &[f64], penalty_w: f64) -> Vec<f64> {
        let mut rhs = vec![0.0; self.n];
        let nf = self.target.len();
        let mut vals = std::mem::take(&mut self.csc);
        vals.values_mut().fill(0.0);
        {
            let v = vals.values_mut();
            for f in 0..nf {
                let (lv, g) = self.face(f);
                let nl = lv.len();
                let s = w[f] * self.area[f];
                let mp = &self.map[self.map_off[f] as usize..self.map_off[f + 1] as usize];
                let t = &self.target[f];
                let mut k = 0;
                for a in 0..nl {
                    let mut ra = 0.0;
                    for r in 0..4 {
                        ra += g[r * nl + a] * t[r];
                    }
                    rhs[lv[a] as usize] += s * ra;
                    for b in a..nl {
                        let mut x = 0.0;
                        for r in 0..4 {
                            x += g[r * nl + a] * g[r * nl + b];
                        }
                        v[mp[k] as usize] += s * x;
                        k += 1;
                    }
                }
            }
        }
        let pw = penalty_w * vals.mean_diag();
        {
            let v = vals.values_mut();
            for (_, co, mp) in &self.pen {
                let mut k = 0;
                for a in 0..co.len() {
                    for b in a..co.len() {
                        v[mp[k] as usize] += pw * co[a] * co[b];
                        k += 1;
                    }
                }
            }
        }
        self.csc = vals;
        rhs
    }

    /// `sum_f w_f A_f |G_f y - T_f|^2` (w = None: all ones).
    pub fn energy(&self, y: &[f64], w: Option<&[f64]>) -> f64 {
        let mut e = 0.0;
        for f in 0..self.target.len() {
            let (lv, g) = self.face(f);
            let nl = lv.len();
            let mut s = 0.0;
            for r in 0..4 {
                let mut x = -self.target[f][r];
                for a in 0..nl {
                    x += g[r * nl + a] * y[lv[a] as usize];
                }
                s += x * x;
            }
            e += w.map_or(1.0, |w| w[f]) * self.area[f] * s;
        }
        e
    }
}

/// A factored system: the assembly, its symbolic analysis and the current factor.
pub(crate) struct Solver {
    pub asm: Assembly,
    sym: Option<Symbolic>,
    chol: Option<Chol>,
}

impl Solver {
    pub fn new(asm: Assembly) -> Self {
        Solver {
            asm,
            sym: None,
            chol: None,
        }
    }

    /// Numeric factorization of the current values (refactor in place after the first).
    fn factorize(&mut self) -> Result<(), AqError> {
        if let Some(c) = self.chol.as_mut() {
            return c.refactor(&self.asm.csc);
        }
        let Some(sym) = self.sym.as_ref() else {
            return Err(AqError::internal("aq_param: no symbolic analysis"));
        };
        self.chol = Some(Chol::factor(sym, &self.asm.csc)?);
        Ok(())
    }

    /// Assembles with weights w and solves. The first call analyzes; later calls refactor with
    /// the same symbolic. A non-SPD factorization is retried once with `1e-10 * mean diag`
    /// added to the diagonal (DESIGN 6.9 "Errors").
    pub fn solve(&mut self, w: &[f64], penalty_w: f64, ctl: &Ctl) -> Result<Vec<f64>, AqError> {
        ctl.check()?;
        let mut rhs = self.asm.assemble(w, penalty_w);
        if self.asm.n == 0 {
            return Ok(rhs);
        }
        if self.sym.is_none() {
            self.sym = Some(Chol::analyze_ctl(&self.asm.csc, Ordering::Auto, Some(ctl))?);
            ctl.check()?;
        }
        match self.factorize() {
            Ok(()) => {}
            Err(e) if e.kind == ErrKind::Numerical => {
                let md = self.asm.csc.mean_diag();
                self.asm.csc.add_diag(1e-10 * md.max(f64::MIN_POSITIVE));
                self.factorize()?;
            }
            Err(e) => return Err(e),
        }
        ctl.check()?;
        let Some(c) = self.chol.as_ref() else {
            return Err(AqError::internal("aq_param: no factor"));
        };
        c.try_solve(&mut rhs)?;
        if rhs.iter().any(|x| !x.is_finite()) {
            return Err(AqError::numerical("aq_param: solution is not finite"));
        }
        Ok(rhs)
    }
}

/// `N_pred = sum_f A_f det J_f` for per-corner uv.
pub(crate) fn n_pred(m: &TriMesh, fr: &Frames, uv: &[[V2; 3]]) -> f64 {
    (0..m.tris.len())
        .map(|f| fr.area[f] * det2(&jacobian(fr, m, f as u32, &uv[f])))
        .sum()
}

/// Singular values (largest first) of a 2x2 matrix.
pub(crate) fn singular_values(j: &[[f64; 2]; 2]) -> (f64, f64) {
    let e = 0.5 * (j[0][0] + j[1][1]);
    let f = 0.5 * (j[0][0] - j[1][1]);
    let g = 0.5 * (j[1][0] + j[0][1]);
    let h = 0.5 * (j[1][0] - j[0][1]);
    let q = e.hypot(h);
    let r = f.hypot(g);
    (q + r, (q - r).abs())
}

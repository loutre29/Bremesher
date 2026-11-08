//! Area-weighted Jacobian least squares with fixed vertices (DESIGN 6.12 step 2, and the
//! global step of step 3).
//!
//! Minimizes `sum_f A_f |J_f - T_f|^2` over the free vertices, where `J_f` is the Jacobian of
//! the map in the C4 frame of face f. The normal equations are the cotan stiffness matrix
//! `K_ij = sum_f A_f g_i . g_j` (assembled exactly from the hat gradients, so it is the true
//! energy) with the right-hand side `b_i = sum_f A_f g_i . T_f[c]` per coordinate c. The
//! matrix does not depend on the targets, so it is factored once (aq_linalg Cholesky) and
//! reused by every global step. Degenerate faces (`2A <= 1e-14 l_max^2`) are left out.

use aq_core::alloc::try_vec;
use aq_core::{AqError, NONE, TriMesh, V2};
use aq_linalg::{Chol, Triplets};

use crate::geom::{FaceGeo, M2, face_geo, input_err};

pub(crate) struct Stiffness {
    /// Vertex -> free unknown, or NONE (fixed or unused).
    pub idx: Vec<u32>,
    pub geo: Vec<Option<FaceGeo>>,
    chol: Option<Chol>,
    nfree: usize,
}

impl Stiffness {
    pub fn new(mesh: &TriMesh, fixed: &[bool]) -> Result<Self, AqError> {
        let nv = mesh.pos.len();
        if fixed.len() != nv {
            return Err(input_err("fixed flags do not match the mesh"));
        }
        let mut idx: Vec<u32> = try_vec(nv, NONE)?;
        let mut used = try_vec(nv, false)?;
        for t in &mesh.tris {
            for &v in t {
                used[v as usize] = true;
            }
        }
        let mut nfree = 0usize;
        for v in 0..nv {
            if used[v] && !fixed[v] {
                idx[v] = nfree as u32;
                nfree += 1;
            }
        }
        let geo: Vec<Option<FaceGeo>> = (0..mesh.tris.len() as u32).map(|f| face_geo(mesh.tri_pos(f))).collect();
        if nfree == 0 {
            return Ok(Stiffness { idx, geo, chol: None, nfree });
        }
        let mut tr = Triplets::with_capacity(nfree, 6 * mesh.tris.len())?;
        for (f, t) in mesh.tris.iter().enumerate() {
            let Some(fg) = &geo[f] else { continue };
            for a in 0..3 {
                let ia = idx[t[a] as usize];
                if ia == NONE {
                    continue;
                }
                for b in a..3 {
                    let ib = idx[t[b] as usize];
                    if ib == NONE {
                        continue;
                    }
                    let w = fg.area * (fg.g[a][0] * fg.g[b][0] + fg.g[a][1] * fg.g[b][1]);
                    tr.push(ia as usize, ib as usize, w);
                }
            }
        }
        let k = tr.to_csc()?;
        let chol = Chol::new(&k)?;
        Ok(Stiffness { idx, geo, chol: Some(chol), nfree })
    }

    /// Minimizer for the per-face targets; fixed vertices keep their value from `x`.
    pub fn solve(&self, mesh: &TriMesh, x: &[V2], targets: &[M2]) -> Result<Vec<V2>, AqError> {
        let mut out = x.to_vec();
        let Some(chol) = &self.chol else { return Ok(out) };
        if targets.len() != mesh.tris.len() || x.len() != mesh.pos.len() {
            return Err(input_err("least squares: size mismatch"));
        }
        let mut rhs = vec![vec![0.0; self.nfree], vec![0.0; self.nfree]];
        for (f, t) in mesh.tris.iter().enumerate() {
            let Some(fg) = &self.geo[f] else { continue };
            let tg = &targets[f];
            for a in 0..3 {
                let ia = self.idx[t[a] as usize];
                if ia == NONE {
                    continue;
                }
                let ga = fg.g[a];
                for c in 0..2 {
                    rhs[c][ia as usize] += fg.area * (ga[0] * tg[c][0] + ga[1] * tg[c][1]);
                }
                for (b, &tb) in t.iter().enumerate() {
                    let vb = tb as usize;
                    if self.idx[vb] != NONE {
                        continue;
                    }
                    let w = fg.area * (ga[0] * fg.g[b][0] + ga[1] * fg.g[b][1]);
                    for c in 0..2 {
                        rhs[c][ia as usize] -= w * x[vb][c];
                    }
                }
            }
        }
        chol.try_solve_multi(&mut rhs)?;
        for (v, &i) in self.idx.iter().enumerate() {
            if i != NONE {
                out[v] = [rhs[0][i as usize], rhs[1][i as usize]];
            }
        }
        Ok(out)
    }
}

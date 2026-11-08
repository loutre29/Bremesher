//! Cleanup (DESIGN 6.13): doublet removal and 3-3 diagonal collapse.

use aq_core::math;
use aq_core::{NONE, QuadMesh, V3};

use crate::adj::newell;
use crate::{VF_BOUNDARY, VF_CHAIN_MASK, VF_CONE, VF_CORNER};

/// Cleanup parameters. In quad units h = 1 (DESIGN C14).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CleanupParams {
    /// Target edge length.
    pub h: f64,
    /// A 3-3 diagonal pair is collapsed on quads of area below `small_area * h^2` (0.3).
    pub small_area: f64,
    /// Passes of the collapse sweep (each pass revisits the small quads).
    pub max_passes: u32,
}

impl Default for CleanupParams {
    fn default() -> Self {
        CleanupParams { h: 1.0, small_area: 0.3, max_passes: 8 }
    }
}

/// What [`cleanup`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CleanupReport {
    /// Interior valence-2 vertices removed (each merges two quads into one).
    pub doublets: u32,
    /// 3-3 diagonal pairs collapsed (each removes one quad and one vertex).
    pub diag_collapses: u32,
    pub faces_before: u32,
    pub faces_after: u32,
    pub verts_before: u32,
    pub verts_after: u32,
    /// Old vertex id to new vertex id (NONE for a removed vertex). Use it to remap any
    /// per-vertex array (for example `chain_of_vert`).
    pub vert_map: Vec<u32>,
    /// True when the mesh failed `QuadMesh::validate` and was left untouched.
    pub skipped: bool,
}

/// Cleanup with the default parameters (h = 1, area threshold 0.3 h^2).
pub fn cleanup(q: &mut QuadMesh) -> CleanupReport {
    cleanup_with(q, &CleanupParams::default())
}

/// Protected vertices are never removed or merged: chain, corner and boundary vertices.
const PROTECTED: u32 = VF_CHAIN_MASK | VF_CORNER | VF_BOUNDARY;

struct Work<'a> {
    pos: &'a mut Vec<V3>,
    vflag: &'a mut Vec<u32>,
    faces: Vec<Vec<u32>>,
    face_alive: Vec<bool>,
    vf: Vec<Vec<u32>>,
    vert_alive: Vec<bool>,
}

impl Work<'_> {
    fn idx(fv: &[u32], v: u32) -> Option<usize> {
        fv.iter().position(|&w| w == v)
    }

    /// True when every edge at v is shared by two of its faces.
    fn interior(&self, v: u32) -> bool {
        let fs = &self.vf[v as usize];
        if fs.is_empty() {
            return false;
        }
        for &f in fs {
            let fv = &self.faces[f as usize];
            let Some(k) = Self::idx(fv, v) else { return false };
            let b = fv[(k + 1) % fv.len()];
            let twin = fs.iter().any(|&g| {
                let gv = &self.faces[g as usize];
                Self::idx(gv, v).is_some_and(|j| gv[(j + gv.len() - 1) % gv.len()] == b)
            });
            if !twin {
                return false;
            }
        }
        true
    }

    /// Edge neighbours of v, sorted.
    fn nbrs(&self, v: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for &f in &self.vf[v as usize] {
            let fv = &self.faces[f as usize];
            if let Some(k) = Self::idx(fv, v) {
                out.push(fv[(k + 1) % fv.len()]);
                out.push(fv[(k + fv.len() - 1) % fv.len()]);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn removable(&self, v: u32) -> bool {
        self.vert_alive[v as usize] && self.vflag[v as usize] & PROTECTED == 0
    }

    fn remove_vf(&mut self, v: u32, f: u32) {
        self.vf[v as usize].retain(|&g| g != f);
    }

    /// Removes interior valence-2 vertex v by merging its two quads. Returns the vertices whose
    /// valence dropped.
    fn try_doublet(&mut self, v: u32) -> Option<[u32; 2]> {
        if !self.removable(v) || self.vf[v as usize].len() != 2 {
            return None;
        }
        let (f1, f2) = (self.vf[v as usize][0], self.vf[v as usize][1]);
        let (q1, q2) = (&self.faces[f1 as usize], &self.faces[f2 as usize]);
        if q1.len() != 4 || q2.len() != 4 {
            return None;
        }
        let k1 = Self::idx(q1, v)?;
        let k2 = Self::idx(q2, v)?;
        // q1 = [v, b, c, a], q2 must be [v, a, d, b].
        let (b, c, a) = (q1[(k1 + 1) % 4], q1[(k1 + 2) % 4], q1[(k1 + 3) % 4]);
        let (a2, d, b2) = (q2[(k2 + 1) % 4], q2[(k2 + 2) % 4], q2[(k2 + 3) % 4]);
        if a2 != a || b2 != b || c == d {
            return None;
        }
        self.faces[f1 as usize] = vec![b, c, a, d];
        self.face_alive[f2 as usize] = false;
        self.faces[f2 as usize].clear();
        self.remove_vf(a, f2);
        self.remove_vf(b, f2);
        for g in self.vf[d as usize].iter_mut() {
            if *g == f2 {
                *g = f1;
            }
        }
        self.vf[d as usize].sort_unstable();
        self.vf[d as usize].dedup();
        self.vf[v as usize].clear();
        self.vert_alive[v as usize] = false;
        Some([a, b])
    }

    fn remove_doublets(&mut self) -> u32 {
        let mut n = 0;
        let mut stack: Vec<u32> = (0..self.vf.len() as u32).rev().collect();
        while let Some(v) = stack.pop() {
            if self.vf[v as usize].len() == 2
                && self.interior(v)
                && let Some([a, b]) = self.try_doublet(v)
            {
                n += 1;
                stack.push(b);
                stack.push(a);
            }
        }
        n
    }

    fn quad_area(&self, f: u32) -> f64 {
        0.5 * math::norm(newell(self.pos, &self.faces[f as usize]))
    }

    /// Collapses the diagonal (a, c) of quad f = [a, b, c, d] (rotated so that a is at k).
    fn try_collapse(&mut self, f: u32, k: usize) -> bool {
        let fv = &self.faces[f as usize];
        if fv.len() != 4 {
            return false;
        }
        let (a, b, c, d) = (fv[k], fv[(k + 1) % 4], fv[(k + 2) % 4], fv[(k + 3) % 4]);
        if !self.removable(a) || !self.removable(c) {
            return false;
        }
        if self.vf[a as usize].len() != 3 || self.vf[c as usize].len() != 3 {
            return false;
        }
        if !self.interior(a) || !self.interior(c) {
            return false;
        }
        // b and d keep at least 3 faces (interior) or 1 face (boundary).
        for w in [b, d] {
            let nfw = self.vf[w as usize].len();
            let need = if self.interior(w) { 4 } else { 2 };
            if nfw < need {
                return false;
            }
        }
        // Link condition: a and c share only face f and only the neighbours b and d.
        let shared_faces =
            self.vf[a as usize].iter().filter(|g| self.vf[c as usize].contains(g)).count();
        if shared_faces != 1 {
            return false;
        }
        let na = self.nbrs(a);
        let nc = self.nbrs(c);
        let common: Vec<u32> = na.iter().copied().filter(|x| nc.binary_search(x).is_ok()).collect();
        let mut bd = [b, d];
        bd.sort_unstable();
        if common != bd {
            return false;
        }
        // Geometry: no affected face may flip.
        let others: Vec<u32> = self.vf[a as usize]
            .iter()
            .chain(self.vf[c as usize].iter())
            .copied()
            .filter(|&g| g != f)
            .collect();
        let old_n: Vec<V3> =
            others.iter().map(|&g| newell(self.pos, &self.faces[g as usize])).collect();
        let pa = self.pos[a as usize];
        let m = math::scale(math::add(pa, self.pos[c as usize]), 0.5);
        self.pos[a as usize] = m;
        let mut ok = true;
        for (i, &g) in others.iter().enumerate() {
            let gv: Vec<u32> =
                self.faces[g as usize].iter().map(|&w| if w == c { a } else { w }).collect();
            let nn = newell(self.pos, &gv);
            if math::dot(nn, old_n[i]) <= 0.0 {
                ok = false;
                break;
            }
            for j in 0..gv.len() {
                if math::dot(crate::adj::corner_normal(self.pos, &gv, j), old_n[i]) <= 0.0 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
        if !ok {
            self.pos[a as usize] = pa;
            return false;
        }
        // Apply.
        let cfaces = std::mem::take(&mut self.vf[c as usize]);
        for &g in &cfaces {
            if g == f {
                continue;
            }
            for w in self.faces[g as usize].iter_mut() {
                if *w == c {
                    *w = a;
                }
            }
            self.vf[a as usize].push(g);
        }
        self.remove_vf(a, f);
        self.remove_vf(b, f);
        self.remove_vf(d, f);
        self.vf[a as usize].sort_unstable();
        self.vf[a as usize].dedup();
        self.face_alive[f as usize] = false;
        self.faces[f as usize].clear();
        self.vert_alive[c as usize] = false;
        self.vflag[a as usize] &= !VF_CONE;
        true
    }

    fn collapse_pass(&mut self, thr: f64) -> u32 {
        let mut cand: Vec<(f64, u32)> = (0..self.faces.len() as u32)
            .filter(|&f| self.face_alive[f as usize] && self.faces[f as usize].len() == 4)
            .map(|f| (self.quad_area(f), f))
            .filter(|&(a, _)| a < thr)
            .collect();
        cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let mut n = 0;
        for &(_, f) in &cand {
            if !self.face_alive[f as usize] || self.faces[f as usize].len() != 4 {
                continue;
            }
            if self.quad_area(f) >= thr {
                continue;
            }
            for k in 0..2 {
                if self.try_collapse(f, k) {
                    n += 1;
                    break;
                }
            }
        }
        n
    }
}

/// Cleanup (DESIGN 6.13): removes interior valence-2 vertices (merging their two quads),
/// collapses 3-3 diagonal pairs (two opposite interior valence-3 vertices) on quads of area
/// below `small_area * h^2` into their midpoint, removes doublets again, and compacts the
/// mesh. Chain, corner and boundary vertices (vflag) are never removed. Infallible: an
/// invalid mesh is left untouched (`skipped`).
pub fn cleanup_with(q: &mut QuadMesh, p: &CleanupParams) -> CleanupReport {
    let nv = q.n_verts();
    let nf = q.n_faces();
    let mut rep = CleanupReport {
        faces_before: nf as u32,
        verts_before: nv as u32,
        faces_after: nf as u32,
        verts_after: nv as u32,
        vert_map: (0..nv as u32).collect(),
        ..Default::default()
    };
    if q.validate().is_err() {
        rep.skipped = true;
        return rep;
    }
    let faces: Vec<Vec<u32>> = (0..nf).map(|f| q.face(f).to_vec()).collect();
    let mut vf: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (f, fv) in faces.iter().enumerate() {
        for &v in fv {
            vf[v as usize].push(f as u32);
        }
    }
    let face_comp = std::mem::take(&mut q.face_comp);
    let mut w = Work {
        pos: &mut q.pos,
        vflag: &mut q.vflag,
        faces,
        face_alive: vec![true; nf],
        vf,
        vert_alive: vec![true; nv],
    };
    rep.doublets = w.remove_doublets();
    let thr = p.small_area * p.h * p.h;
    if thr > 0.0 && thr.is_finite() {
        for _ in 0..p.max_passes {
            let n = w.collapse_pass(thr);
            rep.diag_collapses += n;
            if n == 0 {
                break;
            }
            rep.doublets += w.remove_doublets();
        }
    }
    rep.doublets += w.remove_doublets();

    // Compact. Vertices used by no face stay if they were already unused (ids are kept
    // stable for them); removed vertices are dropped.
    let Work { faces, face_alive, vert_alive, .. } = w;
    let mut map = vec![NONE; nv];
    let mut next = 0u32;
    for v in 0..nv {
        if vert_alive[v] {
            map[v] = next;
            next += 1;
        }
    }
    let old_pos = std::mem::take(&mut q.pos);
    let old_flag = std::mem::take(&mut q.vflag);
    q.pos = (0..nv).filter(|&v| vert_alive[v]).map(|v| old_pos[v]).collect();
    q.vflag = (0..nv).filter(|&v| vert_alive[v]).map(|v| old_flag[v]).collect();
    q.face_off = vec![0];
    q.face_verts.clear();
    q.face_comp = Vec::with_capacity(nf);
    for f in 0..nf {
        if !face_alive[f] {
            continue;
        }
        let mapped: Vec<u32> = faces[f].iter().map(|&v| map[v as usize]).collect();
        q.push_face(&mapped, face_comp.get(f).copied().unwrap_or(0));
    }
    rep.faces_after = q.n_faces() as u32;
    rep.verts_after = q.n_verts() as u32;
    rep.vert_map = map;
    rep
}

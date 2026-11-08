//! Relaxation (DESIGN 6.13).

use rayon::prelude::*;

use aq_core::math;
use aq_core::{AqError, Ctl, NONE, Projector, QuadMesh, Stage, V3};

use crate::adj::{VertFaces, area_centroid, corner_normal};
use crate::chain::{OnPoly, Polyline};
use crate::{COS_MAX_TURN, PostParams, PostProxy, RelaxReport, VF_CONE, VF_CORNER};

/// A chain vertex farther than this from its polyline at start-up is kept fixed instead of
/// being snapped onto it (quad units).
pub const CHAIN_SNAP_MAX: f64 = 1e-3;

/// The arc-length window of a chain slide is this factor times the distance to the target.
const CHAIN_WINDOW: f64 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Fixed,
    Interior,
    Chain,
}

fn bad(msg: impl Into<String>) -> AqError {
    AqError::invalid_input(msg).at(Stage::Post)
}

struct Ctx<'a> {
    face_off: &'a [u32],
    face_verts: &'a [u32],
    vf: &'a VertFaces,
    role: &'a [Role],
    normals: &'a [V3],
}

impl Ctx<'_> {
    #[inline]
    fn face(&self, f: u32) -> &[u32] {
        let a = self.face_off[f as usize] as usize;
        let b = self.face_off[f as usize + 1] as usize;
        &self.face_verts[a..b]
    }

    /// Reference surface normal of face f: the mean proxy normal at the hints of its free
    /// (interior) corners, or of all its corners when none is free.
    fn ref_normal(&self, f: u32, hint: &[u32]) -> Option<V3> {
        let fv = self.face(f);
        let mut n = [0.0; 3];
        let mut cnt = 0;
        for &w in fv {
            if self.role[w as usize] == Role::Interior {
                n = math::add(n, self.normals[hint[w as usize] as usize]);
                cnt += 1;
            }
        }
        if cnt == 0 {
            for &w in fv {
                n = math::add(n, self.normals[hint[w as usize] as usize]);
            }
        }
        math::normalize(n)
    }

    /// Smallest cosine between a corner normal and the reference normal over the corners whose
    /// normal depends on v (the corner at v and its two neighbours in each incident face). A
    /// degenerate corner counts as -1.
    fn min_cos(&self, v: usize, pos: &[V3], hint: &[u32]) -> f64 {
        let mut m = f64::INFINITY;
        for &(f, k) in self.vf.of(v) {
            let fv = self.face(f);
            let n = fv.len();
            let nr = self.ref_normal(f, hint);
            let ks = [(k as usize + n - 1) % n, k as usize, (k as usize + 1) % n];
            for &j in &ks {
                let cn = corner_normal(pos, fv, j);
                let l = math::norm(cn);
                let c = match nr {
                    Some(nr) if l > 0.0 && l.is_finite() => math::dot(cn, nr) / l,
                    _ => -1.0,
                };
                m = m.min(c);
            }
        }
        m
    }

    /// Area-weighted mean of the incident face centroids.
    fn target(&self, v: usize, pos: &[V3]) -> Option<V3> {
        let mut c = [0.0; 3];
        let mut w = 0.0;
        for &(f, _) in self.vf.of(v) {
            let (a, ce) = area_centroid(pos, self.face(f));
            c = math::add(c, math::scale(ce, a));
            w += a;
        }
        if w > 0.0 && w.is_finite() { Some(math::scale(c, 1.0 / w)) } else { None }
    }
}

/// S11 relaxation (DESIGN 6.13). `chain_of_vert[v]` is the proxy chain id (index into
/// `px.feats.chains`) of quad vertex v, or NONE. The quad mesh must pass
/// `QuadMesh::validate`.
pub fn relax(
    q: &mut QuadMesh,
    px: &PostProxy,
    chain_of_vert: &[u32],
    p: &PostParams,
    ctl: &Ctl,
) -> Result<RelaxReport, AqError> {
    ctl.progress(Stage::Post, 0.0);
    q.validate().map_err(|e| e.at(Stage::Post))?;
    let nv = q.n_verts();
    let nf_px = px.mesh.tris.len();
    if chain_of_vert.len() != nv {
        return Err(bad(format!("chain_of_vert has {} entries for {nv} vertices", chain_of_vert.len())));
    }
    if nf_px == 0 {
        return Err(bad("empty proxy"));
    }
    if px.frames.n.len() != nf_px || px.bvh.n_faces() != nf_px || px.topo.twin.len() != 3 * nf_px {
        return Err(bad("proxy frames, topology or BVH do not match the proxy mesh"));
    }
    let vf = VertFaces::build(q).map_err(|e| e.at(Stage::Post))?;

    // Chain polylines, built for the chains in use.
    let chains = &px.feats.chains;
    let mut polys: Vec<Option<Polyline>> = (0..chains.len()).map(|_| None).collect();
    for &c in chain_of_vert {
        if c == NONE {
            continue;
        }
        let ci = c as usize;
        if ci >= chains.len() {
            return Err(bad(format!("chain id {c} out of range ({} chains)", chains.len())));
        }
        if polys[ci].is_none() {
            let pl = Polyline::new(px.mesh, px.topo, &chains[ci])
                .ok_or_else(|| bad(format!("chain {c} has no valid polyline")))?;
            polys[ci] = Some(pl);
        }
    }

    // Start-up: hint faces (BVH) and chain positions, in parallel (pure per vertex).
    let pos0 = q.pos.clone();
    let init: Vec<(u32, Option<(OnPoly, f64)>)> = (0..nv)
        .into_par_iter()
        .map(|v| {
            let c = chain_of_vert[v];
            if c != NONE
                && let Some(pl) = polys[c as usize].as_ref()
                && let Some((o, d)) = pl.closest(pos0[v])
            {
                return (pl.seg_face[o.seg as usize], Some((o, d)));
            }
            (px.bvh.closest(pos0[v]).0.face, None)
        })
        .collect();

    let mut rep = RelaxReport::default();
    let mut role = vec![Role::Fixed; nv];
    let mut cone = vec![false; nv];
    let mut hint = vec![0u32; nv];
    let mut on: Vec<Option<OnPoly>> = vec![None; nv];
    for v in 0..nv {
        let (h, ch) = init[v];
        if h == NONE || h as usize >= nf_px {
            return Err(AqError::internal(format!("no proxy face for vertex {v}")).at(Stage::Post));
        }
        hint[v] = h;
        let fl = q.vflag[v];
        let r = if vf.of(v).is_empty() || fl & VF_CORNER != 0 {
            Role::Fixed
        } else if chain_of_vert[v] != NONE {
            match ch {
                Some((o, d)) if d <= CHAIN_SNAP_MAX => {
                    if d > 0.0 {
                        rep.chain_snapped += 1;
                        rep.max_snap = rep.max_snap.max(d);
                    }
                    q.pos[v] = o.pos;
                    on[v] = Some(o);
                    Role::Chain
                }
                _ => {
                    rep.chain_off += 1;
                    Role::Fixed
                }
            }
        } else if vf.is_bnd[v] {
            Role::Fixed
        } else {
            Role::Interior
        };
        role[v] = r;
        if r != Role::Fixed && fl & VF_CONE != 0 {
            cone[v] = true;
            rep.cone_verts += 1;
        }
        match r {
            Role::Fixed => rep.fixed_verts += 1,
            Role::Interior => rep.interior_verts += 1,
            Role::Chain => rep.chain_verts += 1,
        }
    }

    let proj = Projector::new(px.mesh, px.topo, px.frames, px.bvh);
    let QuadMesh { pos, face_off, face_verts, .. } = q;
    let cx = Ctx { face_off, face_verts, vf: &vf, role: &role, normals: &px.frames.n };

    for it in 0..p.iters {
        let nodes_frozen = it < p.freeze_nodes_iters;
        for v in 0..nv {
            if v % 1024 == 0 {
                ctl.check().map_err(|e| e.at(Stage::Post))?;
            }
            let r = role[v];
            if r == Role::Fixed || (cone[v] && nodes_frozen) {
                continue;
            }
            let Some(c) = cx.target(v, pos) else { continue };
            let pv = pos[v];
            let (np, nh, no) = match r {
                Role::Interior => {
                    let n = px.frames.n[hint[v] as usize];
                    let d = math::sub(c, pv);
                    let dt = math::sub(d, math::scale(n, math::dot(d, n)));
                    let t = math::add(pv, dt);
                    match proj.project(t, hint[v], n) {
                        Some((sp, qp)) => (qp, sp.face, None),
                        None => {
                            rep.projection_failed += 1;
                            continue;
                        }
                    }
                }
                Role::Chain => {
                    let (Some(cur), Some(pl)) =
                        (on[v], polys[chain_of_vert[v] as usize].as_ref())
                    else {
                        continue;
                    };
                    let rad = CHAIN_WINDOW * math::dist(c, pv);
                    match pl.closest_local(c, &cur, rad) {
                        Some(o) => (o.pos, pl.seg_face[o.seg as usize], Some(o)),
                        None => continue,
                    }
                }
                Role::Fixed => continue,
            };
            if np == pv || !math::is_finite3(np) {
                continue;
            }
            let before = cx.min_cos(v, pos, &hint);
            let oh = hint[v];
            pos[v] = np;
            hint[v] = nh;
            let after = cx.min_cos(v, pos, &hint);
            if after >= COS_MAX_TURN || after > before {
                rep.moved += 1;
                rep.max_step = rep.max_step.max(math::dist(pv, np));
                if no.is_some() {
                    on[v] = no;
                }
            } else {
                pos[v] = pv;
                hint[v] = oh;
                rep.rejected_normal += 1;
            }
        }
        rep.iters = it + 1;
        ctl.progress(Stage::Post, (it + 1) as f32 / p.iters.max(1) as f32);
    }

    let mut sum = 0.0;
    let mut cnt = 0u64;
    for v in 0..nv {
        let d = math::dist(pos0[v], pos[v]);
        rep.max_disp = rep.max_disp.max(d);
        if role[v] != Role::Fixed {
            sum += d;
            cnt += 1;
        }
    }
    rep.mean_disp = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    Ok(rep)
}

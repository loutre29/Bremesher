//! Feature isolation (DESIGN 6.4): after it every face has at most 1 chain edge and no
//! non-chain edge joins two chain vertices.

use aq_core::math;
use aq_core::{AqError, ErrKind, Frames, NONE, Projector, Stage, Topology};

use crate::Proxy;
use crate::chains::{CornerRule, build_feature_set, count_bridge_edges, count_multi_chain_faces};
use crate::dynmesh::{DynMesh, NewVert, min_angle, try_copy};

const MAX_PASSES: usize = 4;
/// Relaxation sweeps over the inserted vertices.
const RELAX_SWEEPS: usize = 4;

/// [`isolate_features_with`] without re-projection: new vertices stay at the edge midpoint
/// (or face centroid) on the proxy surface, and their source tracking is copied from the
/// first endpoint (`Proxy` carries no source mesh to project onto).
pub fn isolate_features(px: &mut Proxy) -> Result<u32, AqError> {
    isolate_features_with(px, None)
}

/// Feature isolation, repeated until stable (at most 4 passes):
/// * every non-chain edge whose two endpoints lie on chains (bridge edge) is split at its
///   midpoint, then re-projected with `proj` when given (hint = the first endpoint's source
///   face);
/// * every face with 3 chain edges gets its centroid inserted (a face with exactly 2 chain
///   edges has a bridge edge, which the first rule splits).
///
/// The inserted vertices are then relaxed (up to 4 sweeps): each moves toward its 1-ring
/// centroid (line search 1, 1/2, 1/4, 1/8, re-projected) when that raises the smallest angle
/// of its faces without flipping one.
///
/// Chains, chain kinds and corners are kept (chain edges are never split). Returns the number
/// of inserted vertices. `ErrKind::Invariant` when the postcondition still fails after 4
/// passes; the proxy is then left unchanged.
pub fn isolate_features_with(px: &mut Proxy, proj: Option<&Projector>) -> Result<u32, AqError> {
    let m = &px.mesh;
    let t = &px.topo;
    let nh = 3 * m.tris.len();
    let nv0 = m.pos.len();
    if px.feats.he_kind.len() != nh
        || px.feats.he_chain.len() != nh
        || px.feats.vert_corner.len() != nv0
        || px.topo.twin.len() != nh
        || px.src_face.len() != nv0
        || px.src_bary.len() != nv0
        || px.density.len() != nv0
    {
        return Err(AqError::new(Stage::Proxy, ErrKind::InvalidInput, "isolate_features: inconsistent proxy"));
    }
    if count_bridge_edges(m, t, &px.feats) == 0 && count_multi_chain_faces(m, &px.feats) == 0 {
        return Ok(0);
    }
    let pe = |e: AqError| e.at(Stage::Proxy);
    let mut dm = DynMesh::from_mesh(m, t, try_copy(&px.feats.he_kind).map_err(pe)?, try_copy(&px.feats.he_chain).map_err(pe)?)
        .map_err(pe)?;
    dm.vsrc_f = try_copy(&px.src_face).map_err(pe)?;
    dm.vsrc_b = try_copy(&px.src_bary).map_err(pe)?;
    let mut density = try_copy(&px.density).map_err(pe)?;
    let mut fan: Vec<u32> = Vec::new();
    let mut inserted = 0u32;
    let mut new_verts: Vec<u32> = Vec::new();
    for _pass in 0..MAX_PASSES {
        let nv = dm.pos.len();
        let deg: Vec<u32> = (0..nv as u32).map(|v| if dm.vdead[v as usize] { 0 } else { dm.chain_degree(v, &mut fan) }).collect();
        let mut bridges: Vec<(u32, u32)> = Vec::new();
        let mut triple: Vec<u32> = Vec::new();
        for f in 0..dm.tris.len() as u32 {
            if !dm.alive(f) {
                continue;
            }
            let mut nchain = 0;
            for k in 0..3 {
                let h = 3 * f + k;
                if dm.is_chain(h) {
                    nchain += 1;
                    continue;
                }
                let tw = dm.twin[h as usize];
                if tw != NONE && h > tw {
                    continue;
                }
                let (a, b) = (dm.tail(h), dm.head(h));
                if deg[a as usize] > 0 && deg[b as usize] > 0 {
                    bridges.push((a, b));
                }
            }
            if nchain == 3 {
                triple.push(f);
            }
        }
        if bridges.is_empty() && triple.is_empty() {
            break;
        }
        for &(a, b) in &bridges {
            let h = dm.find_he(a, b);
            let h = if h != NONE { h } else { dm.find_he(b, a) };
            if h == NONE {
                continue;
            }
            let pm = math::scale(math::add(dm.pos[a as usize], dm.pos[b as usize]), 0.5);
            let mut n = dm.face_cross(h / 3);
            let tw = dm.twin[h as usize];
            if tw != NONE {
                n = math::add(n, dm.face_cross(tw / 3));
            }
            let (pos, sf, sb) = place(proj, pm, dm.vsrc_f[a as usize], dm.vsrc_b[a as usize], n);
            let nv = NewVert { pos, chain: (NONE, 0.0), src_f: sf, src_b: sb, lv: 1.0 };
            let d = 0.5 * (density[a as usize] + density[b as usize]);
            let r = dm.split_edge(a, b, nv);
            density.resize(dm.pos.len(), d);
            if let Ok(v) = r {
                inserted += 1;
                new_verts.push(v);
            }
        }
        for &f in &triple {
            if !dm.alive(f) {
                continue;
            }
            let tr = dm.tris[f as usize];
            if (0..3).any(|k| !dm.is_chain(3 * f + k)) {
                continue;
            }
            let pm = math::scale(
                math::add(math::add(dm.pos[tr[0] as usize], dm.pos[tr[1] as usize]), dm.pos[tr[2] as usize]),
                1.0 / 3.0,
            );
            let n = dm.face_cross(f);
            let (pos, sf, sb) = place(proj, pm, dm.vsrc_f[tr[0] as usize], dm.vsrc_b[tr[0] as usize], n);
            let nv = NewVert { pos, chain: (NONE, 0.0), src_f: sf, src_b: sb, lv: 1.0 };
            let d = (density[tr[0] as usize] + density[tr[1] as usize] + density[tr[2] as usize]) / 3.0;
            let r = dm.insert_in_face(f, nv);
            density.resize(dm.pos.len(), d);
            if let Ok(v) = r {
                inserted += 1;
                new_verts.push(v);
            }
        }
    }
    relax(&mut dm, &new_verts, proj, &mut fan);
    let vmap = dm.compact().map_err(pe)?;
    let mut dens2 = aq_core::alloc::try_vec(dm.pos.len(), 1.0).map_err(pe)?;
    for (v, &nvid) in vmap.iter().enumerate() {
        if nvid != NONE {
            dens2[nvid as usize] = density.get(v).copied().unwrap_or(1.0);
        }
    }
    let mesh = dm.to_trimesh().map_err(pe)?;
    let topo = Topology::build(&mesh).map_err(|e| e.at(Stage::Proxy))?;
    let frames = Frames::new(&mesh, &topo).map_err(|e| e.at(Stage::Proxy))?;
    // Keep the existing corners (old vertex ids map through vmap; new vertices are not corners).
    let mut corner = vec![false; mesh.pos.len()];
    for (v, &nvid) in vmap.iter().enumerate() {
        if nvid != NONE && px.feats.vert_corner.get(v).copied().unwrap_or(false) {
            corner[nvid as usize] = true;
        }
    }
    let zero = |_: u32| 0.0;
    let rule = CornerRule { corner_deg: 50.0, sym_corner_deg: 70.0, window: &zero, fixed: Some(&corner) };
    let feats = build_feature_set(&mesh, &topo, dm.kind.clone(), &rule)?;
    if count_bridge_edges(&mesh, &topo, &feats) != 0 || count_multi_chain_faces(&mesh, &feats) != 0 {
        return Err(AqError::new(Stage::Proxy, ErrKind::Invariant, "feature isolation did not converge in 4 passes"));
    }
    px.mesh = mesh;
    px.topo = topo;
    px.frames = frames;
    px.feats = feats;
    px.src_face = dm.vsrc_f;
    px.src_bary = dm.vsrc_b;
    px.density = dens2;
    Ok(inserted)
}

/// Moves each inserted vertex toward the centroid of its 1-ring (re-projected with `proj`,
/// else kept in the tangent plane) when that raises the smallest angle of its faces without
/// flipping any of them. A midpoint inserted between two chains that run closer than the
/// edge length otherwise leaves needle faces on the side of the shorter neighbor.
fn relax(dm: &mut DynMesh, verts: &[u32], proj: Option<&Projector>, fan: &mut Vec<u32>) {
    let mut nb: Vec<u32> = Vec::new();
    let mut fan2: Vec<u32> = Vec::new();
    for _ in 0..RELAX_SWEEPS {
        let mut moved = 0usize;
        for &v in verts {
            if dm.vdead[v as usize] {
                continue;
            }
            let bnd = dm.neighbors(v, &mut fan2, &mut nb);
            if bnd || nb.is_empty() {
                continue;
            }
            dm.fan(v, fan);
            let p = dm.pos[v as usize];
            let mut c = [0.0; 3];
            for &u in nb.iter() {
                c = math::add(c, dm.pos[u as usize]);
            }
            let c = math::scale(c, 1.0 / nb.len() as f64);
            let mut n = [0.0; 3];
            for &h in fan.iter() {
                n = math::add(n, dm.face_cross(h / 3));
            }
            let nn = math::norm(n);
            let n = if nn > 0.0 && nn.is_finite() { math::scale(n, 1.0 / nn) } else { [0.0; 3] };
            // Line search toward the centroid: the centroid itself can lie beyond a nearby
            // chain, where the faces would fold.
            let mut old_min = f64::INFINITY;
            for &h in fan.iter() {
                let tr = dm.tris[(h / 3) as usize];
                let po = tr.map(|x| dm.pos[x as usize]);
                old_min = old_min.min(min_angle(po[0], po[1], po[2]));
            }
            for t in [1.0, 0.5, 0.25, 0.125] {
                let target = math::add(p, math::scale(math::sub(c, p), t));
                let (q, sf, sb) = match proj.and_then(|pr| pr.project(target, dm.vsrc_f[v as usize], n)) {
                    Some((sp, q)) => (q, sp.face, sp.bary),
                    None => {
                        let d = math::sub(target, p);
                        (math::sub(target, math::scale(n, math::dot(d, n))), dm.vsrc_f[v as usize], dm.vsrc_b[v as usize])
                    }
                };
                let mut new_min = f64::INFINITY;
                let mut ok = true;
                for &h in fan.iter() {
                    let tr = dm.tris[(h / 3) as usize];
                    let po = tr.map(|x| dm.pos[x as usize]);
                    let pn = tr.map(|x| if x == v { q } else { dm.pos[x as usize] });
                    let no = math::cross(math::sub(po[1], po[0]), math::sub(po[2], po[0]));
                    let nw = math::cross(math::sub(pn[1], pn[0]), math::sub(pn[2], pn[0]));
                    if math::dot(no, nw) <= 0.0 {
                        ok = false;
                        break;
                    }
                    new_min = new_min.min(min_angle(pn[0], pn[1], pn[2]));
                }
                if ok && new_min > old_min {
                    dm.pos[v as usize] = q;
                    if proj.is_some() {
                        dm.vsrc_f[v as usize] = sf;
                        dm.vsrc_b[v as usize] = sb;
                    }
                    moved += 1;
                    break;
                }
            }
        }
        if moved == 0 {
            break;
        }
    }
}

fn place(proj: Option<&Projector>, p: aq_core::V3, hint: u32, hint_b: aq_core::V3, n: aq_core::V3) -> (aq_core::V3, u32, aq_core::V3) {
    if let Some(pr) = proj
        && let Some((sp, q)) = pr.project(p, hint, n)
    {
        return (q, sp.face, sp.bary);
    }
    (p, hint, hint_b)
}

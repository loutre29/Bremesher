//! Steps 2 to 4 of DESIGN 6.14: greedy pairing, topological split, relaxation.

use std::cmp::Ordering;

use rayon::prelude::*;

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::math;
use aq_core::{AqError, Ctl, ErrKind, FeatureKind, NONE, Projector, QuadMesh, Stage, V3};

use crate::{
    PAIR_MAX_DIHEDRAL_DEG, PAIR_MAX_QUALITY_DEG, RELAX_LAMBDA, SfTri, project_onto, VFLAG_BOUNDARY, VFLAG_CHAIN_MASK,
    VFLAG_CORNER, VFLAG_KIND_SHIFT, VFLAG_SF, ferr,
};

#[inline]
fn next(h: u32) -> u32 {
    3 * (h / 3) + (h % 3 + 1) % 3
}

#[inline]
fn prev(h: u32) -> u32 {
    3 * (h / 3) + (h % 3 + 2) % 3
}

fn be(e: AqError) -> AqError {
    e.at(Stage::Fallback)
}

/// Interior angles of a triangle at its 3 corners (radians).
fn corner_angles(p: [V3; 3]) -> [f64; 3] {
    let ang = |a: V3, b: V3, c: V3| math::angle_between(math::sub(b, a), math::sub(c, a));
    [ang(p[0], p[1], p[2]), ang(p[1], p[2], p[0]), ang(p[2], p[0], p[1])]
}

/// Result of the greedy pairing.
pub(crate) struct Pairing {
    /// Per face: the face it is paired with, or NONE.
    pub mate: Vec<u32>,
    /// Per face: its halfedge on the shared diagonal, or NONE.
    pub diag: Vec<u32>,
}

/// DESIGN 6.14 step 2. Candidates: interior non-chain edges whose two faces have a dihedral
/// below [`PAIR_MAX_DIHEDRAL_DEG`] and whose quad has a quality (max |corner - 90 deg|) below
/// [`PAIR_MAX_QUALITY_DEG`]. Sorted by (quality, halfedge id), each face is paired at most once.
pub(crate) fn pair(t: &SfTri, ctl: &Ctl) -> Result<Pairing, AqError> {
    let nf = t.tris.len();
    let tp = |f: usize| -> [V3; 3] {
        let [a, b, c] = t.tris[f];
        [t.pos[a as usize], t.pos[b as usize], t.pos[c as usize]]
    };
    let geo: Vec<(Option<V3>, [f64; 3])> = (0..nf)
        .into_par_iter()
        .map(|f| {
            let p = tp(f);
            let n = math::normalize(math::cross(math::sub(p[1], p[0]), math::sub(p[2], p[0])));
            (n, corner_angles(p))
        })
        .collect();
    let cos_max = PAIR_MAX_DIHEDRAL_DEG.to_radians().cos();
    let q_max = PAIR_MAX_QUALITY_DEG.to_radians();
    let right = std::f64::consts::FRAC_PI_2;
    let mut cand: Vec<(f64, u32)> = try_with_capacity(3 * nf / 2 + 1).map_err(be)?;
    for h in 0..3 * nf as u32 {
        let tw = t.twin[h as usize];
        if tw == NONE || tw < h || t.hchain[h as usize] != NONE {
            continue;
        }
        let (f, g) = ((h / 3) as usize, (tw / 3) as usize);
        let (Some(nf_), Some(ng)) = (geo[f].0, geo[g].0) else { continue };
        if !(math::dot(nf_, ng) >= cos_max) {
            continue;
        }
        let (kf, kg) = ((h % 3) as usize, (tw % 3) as usize);
        let (af, ag) = (geo[f].1, geo[g].1);
        // Quad (b, c, a, d) with h = a -> b in f and tw = b -> a in g.
        let corners = [af[(kf + 1) % 3] + ag[kg], af[(kf + 2) % 3], af[kf] + ag[(kg + 1) % 3], ag[(kg + 2) % 3]];
        let q = corners.iter().map(|&x| (x - right).abs()).fold(0.0, f64::max);
        if q < q_max {
            cand.push((q, h));
        }
    }
    ctl.check()?;
    cand.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
    let mut mate = try_vec(nf, NONE).map_err(be)?;
    let mut diag = try_vec(nf, NONE).map_err(be)?;
    for &(_, h) in &cand {
        let tw = t.twin[h as usize];
        let (f, g) = ((h / 3) as usize, (tw / 3) as usize);
        if mate[f] == NONE && mate[g] == NONE && f != g {
            mate[f] = g as u32;
            mate[g] = f as u32;
            diag[f] = h;
            diag[g] = tw;
        }
    }
    Ok(Pairing { mate, diag })
}

/// Quad count the split will give: 4 per pair, 3 per unpaired triangle.
pub(crate) fn predicted_count(t: &SfTri, ctl: &Ctl) -> Result<f64, AqError> {
    let pr = pair(t, ctl)?;
    let paired = pr.mate.iter().filter(|&&m| m != NONE).count();
    Ok((2 * paired + 3 * (pr.mate.len() - paired)) as f64)
}

/// Per-vertex attributes of the quad mesh.
struct QVert {
    pos: Vec<V3>,
    hint: Vec<u32>,
    fixed: Vec<bool>,
    chain: Vec<u32>,
    kind: Vec<FeatureKind>,
    corner: Vec<bool>,
    boundary: Vec<bool>,
}

fn is_feat(t: &SfTri, h: u32) -> bool {
    t.twin[h as usize] == NONE || t.hchain[h as usize] != NONE
}

fn kind_of(t: &SfTri, h: u32) -> FeatureKind {
    let c = t.hchain[h as usize];
    if c != NONE {
        t.chain_kind.get(c as usize).copied().unwrap_or(FeatureKind::Hard)
    } else if t.twin[h as usize] == NONE {
        FeatureKind::Boundary
    } else {
        FeatureKind::None
    }
}

/// Badness of quad `q`: its folded corners (corners that do not turn positively about the
/// quad normal `(p2 - p0) x (p3 - p1)`), plus one if the quad normal is not positive against
/// the parent normal, plus one if it is not positive against `n_src` (the source normal at
/// the quad's center vertex). 6 for a degenerate quad.
fn badness(pos: &[V3], q: &[u32; 4], n_parent: V3, n_src: V3) -> u32 {
    let p = [pos[q[0] as usize], pos[q[1] as usize], pos[q[2] as usize], pos[q[3] as usize]];
    let nq = math::cross(math::sub(p[2], p[0]), math::sub(p[3], p[1]));
    let lq = math::norm(nq);
    if !(lq > 0.0) {
        return 6;
    }
    let mut bad = 0;
    for i in 0..4 {
        let e1 = math::sub(p[(i + 1) % 4], p[i]);
        let e0 = math::sub(p[(i + 3) % 4], p[i]);
        let c = math::dot(math::cross(e1, e0), nq);
        if !(c > 1e-12 * math::norm(e1) * math::norm(e0) * lq) {
            bad += 1;
        }
    }
    if !(math::dot(nq, n_parent) > 0.0) {
        bad += 1;
    }
    if !(math::dot(nq, n_src) > 0.0) {
        bad += 1;
    }
    bad
}

/// Steps 2 to 4 of DESIGN 6.14 on a remeshed triangle mesh, then the output checks
/// (QuadMesh::validate and the Euler characteristic `chi`).
pub(crate) fn quadrangulate(t: &SfTri, proj: &Projector, relax_iters: u32, chi: i64, ctl: &Ctl) -> Result<QuadMesh, AqError> {
    let nf = t.tris.len();
    let nv = t.pos.len();
    let nh = 3 * nf;
    if nf == 0
        || t.twin.len() != nh
        || t.hchain.len() != nh
        || t.vhint.len() != nv
        || t.vcorner.len() != nv
        || t.tris.iter().any(|tr| tr.iter().any(|&v| v as usize >= nv))
    {
        return Err(ferr(ErrKind::Internal, "SF: inconsistent remeshed triangle mesh"));
    }
    let pr = pair(t, ctl)?;

    // Original vertices.
    let mut qv = QVert {
        pos: try_with_capacity(nv + nh).map_err(be)?,
        hint: try_with_capacity(nv + nh).map_err(be)?,
        fixed: try_with_capacity(nv + nh).map_err(be)?,
        chain: try_with_capacity(nv + nh).map_err(be)?,
        kind: try_with_capacity(nv + nh).map_err(be)?,
        corner: try_with_capacity(nv + nh).map_err(be)?,
        boundary: try_with_capacity(nv + nh).map_err(be)?,
    };
    qv.pos.extend_from_slice(&t.pos);
    qv.hint.extend_from_slice(&t.vhint);
    qv.fixed.resize(nv, false);
    qv.chain.resize(nv, NONE);
    qv.kind.resize(nv, FeatureKind::None);
    qv.corner.resize(nv, false);
    qv.boundary.resize(nv, false);
    for h in 0..nh as u32 {
        if !is_feat(t, h) {
            continue;
        }
        let f = (h / 3) as usize;
        let k = (h % 3) as usize;
        let ends = [t.tris[f][k], t.tris[f][(k + 1) % 3]];
        let c = t.hchain[h as usize];
        let kd = kind_of(t, h);
        for v in ends {
            let v = v as usize;
            qv.fixed[v] = true;
            if t.twin[h as usize] == NONE {
                qv.boundary[v] = true;
            }
            if c != NONE && (qv.chain[v] == NONE || c < qv.chain[v]) {
                qv.chain[v] = c;
                qv.kind[v] = kd;
            } else if qv.chain[v] == NONE && qv.kind[v] == FeatureKind::None {
                qv.kind[v] = kd;
            }
        }
    }
    for v in 0..nv {
        qv.corner[v] = t.vcorner[v] && qv.fixed[v];
    }

    // Face normals (unit, zero when degenerate).
    let fnorm: Vec<V3> = (0..nf)
        .into_par_iter()
        .map(|f| {
            let [a, b, c] = t.tris[f];
            let (pa, pb, pc) = (t.pos[a as usize], t.pos[b as usize], t.pos[c as usize]);
            math::normalize(math::cross(math::sub(pb, pa), math::sub(pc, pa))).unwrap_or([0.0; 3])
        })
        .collect();

    // Edge midpoints: one per undirected edge that is not a paired diagonal.
    // seeds: (position, hint, reference normal) of each new vertex, projected below.
    let mut seeds: Vec<(V3, u32, V3)> = try_with_capacity(nh).map_err(be)?;
    let mut emid = try_vec(nh, NONE).map_err(be)?;
    for h in 0..nh as u32 {
        let tw = t.twin[h as usize];
        if tw != NONE && tw < h {
            continue;
        }
        let f = (h / 3) as usize;
        if pr.diag[f] == h {
            continue;
        }
        let k = (h % 3) as usize;
        let (a, b) = (t.tris[f][k], t.tris[f][(k + 1) % 3]);
        let id = (nv + seeds.len()) as u32;
        emid[h as usize] = id;
        let mut n = fnorm[f];
        if tw != NONE {
            emid[tw as usize] = id;
            n = math::add(n, fnorm[(tw / 3) as usize]);
        }
        let p = math::scale(math::add(t.pos[a as usize], t.pos[b as usize]), 0.5);
        seeds.push((p, t.vhint[a as usize], n));
        let feat = is_feat(t, h);
        qv.fixed.push(feat);
        qv.chain.push(t.hchain[h as usize]);
        qv.kind.push(kind_of(t, h));
        qv.corner.push(false);
        qv.boundary.push(tw == NONE);
    }

    // Faces and centers. ref_n: per quad, the normal of its parent triangle or pair.
    let n_pairs = pr.mate.iter().filter(|&&m| m != NONE).count() / 2;
    let n_quads = 4 * n_pairs + 3 * (nf - 2 * n_pairs);
    let mut quads: Vec<[u32; 4]> = try_with_capacity(n_quads).map_err(be)?;
    let mut ref_n: Vec<V3> = try_with_capacity(n_quads).map_err(be)?;
    let push_center = |p: V3, hint: u32, n: V3, seeds: &mut Vec<(V3, u32, V3)>, qv: &mut QVert| -> u32 {
        let id = (nv + seeds.len()) as u32;
        seeds.push((p, hint, n));
        qv.fixed.push(false);
        qv.chain.push(NONE);
        qv.kind.push(FeatureKind::None);
        qv.corner.push(false);
        qv.boundary.push(false);
        id
    };
    for f in 0..nf {
        let m = pr.mate[f];
        if m == NONE {
            let [v0, v1, v2] = t.tris[f];
            let p = math::scale(
                math::add(math::add(t.pos[v0 as usize], t.pos[v1 as usize]), t.pos[v2 as usize]),
                1.0 / 3.0,
            );
            let ctr = push_center(p, t.vhint[v0 as usize], fnorm[f], &mut seeds, &mut qv);
            let vs = [v0, v1, v2];
            for k in 0..3 {
                let e_out = emid[3 * f + k];
                let e_in = emid[3 * f + (k + 2) % 3];
                quads.push([vs[k], e_out, ctr, e_in]);
                ref_n.push(fnorm[f]);
            }
        } else if (f as u32) < m {
            let h = pr.diag[f];
            let tw = t.twin[h as usize];
            let ends = |x: u32| -> u32 { t.tris[(x / 3) as usize][(x % 3) as usize] };
            let e = [next(h), prev(h), next(tw), prev(tw)];
            let q = [ends(e[0]), ends(e[1]), ends(e[2]), ends(e[3])]; // b, c, a, d
            let mut p = [0.0; 3];
            for &v in &q {
                p = math::add(p, t.pos[v as usize]);
            }
            let p = math::scale(p, 0.25);
            let n = math::normalize(math::add(fnorm[f], fnorm[m as usize])).unwrap_or(fnorm[f]);
            let ctr = push_center(p, t.vhint[q[0] as usize], n, &mut seeds, &mut qv);
            for i in 0..4 {
                quads.push([q[i], emid[e[i] as usize], ctr, emid[e[(i + 3) % 4] as usize]]);
                ref_n.push(n);
            }
        }
    }
    if quads.iter().any(|q| q.contains(&NONE)) {
        return Err(ferr(ErrKind::Internal, "SF: split produced an unassigned midpoint"));
    }
    ctl.check()?;

    // Project the new vertices.
    let projected: Vec<(V3, u32)> = seeds
        .par_iter()
        .map(|&(p, hint, n)| match project_onto(proj, p, hint, n) {
            Some((sp, q)) => (q, sp.face),
            None => (p, hint),
        })
        .collect();
    for (p, hint) in projected {
        qv.pos.push(p);
        qv.hint.push(hint);
    }
    let nqv = qv.pos.len();

    relax(&mut qv, &quads, &ref_n, proj, relax_iters, ctl)?;

    // Output.
    let mut q = QuadMesh::new();
    q.pos = qv.pos;
    q.face_off = try_with_capacity(quads.len() + 1).map_err(be)?;
    q.face_off.push(0);
    q.face_verts = try_with_capacity(4 * quads.len()).map_err(be)?;
    for qd in &quads {
        q.face_verts.extend_from_slice(qd);
        q.face_off.push(q.face_verts.len() as u32);
    }
    q.face_comp = try_vec(quads.len(), 0u32).map_err(be)?;
    q.vflag = (0..nqv)
        .map(|v| {
            let mut fl = VFLAG_SF;
            let c = qv.chain[v];
            if c != NONE {
                fl |= (c.saturating_add(1)).min(VFLAG_CHAIN_MASK);
            }
            fl |= ((qv.kind[v] as u32) & 0xF) << VFLAG_KIND_SHIFT;
            if qv.corner[v] {
                fl |= VFLAG_CORNER;
            }
            if qv.boundary[v] {
                fl |= VFLAG_BOUNDARY;
            }
            fl
        })
        .collect();
    q.validate().map_err(|e| ferr(ErrKind::Topology, format!("SF output: {}", e.msg)))?;
    // Euler characteristic.
    let mut edges: Vec<(u32, u32)> = try_with_capacity(4 * quads.len()).map_err(be)?;
    for qd in &quads {
        for i in 0..4 {
            let (a, b) = (qd[i], qd[(i + 1) % 4]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mut used = try_vec(nqv, false).map_err(be)?;
    for &v in &q.face_verts {
        used[v as usize] = true;
    }
    let nv_used = used.iter().filter(|&&u| u).count() as i64;
    let euler = nv_used - edges.len() as i64 + quads.len() as i64;
    if euler != chi {
        return Err(ferr(ErrKind::Topology, format!("SF output Euler characteristic {euler} != {chi}")));
    }
    Ok(q)
}

/// DESIGN 6.14 step 4: `iters` Jacobi iterations; each moves every free vertex by
/// [`RELAX_LAMBDA`] toward the area-weighted centroid of its quads, in the tangent plane of
/// the source, then projects it. A quad whose badness (folded corners, normal against its
/// parent or the source) would grow has its moved vertices reverted, until no quad gets worse.
fn relax(qv: &mut QVert, quads: &[[u32; 4]], ref_n: &[V3], proj: &Projector, iters: u32, ctl: &Ctl) -> Result<(), AqError> {
    let nv = qv.pos.len();
    // Vertex -> quads (CSR).
    let mut off = try_vec(nv + 1, 0u32).map_err(be)?;
    for q in quads {
        for &v in q {
            off[v as usize + 1] += 1;
        }
    }
    for v in 0..nv {
        off[v + 1] += off[v];
    }
    let mut fill = off.clone();
    let mut adj = try_vec(4 * quads.len(), 0u32).map_err(be)?;
    for (qi, q) in quads.iter().enumerate() {
        for &v in q {
            adj[fill[v as usize] as usize] = qi as u32;
            fill[v as usize] += 1;
        }
    }
    let nrm = proj.frames.n.as_slice();
    for it in 0..iters {
        ctl.check()?;
        let old = qv.pos.clone();
        let (cen, area): (Vec<V3>, Vec<f64>) = quads
            .par_iter()
            .map(|q| {
                let p = [old[q[0] as usize], old[q[1] as usize], old[q[2] as usize], old[q[3] as usize]];
                let c = math::scale(math::add(math::add(p[0], p[1]), math::add(p[2], p[3])), 0.25);
                let a = 0.5 * math::norm(math::cross(math::sub(p[2], p[0]), math::sub(p[3], p[1])));
                (c, a)
            })
            .unzip();
        let moved: Vec<Option<(V3, u32)>> = (0..nv)
            .into_par_iter()
            .map(|v| {
                if qv.fixed[v] {
                    return None;
                }
                let (s, e) = (off[v] as usize, off[v + 1] as usize);
                let mut acc = [0.0; 3];
                let mut w = 0.0;
                for &qi in &adj[s..e] {
                    let a = area[qi as usize];
                    acc = math::add(acc, math::scale(cen[qi as usize], a));
                    w += a;
                }
                if !(w > 0.0) {
                    return None;
                }
                let target = math::scale(acc, 1.0 / w);
                let p = old[v];
                let hint = qv.hint[v];
                let n = nrm.get(hint as usize).copied().unwrap_or([0.0; 3]);
                let mut d = math::sub(target, p);
                d = math::sub(d, math::scale(n, math::dot(d, n)));
                let cand = math::add(p, math::scale(d, RELAX_LAMBDA));
                proj.project(cand, hint, n).map(|(sp, q)| (q, sp.face))
            })
            .collect();
        let mut newp = old.clone();
        let mut newh = qv.hint.clone();
        let mut is_moved = vec![false; nv];
        for (v, m) in moved.iter().enumerate() {
            if let Some((p, h)) = *m {
                newp[v] = p;
                newh[v] = h;
                is_moved[v] = true;
            }
        }
        let src_n = |hint: &[u32], q: &[u32; 4]| nrm.get(hint[q[2] as usize] as usize).copied().unwrap_or([0.0; 3]);
        let old_bad: Vec<u32> = quads
            .par_iter()
            .zip(ref_n.par_iter())
            .map(|(q, &n)| badness(&old, q, n, src_n(&qv.hint, q)))
            .collect();
        // Guard: revert until no quad has more bad corners than before.
        let mut dirty: Vec<u32> = (0..quads.len() as u32).collect();
        let mut stamp = vec![u32::MAX; quads.len()];
        let mut round = 0u32;
        while !dirty.is_empty() {
            let mut revert: Vec<u32> = Vec::new();
            for &qi in &dirty {
                let q = &quads[qi as usize];
                if badness(&newp, q, ref_n[qi as usize], src_n(&newh, q)) > old_bad[qi as usize] {
                    for &v in q {
                        if is_moved[v as usize] {
                            revert.push(v);
                        }
                    }
                }
            }
            revert.sort_unstable();
            revert.dedup();
            let mut next_dirty = Vec::new();
            for &v in &revert {
                let v = v as usize;
                is_moved[v] = false;
                newp[v] = old[v];
                newh[v] = qv.hint[v];
                for &qi in &adj[off[v] as usize..off[v + 1] as usize] {
                    if stamp[qi as usize] != round {
                        stamp[qi as usize] = round;
                        next_dirty.push(qi);
                    }
                }
            }
            next_dirty.sort_unstable();
            dirty = next_dirty;
            round += 1;
        }
        qv.pos = newp;
        qv.hint = newh;
        ctl.progress(Stage::Fallback, 0.6 + 0.4 * (it + 1) as f32 / iters.max(1) as f32);
    }
    Ok(())
}

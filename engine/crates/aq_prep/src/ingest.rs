//! S0 ingest (DESIGN 6.4 "Ingest"): validate, collapse degenerate faces, split non-manifold
//! edges and vertices, orient, map features.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use aq_core::math;
use aq_core::tol::ZERO_AREA_REL;
use aq_core::{AqError, Ctl, ErrKind, FeatureKind, NONE, Stage, Topology, TriMesh, V3};

use crate::chains::{CornerRule, build_feature_set};
use crate::dynmesh::try_copy;
use crate::{IngestReport, Ingested};

fn bad(msg: impl Into<String>) -> AqError {
    AqError::new(Stage::Ingest, ErrKind::InvalidInput, msg)
}

/// `vec![x; n]` failing with `ErrKind::Budget` (DESIGN 3.4).
fn tvec<T: Clone>(n: usize, x: T) -> Result<Vec<T>, AqError> {
    aq_core::alloc::try_vec(n, x).map_err(|e| e.at(Stage::Ingest))
}

/// `Vec::with_capacity(n)` failing with `ErrKind::Budget` (DESIGN 3.4).
fn tcap<T>(n: usize) -> Result<Vec<T>, AqError> {
    aq_core::alloc::try_with_capacity(n).map_err(|e| e.at(Stage::Ingest))
}

/// Cancel and timeout check of the long loops, every `CHECK_EVERY` items.
const CHECK_EVERY: usize = 1 << 16;

fn tri_cross(pos: &[V3], t: [u32; 3]) -> V3 {
    let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
    math::cross(math::sub(b, a), math::sub(c, a))
}

fn tri_area(pos: &[V3], t: [u32; 3]) -> f64 {
    0.5 * math::norm(tri_cross(pos, t))
}

fn has_repeat(t: [u32; 3]) -> bool {
    t[0] == t[1] || t[1] == t[2] || t[0] == t[2]
}

/// Union-find with path halving.
fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        let g = p[p[x as usize] as usize];
        p[x as usize] = g;
        x = g;
    }
    x
}

/// Kind of an edge marked twice.
fn combine(old: FeatureKind, new: FeatureKind) -> FeatureKind {
    use FeatureKind::*;
    match (old, new) {
        (None, k) | (k, None) => k,
        (Hard, Guide) | (Guide, Hard) => GuideHard,
        (a, b) => a.max(b),
    }
}

/// S0 ingest. See the crate documentation and DESIGN 6.4.
///
/// * `InvalidInput` for: a NaN or Inf position, an index out of range, no triangle, a feature
///   pair with an out-of-range vertex, `feat_kind.len() != feat_pairs.len()`, a kind above 5.
/// * Kind 0 (None) pairs are ignored. Kind 4 (Boundary) on an interior edge is read as Hard.
/// * Vertices used by no face are removed.
pub fn ingest(mesh: TriMesh, feat_pairs: &[[u32; 2]], feat_kind: &[u8], ctl: &Ctl) -> Result<Ingested, AqError> {
    ctl.progress(Stage::Ingest, 0.0);
    let TriMesh { pos, tris } = mesh;
    let nv = pos.len();
    let nf = tris.len();
    if nf == 0 {
        return Err(bad("the mesh has no triangle"));
    }
    if nv >= NONE as usize / 2 || nf.saturating_mul(3) >= NONE as usize / 2 {
        return Err(bad("mesh too large for u32 ids"));
    }
    for (i, p) in pos.iter().enumerate() {
        if !math::is_finite3(*p) {
            return Err(bad(format!("vertex {i} is not finite")));
        }
    }
    for (f, t) in tris.iter().enumerate() {
        for &v in t {
            if v as usize >= nv {
                return Err(bad(format!("face {f} index {v} out of range ({nv} vertices)")));
            }
        }
    }
    if feat_kind.len() != feat_pairs.len() {
        return Err(bad(format!("{} feature pairs but {} kinds", feat_pairs.len(), feat_kind.len())));
    }
    let mut feats: Vec<(u32, u32, FeatureKind)> = Vec::with_capacity(feat_pairs.len());
    for (i, (pr, &k)) in feat_pairs.iter().zip(feat_kind).enumerate() {
        if pr[0] as usize >= nv || pr[1] as usize >= nv {
            return Err(bad(format!("feature pair {i} ({}, {}) out of range", pr[0], pr[1])));
        }
        let kind = FeatureKind::from_u8(k).ok_or_else(|| bad(format!("feature pair {i} has kind {k}")))?;
        if kind != FeatureKind::None {
            feats.push((pr[0], pr[1], kind));
        }
    }
    let mut rep = IngestReport::default();
    let mut tris = tris;
    let mut alive = tvec(nf, true)?;

    // Step 2: faces with a repeated index are already collapsed: remove them.
    for f in 0..nf {
        if has_repeat(tris[f]) {
            alive[f] = false;
            rep.collapsed_faces += 1;
        }
    }
    let mut vrep: Vec<u32> = tcap(nv)?;
    vrep.extend(0..nv as u32);
    let diag = bbox_diag(&pos);
    let thr = ZERO_AREA_REL * diag * diag;
    collapse_degenerate(&pos, &mut tris, &mut alive, &mut vrep, thr, &mut rep, ctl)?;
    drop_duplicates(&tris, &mut alive, &mut rep)?;
    ctl.check()?;
    ctl.progress(Stage::Ingest, 0.3);

    // Compact faces.
    let mut face_src: Vec<u32> = tcap(nf)?;
    let mut ftris: Vec<[u32; 3]> = tcap(nf)?;
    for f in 0..nf {
        if alive[f] {
            face_src.push(f as u32);
            ftris.push(tris[f]);
        }
    }
    drop(tris);
    if ftris.is_empty() {
        return Err(bad("no valid triangle left after removing degenerate faces"));
    }
    let nf = ftris.len();
    let mut area: Vec<f64> = tcap(nf)?;
    area.extend(ftris.iter().map(|&t| tri_area(&pos, t)));
    let mut normal: Vec<V3> = tcap(nf)?;
    normal.extend(ftris.iter().map(|&t| math::normalize(tri_cross(&pos, t)).unwrap_or([0.0; 3])));

    // Step 3: glue face-edge occurrences (2 per manifold edge, the best pair otherwise).
    let mut glue = glue_occurrences(&ftris, &normal, &mut rep)?;
    ctl.check()?;

    // Step 5 (orientation) before vertex splitting: BFS over glued edges.
    let flip = orient(&pos, &ftris, &area, &mut glue, &mut rep, ctl)?;
    ctl.check()?;
    let mut ftris2 = try_copy(&ftris).map_err(|e| e.at(Stage::Ingest))?;
    for f in 0..nf {
        if flip[f] {
            let t = ftris[f];
            ftris2[f] = [t[0], t[2], t[1]];
            rep.flipped_faces += 1;
        }
    }
    // Remap occurrence ids for flipped faces: old k -> new (2 - k).
    let occ_map = |o: u32| -> u32 {
        let f = o / 3;
        let k = o % 3;
        if flip[f as usize] { 3 * f + (2 - k) } else { o }
    };
    let mut glue2 = tvec(3 * nf, NONE)?;
    for o in 0..(3 * nf) as u32 {
        let p = glue[o as usize];
        if p != NONE {
            glue2[occ_map(o) as usize] = occ_map(p);
        }
    }
    drop(glue);
    let mut tris = ftris2;
    let mut pos = pos;
    // vert_work[v]: working (input) id of each vertex.
    let mut vert_work: Vec<u32> = tcap(nv)?;
    vert_work.extend(0..nv as u32);
    ctl.check()?;
    ctl.progress(Stage::Ingest, 0.5);

    // Step 4 (and 3): one fan per vertex; then detach faces that still share an edge badly.
    for round in 0..4 {
        split_vertices(&mut pos, &mut tris, &glue2, &mut vert_work, &mut rep, ctl)?;
        ctl.check()?;
        let bad_faces = bad_edge_faces(&tris)?;
        ctl.check()?;
        if bad_faces.is_empty() {
            break;
        }
        if round == 3 {
            return Err(AqError::new(Stage::Ingest, ErrKind::Topology, "could not make the mesh manifold"));
        }
        for &f in &bad_faces {
            for k in 0..3 {
                let o = 3 * f + k;
                let p = glue2[o as usize];
                if p != NONE {
                    glue2[p as usize] = NONE;
                    glue2[o as usize] = NONE;
                }
            }
            rep.split_edges += 1;
        }
    }
    ctl.check()?;

    // Remove unreferenced vertices.
    let mut used = tvec(pos.len(), false)?;
    for t in &tris {
        for &v in t {
            used[v as usize] = true;
        }
    }
    let mut vmap = tvec(pos.len(), NONE)?;
    let mut out_pos = tcap(pos.len())?;
    let mut vert_src = tcap(pos.len())?;
    for v in 0..pos.len() {
        if used[v] {
            vmap[v] = out_pos.len() as u32;
            out_pos.push(pos[v]);
            vert_src.push(vert_work[v]);
        }
    }
    for t in tris.iter_mut() {
        for v in t.iter_mut() {
            *v = vmap[*v as usize];
        }
    }
    let out = TriMesh { pos: out_pos, tris };
    ctl.check()?;
    let topo = Topology::build(&out).map_err(|e| e.at(Stage::Ingest))?;
    rep.components = topo.n_comp;
    ctl.check()?;
    ctl.progress(Stage::Ingest, 0.8);

    // Step 6: features.
    let mut he_kind = tvec(3 * out.tris.len(), FeatureKind::None)?;
    map_features(&out, &topo, &vert_src, &mut vrep, &feats, &mut he_kind, &mut rep, ctl)?;
    ctl.check()?;
    for h in 0..he_kind.len() {
        if topo.twin[h] == NONE {
            if he_kind[h] != FeatureKind::Symmetry {
                he_kind[h] = FeatureKind::Boundary;
            }
        } else if he_kind[h] == FeatureKind::Boundary {
            he_kind[h] = FeatureKind::Hard;
        }
    }
    let zero = |_: u32| 0.0;
    let rule = CornerRule { corner_deg: 50.0, sym_corner_deg: 70.0, window: &zero, fixed: None };
    let fs = build_feature_set(&out, &topo, he_kind, &rule).map_err(|e| e.at(Stage::Ingest))?;
    ctl.check()?;
    ctl.progress(Stage::Ingest, 1.0);
    Ok(Ingested { mesh: out, topo, feats: fs, vert_src, face_src, report: rep })
}

fn bbox_diag(pos: &[V3]) -> f64 {
    let Some(&first) = pos.first() else { return 0.0 };
    let mut lo = first;
    let mut hi = first;
    for p in pos {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    math::dist(lo, hi)
}

/// Step 2: collapse zero-area faces topologically (link condition), else delete them once no
/// collapse makes progress.
fn collapse_degenerate(
    pos: &[V3],
    tris: &mut [[u32; 3]],
    alive: &mut [bool],
    vrep: &mut [u32],
    thr: f64,
    rep: &mut IngestReport,
    ctl: &Ctl,
) -> Result<(), AqError> {
    let nf = tris.len();
    let is_deg = |t: [u32; 3]| -> bool { has_repeat(t) || !(tri_area(pos, t) >= thr) };
    if !(0..nf).any(|f| alive[f] && is_deg(tris[f])) {
        return Ok(());
    }
    let mut vfaces: Vec<Vec<u32>> = tvec(pos.len(), Vec::new())?;
    for f in 0..nf {
        if alive[f] {
            for &v in &tris[f] {
                vfaces[v as usize].push(f as u32);
            }
        }
    }
    let mut na: Vec<u32> = Vec::new();
    let mut nb: Vec<u32> = Vec::new();
    let mut opp: Vec<u32> = Vec::new();
    for _pass in 0..32 {
        ctl.check()?;
        let cand: Vec<u32> = (0..nf as u32).filter(|&f| alive[f as usize] && is_deg(tris[f as usize])).collect();
        if cand.is_empty() {
            return Ok(());
        }
        let mut progress = false;
        for (ci, &f) in cand.iter().enumerate() {
            if ci % 4096 == 4095 {
                ctl.check()?;
            }
            let f = f as usize;
            if !alive[f] {
                continue;
            }
            let t = tris[f];
            if has_repeat(t) {
                alive[f] = false;
                rep.collapsed_faces += 1;
                progress = true;
                continue;
            }
            if !is_deg(t) {
                continue;
            }
            let mut edges: Vec<(f64, usize)> =
                (0..3).map(|k| (math::dist2(pos[t[k] as usize], pos[t[(k + 1) % 3] as usize]), k)).collect();
            edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for &(_, k) in &edges {
                let (x, y) = (t[k], t[(k + 1) % 3]);
                // Link condition: N(x) and N(y) share exactly the opposite vertices of the
                // faces containing both.
                let collect = |v: u32, out: &mut Vec<u32>, tris: &[[u32; 3]], alive: &[bool]| {
                    out.clear();
                    for &g in &vfaces[v as usize] {
                        if alive[g as usize] {
                            for &w in &tris[g as usize] {
                                if w != v {
                                    out.push(w);
                                }
                            }
                        }
                    }
                    out.sort_unstable();
                    out.dedup();
                };
                collect(x, &mut na, tris, alive);
                collect(y, &mut nb, tris, alive);
                opp.clear();
                for &g in &vfaces[x as usize] {
                    let tg = tris[g as usize];
                    if alive[g as usize] && tg.contains(&y) {
                        for &w in &tg {
                            if w != x && w != y {
                                opp.push(w);
                            }
                        }
                    }
                }
                opp.sort_unstable();
                opp.dedup();
                let mut common: Vec<u32> = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < na.len() && j < nb.len() {
                    match na[i].cmp(&nb[j]) {
                        Ordering::Less => i += 1,
                        Ordering::Greater => j += 1,
                        Ordering::Equal => {
                            common.push(na[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                if common != opp {
                    continue;
                }
                // Merge the higher id into the lower one (its position is kept).
                let (keep, rem) = if x < y { (x, y) } else { (y, x) };
                let faces = std::mem::take(&mut vfaces[rem as usize]);
                for g in faces {
                    let gi = g as usize;
                    if !alive[gi] {
                        continue;
                    }
                    if tris[gi].contains(&keep) {
                        alive[gi] = false;
                        rep.collapsed_faces += 1;
                    } else {
                        for w in tris[gi].iter_mut() {
                            if *w == rem {
                                *w = keep;
                            }
                        }
                        vfaces[keep as usize].push(g);
                    }
                }
                vrep[rem as usize] = keep;
                progress = true;
                break;
            }
        }
        if !progress {
            break;
        }
    }
    for f in 0..nf {
        if alive[f] && is_deg(tris[f]) {
            alive[f] = false;
            rep.dropped_faces += 1;
        }
    }
    Ok(())
}

/// Deletes faces with the same vertex set as an earlier face.
fn drop_duplicates(tris: &[[u32; 3]], alive: &mut [bool], rep: &mut IngestReport) -> Result<(), AqError> {
    let mut keys: Vec<([u32; 3], u32)> = tcap(tris.len())?;
    keys.extend((0..tris.len()).filter(|&f| alive[f]).map(|f| {
        let mut k = tris[f];
        k.sort_unstable();
        (k, f as u32)
    }));
    keys.sort_unstable();
    for w in keys.windows(2) {
        if w[0].0 == w[1].0 && alive[w[1].1 as usize] {
            alive[w[1].1 as usize] = false;
            rep.dropped_faces += 1;
        }
    }
    Ok(())
}

/// Glues the face-edge occurrences: 2 per edge, the most consistent pair on a non-manifold
/// edge. Returns the partner occurrence (3f + k) of every occurrence or NONE.
fn glue_occurrences(tris: &[[u32; 3]], normal: &[V3], rep: &mut IngestReport) -> Result<Vec<u32>, AqError> {
    let nf = tris.len();
    let mut occ: Vec<(u64, u32)> = tcap(3 * nf)?;
    for f in 0..nf {
        for k in 0..3 {
            let (u, v) = (tris[f][k], tris[f][(k + 1) % 3]);
            let key = ((u.min(v) as u64) << 32) | u.max(v) as u64;
            occ.push((key, (3 * f + k) as u32));
        }
    }
    occ.sort_unstable();
    let mut glue = tvec(3 * nf, NONE)?;
    let dir = |o: u32| -> (u32, u32) {
        let f = (o / 3) as usize;
        let k = (o % 3) as usize;
        (tris[f][k], tris[f][(k + 1) % 3])
    };
    let mut i = 0;
    while i < occ.len() {
        let mut j = i;
        while j < occ.len() && occ[j].0 == occ[i].0 {
            j += 1;
        }
        let g = &occ[i..j];
        if g.len() == 2 {
            glue[g[0].1 as usize] = g[1].1;
            glue[g[1].1 as usize] = g[0].1;
        } else if g.len() >= 3 {
            rep.split_edges += (g.len() - 2) as u32;
            let mut best = (f64::NEG_INFINITY, 0usize, 1usize);
            for a in 0..g.len() {
                for b in a + 1..g.len() {
                    let (oa, ob) = (g[a].1, g[b].1);
                    let same = dir(oa) == dir(ob);
                    let d = math::dot(normal[(oa / 3) as usize], normal[(ob / 3) as usize]);
                    let s = if same { -d } else { d };
                    if s > best.0 {
                        best = (s, a, b);
                    }
                }
            }
            let (oa, ob) = (g[best.1].1, g[best.2].1);
            glue[oa as usize] = ob;
            glue[ob as usize] = oa;
        }
        i = j;
    }
    Ok(glue)
}

/// Orientation BFS (DESIGN 6.4 step 5): from the largest face of each component, flip the
/// faces whose glued edge runs the same way twice. Edges that cannot be made consistent
/// (non-orientable) are unglued. Closed components with a negative signed volume are flipped
/// as a whole. Returns the flip flag per face.
fn orient(
    pos: &[V3],
    tris: &[[u32; 3]],
    area: &[f64],
    glue: &mut [u32],
    rep: &mut IngestReport,
    ctl: &Ctl,
) -> Result<Vec<bool>, AqError> {
    let nf = tris.len();
    let mut order: Vec<u32> = tcap(nf)?;
    order.extend(0..nf as u32);
    order.sort_by(|&a, &b| area[b as usize].total_cmp(&area[a as usize]).then(a.cmp(&b)));
    ctl.check()?;
    let mut comp = tvec(nf, NONE)?;
    let mut flip = tvec(nf, false)?;
    let mut ncomp = 0u32;
    let mut queue: Vec<u32> = Vec::new();
    let mut popped = 0usize;
    for &root in &order {
        if comp[root as usize] != NONE {
            continue;
        }
        comp[root as usize] = ncomp;
        queue.clear();
        queue.push(root);
        let mut qi = 0;
        while qi < queue.len() {
            let f = queue[qi] as usize;
            qi += 1;
            popped += 1;
            if popped.is_multiple_of(CHECK_EVERY) {
                ctl.check()?;
            }
            for k in 0..3 {
                let o = 3 * f + k;
                let p = glue[o];
                if p == NONE {
                    continue;
                }
                let g = (p / 3) as usize;
                let j = (p % 3) as usize;
                let same = tris[f][k] == tris[g][j];
                let want = flip[f] ^ same;
                if comp[g] == NONE {
                    comp[g] = ncomp;
                    flip[g] = want;
                    queue.push(g as u32);
                } else if flip[g] != want {
                    glue[o] = NONE;
                    glue[p as usize] = NONE;
                    rep.split_edges += 1;
                }
            }
        }
        ncomp += 1;
    }
    let mut closed = tvec(ncomp as usize, true)?;
    let mut vol = tvec(ncomp as usize, 0.0f64)?;
    for f in 0..nf {
        let c = comp[f] as usize;
        if (0..3).any(|k| glue[3 * f + k] == NONE) {
            closed[c] = false;
        }
        let t = tris[f];
        let (a, b, cc) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let v = math::dot(a, math::cross(b, cc));
        vol[c] += if flip[f] { -v } else { v };
    }
    for f in 0..nf {
        let c = comp[f] as usize;
        if closed[c] && vol[c] < 0.0 {
            flip[f] = !flip[f];
        }
    }
    Ok(flip)
}

/// Step 4: gives every fan (corners connected through glued edges) its own vertex. The first
/// corner (lowest id) of a vertex keeps the vertex id; other fans get copies.
fn split_vertices(
    pos: &mut Vec<V3>,
    tris: &mut [[u32; 3]],
    glue: &[u32],
    vert_work: &mut Vec<u32>,
    rep: &mut IngestReport,
    ctl: &Ctl,
) -> Result<(), AqError> {
    let nf = tris.len();
    let nc = 3 * nf;
    let mut uf: Vec<u32> = tcap(nc)?;
    uf.extend(0..nc as u32);
    let corner_at = |tris: &[[u32; 3]], g: usize, v: u32| -> Option<usize> { (0..3).find(|&i| tris[g][i] == v) };
    for o in 0..nc {
        if o % CHECK_EVERY == CHECK_EVERY - 1 {
            ctl.check()?;
        }
        let p = glue[o] as usize;
        if glue[o] == NONE || p < o {
            continue;
        }
        let f = o / 3;
        let k = o % 3;
        let g = p / 3;
        let (u, v) = (tris[f][k], tris[f][(k + 1) % 3]);
        if let (Some(gu), Some(gv)) = (corner_at(tris, g, u), corner_at(tris, g, v)) {
            let a = find(&mut uf, (3 * f + k) as u32);
            let b = find(&mut uf, (3 * g + gu) as u32);
            uf[a.max(b) as usize] = a.min(b);
            let a = find(&mut uf, (3 * f + (k + 1) % 3) as u32);
            let b = find(&mut uf, (3 * g + gv) as u32);
            uf[a.max(b) as usize] = a.min(b);
        }
    }
    let mut root_vert = tvec(nc, NONE)?;
    let mut claimed = tvec(pos.len(), false)?;
    for c in 0..nc {
        if c % CHECK_EVERY == CHECK_EVERY - 1 {
            ctl.check()?;
        }
        let f = c / 3;
        let k = c % 3;
        let v = tris[f][k];
        let r = find(&mut uf, c as u32) as usize;
        if root_vert[r] == NONE {
            if !claimed[v as usize] {
                claimed[v as usize] = true;
                root_vert[r] = v;
            } else {
                let nvid = pos.len() as u32;
                pos.push(pos[v as usize]);
                vert_work.push(vert_work[v as usize]);
                claimed.push(true);
                root_vert[r] = nvid;
                rep.split_verts += 1;
            }
        }
        tris[f][k] = root_vert[r];
    }
    Ok(())
}

/// Faces that still make an edge non-manifold (a directed edge used twice, or an undirected
/// edge with more than 2 faces): every such face except the lowest id of each group.
fn bad_edge_faces(tris: &[[u32; 3]]) -> Result<Vec<u32>, AqError> {
    let mut und: Vec<(u64, u32, bool)> = tcap(3 * tris.len())?;
    for (f, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (u, v) = (t[k], t[(k + 1) % 3]);
            let key = ((u.min(v) as u64) << 32) | u.max(v) as u64;
            und.push((key, f as u32, u < v));
        }
    }
    und.sort_unstable();
    let mut out: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < und.len() {
        let mut j = i;
        while j < und.len() && und[j].0 == und[i].0 {
            j += 1;
        }
        let g = &und[i..j];
        let badg = g.len() > 2 || (g.len() == 2 && g[0].2 == g[1].2);
        if badg {
            let mut fs: Vec<u32> = g.iter().map(|x| x.1).collect();
            fs.sort_unstable();
            out.extend_from_slice(&fs[1..]);
        }
        i = j;
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[derive(Clone, Copy, PartialEq)]
struct HeapItem(f64, u32);
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap on distance, then vertex id.
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Step 6: feature pairs to halfedges. An edge marks both halves; a non-edge pair is routed
/// along the shortest edge path inside the 2-ring neighbourhoods of its endpoints.
fn map_features(
    m: &TriMesh,
    t: &Topology,
    vert_src: &[u32],
    vrep: &mut [u32],
    feats: &[(u32, u32, FeatureKind)],
    he_kind: &mut [FeatureKind],
    rep: &mut IngestReport,
    ctl: &Ctl,
) -> Result<(), AqError> {
    if feats.is_empty() {
        return Ok(());
    }
    let nh = 3 * m.tris.len();
    let mut edge_keys: Vec<(u64, u32)> = tcap(nh)?;
    for h in 0..nh as u32 {
        let a = vert_src[t.tail(m, h) as usize];
        let b = vert_src[t.head(m, h) as usize];
        edge_keys.push((((a.min(b) as u64) << 32) | a.max(b) as u64, h));
    }
    ctl.check()?;
    edge_keys.sort_unstable();
    ctl.check()?;
    // Copies of each working vertex.
    let mut copies: Vec<(u32, u32)> = tcap(vert_src.len())?;
    copies.extend(vert_src.iter().enumerate().map(|(i, &w)| (w, i as u32)));
    copies.sort_unstable();
    ctl.check()?;
    let copies_of = |w: u32| -> Vec<u32> {
        let lo = copies.partition_point(|x| x.0 < w);
        copies[lo..].iter().take_while(|x| x.0 == w).map(|x| x.1).collect()
    };
    let mark = |he_kind: &mut [FeatureKind], h: u32, k: FeatureKind| {
        he_kind[h as usize] = combine(he_kind[h as usize], k);
        let tw = t.twin[h as usize];
        if tw != NONE {
            he_kind[tw as usize] = combine(he_kind[tw as usize], k);
        }
    };
    for (fi, &(a0, b0, kind)) in feats.iter().enumerate() {
        if fi % 1024 == 1023 {
            ctl.check()?;
        }
        let a = find(vrep, a0);
        let b = find(vrep, b0);
        if a == b {
            rep.dropped_features += 1;
            continue;
        }
        let key = ((a.min(b) as u64) << 32) | a.max(b) as u64;
        let lo = edge_keys.partition_point(|x| x.0 < key);
        let hs: Vec<u32> = edge_keys[lo..].iter().take_while(|x| x.0 == key).map(|x| x.1).collect();
        if !hs.is_empty() {
            for h in hs {
                mark(he_kind, h, kind);
            }
            continue;
        }
        let ca = copies_of(a);
        let cb = copies_of(b);
        if ca.is_empty() || cb.is_empty() {
            rep.dropped_features += 1;
            continue;
        }
        match route(m, t, &ca, &cb) {
            Some(path) => {
                for h in path {
                    mark(he_kind, h, kind);
                }
                rep.rerouted_features += 1;
            }
            None => rep.dropped_features += 1,
        }
    }
    Ok(())
}

/// Neighbours of v (one ring).
fn ring(m: &TriMesh, t: &Topology, v: u32, out: &mut Vec<(u32, u32)>) {
    out.clear();
    for h in t.fan_out(v) {
        out.push((t.head(m, h), h));
        if t.twin[Topology::prev(h) as usize] == NONE {
            // Last face of an open fan: the incoming boundary edge.
            let p = Topology::prev(h);
            out.push((t.tail(m, p), NONE));
        }
    }
}

/// Shortest edge path (Euclidean) from any vertex of `from` to any of `to`, restricted to the
/// vertices within 2 rings of `from` (so the pair is at most 2 rings apart). Returns the
/// halfedges along the path.
fn route(m: &TriMesh, t: &Topology, from: &[u32], to: &[u32]) -> Option<Vec<u32>> {
    let mut region: Vec<u32> = Vec::new();
    let mut buf = Vec::new();
    {
        let seeds = from;
        let mut layer: Vec<u32> = seeds.to_vec();
        region.extend_from_slice(seeds);
        for _ in 0..2 {
            let mut next = Vec::new();
            for &v in &layer {
                ring(m, t, v, &mut buf);
                for &(w, _) in &buf {
                    next.push(w);
                }
            }
            next.sort_unstable();
            next.dedup();
            region.extend_from_slice(&next);
            layer = next;
        }
    }
    region.sort_unstable();
    region.dedup();
    let idx = |v: u32| region.binary_search(&v).ok();
    let n = region.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![NONE; n];
    let mut heap = BinaryHeap::new();
    for &s in from {
        if let Some(i) = idx(s) {
            dist[i] = 0.0;
            heap.push(HeapItem(0.0, s));
        }
    }
    let mut goal = NONE;
    while let Some(HeapItem(d, v)) = heap.pop() {
        let Some(iv) = idx(v) else { continue };
        if d > dist[iv] {
            continue;
        }
        if to.contains(&v) {
            goal = v;
            break;
        }
        ring(m, t, v, &mut buf);
        for &(w, _) in &buf {
            let Some(iw) = idx(w) else { continue };
            let nd = d + math::dist(m.pos[v as usize], m.pos[w as usize]);
            if nd < dist[iw] {
                dist[iw] = nd;
                prev[iw] = v;
                heap.push(HeapItem(nd, w));
            }
        }
    }
    if goal == NONE {
        return None;
    }
    let mut path = Vec::new();
    let mut v = goal;
    let mut guard = 0;
    while let Some(iv) = idx(v) {
        let u = prev[iv];
        if u == NONE {
            break;
        }
        let h = t.find_he(m, u, v).or_else(|| t.find_he(m, v, u))?;
        path.push(h);
        v = u;
        guard += 1;
        if guard > n {
            return None;
        }
    }
    path.reverse();
    Some(path)
}

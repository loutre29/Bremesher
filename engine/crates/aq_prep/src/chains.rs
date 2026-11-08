//! Chains and corners from per-halfedge feature kinds (DESIGN 6.4 "Chain classification").

use aq_core::math;
use aq_core::{AqError, Chain, FeatureKind, FeatureSet, NONE, Stage, Topology, TriMesh, V3};

/// How corners are found along a chain.
pub struct CornerRule<'a> {
    /// Turning angle (degrees) for Hard, Guide, GuideHard and Boundary chains.
    pub corner_deg: f64,
    /// Turning angle (degrees) for Symmetry chains.
    pub sym_corner_deg: f64,
    /// Window length on each side of a vertex (0.5 h). A window of 0 measures the angle
    /// between the two chain edges at the vertex.
    pub window: &'a dyn Fn(u32) -> f64,
    /// When set, the turning-angle test is replaced by these per-vertex corner flags (used to
    /// keep the corners of an existing classification).
    pub fixed: Option<&'a [bool]>,
}

/// `hard_mode` (DESIGN 6.4): 0 drops Hard (GuideHard becomes Guide), 1 turns Hard and
/// GuideHard into Guide, 2 (or more) keeps everything.
pub fn apply_hard_mode(kinds: &mut [FeatureKind], mode: u8) {
    if mode >= 2 {
        return;
    }
    for k in kinds.iter_mut() {
        *k = match (*k, mode) {
            (FeatureKind::Hard, 0) => FeatureKind::None,
            (FeatureKind::Hard, _) => FeatureKind::Guide,
            (FeatureKind::GuideHard, _) => FeatureKind::Guide,
            (x, _) => x,
        };
    }
}

fn is_feat(k: FeatureKind) -> bool {
    k != FeatureKind::None
}

/// Number of feature edges at every vertex (canonical halfedges counted once).
pub fn chain_vertex_degree(m: &TriMesh, t: &Topology, he_kind: &[FeatureKind]) -> Vec<u32> {
    let mut deg = vec![0u32; m.pos.len()];
    for h in 0..(3 * m.tris.len()) as u32 {
        if he_kind.get(h as usize).copied().is_some_and(is_feat) && t.is_canonical(h) {
            deg[t.tail(m, h) as usize] += 1;
            deg[t.head(m, h) as usize] += 1;
        }
    }
    deg
}

/// Number of non-chain edges whose two endpoints both lie on chains (bridge edges).
pub fn count_bridge_edges(m: &TriMesh, t: &Topology, fs: &FeatureSet) -> usize {
    let deg = chain_vertex_degree(m, t, &fs.he_kind);
    (0..(3 * m.tris.len()) as u32)
        .filter(|&h| {
            t.is_canonical(h)
                && !is_feat(fs.he_kind[h as usize])
                && deg[t.tail(m, h) as usize] > 0
                && deg[t.head(m, h) as usize] > 0
        })
        .count()
}

/// Number of faces with 2 or more chain edges.
pub fn count_multi_chain_faces(m: &TriMesh, fs: &FeatureSet) -> usize {
    (0..m.tris.len())
        .filter(|&f| (0..3).filter(|&k| is_feat(fs.he_kind[3 * f + k])).count() >= 2)
        .count()
}

/// Builds a [`FeatureSet`] (chains, corners) from per-halfedge kinds.
///
/// The kinds are first normalized: a boundary halfedge carries Symmetry when it was given
/// Symmetry and Boundary otherwise; both halves of an interior edge get the larger kind of the
/// two. Chains are maximal runs between breaker vertices (chain degree other than 2, a change
/// of kind, or a change between boundary and interior edges); a closed run without breakers
/// is one closed chain. Chain order: runs from breakers in increasing vertex id, then closed
/// runs by their lowest halfedge. Boundary chains follow the boundary halfedges. Corners are
/// the breakers plus the turning-angle corners of `rule` (non-maximum suppressed within the
/// window); a closed chain with corners is rotated so that it starts at a corner.
pub fn build_feature_set(
    m: &TriMesh,
    t: &Topology,
    mut he_kind: Vec<FeatureKind>,
    rule: &CornerRule,
) -> Result<FeatureSet, AqError> {
    let nh = 3 * m.tris.len();
    let nv = m.pos.len();
    if he_kind.len() != nh || t.twin.len() != nh {
        return Err(AqError::invariant("build_feature_set: size mismatch").at(Stage::Proxy));
    }
    for h in 0..nh {
        let tw = t.twin[h];
        if tw == NONE {
            if he_kind[h] != FeatureKind::Symmetry {
                he_kind[h] = FeatureKind::Boundary;
            }
        } else if (tw as usize) > h {
            let k = he_kind[h].max(he_kind[tw as usize]);
            let k = if k == FeatureKind::Boundary { FeatureKind::Hard } else { k };
            he_kind[h] = k;
            he_kind[tw as usize] = k;
        }
    }
    // Canonical feature halfedges per vertex (CSR).
    let mut cnt = vec![0u32; nv + 1];
    for h in 0..nh as u32 {
        if is_feat(he_kind[h as usize]) && t.is_canonical(h) {
            cnt[t.tail(m, h) as usize + 1] += 1;
            cnt[t.head(m, h) as usize + 1] += 1;
        }
    }
    for v in 0..nv {
        cnt[v + 1] += cnt[v];
    }
    let mut fill = cnt.clone();
    let mut inc = vec![0u32; cnt[nv] as usize];
    for h in 0..nh as u32 {
        if is_feat(he_kind[h as usize]) && t.is_canonical(h) {
            for v in [t.tail(m, h), t.head(m, h)] {
                inc[fill[v as usize] as usize] = h;
                fill[v as usize] += 1;
            }
        }
    }
    let inc_of = |v: u32| &inc[cnt[v as usize] as usize..cnt[v as usize + 1] as usize];
    let is_bnd = |h: u32| t.twin[h as usize] == NONE;
    let breaker: Vec<bool> = (0..nv as u32)
        .map(|v| {
            let e = inc_of(v);
            match e.len() {
                0 => false,
                2 => he_kind[e[0] as usize] != he_kind[e[1] as usize] || is_bnd(e[0]) != is_bnd(e[1]),
                _ => true,
            }
        })
        .collect();
    let orient = |e: u32, v: u32| -> Option<u32> {
        if t.tail(m, e) == v {
            Some(e)
        } else if !is_bnd(e) && t.head(m, e) == v {
            Some(t.twin[e as usize])
        } else {
            None
        }
    };
    let mut visited = vec![false; nh];
    let mut runs: Vec<(Vec<u32>, bool)> = Vec::new();
    for v in 0..nv as u32 {
        if !breaker[v as usize] {
            continue;
        }
        for &e in inc_of(v) {
            if visited[e as usize] {
                continue;
            }
            // A boundary edge is walked from its tail only.
            let Some(h0) = orient(e, v) else { continue };
            visited[e as usize] = true;
            let mut hes = vec![h0];
            let mut cur = t.head(m, h0);
            let mut guard = 0usize;
            while !breaker[cur as usize] {
                guard += 1;
                if guard > nh {
                    return Err(AqError::internal("chain walk does not terminate").at(Stage::Proxy));
                }
                let Some(&next) = inc_of(cur).iter().find(|&&x| !visited[x as usize]) else {
                    break;
                };
                let Some(h) = orient(next, cur) else {
                    return Err(AqError::invariant(format!("boundary chain reversed at vertex {cur}"))
                        .at(Stage::Proxy));
                };
                visited[next as usize] = true;
                hes.push(h);
                cur = t.head(m, h);
            }
            runs.push((hes, false));
        }
    }
    for e in 0..nh as u32 {
        if visited[e as usize] || !is_feat(he_kind[e as usize]) || !t.is_canonical(e) {
            continue;
        }
        let start = t.tail(m, e);
        visited[e as usize] = true;
        let mut hes = vec![e];
        let mut cur = t.head(m, e);
        let mut guard = 0usize;
        while cur != start {
            guard += 1;
            if guard > nh {
                return Err(AqError::internal("closed chain walk does not terminate").at(Stage::Proxy));
            }
            let Some(&next) = inc_of(cur).iter().find(|&&x| !visited[x as usize]) else {
                return Err(AqError::invariant(format!("closed chain broken at vertex {cur}")).at(Stage::Proxy));
            };
            let Some(h) = orient(next, cur) else {
                return Err(AqError::invariant(format!("boundary chain reversed at vertex {cur}")).at(Stage::Proxy));
            };
            visited[next as usize] = true;
            hes.push(h);
            cur = t.head(m, h);
        }
        runs.push((hes, true));
    }
    let mut fs = FeatureSet {
        he_kind,
        he_chain: vec![NONE; nh],
        chains: Vec::with_capacity(runs.len()),
        vert_corner: vec![false; nv],
    };
    for (ci, (mut hes, closed)) in runs.into_iter().enumerate() {
        let kind = fs.he_kind[hes[0] as usize];
        let lim = if kind == FeatureKind::Symmetry { rule.sym_corner_deg } else { rule.corner_deg };
        let turn = match rule.fixed {
            Some(fx) => hes.iter().map(|&h| fx.get(t.tail(m, h) as usize).copied().unwrap_or(false)).collect(),
            None => turning_corners(m, t, &hes, closed, lim.to_radians(), rule.window),
        };
        let n = hes.len();
        let mut corners: Vec<u32> = Vec::new();
        for i in 0..n {
            let v = t.tail(m, hes[i]);
            if breaker[v as usize] || turn[i] {
                corners.push(i as u32);
            }
        }
        if closed && !corners.is_empty() && corners[0] != 0 {
            let s = corners[0] as usize;
            hes.rotate_left(s);
            for c in corners.iter_mut() {
                *c = ((*c as usize + n - s) % n) as u32;
            }
            corners.sort_unstable();
        }
        for (i, &h) in hes.iter().enumerate() {
            fs.he_chain[h as usize] = ci as u32;
            let tw = t.twin[h as usize];
            if tw != NONE {
                fs.he_chain[tw as usize] = ci as u32;
            }
            if corners.binary_search(&(i as u32)).is_ok() {
                fs.vert_corner[t.tail(m, h) as usize] = true;
            }
        }
        fs.chains.push(Chain { hes, kind, closed, corners });
    }
    for v in 0..nv {
        if breaker[v] {
            fs.vert_corner[v] = true;
        }
    }
    Ok(fs)
}

/// Turning-angle corners along one chain: `out[i]` is true when the vertex `tail(hes[i])` is a
/// corner. Position 0 of an open chain is never tested (it is a breaker).
fn turning_corners(
    m: &TriMesh,
    t: &Topology,
    hes: &[u32],
    closed: bool,
    lim: f64,
    window: &dyn Fn(u32) -> f64,
) -> Vec<bool> {
    let n = hes.len();
    let mut out = vec![false; n];
    if n < 2 && !closed {
        return out;
    }
    let vec_of = |h: u32| -> V3 { math::sub(m.pos[t.head(m, h) as usize], m.pos[t.tail(m, h) as usize]) };
    let ev: Vec<V3> = hes.iter().map(|&h| vec_of(h)).collect();
    let el: Vec<f64> = ev.iter().map(|&e| math::norm(e)).collect();
    // Cumulative arc length at the start of edge i.
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + el[i];
    }
    let total = cum[n];
    let first = if closed { 0 } else { 1 };
    let mut ang = vec![-1.0f64; n];
    let mut win = vec![0.0f64; n];
    let max_side = if closed { n.saturating_sub(1).max(1) } else { n };
    for i in first..n {
        let w = window(t.tail(m, hes[i]));
        let w = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        win[i] = w;
        // Backward: edges i-1, i-2, ...
        let mut back = [0.0; 3];
        let mut rem = w;
        let mut j = i;
        let mut steps = 0;
        loop {
            if j == 0 {
                if !closed {
                    break;
                }
                j = n;
            }
            j -= 1;
            let frac = if steps == 0 || el[j] <= rem { 1.0 } else { (rem / el[j]).max(0.0) };
            back = math::add(back, math::scale(ev[j], frac));
            rem -= el[j];
            steps += 1;
            if rem <= 0.0 || steps >= max_side {
                break;
            }
        }
        let mut fwd = [0.0; 3];
        let mut rem = w;
        let mut j = i;
        let mut steps = 0;
        loop {
            if j >= n {
                if !closed {
                    break;
                }
                j = 0;
            }
            let frac = if steps == 0 || el[j] <= rem { 1.0 } else { (rem / el[j]).max(0.0) };
            fwd = math::add(fwd, math::scale(ev[j], frac));
            rem -= el[j];
            steps += 1;
            j += 1;
            if rem <= 0.0 || steps >= max_side {
                break;
            }
        }
        if steps == 0 {
            continue;
        }
        ang[i] = math::angle_between(back, fwd);
    }
    // Arc distance between positions i and j along the chain (shorter way round when closed).
    let dist = |i: usize, j: usize| -> f64 {
        let d = (cum[i] - cum[j]).abs();
        if closed { d.min(total - d) } else { d }
    };
    for i in first..n {
        if !(ang[i] >= lim) {
            continue;
        }
        // Non-maximum suppression over half the window: enough to merge the neighbours of
        // one kink (their windowed angle drops below the limit before 0.5 w), while two real
        // corners one window apart both survive.
        let w = 0.5 * win[i];
        let mut keep = true;
        if w > 0.0 {
            // Scan neighbours within the window on both sides.
            for dir in [-1i64, 1] {
                let mut j = i as i64;
                for _ in 0..n {
                    j += dir;
                    if j < 0 || j >= n as i64 {
                        if !closed {
                            break;
                        }
                        j = j.rem_euclid(n as i64);
                    }
                    let ju = j as usize;
                    if ju == i || dist(i, ju) > w {
                        break;
                    }
                    if ang[ju] > ang[i] || (ang[ju] == ang[i] && ju < i) {
                        keep = false;
                        break;
                    }
                }
                if !keep {
                    break;
                }
            }
        }
        out[i] = keep;
    }
    out
}

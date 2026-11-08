//! Edge loop closure checker (DESIGN 6.15 `loop_closure`).
//!
//! Every undirected edge belongs to exactly one edge line: the line runs straight through
//! every regular vertex and stops elsewhere.
//! * An interior edge continues through a regular interior vertex (valence 4, a closed fan
//!   of 4 clean edges) to the opposite edge, and stops at a boundary or irregular vertex.
//! * A boundary edge continues along the boundary through a regular boundary vertex
//!   (valence 3) and stops at a corner (valence 2) or an irregular boundary vertex, so the
//!   boundary rings of an open cylinder are closed loops.
//!
//! A line is closed when the walk returns to its start edge, open otherwise.
//!
//! With an axis: only lines whose length-weighted mean |cos| between their edges and the
//! axis is below 0.3 (circumferential lines) are counted, and helix suspects are searched
//! among them: a counted line whose azimuth about the axis (through the mean vertex of the
//! mesh) completes a full turn with an axial drift above 0.5 h, h = the mean edge length.

use aq_core::math;
use aq_core::{NONE, QuadMesh, V3};
use serde::{Deserialize, Serialize};

use crate::poly::PolyTopo;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LoopReport {
    /// Edge lines counted (circumferential ones only when an axis is given).
    pub loops: u32,
    /// Closed lines among them.
    pub closed: u32,
    /// closed / loops (0 when there is no line).
    pub closed_frac: f64,
    pub helix_suspects: u32,
}

/// Threshold on the mean |cos(edge, axis)| for a circumferential line.
pub const CIRC_COS: f64 = 0.3;
/// Axial drift per turn, in h, above which a line is a helix suspect.
pub const HELIX_DRIFT_H: f64 = 0.5;

/// A walking direction: halfedge `he` traversed tail to head (`fwd`) or head to tail
/// (only used for boundary halfedges, which have no twin).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Dir {
    he: u32,
    fwd: bool,
}

impl Dir {
    fn from(self, t: &PolyTopo) -> u32 {
        if self.fwd { t.tail[self.he as usize] } else { t.head[self.he as usize] }
    }
    fn to(self, t: &PolyTopo) -> u32 {
        if self.fwd { t.head[self.he as usize] } else { t.tail[self.he as usize] }
    }
    fn rev(self, t: &PolyTopo) -> Dir {
        let tw = t.twin[self.he as usize];
        if tw != NONE { Dir { he: tw, fwd: true } } else { Dir { he: self.he, fwd: !self.fwd } }
    }
}

/// The continuation of a line past the end vertex of `d`, if that vertex is regular.
fn step(t: &PolyTopo, d: Dir) -> Option<Dir> {
    let h = d.he as usize;
    let v = d.to(t);
    let guard = t.census.valence[v as usize] as usize + 2;
    if t.twin[h] != NONE {
        // Interior edge, always walked forward.
        if !d.fwd || !t.regular_interior(v, d.he) {
            return None;
        }
        let o0 = t.twin[h];
        let o1 = t.twin[t.prev[o0 as usize] as usize];
        let o2 = t.twin[t.prev[o1 as usize] as usize];
        return Some(Dir { he: o2, fwd: true });
    }
    if t.edge_count[t.edge[h] as usize] != 1 || !t.regular_boundary(v) {
        return None;
    }
    if d.fwd {
        // Arriving at head v along the boundary: rotate to the outgoing boundary halfedge.
        let mut x = t.next[h];
        for _ in 0..guard {
            let tw = t.twin[x as usize];
            if tw == NONE {
                return (t.edge_count[t.edge[x as usize] as usize] == 1 && x != d.he)
                    .then_some(Dir { he: x, fwd: true });
            }
            x = t.next[tw as usize];
        }
        None
    } else {
        // Arriving at tail v walking backward: rotate to the incoming boundary halfedge.
        let mut x = t.prev[h];
        for _ in 0..guard {
            let tw = t.twin[x as usize];
            if tw == NONE {
                return (t.edge_count[t.edge[x as usize] as usize] == 1 && x != d.he)
                    .then_some(Dir { he: x, fwd: false });
            }
            x = t.prev[tw as usize];
        }
        None
    }
}

/// One edge line: its vertex sequence and whether it is closed (then the last vertex
/// equals the first).
struct Line {
    verts: Vec<u32>,
    closed: bool,
}

fn walk_lines(t: &PolyTopo) -> Vec<Line> {
    let mut visited = vec![false; t.n_edges];
    let mut lines = Vec::new();
    for e in 0..t.n_edges {
        if visited[e] {
            continue;
        }
        visited[e] = true;
        let d0 = Dir { he: t.edge_he[e], fwd: true };
        let mut fwd_verts = vec![d0.from(t), d0.to(t)];
        let mut closed = false;
        let mut cur = d0;
        while let Some(nd) = step(t, cur) {
            let ne = t.edge[nd.he as usize] as usize;
            if ne == e {
                closed = true;
                break;
            }
            if visited[ne] {
                break;
            }
            visited[ne] = true;
            fwd_verts.push(nd.to(t));
            cur = nd;
        }
        if closed {
            lines.push(Line { verts: fwd_verts, closed: true });
            continue;
        }
        let mut back: Vec<u32> = Vec::new();
        let mut cur = d0.rev(t);
        while let Some(nd) = step(t, cur) {
            let ne = t.edge[nd.he as usize] as usize;
            if visited[ne] {
                break;
            }
            visited[ne] = true;
            back.push(nd.to(t));
            cur = nd;
        }
        back.reverse();
        back.extend_from_slice(&fwd_verts);
        lines.push(Line { verts: back, closed: false });
    }
    lines
}

fn basis(axis: V3) -> Option<(V3, V3, V3)> {
    let a = math::normalize(axis)?;
    let helper = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let b = math::normalize(math::cross(a, helper))?;
    let c = math::cross(a, b);
    Some((a, b, c))
}

/// True when the polyline completes a turn about the axis with an axial drift above
/// `max_drift`.
fn is_helix(pts: &[V3], origin: V3, (a, b, c): (V3, V3, V3), max_drift: f64) -> bool {
    let mut phis = Vec::with_capacity(pts.len());
    let mut zs = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    let mut last: Option<f64> = None;
    for &p in pts {
        let r = math::sub(p, origin);
        let (x, y) = (math::dot(r, b), math::dot(r, c));
        if x == 0.0 && y == 0.0 {
            continue;
        }
        let phi = y.atan2(x);
        if let Some(l) = last {
            acc += aq_core::wrap_pi(phi - l);
        }
        last = Some(phi);
        phis.push(acc);
        zs.push(math::dot(r, a));
    }
    let Some(&total) = phis.last() else { return false };
    let sign = if total >= 0.0 { 1.0 } else { -1.0 };
    let tau = std::f64::consts::TAU;
    let mut z_prev = match zs.first() {
        Some(&z) => z,
        None => return false,
    };
    let mut k = 1.0;
    for i in 1..phis.len() {
        let (p0, p1) = (sign * phis[i - 1], sign * phis[i]);
        while p1 >= k * tau {
            let s = if p1 > p0 { ((k * tau - p0) / (p1 - p0)).clamp(0.0, 1.0) } else { 1.0 };
            let z = zs[i - 1] + s * (zs[i] - zs[i - 1]);
            if (z - z_prev).abs() > max_drift {
                return true;
            }
            z_prev = z;
            k += 1.0;
        }
    }
    false
}

/// Edge loop closure statistics of q (see the module rules). `axis` selects the
/// circumferential lines and enables the helix test.
pub fn loop_closure(q: &QuadMesh, axis: Option<V3>) -> LoopReport {
    let t = PolyTopo::build(q);
    let lines = walk_lines(&t);
    let frame = axis.and_then(basis);
    let (origin, h) = {
        let used: Vec<usize> = (0..q.pos.len()).filter(|&v| t.census.valence[v] > 0).collect();
        let mut o = [0.0; 3];
        for &v in &used {
            o = math::add(o, q.pos[v]);
        }
        let o = if used.is_empty() { o } else { math::scale(o, 1.0 / used.len() as f64) };
        let lens: Vec<f64> =
            t.census.edges.iter().map(|&([a, b], _)| math::dist(q.pos[a as usize], q.pos[b as usize])).collect();
        let h = if lens.is_empty() { 0.0 } else { lens.iter().sum::<f64>() / lens.len() as f64 };
        (o, h)
    };
    let mut rep = LoopReport::default();
    for l in &lines {
        let pts: Vec<V3> = l.verts.iter().map(|&v| q.pos[v as usize]).collect();
        if let Some(fr) = frame {
            let (mut num, mut den) = (0.0, 0.0);
            for w in pts.windows(2) {
                let d = math::sub(w[1], w[0]);
                num += math::dot(d, fr.0).abs();
                den += math::norm(d);
            }
            let mean_cos = if den > 0.0 { num / den } else { 1.0 };
            if mean_cos >= CIRC_COS {
                continue;
            }
            if h > 0.0 && is_helix(&pts, origin, fr, HELIX_DRIFT_H * h) {
                rep.helix_suspects += 1;
            }
        }
        rep.loops += 1;
        if l.closed {
            rep.closed += 1;
        }
    }
    rep.closed_frac = if rep.loops == 0 { 0.0 } else { rep.closed as f64 / rep.loops as f64 };
    rep
}

//! Shape diameter for the thickness cap (DESIGN 6.5 "Thickness cap (M2)", WP-20).
//!
//! `d(v)`: 5 rays from v into the solid, one along `-n_v` (area-weighted vertex normal) and
//! four at 15 degrees from it (a 30 degree cone, azimuths 0, 90, 180 and 270 degrees in a
//! tangent frame fixed by the normal). A ray counts only when its first hit is a **back face**
//! (`n_f . dir > 0`), so a ray leaving one sheet of a layered surface and meeting the front of
//! another sheet does not count; faces incident to v are skipped. With at least 3 hits, `d(v)`
//! is the median hit distance; otherwise `d(v)` is infinite (an open sheet, a mostly unclosed
//! part). The values are then filtered by the median over the closed 1-ring, which removes
//! single-vertex outliers (self-intersections, ray leaks through cracks). Last, plate interiors
//! get infinity ([`plate_filter`]): only tubes and rims are capped.
//!
//! The cap itself (`g <= pi d / ring_min`) lives in [`crate::sizing`]. Everything is in the
//! units of the mesh (quad units in the pipeline). The ray BVH is local to this crate: aq_core's
//! BVH answers closest-point queries only.

use rayon::prelude::*;

use aq_core::math::{self, cross, dot, sub};
use aq_core::{AqError, Ctl, TriMesh, V3};

use crate::{ProxyRef, restamp};

/// Rays per vertex (DESIGN 6.5).
pub const THICK_RAYS: usize = 5;
/// Angle of the 4 outer rays from `-n`, radians (half of the 30 degree cone).
pub const THICK_CONE_HALF: f64 = 15.0 * std::f64::consts::PI / 180.0;
/// Hits needed for a finite diameter (a majority of the rays).
pub const THICK_MIN_HITS: usize = 3;
/// Longest ray, in the units of the mesh (quad units): a hit farther than this counts as a miss.
/// With ring_min 8 it is a cap of 3.9 h, at the top of the g clamp, so a longer diameter can
/// never bind; the bound keeps each ray local.
pub const THICK_D_MAX: f64 = 10.0;

const LEAF: usize = 4;

#[derive(Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    /// Leaf: first index into `order`; inner: index of the left child (right = left + 1).
    first: u32,
    /// Leaf: number of triangles; inner: 0.
    count: u32,
}

/// Bounding volume hierarchy for first-hit ray queries.
pub struct RayBvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
    tris: Vec<[V3; 3]>,
    normals: Vec<V3>,
}

fn box_of(tris: &[[V3; 3]], ids: &[u32]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in ids {
        for p in &tris[i as usize] {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    (lo, hi)
}

/// Slab test: entry distance of the ray into the box, or None when it misses within `tmax`.
#[inline]
fn slab(lo: V3, hi: V3, o: V3, inv: V3, tmax: f64) -> Option<f64> {
    let mut t0 = 0.0f64;
    let mut t1 = tmax;
    for k in 0..3 {
        let a = (lo[k] - o[k]) * inv[k];
        let b = (hi[k] - o[k]) * inv[k];
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        // NaN (0 * inf) leaves the bound unchanged.
        if a > t0 {
            t0 = a;
        }
        if b < t1 {
            t1 = b;
        }
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// Moller-Trumbore; returns t > 0 or None.
#[inline]
fn hit_tri(o: V3, d: V3, t: &[V3; 3]) -> Option<f64> {
    let e1 = sub(t[1], t[0]);
    let e2 = sub(t[2], t[0]);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, t[0]);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let tt = dot(e2, q) * inv;
    if tt > 0.0 && tt.is_finite() { Some(tt) } else { None }
}

impl RayBvh {
    /// Builds the hierarchy over the triangles of `m` (median split on the largest centroid
    /// extent, ties by id, so the result does not depend on the thread count).
    pub fn build(m: &TriMesh) -> Result<Self, AqError> {
        let nf = m.tris.len();
        let mut tris: Vec<[V3; 3]> = aq_core::alloc::try_with_capacity(nf)?;
        let mut normals: Vec<V3> = aq_core::alloc::try_with_capacity(nf)?;
        for (f, t) in m.tris.iter().enumerate() {
            let mut q = [[0.0; 3]; 3];
            for k in 0..3 {
                q[k] = *m
                    .pos
                    .get(t[k] as usize)
                    .ok_or_else(|| AqError::invalid_input(format!("RayBvh: face {f} index out of range")))?;
            }
            tris.push(q);
            normals.push(math::normalize(cross(sub(q[1], q[0]), sub(q[2], q[0]))).unwrap_or([0.0; 3]));
        }
        let cent: Vec<V3> = tris
            .iter()
            .map(|t| {
                [
                    (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                    (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                    (t[0][2] + t[1][2] + t[2][2]) / 3.0,
                ]
            })
            .collect();
        let mut order: Vec<u32> = (0..nf as u32).collect();
        let mut nodes: Vec<Node> = aq_core::alloc::try_with_capacity(2 * nf.div_ceil(LEAF) + 1)?;
        if nf == 0 {
            return Ok(RayBvh { nodes, order, tris, normals });
        }
        let (lo, hi) = box_of(&tris, &order);
        nodes.push(Node { lo, hi, first: 0, count: nf as u32 });
        let mut stack: Vec<(usize, usize, usize)> = vec![(0, 0, nf)];
        while let Some((ni, s, e)) = stack.pop() {
            let n = e - s;
            if n <= LEAF {
                nodes[ni].first = s as u32;
                nodes[ni].count = n as u32;
                continue;
            }
            let mut clo = [f64::INFINITY; 3];
            let mut chi = [f64::NEG_INFINITY; 3];
            for &i in &order[s..e] {
                let c = cent[i as usize];
                for k in 0..3 {
                    clo[k] = clo[k].min(c[k]);
                    chi[k] = chi[k].max(c[k]);
                }
            }
            let mut axis = 0;
            for k in 1..3 {
                if chi[k] - clo[k] > chi[axis] - clo[axis] {
                    axis = k;
                }
            }
            let mid = n / 2;
            order[s..e].select_nth_unstable_by(mid, |&a, &b| {
                cent[a as usize][axis].total_cmp(&cent[b as usize][axis]).then(a.cmp(&b))
            });
            let left = nodes.len();
            let (llo, lhi) = box_of(&tris, &order[s..s + mid]);
            let (rlo, rhi) = box_of(&tris, &order[s + mid..e]);
            nodes.push(Node { lo: llo, hi: lhi, first: 0, count: 0 });
            nodes.push(Node { lo: rlo, hi: rhi, first: 0, count: 0 });
            nodes[ni].first = left as u32;
            nodes[ni].count = 0;
            stack.push((left + 1, s + mid, e));
            stack.push((left, s, s + mid));
        }
        Ok(RayBvh { nodes, order, tris, normals })
    }

    /// First hit of the ray `o + t d` (`t_min < t <= t_max`) among the faces not rejected by
    /// `skip`: `(t, face, back)` where `back` is `n_f . d > 0`. Ties are broken by the lowest
    /// face id.
    pub fn first_hit(&self, o: V3, d: V3, t_min: f64, t_max: f64, skip: &dyn Fn(u32) -> bool) -> Option<(f64, u32, bool)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let mut best: Option<(f64, u32)> = None;
        let mut tmax = t_max;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            match slab(n.lo, n.hi, o, inv, tmax) {
                Some(t0) if t0 <= tmax => {}
                _ => continue,
            }
            if n.count > 0 {
                for &f in &self.order[n.first as usize..(n.first + n.count) as usize] {
                    if skip(f) {
                        continue;
                    }
                    if let Some(t) = hit_tri(o, d, &self.tris[f as usize])
                        && t > t_min
                        && t <= tmax
                    {
                        let better = match best {
                            None => true,
                            Some((bt, bf)) => t < bt || (t == bt && f < bf),
                        };
                        if better {
                            best = Some((t, f));
                            tmax = t;
                        }
                    }
                }
            } else {
                let l = n.first;
                stack.push(l + 1);
                stack.push(l);
            }
        }
        best.map(|(t, f)| (t, f, dot(self.normals[f as usize], d) > 0.0))
    }
}

/// Median of a small sorted slice.
fn median_sorted(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        f64::INFINITY
    } else if n % 2 == 1 || x[n / 2 - 1].is_infinite() || x[n / 2].is_infinite() {
        x[n / 2]
    } else {
        0.5 * (x[n / 2 - 1] + x[n / 2])
    }
}

/// Unit tangent frame (t1, t2) of a unit normal n, fixed by n alone.
fn tangent_frame(n: V3) -> (V3, V3) {
    let a = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let t1 = math::normalize(cross(n, a)).unwrap_or([1.0, 0.0, 0.0]);
    (t1, cross(n, t1))
}

/// Raw shape diameter of every vertex of `m` (see the module doc), before the 1-ring filter.
/// `vn` are unit vertex normals (zero for an isolated vertex, which gets infinity), `fans`
/// the incident faces of each vertex in CSR form (`fan_off`, `fan`).
fn raw_diameter(
    m: &TriMesh,
    bvh: &RayBvh,
    vn: &[V3],
    fan_off: &[usize],
    fan: &[u32],
    ctl: &Ctl,
) -> Result<Vec<f64>, AqError> {
    let nv = m.pos.len();
    let mut out = aq_core::alloc::try_vec(nv, f64::INFINITY).map_err(restamp)?;
    let (cs, sn) = (THICK_CONE_HALF.cos(), THICK_CONE_HALF.sin());
    const CHUNK: usize = 4096;
    for (ci, chunk) in out.chunks_mut(CHUNK).enumerate() {
        ctl.check()?;
        let base = ci * CHUNK;
        chunk.par_iter_mut().enumerate().with_min_len(64).for_each(|(k, dv)| {
            let v = base + k;
            let n = vn[v];
            if n == [0.0; 3] {
                return;
            }
            let own = &fan[fan_off[v]..fan_off[v + 1]];
            let skip = |f: u32| own.contains(&f);
            let (t1, t2) = tangent_frame(n);
            let o = m.pos[v];
            let mut hits = [f64::INFINITY; THICK_RAYS];
            let mut nh = 0;
            for r in 0..THICK_RAYS {
                let d = if r == 0 {
                    math::scale(n, -1.0)
                } else {
                    let a = (r - 1) as f64 * std::f64::consts::FRAC_PI_2;
                    let side = math::add(math::scale(t1, a.cos()), math::scale(t2, a.sin()));
                    math::add(math::scale(n, -cs), math::scale(side, sn))
                };
                if let Some((t, _, true)) = bvh.first_hit(o, d, 1e-9, THICK_D_MAX, &skip) {
                    hits[nh] = t;
                    nh += 1;
                }
            }
            if nh >= THICK_MIN_HITS {
                let h = &mut hits[..nh];
                h.sort_by(f64::total_cmp);
                *dv = median_sorted(h);
            }
        });
    }
    Ok(out)
}

/// Shape diameter `d(v)` of every proxy vertex (module doc), 1-ring median filtered.
/// Infinity where fewer than 3 of the 5 rays meet a back face.
pub fn shape_diameter(px: &ProxyRef, ctl: &Ctl) -> Result<Vec<f64>, AqError> {
    crate::check_proxy(px)?;
    shape_diameter_mesh(px.mesh, &px.frames.n, &px.frames.area, ctl)
}

/// [`shape_diameter`] on a bare triangle mesh with its face normals and areas (used by the
/// engine on input components before any proxy exists).
pub fn shape_diameter_mesh(m: &TriMesh, face_n: &[V3], face_area: &[f64], ctl: &Ctl) -> Result<Vec<f64>, AqError> {
    let nv = m.pos.len();
    let nf = m.tris.len();
    if face_n.len() != nf || face_area.len() != nf {
        return Err(crate::bad("shape_diameter: normals or areas do not match the faces"));
    }
    if m.tris.iter().flatten().any(|&v| v as usize >= nv) {
        return Err(crate::bad("shape_diameter: vertex index out of range"));
    }
    // Vertex normals (area weighted) and fans.
    let mut acc = aq_core::alloc::try_vec(nv, [0.0f64; 3]).map_err(restamp)?;
    let mut deg = aq_core::alloc::try_vec(nv + 1, 0usize).map_err(restamp)?;
    for (f, t) in m.tris.iter().enumerate() {
        for &v in t {
            acc[v as usize] = math::add(acc[v as usize], math::scale(face_n[f], face_area[f]));
            deg[v as usize + 1] += 1;
        }
    }
    for v in 0..nv {
        deg[v + 1] += deg[v];
    }
    let fan_off = deg;
    let mut fill = fan_off.clone();
    let mut fan = aq_core::alloc::try_vec(fan_off[nv], 0u32).map_err(restamp)?;
    for (f, t) in m.tris.iter().enumerate() {
        for &v in t {
            fan[fill[v as usize]] = f as u32;
            fill[v as usize] += 1;
        }
    }
    let vn: Vec<V3> = acc.iter().map(|a| math::normalize(*a).unwrap_or([0.0; 3])).collect();
    let bvh = RayBvh::build(m).map_err(restamp)?;
    let raw = raw_diameter(m, &bvh, &vn, &fan_off, &fan, ctl)?;
    ctl.check()?;
    // 1-ring median filter over the closed ring (vertex neighbours through the fan faces).
    let mut out = aq_core::alloc::try_vec(nv, f64::INFINITY).map_err(restamp)?;
    let mut ring: Vec<u32> = Vec::with_capacity(16);
    let mut vals: Vec<f64> = Vec::with_capacity(16);
    for v in 0..nv {
        ring.clear();
        for &f in &fan[fan_off[v]..fan_off[v + 1]] {
            for &w in &m.tris[f as usize] {
                ring.push(w);
            }
        }
        ring.sort_unstable();
        ring.dedup();
        vals.clear();
        vals.extend(ring.iter().map(|&w| raw[w as usize]));
        if vals.is_empty() {
            out[v] = raw[v];
            continue;
        }
        vals.sort_by(f64::total_cmp);
        out[v] = median_sorted(&vals);
    }
    ctl.check()?;
    plate_filter(m, &vn, &mut out, ctl)?;
    Ok(out)
}

/// Smallest normal turn, within geodesic distance `TURN_RADIUS * d(v)` of v, that marks v as
/// part of a tube or a rim (cos 60 degrees).
pub const TURN_COS: f64 = 0.5;
/// Radius of the turn test in units of d(v).
pub const TURN_RADIUS: f64 = 1.0;

/// Tube or plate (WP-20). A thin **tube** of diameter d turns its normal by about 115 degrees
/// within a geodesic distance d; the interior of a thin **plate** (a solidified sheet, a shell
/// with a thickness) does not turn at all there, and only its rim does. The cap
/// `pi d / ring_min` counts quads around a round cross-section, which a plate does not have:
/// its cross-section is as long as the plate is wide. So a vertex whose area-weighted normal
/// stays within 60 degrees of every normal met within geodesic distance d(v) (Dijkstra over the
/// mesh edges) gets d = infinity. Rims and tubes keep their diameter, and the gradation spreads
/// the rim size into the plate.
fn plate_filter(m: &TriMesh, vn: &[V3], d: &mut [f64], ctl: &Ctl) -> Result<(), AqError> {
    let nv = m.pos.len();
    // Undirected edge graph, CSR.
    let mut deg = aq_core::alloc::try_vec(nv + 1, 0usize).map_err(restamp)?;
    let mut edges: Vec<(u32, u32)> = aq_core::alloc::try_with_capacity(3 * m.tris.len()).map_err(restamp)?;
    for t in &m.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    for &(a, b) in &edges {
        deg[a as usize + 1] += 1;
        deg[b as usize + 1] += 1;
    }
    for v in 0..nv {
        deg[v + 1] += deg[v];
    }
    let off = deg;
    let mut fill = off.clone();
    let mut nbr = aq_core::alloc::try_vec(off[nv], (0u32, 0.0f64)).map_err(restamp)?;
    for &(a, b) in &edges {
        let l = math::dist(m.pos[a as usize], m.pos[b as usize]);
        nbr[fill[a as usize]] = (b, l);
        fill[a as usize] += 1;
        nbr[fill[b as usize]] = (a, l);
        fill[b as usize] += 1;
    }
    drop(edges);
    let cand: Vec<u32> = (0..nv as u32).filter(|&v| d[v as usize].is_finite()).collect();
    let mut plate = vec![false; cand.len()];
    const CHUNK: usize = 2048;
    for (ci, chunk) in plate.chunks_mut(CHUNK).enumerate() {
        ctl.check()?;
        let base = ci * CHUNK;
        chunk
            .par_iter_mut()
            .enumerate()
            .with_min_len(32)
            .for_each_init(
                || (vec![f64::INFINITY; nv], Vec::<u32>::new(), std::collections::BinaryHeap::<std::cmp::Reverse<(u64, u32)>>::new()),
                |(dist, touched, heap), (k, out)| {
                    let v = cand[base + k] as usize;
                    let n0 = vn[v];
                    let r = TURN_RADIUS * d[v];
                    let mut turned = false;
                    dist[v] = 0.0;
                    touched.push(v as u32);
                    heap.push(std::cmp::Reverse((0f64.to_bits(), v as u32)));
                    while let Some(std::cmp::Reverse((bits, w))) = heap.pop() {
                        let w = w as usize;
                        let dw = f64::from_bits(bits);
                        if dw > dist[w] {
                            continue;
                        }
                        if dot(n0, vn[w]) < TURN_COS {
                            turned = true;
                            break;
                        }
                        for &(x, l) in &nbr[off[w]..off[w + 1]] {
                            let nd = dw + l;
                            let x = x as usize;
                            if nd <= r && nd < dist[x] {
                                if dist[x].is_infinite() {
                                    touched.push(x as u32);
                                }
                                dist[x] = nd;
                                heap.push(std::cmp::Reverse((nd.to_bits(), x as u32)));
                            }
                        }
                    }
                    for &t in touched.iter() {
                        dist[t as usize] = f64::INFINITY;
                    }
                    touched.clear();
                    heap.clear();
                    *out = !turned;
                },
            );
    }
    for (i, &v) in cand.iter().enumerate() {
        if plate[i] {
            d[v as usize] = f64::INFINITY;
        }
    }
    Ok(())
}

/// Largest triangle count [`subdivide_to`] produces (the refinement stops before it).
pub const PLAN_MAX_TRIS: usize = 400_000;

/// Densest sampling [`thickness_plan`] subdivides to, in triangles per h^2 (edges of about
/// 0.35 h): a single long edge in an otherwise fine input must not multiply it by 16.
pub const PLAN_TRIS_PER_H2: f64 = 32.0;

/// Uniform 1-to-4 midpoint subdivision until the longest edge is at most `max_edge`, at most
/// `max_tris` (and [`PLAN_MAX_TRIS`]) triangles. Coarse inputs (a few long triangles across a
/// thin hem) would otherwise give each ray origin a large area; the surface itself does not
/// change.
pub fn subdivide_to(m: &TriMesh, max_edge: f64, max_tris: usize) -> TriMesh {
    let max_tris = max_tris.min(PLAN_MAX_TRIS);
    let mut cur = m.clone();
    loop {
        let longest = cur
            .tris
            .iter()
            .flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3])))
            .map(|(a, b)| math::dist(cur.pos[a as usize], cur.pos[b as usize]))
            .fold(0.0f64, f64::max);
        if longest.is_nan() || longest <= max_edge || 4 * cur.tris.len() > max_tris {
            return cur;
        }
        let mut edges: Vec<(u32, u32)> = cur
            .tris
            .iter()
            .flat_map(|t| (0..3).map(move |k| (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]))))
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let nv0 = cur.pos.len() as u32;
        let mut pos = cur.pos.clone();
        for &(a, b) in &edges {
            pos.push(math::lerp(cur.pos[a as usize], cur.pos[b as usize], 0.5));
        }
        let mid = |a: u32, b: u32| -> u32 {
            let key = (a.min(b), a.max(b));
            nv0 + edges.binary_search(&key).unwrap_or(0) as u32
        };
        let mut tris = Vec::with_capacity(4 * cur.tris.len());
        for t in &cur.tris {
            let (ab, bc, ca) = (mid(t[0], t[1]), mid(t[1], t[2]), mid(t[2], t[0]));
            tris.push([t[0], ab, ca]);
            tris.push([ab, t[1], bc]);
            tris.push([ca, bc, t[2]]);
            tris.push([ab, bc, ca]);
        }
        cur = TriMesh { pos, tris };
    }
}

/// Result of [`thickness_plan`].
#[derive(Clone, Debug, PartialEq)]
pub struct ThickPlan {
    /// Floor of the thickness caps, in the units of the meshes (infinite: no cap affordable;
    /// [`crate::G_MIN`]: the budget does not bind).
    pub floor: f64,
    /// Quad demand of each component at that floor (sums to n), to share the target by.
    pub demand: Vec<f64>,
    /// Area of each component whose thickness cap is below the uniform size.
    pub thin_area: Vec<f64>,
    /// The uniform size s of the plan (about 1 in quad units).
    pub scale: f64,
}

/// Global thickness plan across components (WP-20, DESIGN 6.5 thickness cap with the "count
/// correction" of 7.1 made global): `meshes` in quad units (h0 = 1), `measure[c]` false for
/// components to count by area only (debris). The shape diameter is measured on each measured
/// component (subdivided to edges of at most 0.5), the floor of the caps `pi d / ring_min` is
/// the one that spends at most `budget * n` quads ([`crate::thickness_floor`], no gradation),
/// and each component's demand is `sum A_v / min(s, max(cap_v, floor))^2`. A component made of
/// thin parts (spectacle frames) thus gets more quads than its area share, and the rest of the
/// target is renormalized.
pub fn thickness_plan(
    meshes: &[&TriMesh],
    measure: &[bool],
    ring_min: u32,
    n: f64,
    budget: f64,
    floor_min: f64,
    ctl: &Ctl,
) -> Result<ThickPlan, AqError> {
    let nc = meshes.len();
    let mut av: Vec<f64> = Vec::new();
    let mut raw: Vec<f64> = Vec::new();
    let mut group: Vec<u32> = Vec::new();
    let mut areas = vec![0.0f64; nc];
    for (c, m) in meshes.iter().enumerate() {
        ctl.check()?;
        let fa: Vec<f64> = (0..m.tris.len() as u32).map(|f| m.face_area(f)).collect();
        areas[c] = fa.iter().sum();
        if !(measure.get(c).copied().unwrap_or(false) && ring_min > 0) {
            av.push(areas[c]);
            raw.push(f64::INFINITY);
            group.push(c as u32);
            continue;
        }
        let sm = subdivide_to(m, 0.5, (PLAN_TRIS_PER_H2 * areas[c]) as usize);
        let fa: Vec<f64> = (0..sm.tris.len() as u32).map(|f| sm.face_area(f)).collect();
        let fnrm: Vec<V3> =
            (0..sm.tris.len() as u32).map(|f| math::normalize(sm.face_cross(f)).unwrap_or([0.0; 3])).collect();
        let d = shape_diameter_mesh(&sm, &fnrm, &fa, ctl)?;
        let mut w = vec![0.0f64; sm.pos.len()];
        for (f, t) in sm.tris.iter().enumerate() {
            for &v in t {
                w[v as usize] += fa[f] / 3.0;
            }
        }
        for v in 0..sm.pos.len() {
            av.push(w[v]);
            raw.push(std::f64::consts::PI * d[v] / ring_min as f64);
            group.push(c as u32);
        }
    }
    let ones = vec![1.0; av.len()];
    let floor = crate::thickness_floor(&av, &ones, &[], &raw, n, budget, ctl)?.max(floor_min);
    let (demand, scale) = if floor.is_finite() {
        crate::thickness_demand(&av, &raw, &group, nc, floor, n)
            .ok_or_else(|| crate::bad("thickness_plan: the caps exceed the target"))?
    } else {
        let tot: f64 = areas.iter().sum();
        (areas.iter().map(|a| n * a / tot).collect(), (tot / n).sqrt())
    };
    let mut thin_area = vec![0.0f64; nc];
    for v in 0..av.len() {
        if raw[v].max(floor) < scale {
            thin_area[group[v] as usize] += av[v];
        }
    }
    Ok(ThickPlan { floor, demand, thin_area, scale })
}

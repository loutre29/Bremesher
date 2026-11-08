//! Synthetic triangle meshes (DESIGN 6.2 "Generators"). All CCW seen from outside (C1), all
//! deterministic. Jitter is a fraction of the local edge length and never moves a boundary
//! vertex off its boundary curve (it only slides along it), so boundaries stay straight lines
//! or exact circles.

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use aq_core::math;
use aq_core::{AqError, FeatureKind, SplitMix64, TriMesh, V3};

use crate::util::{bad, rot2, vertex_normals};

/// A mesh with feature edges in the form `aq_prep::ingest` takes them
/// (`feat_pairs[i]` is a vertex pair, `feat_kind[i]` its [`FeatureKind`] as u8).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SynthMesh {
    pub mesh: TriMesh,
    pub feat_pairs: Vec<[u32; 2]>,
    pub feat_kind: Vec<u8>,
}

impl SynthMesh {
    /// Feature edges as (pair, kind) for [`crate::Surface::with_features`].
    pub fn feature_edges(&self) -> Vec<([u32; 2], FeatureKind)> {
        self.feat_pairs
            .iter()
            .zip(&self.feat_kind)
            .map(|(p, &k)| (*p, FeatureKind::from_u8(k).unwrap_or(FeatureKind::Hard)))
            .collect()
    }
}

fn check_jitter(j: f64) -> Result<(), AqError> {
    if (0.0..0.5).contains(&j) { Ok(()) } else { Err(bad(format!("jitter {j} outside [0, 0.5)"))) }
}

fn check_pos(name: &str, x: f64) -> Result<(), AqError> {
    if x.is_finite() && x > 0.0 { Ok(()) } else { Err(bad(format!("{name} = {x} must be > 0"))) }
}

/// Splits the grid cell (a, b, c, d) (CCW) into two triangles, alternating the diagonal.
fn push_quad(tris: &mut Vec<[u32; 3]>, q: [u32; 4], alt: bool) {
    let [a, b, c, d] = q;
    if alt {
        tris.push([a, b, d]);
        tris.push([b, c, d]);
    } else {
        tris.push([a, b, c]);
        tris.push([a, c, d]);
    }
}

/// Planar grid in z = 0 with `nx` x `ny` square cells of size `cell`, its corner at the origin,
/// rotated by `angle` about +z. Interior vertices are jittered by up to `jitter` (fraction of
/// the cell, in [0, 0.5)); boundary vertices stay on the four straight sides (they slide along
/// them by the same amount, corners fixed). Diagonals alternate. Normal +z.
pub fn plane_grid(nx: u32, ny: u32, cell: f64, angle: f64, jitter: f64, seed: u64) -> Result<TriMesh, AqError> {
    if nx == 0 || ny == 0 {
        return Err(bad("plane_grid needs at least one cell"));
    }
    check_pos("cell", cell)?;
    check_jitter(jitter)?;
    let mut rng = SplitMix64::new(seed);
    let mut pos = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
    for j in 0..=ny {
        for i in 0..=nx {
            let mut x = i as f64;
            let mut y = j as f64;
            let jx = jitter * rng.range_f64(-1.0, 1.0);
            let jy = jitter * rng.range_f64(-1.0, 1.0);
            if i > 0 && i < nx {
                x += jx;
            }
            if j > 0 && j < ny {
                y += jy;
            }
            let p = rot2(angle, [x * cell, y * cell]);
            pos.push([p[0], p[1], 0.0]);
        }
    }
    let id = |i: u32, j: u32| j * (nx + 1) + i;
    let mut tris = Vec::with_capacity((2 * nx * ny) as usize);
    for j in 0..ny {
        for i in 0..nx {
            push_quad(&mut tris, [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], (i + j) % 2 == 1);
        }
    }
    Ok(TriMesh { pos, tris })
}

/// Planar square disk `[0, side]^2` with `n` x `n` cells (see [`plane_grid`]). Its boundary is
/// one loop with 4 corners of 90 degrees ([`crate::Surface::new`] turns it into one Boundary
/// chain with 4 corners).
pub fn square_disk(n: u32, side: f64, jitter: f64, seed: u64) -> Result<TriMesh, AqError> {
    check_pos("side", side)?;
    plane_grid(n, n, side / n.max(1) as f64, 0.0, jitter, seed)
}

/// Height field `z = f(x, y)` over `[-extent, extent]^2` with `n` x `n` cells (no jitter).
pub fn heightfield(n: u32, extent: f64, f: impl Fn(f64, f64) -> f64) -> Result<TriMesh, AqError> {
    check_pos("extent", extent)?;
    let mut m = plane_grid(n, n, 2.0 * extent / n.max(1) as f64, 0.0, 0.0, 0)?;
    for p in m.pos.iter_mut() {
        p[0] -= extent;
        p[1] -= extent;
        p[2] = f(p[0], p[1]);
    }
    Ok(m)
}

/// UV sphere: `n_lat` latitude bands (>= 2), `n_lon` longitudes (>= 3), two poles (+z, -z).
pub fn uv_sphere(n_lat: u32, n_lon: u32, radius: f64) -> Result<TriMesh, AqError> {
    if n_lat < 2 || n_lon < 3 {
        return Err(bad("uv_sphere needs n_lat >= 2 and n_lon >= 3"));
    }
    check_pos("radius", radius)?;
    let mut pos = vec![[0.0, 0.0, radius]];
    for i in 1..n_lat {
        let t = PI * i as f64 / n_lat as f64;
        for j in 0..n_lon {
            let p = TAU * j as f64 / n_lon as f64;
            pos.push([radius * t.sin() * p.cos(), radius * t.sin() * p.sin(), radius * t.cos()]);
        }
    }
    pos.push([0.0, 0.0, -radius]);
    let south = (pos.len() - 1) as u32;
    let ring = |i: u32, j: u32| 1 + (i - 1) * n_lon + (j % n_lon);
    let mut tris = Vec::new();
    for j in 0..n_lon {
        tris.push([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..n_lat - 1 {
        for j in 0..n_lon {
            let (a, b, c, d) = (ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1));
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    for j in 0..n_lon {
        tris.push([south, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)]);
    }
    Ok(TriMesh { pos, tris })
}

/// Icosphere: icosahedron subdivided `level` times (20 * 4^level faces), on a sphere of
/// `radius` centered at the origin.
pub fn icosphere(level: u32, radius: f64) -> Result<TriMesh, AqError> {
    check_pos("radius", radius)?;
    if level > 9 {
        return Err(bad("icosphere level > 9"));
    }
    let g = (1.0 + 5f64.sqrt()) / 2.0;
    let base: [V3; 12] = [
        [-1.0, g, 0.0],
        [1.0, g, 0.0],
        [-1.0, -g, 0.0],
        [1.0, -g, 0.0],
        [0.0, -1.0, g],
        [0.0, 1.0, g],
        [0.0, -1.0, -g],
        [0.0, 1.0, -g],
        [g, 0.0, -1.0],
        [g, 0.0, 1.0],
        [-g, 0.0, -1.0],
        [-g, 0.0, 1.0],
    ];
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let onto = |p: V3| math::scale(math::normalize(p).unwrap_or([0.0, 0.0, 1.0]), radius);
    let mut pos: Vec<V3> = base.iter().map(|&p| onto(p)).collect();
    for _ in 0..level {
        let mut cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut nt = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let mut mid = |x: u32, y: u32| -> u32 {
                let key = (x.min(y), x.max(y));
                *cache.entry(key).or_insert_with(|| {
                    pos.push(onto(math::add(pos[x as usize], pos[y as usize])));
                    (pos.len() - 1) as u32
                })
            };
            let ab = mid(a, b);
            let bc = mid(b, c);
            let ca = mid(c, a);
            nt.push([a, ab, ca]);
            nt.push([b, bc, ab]);
            nt.push([c, ca, bc]);
            nt.push([ab, bc, ca]);
        }
        tris = nt;
    }
    Ok(TriMesh { pos, tris })
}

/// Octahedral lattice sphere: each octant face of the octahedron split into n^2 triangles and
/// projected to the sphere. With `upper_only`, only z >= 0 (the equator is then the boundary).
fn octa_lattice(n: u32, radius: f64, upper_only: bool) -> Result<(TriMesh, Vec<[i64; 3]>), AqError> {
    if n == 0 {
        return Err(bad("octasphere needs n >= 1"));
    }
    check_pos("radius", radius)?;
    let ni = n as i64;
    let mut ids: BTreeMap<[i64; 3], u32> = BTreeMap::new();
    let mut keys: Vec<[i64; 3]> = Vec::new();
    let mut tris = Vec::new();
    let szs: &[i64] = if upper_only { &[1] } else { &[1, -1] };
    for &sz in szs {
        for &(sx, sy) in &[(1i64, 1i64), (-1, 1), (-1, -1), (1, -1)] {
            let mut vid = |i: i64, j: i64| -> u32 {
                let k = [sx * (ni - i - j), sy * i, sz * j];
                *ids.entry(k).or_insert_with(|| {
                    keys.push(k);
                    (keys.len() - 1) as u32
                })
            };
            for i in 0..ni {
                for j in 0..ni - i {
                    let t = [vid(i, j), vid(i + 1, j), vid(i, j + 1)];
                    tris.push(t);
                    if i + j < ni - 1 {
                        let t = [vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)];
                        tris.push(t);
                    }
                }
            }
        }
    }
    let pos: Vec<V3> = keys
        .iter()
        .map(|k| {
            let p = [k[0] as f64, k[1] as f64, k[2] as f64];
            math::scale(math::normalize(p).unwrap_or([0.0, 0.0, 1.0]), radius)
        })
        .collect();
    let mut m = TriMesh { pos, tris };
    // Orient every face outward (normal . centroid > 0).
    for f in 0..m.tris.len() {
        let c = m.face_cross(f as u32);
        let [a, b, d] = m.tri_pos(f as u32);
        let g = math::add(math::add(a, b), d);
        if math::dot(c, g) < 0.0 {
            m.tris[f].swap(1, 2);
        }
    }
    Ok((m, keys))
}

/// Full octahedral sphere (8 n^2 faces). Its great circles z = 0, x = 0 and y = 0 run along
/// mesh edges.
pub fn octasphere(n: u32, radius: f64) -> Result<TriMesh, AqError> {
    Ok(octa_lattice(n, radius, false)?.0)
}

/// Hemisphere z >= 0 of radius `radius` (4 n^2 faces, octahedral lattice): a bisected sphere
/// whose cut chain, the equator (4n edges, one closed loop, no corner), is flagged
/// [`FeatureKind::Symmetry`].
pub fn hemisphere(n: u32, radius: f64) -> Result<SynthMesh, AqError> {
    let (mesh, keys) = octa_lattice(n, radius, true)?;
    let mut pairs = std::collections::BTreeSet::new();
    for t in &mesh.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if keys[a as usize][2] == 0 && keys[b as usize][2] == 0 {
                pairs.insert([a.min(b), a.max(b)]);
            }
        }
    }
    let feat_pairs: Vec<[u32; 2]> = pairs.into_iter().collect();
    let feat_kind = vec![FeatureKind::Symmetry as u8; feat_pairs.len()];
    Ok(SynthMesh { mesh, feat_pairs, feat_kind })
}

/// Side grid of a cylinder of `radius` along +z on [0, length]: `n_around` x `n_along` cells.
/// Returns positions and the vertex id function layout (rings of n_around vertices).
fn cylinder_side(
    radius: f64,
    length: f64,
    n_around: u32,
    n_along: u32,
    jitter: f64,
    seed: u64,
) -> Result<(Vec<V3>, Vec<[u32; 3]>), AqError> {
    if n_around < 3 || n_along == 0 {
        return Err(bad("cylinder needs n_around >= 3 and n_along >= 1"));
    }
    check_pos("radius", radius)?;
    check_pos("length", length)?;
    check_jitter(jitter)?;
    let mut rng = SplitMix64::new(seed);
    let mut pos = Vec::new();
    for j in 0..=n_along {
        for i in 0..n_around {
            let ja = jitter * rng.range_f64(-1.0, 1.0);
            let jz = jitter * rng.range_f64(-1.0, 1.0);
            let a = TAU * (i as f64 + ja) / n_around as f64;
            let mut z = j as f64;
            if j > 0 && j < n_along {
                z += jz;
            }
            pos.push([radius * a.cos(), radius * a.sin(), length * z / n_along as f64]);
        }
    }
    let id = |i: u32, j: u32| j * n_around + (i % n_around);
    let mut tris = Vec::new();
    for j in 0..n_along {
        for i in 0..n_around {
            push_quad(&mut tris, [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], (i + j) % 2 == 1);
        }
    }
    Ok((pos, tris))
}

/// Open cylinder of `radius` along +z on [0, length], `n_around` x `n_along` cells, with
/// triangulation jitter (fraction of a cell; the rim vertices slide along their circles).
/// chi = 0, 2 boundary loops.
pub fn cylinder_open(radius: f64, length: f64, n_around: u32, n_along: u32, jitter: f64, seed: u64) -> Result<TriMesh, AqError> {
    let (pos, tris) = cylinder_side(radius, length, n_around, n_along, jitter, seed)?;
    Ok(TriMesh { pos, tris })
}

/// Zips two concentric closed rings (vertex ids with their angles, both increasing over one
/// turn) into a triangle band. `inner` may be a single centre vertex.
fn zip_rings(tris: &mut Vec<[u32; 3]>, inner: &[(u32, f64)], outer: &[(u32, f64)]) {
    let (a, b) = (inner.len(), outer.len());
    let ang = |r: &[(u32, f64)], i: usize| -> f64 {
        let n = r.len();
        if n == 1 {
            return f64::INFINITY;
        }
        r[i % n].1 + TAU * (i / n) as f64
    };
    let (mut i, mut j) = (0usize, 0usize);
    while i < a || j < b {
        let adv_inner = a > 1 && i < a && (j >= b || ang(inner, i + 1) < ang(outer, j + 1));
        if adv_inner {
            tris.push([inner[i % a].0, inner[(i + 1) % a].0, outer[j % b].0]);
            i += 1;
        } else {
            tris.push([inner[i % a].0, outer[(j + 1) % b].0, outer[j % b].0]);
            j += 1;
        }
        if a == 1 && j >= b {
            break;
        }
    }
}

/// Closed cylinder: the open cylinder plus two flat caps made of concentric rings (radial
/// spacing close to the circumferential one). chi = 2. Cap interior vertices are jittered in
/// their plane.
pub fn cylinder_closed(radius: f64, length: f64, n_around: u32, n_along: u32, jitter: f64, seed: u64) -> Result<TriMesh, AqError> {
    let (mut pos, mut tris) = cylinder_side(radius, length, n_around, n_along, jitter, seed)?;
    let mut rng = SplitMix64::new(seed ^ 0xC0FF_EE00_D15C_0001);
    let nr = ((n_around as f64 / TAU).round() as u32).max(1);
    for cap in 0..2u32 {
        let z = if cap == 0 { 0.0 } else { length };
        let rim_j = if cap == 0 { 0 } else { n_along };
        let side_n = n_around;
        let start = tris.len();
        // Rim ring: the side's first or last ring, with its actual angles.
        let mut rim: Vec<(u32, f64)> = (0..side_n)
            .map(|i| {
                let v = rim_j * side_n + i;
                let p = pos[v as usize];
                (v, p[1].atan2(p[0]))
            })
            .collect();
        // Angles increasing over one turn starting at vertex 0 (angle near 0).
        let a0 = rim[0].1;
        for r in rim.iter_mut() {
            r.1 = (r.1 - a0).rem_euclid(TAU) + a0;
        }
        let mut outer = rim;
        for k in (0..nr).rev() {
            let inner: Vec<(u32, f64)> = if k == 0 {
                pos.push([0.0, 0.0, z]);
                vec![((pos.len() - 1) as u32, 0.0)]
            } else {
                let cnt = ((n_around as f64 * k as f64 / nr as f64).round() as u32).max(3);
                let rr = radius * k as f64 / nr as f64;
                let phase = 0.5 * TAU / cnt as f64 * 0.37;
                (0..cnt)
                    .map(|i| {
                        let a = phase + TAU * i as f64 / cnt as f64;
                        let ja = jitter * rng.range_f64(-1.0, 1.0) * TAU / cnt as f64;
                        let jr = jitter * rng.range_f64(-1.0, 1.0) * radius / nr as f64;
                        let (aa, r2) = (a + ja, rr + jr);
                        pos.push([r2 * aa.cos(), r2 * aa.sin(), z]);
                        ((pos.len() - 1) as u32, a)
                    })
                    .collect()
            };
            zip_rings(&mut tris, &inner, &outer);
            outer = inner;
        }
        // Orient the cap: normal -z at the bottom, +z at the top.
        let want = if cap == 0 { -1.0 } else { 1.0 };
        for t in tris[start..].iter_mut() {
            let [p0, p1, p2] = [pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]];
            let c = math::cross(math::sub(p1, p0), math::sub(p2, p0));
            if c[2] * want < 0.0 {
                t.swap(1, 2);
            }
        }
    }
    Ok(TriMesh { pos, tris })
}

/// Torus about the z axis with major radius `big_r` and minor radius `r`: `nu` cells around
/// z, `nv` around the tube, jittered in both angles (fraction of a cell). chi = 0.
pub fn torus(big_r: f64, r: f64, nu: u32, nv: u32, jitter: f64, seed: u64) -> Result<TriMesh, AqError> {
    if nu < 3 || nv < 3 {
        return Err(bad("torus needs nu, nv >= 3"));
    }
    check_pos("r", r)?;
    if big_r.partial_cmp(&r) != Some(std::cmp::Ordering::Greater) {
        return Err(bad("torus needs big_r > r"));
    }
    check_jitter(jitter)?;
    let mut rng = SplitMix64::new(seed);
    let mut pos = Vec::with_capacity((nu * nv) as usize);
    for i in 0..nu {
        for j in 0..nv {
            let a = TAU * (i as f64 + jitter * rng.range_f64(-1.0, 1.0)) / nu as f64;
            let b = TAU * (j as f64 + jitter * rng.range_f64(-1.0, 1.0)) / nv as f64;
            let rr = big_r + r * b.cos();
            pos.push([rr * a.cos(), rr * a.sin(), r * b.sin()]);
        }
    }
    let id = |i: u32, j: u32| (i % nu) * nv + (j % nv);
    let mut tris = Vec::with_capacity((2 * nu * nv) as usize);
    for i in 0..nu {
        for j in 0..nv {
            push_quad(&mut tris, [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], (i + j) % 2 == 1);
        }
    }
    Ok(TriMesh { pos, tris })
}

/// The 6 cube faces as (normal axis, at max, u axis, v axis) with u x v = outward normal.
pub(crate) const CUBE_FACES: [(usize, bool, usize, usize); 6] = [
    (0, true, 1, 2),
    (0, false, 2, 1),
    (1, true, 2, 0),
    (1, false, 0, 2),
    (2, true, 0, 1),
    (2, false, 1, 0),
];

/// Integer lattice of the cube surface [0, l]^3 with l x l squares per face (2 triangles
/// each, diagonals alternating), outward CCW. Returns the lattice keys, triangles and the
/// cube face (0..6, [`CUBE_FACES`]) of each triangle.
pub(crate) fn cube_lattice(l: i64) -> (Vec<[i64; 3]>, Vec<[u32; 3]>, Vec<u8>) {
    let mut ids: BTreeMap<[i64; 3], u32> = BTreeMap::new();
    let mut keys = Vec::new();
    let mut tris = Vec::new();
    let mut tri_face = Vec::new();
    for (fi, &(na, at_max, ua, va)) in CUBE_FACES.iter().enumerate() {
        for i in 0..l {
            for j in 0..l {
                let mut vid = |a: i64, b: i64| -> u32 {
                    let mut k = [0i64; 3];
                    k[na] = if at_max { l } else { 0 };
                    k[ua] = a;
                    k[va] = b;
                    *ids.entry(k).or_insert_with(|| {
                        keys.push(k);
                        (keys.len() - 1) as u32
                    })
                };
                let q = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)];
                push_quad(&mut tris, q, (i + j) % 2 == 1);
                tri_face.push(fi as u8);
                tri_face.push(fi as u8);
            }
        }
    }
    (keys, tris, tri_face)
}

/// Number of coordinates of a cube lattice key at 0 or l.
pub(crate) fn extreme_axes(k: &[i64; 3], l: i64) -> [bool; 3] {
    [k[0] == 0 || k[0] == l, k[1] == 0 || k[1] == l, k[2] == 0 || k[2] == l]
}

/// Cube surface `[-size/2, size/2]^3` with `n` x `n` cells per face and its 12 sharp edges as
/// [`FeatureKind::Hard`] feature edges (12 chains meeting at the 8 corners). Jitter moves face
/// interior vertices inside their face plane and cube-edge vertices along their edge, so faces
/// stay planar and edges straight; corners are exact.
pub fn cube_sharp(n: u32, size: f64, jitter: f64, seed: u64) -> Result<SynthMesh, AqError> {
    if n == 0 {
        return Err(bad("cube_sharp needs n >= 1"));
    }
    check_pos("size", size)?;
    check_jitter(jitter)?;
    let l = n as i64;
    let (keys, tris, _) = cube_lattice(l);
    let mut rng = SplitMix64::new(seed);
    let step = size / n as f64;
    let pos: Vec<V3> = keys
        .iter()
        .map(|k| {
            let ext = extreme_axes(k, l);
            let mut p = [0.0; 3];
            for a in 0..3 {
                let jit = jitter * rng.range_f64(-1.0, 1.0);
                let x = if ext[a] { k[a] as f64 } else { k[a] as f64 + jit };
                p[a] = x * step - 0.5 * size;
            }
            p
        })
        .collect();
    let mut pairs = std::collections::BTreeSet::new();
    for t in &tris {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            let (ka, kb) = (&keys[a as usize], &keys[b as usize]);
            let (ea, eb) = (extreme_axes(ka, l), extreme_axes(kb, l));
            let common = (0..3).filter(|&x| ea[x] && eb[x] && ka[x] == kb[x]).count();
            if common >= 2 {
                pairs.insert([a.min(b), a.max(b)]);
            }
        }
    }
    let feat_pairs: Vec<[u32; 2]> = pairs.into_iter().collect();
    let feat_kind = vec![FeatureKind::Hard as u8; feat_pairs.len()];
    Ok(SynthMesh { mesh: TriMesh { pos, tris }, feat_pairs, feat_kind })
}

/// Concatenates meshes (vertex ids offset), e.g. two parallel planes.
pub fn merge(meshes: &[TriMesh]) -> TriMesh {
    let mut out = TriMesh::default();
    for m in meshes {
        let off = out.pos.len() as u32;
        out.pos.extend_from_slice(&m.pos);
        out.tris.extend(m.tris.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    }
    out
}

/// Noise adder: displaces every vertex along its area-weighted normal by a smooth band-limited
/// noise of the given `wavelength`: `amp * sum_k sin(2 pi d_k . x / wavelength + phi_k) / sqrt(8)`
/// over 8 random unit directions d_k and phases phi_k (RMS amp/sqrt(2), like one sine of
/// amplitude amp). Deterministic in `seed`. Boundary vertices move too (along their normal).
pub fn add_noise(m: &mut TriMesh, amp: f64, wavelength: f64, seed: u64) -> Result<(), AqError> {
    if !amp.is_finite() {
        return Err(bad("noise amplitude not finite"));
    }
    check_pos("wavelength", wavelength)?;
    let mut rng = SplitMix64::new(seed);
    const K: usize = 8;
    let mut waves = [([0.0; 3], 0.0); K];
    for w in waves.iter_mut() {
        let z = rng.range_f64(-1.0, 1.0);
        let a = rng.range_f64(0.0, TAU);
        let s = (1.0 - z * z).max(0.0).sqrt();
        *w = ([s * a.cos(), s * a.sin(), z], rng.range_f64(0.0, TAU));
    }
    let n = vertex_normals(m);
    let k = TAU / wavelength;
    let norm = 1.0 / (K as f64).sqrt();
    for (p, nv) in m.pos.iter_mut().zip(&n) {
        let s: f64 = waves.iter().map(|(d, ph)| (k * math::dot(*d, *p) + ph).sin()).sum();
        *p = math::add(*p, math::scale(*nv, amp * norm * s));
    }
    Ok(())
}

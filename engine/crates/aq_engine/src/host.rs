//! Host-side steps around the engine call (DESIGN 9): what the add-on does before and after
//! `aq_remesh`, shared by the CLI and the tests so that both represent the add-on.
//!
//! * [`normalize`]: scale about the origin so the bounding box diagonal is 10 (the add-on's
//!   `WORK_DIAG`; it scales with `Matrix.Scale`, no translation, so symmetry planes through
//!   the origin stay in place).
//! * [`auto_features`]: edges with a dihedral angle above a threshold, as kind 1 (Hard).
//! * [`upsample`]: 1-to-4 subdivision with shared midpoints.
//! * [`reproject`]: every output vertex moved to its closest point on the input (BVH).

use aq_core::math;
use aq_core::{Bvh, QuadMesh, TriMesh};

/// The add-on's working bounding box diagonal.
pub const WORK_DIAG: f64 = 10.0;

/// Scales `m` about the origin so its bounding box diagonal is [`WORK_DIAG`]. Returns the
/// factor applied (1 when the diagonal is zero or not finite).
pub fn normalize(m: &mut TriMesh) -> f64 {
    let d = m.bbox_diag();
    if !(d.is_finite() && d > 0.0) {
        return 1.0;
    }
    let s = WORK_DIAG / d;
    for p in m.pos.iter_mut() {
        *p = math::scale(*p, s);
    }
    s
}

/// Multiplies every position of a quad mesh by `s` (the inverse of [`normalize`] is `1/s`).
pub fn scale_quads(q: &mut QuadMesh, s: f64) {
    for p in q.pos.iter_mut() {
        *p = math::scale(*p, s);
    }
}

/// Undirected edges with exactly two incident faces whose normals differ by more than
/// `angle_deg`, as sorted vertex pairs (kind 1, Hard). Edges of degenerate faces and
/// non-manifold edges are skipped.
pub fn auto_features(m: &TriMesh, angle_deg: f64) -> Vec<[u32; 2]> {
    let mut e: Vec<(u32, u32, u32)> = Vec::with_capacity(3 * m.tris.len());
    for (f, t) in m.tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if a != b {
                e.push((a.min(b), a.max(b), f as u32));
            }
        }
    }
    e.sort_unstable();
    // `TriMesh::face_normal` (C1) indexes without checks: faces with an out-of-range index
    // are skipped first.
    let nv = m.pos.len();
    let normal = |f: u32| -> Option<[f64; 3]> {
        let t = m.tris.get(f as usize)?;
        if t.iter().any(|&i| i as usize >= nv) {
            return None;
        }
        m.face_normal(f)
    };
    let cos_lim = angle_deg.to_radians().cos();
    let mut out = Vec::new();
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j].0 == e[i].0 && e[j].1 == e[i].1 {
            j += 1;
        }
        if j - i == 2
            && let (Some(n1), Some(n2)) = (normal(e[i].2), normal(e[i + 1].2))
            && math::dot(n1, n2) < cos_lim
        {
            out.push([e[i].0, e[i].1]);
        }
        i = j;
    }
    out
}

/// 1-to-4 subdivision, `k` times: each triangle (a, b, c) gets the midpoints of its edges
/// (shared between neighbours) and becomes (a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca).
/// Feature pairs on a subdivided edge are split into their two halves.
pub fn upsample(m: &TriMesh, feats: &[[u32; 2]], k: u32) -> (TriMesh, Vec<[u32; 2]>) {
    let (m, f, _) = upsample_with(m, feats, None, k);
    (m, f)
}

/// [`upsample`] that also carries a per-vertex density (a midpoint gets the mean of its edge
/// ends).
pub fn upsample_with(
    m: &TriMesh,
    feats: &[[u32; 2]],
    density: Option<&[f32]>,
    k: u32,
) -> (TriMesh, Vec<[u32; 2]>, Option<Vec<f32>>) {
    let mut cur = m.clone();
    let mut fe: Vec<[u32; 2]> = feats.to_vec();
    let mut dens: Option<Vec<f32>> = density.map(|d| d.to_vec());
    for _ in 0..k {
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(3 * cur.tris.len());
        for t in &cur.tris {
            for i in 0..3 {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                edges.push((a.min(b), a.max(b)));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let nv = cur.pos.len() as u32;
        let mut pos = cur.pos.clone();
        for &(a, b) in &edges {
            let pa = cur.pos.get(a as usize).copied().unwrap_or([0.0; 3]);
            let pb = cur.pos.get(b as usize).copied().unwrap_or([0.0; 3]);
            pos.push(math::lerp(pa, pb, 0.5));
        }
        let mid = |a: u32, b: u32| -> u32 {
            let key = (a.min(b), a.max(b));
            match edges.binary_search(&key) {
                Ok(i) => nv + i as u32,
                Err(_) => a,
            }
        };
        let mut tris = Vec::with_capacity(4 * cur.tris.len());
        for t in &cur.tris {
            let (a, b, c) = (t[0], t[1], t[2]);
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            tris.push([a, ab, ca]);
            tris.push([ab, b, bc]);
            tris.push([ca, bc, c]);
            tris.push([ab, bc, ca]);
        }
        let mut nfe = Vec::with_capacity(2 * fe.len());
        for &[a, b] in &fe {
            let key = (a.min(b), a.max(b));
            if let Ok(i) = edges.binary_search(&key) {
                let m = nv + i as u32;
                nfe.push([a, m]);
                nfe.push([m, b]);
            } else {
                nfe.push([a, b]);
            }
        }
        if let Some(d) = dens.as_mut() {
            for &(a, b) in &edges {
                let da = d.get(a as usize).copied().unwrap_or(1.0);
                let db = d.get(b as usize).copied().unwrap_or(1.0);
                d.push(0.5 * (da + db));
            }
        }
        cur = TriMesh { pos, tris };
        fe = nfe;
    }
    (cur, fe, dens)
}

/// Moves every vertex of `q` to its closest point on `input` (same units). Returns the
/// largest displacement.
pub fn reproject(q: &mut QuadMesh, input: &TriMesh) -> f64 {
    let bvh = Bvh::build(input);
    let mut maxd: f64 = 0.0;
    for p in q.pos.iter_mut() {
        let (sp, c, d) = bvh.closest(*p);
        if sp.face != aq_core::NONE && math::is_finite3(c) {
            *p = c;
            maxd = maxd.max(d);
        }
    }
    maxd
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn cube() -> TriMesh {
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let q = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        let mut tris = Vec::new();
        for f in q {
            tris.push([f[0], f[1], f[2]]);
            tris.push([f[0], f[2], f[3]]);
        }
        TriMesh { pos, tris }
    }

    #[test]
    fn normalize_to_diag_10() {
        let mut m = cube();
        let s = normalize(&mut m);
        assert!((m.bbox_diag() - 10.0).abs() < 1e-12);
        assert!((s - 10.0 / 3f64.sqrt()).abs() < 1e-12);
        assert_eq!(m.pos[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn cube_auto_features() {
        let f = auto_features(&cube(), 30.0);
        assert_eq!(f.len(), 12);
        assert!(auto_features(&cube(), 95.0).is_empty());
    }

    #[test]
    fn upsample_counts() {
        let m = cube();
        let fe = auto_features(&m, 30.0);
        let (u, ufe) = upsample(&m, &fe, 2);
        assert_eq!(u.tris.len(), 12 * 16);
        assert_eq!(ufe.len(), 12 * 4);
        u.validate().unwrap();
        let t = aq_core::Topology::build(&u).unwrap();
        assert_eq!(t.euler_char(&u, 0), 2);
        assert_eq!(auto_features(&u, 30.0).len(), 48);
    }

    #[test]
    fn reproject_onto_plane() {
        let m = TriMesh {
            pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            tris: vec![[0, 1, 2]],
        };
        let mut q = QuadMesh::new();
        q.pos = vec![[0.1, 0.1, 0.5]];
        q.vflag = vec![0];
        let d = reproject(&mut q, &m);
        assert!((d - 0.5).abs() < 1e-12);
        assert_eq!(q.pos[0], [0.1, 0.1, 0.0]);
    }
}

//! S12 merge (DESIGN 7.1 step 7): offset ids, set face_comp and vflag, scale back by h0.

use aq_core::{AqError, QuadMesh, Stage};

/// vflag bits (DESIGN 8.4).
pub const VFLAG_CHAIN_MASK: u32 = (1 << 20) - 1;
pub const VFLAG_KIND_SHIFT: u32 = 20;
pub const VFLAG_KIND_MASK: u32 = 0xF << VFLAG_KIND_SHIFT;
pub const VFLAG_CORNER: u32 = 1 << 24;
pub const VFLAG_CONE: u32 = 1 << 25;
pub const VFLAG_BOUNDARY: u32 = 1 << 26;
pub const VFLAG_SF: u32 = 1 << 27;
/// Bits 28..31 are reserved and always 0 in the output.
pub const VFLAG_RESERVED: u32 = 0xF << 28;

/// What [`merge`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    /// Global chain ids handed out (the number of chains in the output).
    pub chains: u32,
    /// Vertices whose chain id did not fit in 20 bits (their chain field is set to 0).
    pub chain_overflow: u32,
}

/// Merges component meshes (in quad units) into one mesh in input units.
///
/// `parts[i] = (component id, mesh)`. Vertex and face ids are offset in order. Every face of
/// part i gets `face_comp = component id`. The chain field of vflag (bits 0..19, component-local
/// chain id + 1) is renumbered globally: the chains of a part are offset by the number of chain
/// ids used by the parts before it (its largest local id + 1). Reserved bits are cleared.
/// Positions are multiplied by `h0`.
pub fn merge(parts: &[(u32, &QuadMesh)], h0: f64) -> Result<(QuadMesh, MergeReport), AqError> {
    let nv: usize = parts.iter().map(|(_, q)| q.pos.len()).sum();
    let nf: usize = parts.iter().map(|(_, q)| q.n_faces()).sum();
    let nfv: usize = parts.iter().map(|(_, q)| q.face_verts.len()).sum();
    let st = |e: AqError| e.at(Stage::Engine);
    let mut out = QuadMesh {
        pos: aq_core::alloc::try_with_capacity(nv).map_err(st)?,
        face_off: aq_core::alloc::try_with_capacity(nf + 1).map_err(st)?,
        face_verts: aq_core::alloc::try_with_capacity(nfv).map_err(st)?,
        vflag: aq_core::alloc::try_with_capacity(nv).map_err(st)?,
        face_comp: aq_core::alloc::try_with_capacity(nf).map_err(st)?,
    };
    out.face_off.push(0);
    let mut rep = MergeReport::default();
    let mut chain_base: u64 = 0;
    for &(cid, q) in parts {
        if q.n_faces() + 1 != q.face_off.len().max(1) || q.vflag.len() != q.pos.len() {
            return Err(
                AqError::invariant(format!("merge: part {cid} has inconsistent arrays"))
                    .at(Stage::Engine),
            );
        }
        let v_off = out.pos.len() as u64;
        if v_off + q.pos.len() as u64 >= u32::MAX as u64 {
            return Err(AqError::budget("merge: more than 2^32 vertices").at(Stage::Engine));
        }
        let f_off = out.face_verts.len() as u64;
        let mut local_max = 0u32;
        for (&p, &fl) in q.pos.iter().zip(&q.vflag) {
            out.pos.push([p[0] * h0, p[1] * h0, p[2] * h0]);
            let local = fl & VFLAG_CHAIN_MASK;
            local_max = local_max.max(local);
            let mut g = fl & !(VFLAG_CHAIN_MASK | VFLAG_RESERVED);
            if local != 0 {
                let id = chain_base + local as u64;
                if id <= VFLAG_CHAIN_MASK as u64 {
                    g |= id as u32;
                } else {
                    rep.chain_overflow += 1;
                }
            }
            out.vflag.push(g);
        }
        for f in 0..q.n_faces() {
            for &v in q.face(f) {
                out.face_verts.push((v as u64 + v_off) as u32);
            }
            let end = f_off + q.face_off[f + 1] as u64;
            if end >= u32::MAX as u64 {
                return Err(AqError::budget("merge: more than 2^32 face corners").at(Stage::Engine));
            }
            out.face_off.push(end as u32);
            out.face_comp.push(cid);
        }
        chain_base += local_max as u64;
    }
    rep.chains = chain_base.min(u32::MAX as u64) as u32;
    Ok((out, rep))
}

/// Number of vertices used by at least one face (the V of the Euler characteristic).
pub fn used_vertices(q: &QuadMesh) -> usize {
    let mut used = vec![false; q.pos.len()];
    for &v in &q.face_verts {
        if let Some(u) = used.get_mut(v as usize) {
            *u = true;
        }
    }
    used.iter().filter(|&&u| u).count()
}

/// Euler characteristic V - E + F of a polygon mesh (V counts used vertices only).
pub fn euler_char(q: &QuadMesh) -> i64 {
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(q.face_verts.len());
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        for i in 0..fv.len() {
            let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    used_vertices(q) as i64 - edges.len() as i64 + q.n_faces() as i64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn square(z: f64, flags: [u32; 4]) -> QuadMesh {
        let mut q = QuadMesh::new();
        q.pos = vec![[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]];
        q.vflag = flags.to_vec();
        q.push_face(&[0, 1, 2, 3], 0);
        q
    }

    #[test]
    fn offsets_comp_and_scale() {
        let a = square(0.0, [0; 4]);
        let b = square(1.0, [0; 4]);
        let (m, rep) = merge(&[(0, &a), (3, &b)], 2.0).unwrap();
        m.validate().unwrap();
        assert_eq!(m.n_faces(), 2);
        assert_eq!(m.face(1), &[4, 5, 6, 7]);
        assert_eq!(m.face_comp, vec![0, 3]);
        assert_eq!(m.pos[6], [2.0, 2.0, 2.0]);
        assert_eq!(rep.chains, 0);
        assert_eq!(euler_char(&m), 2);
    }

    #[test]
    fn chain_renumbering() {
        let kind = 1u32 << VFLAG_KIND_SHIFT;
        let a = square(
            0.0,
            [1 | kind, 2 | kind | VFLAG_CORNER, 0, VFLAG_SF | (1 << 30)],
        );
        let b = square(1.0, [1 | kind, 0, 3 | VFLAG_BOUNDARY, 0]);
        let (m, rep) = merge(&[(0, &a), (1, &b)], 1.0).unwrap();
        assert_eq!(rep.chains, 2 + 3);
        assert_eq!(m.vflag[0], 1 | kind);
        assert_eq!(m.vflag[1], 2 | kind | VFLAG_CORNER);
        // Reserved bit 30 is cleared.
        assert_eq!(m.vflag[3], VFLAG_SF);
        assert_eq!(m.vflag[4], 3 | kind);
        assert_eq!(m.vflag[6], 5 | VFLAG_BOUNDARY);
    }

    #[test]
    fn chain_overflow_is_counted() {
        let a = square(0.0, [VFLAG_CHAIN_MASK, 0, 0, 0]);
        let b = square(1.0, [1, 0, 0, 0]);
        let (m, rep) = merge(&[(0, &a), (1, &b)], 1.0).unwrap();
        assert_eq!(rep.chain_overflow, 1);
        assert_eq!(m.vflag[4] & VFLAG_CHAIN_MASK, 0);
    }

    #[test]
    fn empty_merge() {
        let (m, rep) = merge(&[], 1.0).unwrap();
        m.validate().unwrap();
        assert_eq!(m.n_faces(), 0);
        assert_eq!(rep, MergeReport::default());
    }
}

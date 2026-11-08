//! S1 helper: one [`ComponentMesh`] per connected component of an [`Ingested`] mesh.

use aq_core::{FeatureKind, NONE, Topology, TriMesh};

use crate::chains::{CornerRule, build_feature_set};
use crate::{ComponentMesh, Ingested};

/// Splits the ingested mesh into its connected components, in the order of
/// `Topology::face_comp` (numbered by their lowest face). Faces keep their relative order and
/// corner order, so per-halfedge feature kinds carry over unchanged; vertices are renumbered
/// in increasing ingested id. `face_src` and `vert_src` map to the **input** ids. The chains
/// of each component are rebuilt with the ingest corner rule (per-vertex turning angle,
/// 50 / 70 degrees). A component whose topology cannot be rebuilt (impossible for a valid
/// `Ingested`) is skipped.
pub fn split_components(ing: &Ingested) -> Vec<ComponentMesh> {
    let m = &ing.mesh;
    let t = &ing.topo;
    let nc = t.n_comp as usize;
    let mut faces: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for (f, &c) in t.face_comp.iter().enumerate() {
        if (c as usize) < nc {
            faces[c as usize].push(f as u32);
        }
    }
    let mut vmap = vec![NONE; m.pos.len()];
    let mut out = Vec::with_capacity(nc);
    for fl in faces {
        let mut verts: Vec<u32> = fl.iter().flat_map(|&f| m.tris[f as usize]).collect();
        verts.sort_unstable();
        verts.dedup();
        for (i, &v) in verts.iter().enumerate() {
            vmap[v as usize] = i as u32;
        }
        let pos = verts.iter().map(|&v| m.pos[v as usize]).collect();
        let tris: Vec<[u32; 3]> = fl
            .iter()
            .map(|&f| {
                let tr = m.tris[f as usize];
                [vmap[tr[0] as usize], vmap[tr[1] as usize], vmap[tr[2] as usize]]
            })
            .collect();
        let mesh = TriMesh { pos, tris };
        let Ok(topo) = Topology::build(&mesh) else { continue };
        let mut he_kind = vec![FeatureKind::None; 3 * fl.len()];
        for (i, &f) in fl.iter().enumerate() {
            for k in 0..3 {
                he_kind[3 * i + k] = ing.feats.he_kind.get((3 * f) as usize + k).copied().unwrap_or_default();
            }
        }
        let zero = |_: u32| 0.0;
        let rule = CornerRule { corner_deg: 50.0, sym_corner_deg: 70.0, window: &zero, fixed: None };
        let Ok(feats) = build_feature_set(&mesh, &topo, he_kind, &rule) else { continue };
        let face_src = fl.iter().map(|&f| ing.face_src.get(f as usize).copied().unwrap_or(NONE)).collect();
        let vert_src = verts.iter().map(|&v| ing.vert_src.get(v as usize).copied().unwrap_or(NONE)).collect();
        let area = mesh.total_area();
        let chi = topo.euler_char(&mesh, 0);
        let closed = topo.boundary_loops.is_empty();
        out.push(ComponentMesh { mesh, topo, feats, face_src, vert_src, area, chi, closed });
    }
    out
}

//! `autoquad remesh`: what the add-on does before and after the engine call (DESIGN 9), so the
//! results represent the add-on.
//!
//! 1. Read the OBJ (polygons fan-triangulated), optionally `--upsample K` (1-to-4, K times).
//! 2. Normalize to a bounding box diagonal of 10 about the origin (the add-on's WORK_DIAG).
//! 3. `--sym`: bisect at the planes through the origin (`aq_eval::bisect`), keep the positive
//!    side unless `--keep-neg`; the cut edges become kind 5 (Symmetry); the target is divided
//!    by 2 per axis (at least 4), as in the add-on.
//! 4. Features: `auto` marks edges with a dihedral above `--hard-angle` as kind 1, `none`, or a
//!    file of pairs.
//! 5. The engine call (in process, `aq_engine::remesh`).
//! 6. `--sym`: snap boundary vertices within 0.3 h of each plane onto it (the add-on's
//!    `snap_seam`), then mirror and weld (`aq_eval::mirror_weld`).
//! 7. `--reproject`: project the vertices onto the (normalized, full) input with the BVH.
//! 8. Scale back, write the OBJ, and the report with its stats block.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};

use aq_core::{CtlState, QuadMesh, TriMesh, V3};
use aq_engine::{Input, Params, Status, host};
use aq_eval::{WeldReport, bisect, mirror_weld};

use crate::stats;

/// Feature source.
#[derive(Clone, Debug, PartialEq)]
pub enum Features {
    None,
    Auto,
    File(PathBuf),
}

/// Options of one remesh run.
#[derive(Clone, Debug)]
pub struct RemeshOpts {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    /// Engine parameters; `target_quads` is the full target (before the symmetry split).
    pub params: Params,
    pub features: Features,
    pub hard_angle: f64,
    pub density: Option<PathBuf>,
    /// Symmetry axes (0 x, 1 y, 2 z).
    pub sym: Vec<usize>,
    pub keep_neg: bool,
    pub upsample: u32,
    pub reproject: bool,
    pub report: Option<PathBuf>,
    /// Loop-closure axis for the stats block.
    pub axis: Option<V3>,
}

impl RemeshOpts {
    pub fn new(input: PathBuf) -> RemeshOpts {
        RemeshOpts {
            input,
            output: None,
            params: Params::default(),
            features: Features::None,
            hard_angle: 30.0,
            density: None,
            sym: Vec::new(),
            keep_neg: false,
            upsample: 0,
            reproject: false,
            report: None,
            axis: None,
        }
    }
}

/// Result of one remesh run.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub status: Status,
    pub error: String,
    /// Output mesh in input units (after mirror and reprojection).
    pub mesh: QuadMesh,
    /// Input mesh in input units (after upsampling).
    pub input: TriMesh,
    /// The feature edges given to the engine (Symmetry cut edges excluded), as segments in
    /// input units (for the coverage gate).
    pub feature_segments: Vec<[V3; 2]>,
    pub weld: Option<WeldReport>,
    pub stats: stats::Stats,
    /// Engine report plus `stats` and `cli` blocks.
    pub report: Value,
    pub seconds: f64,
}

/// Reads feature pairs: one `a b [kind]` per line (0-based vertex ids, kind 1 by default),
/// `#` comments allowed.
pub fn read_feature_file(path: &Path) -> Result<(Vec<[u32; 2]>, Vec<u8>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pairs = Vec::new();
    let mut kinds = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let bad = || format!("{}:{}: expected `a b [kind]`", path.display(), ln + 1);
        if f.len() < 2 || f.len() > 3 {
            return Err(bad());
        }
        let a: u32 = f[0].parse().map_err(|_| bad())?;
        let b: u32 = f[1].parse().map_err(|_| bad())?;
        let k: u8 = if f.len() == 3 {
            f[2].parse().map_err(|_| bad())?
        } else {
            1
        };
        pairs.push([a, b]);
        kinds.push(k);
    }
    Ok((pairs, kinds))
}

/// Reads a density file: one float per vertex, whitespace separated.
pub fn read_density_file(path: &Path, n_verts: usize) -> Result<Vec<f32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Vec<f32> = text
        .split_whitespace()
        .map(|x| {
            x.parse::<f32>()
                .map_err(|_| format!("{}: bad number {x:?}", path.display()))
        })
        .collect::<Result<_, _>>()?;
    if v.len() != n_verts {
        return Err(format!(
            "{}: {} values for {n_verts} vertices",
            path.display(),
            v.len()
        ));
    }
    Ok(v)
}

fn axis_plane(a: usize) -> [f64; 4] {
    let mut p = [0.0; 4];
    p[a] = 1.0;
    p
}

/// Boundary edges of a triangle mesh whose two vertices lie on one of the planes (within
/// `eps`), as `[tail, head]` of their triangle side.
fn plane_boundary_edges(m: &TriMesh, axes: &[usize], eps: f64) -> Vec<[u32; 2]> {
    let mut dir: Vec<(u32, u32)> = Vec::with_capacity(3 * m.tris.len());
    for t in &m.tris {
        for k in 0..3 {
            dir.push((t[k], t[(k + 1) % 3]));
        }
    }
    let mut sorted = dir.clone();
    sorted.sort_unstable();
    let on = |v: u32, a: usize| m.pos.get(v as usize).is_some_and(|p| p[a].abs() <= eps);
    let mut out: Vec<[u32; 2]> = dir
        .iter()
        .filter(|&&(a, b)| sorted.binary_search(&(b, a)).is_err())
        .filter(|&&(a, b)| axes.iter().any(|&ax| on(a, ax) && on(b, ax)))
        .map(|&(a, b)| [a, b])
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Boundary vertices of a polygon mesh (on an edge used by one face side).
fn boundary_vertices(q: &QuadMesh) -> Vec<bool> {
    let mut e: Vec<(u32, u32)> = Vec::new();
    for f in 0..q.n_faces() {
        let fv = q.face(f);
        for i in 0..fv.len() {
            let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.sort_unstable();
    let mut bnd = vec![false; q.pos.len()];
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j] == e[i] {
            j += 1;
        }
        if j - i == 1 {
            for v in [e[i].0, e[i].1] {
                if let Some(b) = bnd.get_mut(v as usize) {
                    *b = true;
                }
            }
        }
        i = j;
    }
    bnd
}

/// Runs one remesh. `Err` is a usage or IO error (no engine call happened).
pub fn run(o: &RemeshOpts) -> Result<Outcome, String> {
    let t0 = Instant::now();
    let mut m = aq_io::read_obj_tri(&o.input).map_err(|e| format!("{}: {e}", o.input.display()))?;
    let (mut pairs, mut kinds) = match &o.features {
        Features::File(p) => read_feature_file(p)?,
        _ => (Vec::new(), Vec::new()),
    };
    let mut density = match &o.density {
        Some(p) => Some(read_density_file(p, m.pos.len())?),
        None => None,
    };
    if o.upsample > 0 {
        let (um, uf, ud) = host::upsample_with(&m, &pairs, density.as_deref(), o.upsample);
        // Split pairs keep their kind.
        let mut uk = Vec::with_capacity(uf.len());
        for (i, _) in pairs.iter().enumerate() {
            let k = kinds.get(i).copied().unwrap_or(1);
            uk.push(k);
            uk.push(k);
        }
        if uk.len() != uf.len() {
            uk = vec![1; uf.len()];
        }
        m = um;
        pairs = uf;
        kinds = uk;
        density = ud;
    }
    let input_units = m.clone();
    let s = host::normalize(&mut m);
    let full_norm = m.clone();
    let mut warnings: Vec<String> = Vec::new();
    let mut engine_mesh = m;
    if !o.sym.is_empty() {
        if !pairs.is_empty() || density.is_some() {
            warnings.push(
                "--sym: feature files and density maps are dropped (vertex ids change at the cut)"
                    .into(),
            );
            pairs.clear();
            kinds.clear();
            density = None;
        }
        for &a in &o.sym {
            let (h, _) = bisect(&engine_mesh, axis_plane(a), !o.keep_neg);
            engine_mesh = h;
        }
    }
    if o.features == Features::Auto {
        pairs = host::auto_features(&engine_mesh, o.hard_angle);
        kinds = vec![1; pairs.len()];
    }
    let inv = 1.0 / s;
    let feature_segments: Vec<[V3; 2]> = pairs
        .iter()
        .filter_map(|&[a, b]| {
            let pa = engine_mesh.pos.get(a as usize)?;
            let pb = engine_mesh.pos.get(b as usize)?;
            Some([
                aq_core::math::scale(*pa, inv),
                aq_core::math::scale(*pb, inv),
            ])
        })
        .collect();
    if !o.sym.is_empty() {
        let eps = 1e-9 * host::WORK_DIAG;
        let cut = plane_boundary_edges(&engine_mesh, &o.sym, eps);
        kinds.extend(std::iter::repeat_n(5u8, cut.len()));
        pairs.extend(cut);
    }
    let full_target = o.params.target_quads;
    let mut p = o.params.clone();
    if !o.sym.is_empty() {
        let div = 1u64 << o.sym.len();
        p.target_quads = ((full_target as f64 / div as f64).round() as u32).max(4);
    }
    let st = CtlState::new();
    let out = aq_engine::remesh(
        Input {
            mesh: engine_mesh,
            feat_pairs: &pairs,
            feat_kind: &kinds,
            density: density.as_deref(),
        },
        &p,
        &st.unbounded(),
    );
    let mut q = out.mesh;
    let mut weld = None;
    if !o.sym.is_empty() && q.n_faces() > 0 {
        let edge_len = (full_norm.total_area() / full_target.max(1) as f64).sqrt();
        for &a in &o.sym {
            let bnd = boundary_vertices(&q);
            for (v, pos) in q.pos.iter_mut().enumerate() {
                if bnd[v] && pos[a].abs() < 0.3 * edge_len {
                    pos[a] = 0.0;
                }
            }
            let (w, r) = mirror_weld(&q, axis_plane(a), 1e-9 * host::WORK_DIAG);
            q = w;
            weld = Some(r);
        }
    }
    if o.reproject && q.n_faces() > 0 {
        host::reproject(&mut q, &full_norm);
    }
    host::scale_quads(&mut q, 1.0 / s);
    if let Some(path) = &o.output
        && q.n_faces() > 0
    {
        aq_io::write_obj_quad(path, &q).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let seconds = t0.elapsed().as_secs_f64();
    let stats = stats::compute(&q, Some(&input_units), Some(full_target), o.axis, weld);
    let mut report = out.report;
    if let Value::Object(r) = &mut report {
        r.insert("stats".into(), stats.to_json());
        r.insert(
            "cli".into(),
            json!({
                "input": o.input.display().to_string(),
                "output": o.output.as_ref().map(|p| p.display().to_string()),
                "target_full": full_target,
                "target_engine": p.target_quads,
                "sym": o.sym.iter().map(|&a| ["x", "y", "z"][a]).collect::<Vec<_>>(),
                "keep_neg": o.keep_neg,
                "upsample": o.upsample,
                "input_tris": input_units.tris.len(),
                "features": feature_segments.len(),
                "reproject": o.reproject,
                "normalize_scale": s,
                "warnings": warnings,
                "total_s": seconds,
            }),
        );
    }
    if let Some(rp) = &o.report {
        write_json(rp, &report)?;
    }
    Ok(Outcome {
        status: out.status,
        error: out.error,
        mesh: q,
        input: input_units,
        feature_segments,
        weld,
        stats,
        report,
        seconds,
    })
}

/// Writes pretty JSON, creating parent folders.
pub fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    if let Some(d) = path.parent()
        && !d.as_os_str().is_empty()
    {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_edges_and_boundary() {
        // A square in the z = 0 plane cut by x = 0 along its left side.
        let m = TriMesh {
            pos: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            tris: vec![[0, 1, 2], [0, 2, 3]],
        };
        let e = plane_boundary_edges(&m, &[0], 1e-12);
        assert_eq!(e, vec![[3, 0]]);
        let mut q = QuadMesh::new();
        q.pos = m.pos.clone();
        q.vflag = vec![0; 4];
        q.push_face(&[0, 1, 2, 3], 0);
        assert_eq!(boundary_vertices(&q), vec![true; 4]);
    }

    #[test]
    fn feature_and_density_files() {
        // Next to the test binary (inside the target folder).
        let exe = std::env::current_exe().unwrap();
        let dir = exe
            .parent()
            .unwrap()
            .join(format!("aq_cli_files_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("f.txt");
        std::fs::write(&f, "# pairs\n0 1\n2 3 5\n\n").unwrap();
        let (p, k) = read_feature_file(&f).unwrap();
        assert_eq!(p, vec![[0, 1], [2, 3]]);
        assert_eq!(k, vec![1, 5]);
        std::fs::write(&f, "0 x\n").unwrap();
        assert!(read_feature_file(&f).is_err());
        let d = dir.join("d.txt");
        std::fs::write(&d, "1 2.5\n0.5").unwrap();
        assert_eq!(read_density_file(&d, 3).unwrap(), vec![1.0, 2.5, 0.5]);
        assert!(read_density_file(&d, 4).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

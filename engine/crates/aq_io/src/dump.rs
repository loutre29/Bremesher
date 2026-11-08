//! Debug dumps of the pipeline stages (DESIGN 6.15). Every dump is a plain OBJ (or JSON)
//! that opens in Blender: faces for meshes, `l` polylines for curves, `p` for points, and
//! `g` groups to separate the parts. All functions write to any `Write`; [`DumpDir`] writes
//! the standard file names into a directory.
//!
//! The dumps are tolerant: entries whose ids are out of range are skipped (they are a
//! debugging aid and must not fail on a broken stage output; the stage validators report
//! the breakage).

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use aq_core::cross::branch_dir;
use aq_core::math;
use aq_core::{AqError, CutGraph, FeatureSet, Frames, Layout, NONE, QComb, QuadMesh, SeamlessParam, TMesh, TriMesh, V3};
use serde::Serialize;

use crate::obj::{map_io, write_f, write_file, write_line, write_quad_obj, write_v};

fn tri_ok(m: &TriMesh, f: usize) -> bool {
    m.tris.get(f).is_some_and(|t| t.iter().all(|&v| (v as usize) < m.pos.len()))
}

fn write_mesh_body<W: Write>(w: &mut W, m: &TriMesh, group: &str) -> Result<(), AqError> {
    for &p in &m.pos {
        write_v(w, p)?;
    }
    write_line(w, &format!("g {group}"))?;
    for f in 0..m.tris.len() {
        if tri_ok(m, f) {
            write_f(w, &m.tris[f])?;
        }
    }
    Ok(())
}

fn write_l<W: Write>(w: &mut W, verts: &[u32]) -> Result<(), AqError> {
    if verts.len() < 2 {
        return Ok(());
    }
    let mut s = String::from("l");
    for &v in verts {
        s.push_str(&format!(" {}", v as u64 + 1));
    }
    write_line(w, &s)
}

fn he_ends(m: &TriMesh, h: u32) -> Option<(u32, u32)> {
    let f = (h / 3) as usize;
    let k = (h % 3) as usize;
    if !tri_ok(m, f) {
        return None;
    }
    Some((m.tris[f][k], m.tris[f][(k + 1) % 3]))
}

/// Turns a halfedge chain into a vertex polyline (consecutive halfedges; a gap starts a new
/// polyline).
fn he_polylines(m: &TriMesh, hes: &[u32]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    for &h in hes {
        let Some((a, b)) = he_ends(m, h) else { continue };
        if cur.last() != Some(&a) {
            if cur.len() >= 2 {
                out.push(std::mem::take(&mut cur));
            }
            cur.clear();
            cur.push(a);
        }
        cur.push(b);
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

/// `proxy.obj`: the proxy mesh, plus one group of polylines per feature chain
/// (`g chain_<id>_<Kind>`) when a feature set is given.
pub fn dump_proxy<W: Write>(w: &mut W, m: &TriMesh, feats: Option<&FeatureSet>) -> Result<(), AqError> {
    write_line(w, "# aq_io proxy dump")?;
    write_mesh_body(w, m, "proxy")?;
    if let Some(fs) = feats {
        for (c, ch) in fs.chains.iter().enumerate() {
            write_line(w, &format!("g chain_{c}_{:?}", ch.kind))?;
            for pl in he_polylines(m, &ch.hes) {
                write_l(w, &pl)?;
            }
        }
    }
    Ok(())
}

/// `field.obj`: per face two line segments through the centroid, the u-axis (branch 0, group
/// `u`) and the v-axis (branch 1, group `v`), of half-length `arm` (when `arm <= 0`:
/// `0.4 * sqrt(area)` per face). `theta` is per face (raw or combed).
pub fn dump_field<W: Write>(w: &mut W, m: &TriMesh, fr: &Frames, theta: &[f64], arm: f64) -> Result<(), AqError> {
    write_line(w, "# aq_io field dump: per face u (branch 0) and v (branch 1) segments")?;
    let nf = m.tris.len().min(theta.len()).min(fr.e1.len()).min(fr.e2.len()).min(fr.area.len());
    let mut segs: [Vec<(V3, V3)>; 2] = [Vec::new(), Vec::new()];
    for (f, &th) in theta.iter().enumerate().take(nf) {
        if !tri_ok(m, f) {
            continue;
        }
        let [a, b, c] = m.tri_pos(f as u32);
        let ctr = math::scale(math::add(math::add(a, b), c), 1.0 / 3.0);
        let s = if arm > 0.0 { arm } else { 0.4 * fr.area[f].max(0.0).sqrt() };
        for (k, seg) in segs.iter_mut().enumerate() {
            let d = branch_dir(fr, f as u32, th, k as u8);
            seg.push((math::sub(ctr, math::scale(d, s)), math::add(ctr, math::scale(d, s))));
        }
    }
    let mut next = 0u32;
    for (k, seg) in segs.iter().enumerate() {
        write_line(w, &format!("g {}", if k == 0 { "u" } else { "v" }))?;
        for &(p, q) in seg {
            write_v(w, p)?;
            write_v(w, q)?;
            write_l(w, &[next, next + 1])?;
            next += 2;
        }
    }
    Ok(())
}

/// `cones.obj`: one point per cone, grouped by index (`g index4_+1`, `g index4_-1`, ...).
pub fn dump_cones<W: Write>(w: &mut W, pos: &[V3], cones: &[(u32, i8)]) -> Result<(), AqError> {
    write_line(w, "# aq_io cones dump: points grouped by index4")?;
    let mut sorted: Vec<(i8, u32)> =
        cones.iter().filter(|(v, _)| (*v as usize) < pos.len()).map(|&(v, i)| (i, v)).collect();
    sorted.sort_unstable();
    for &(_, v) in &sorted {
        write_v(w, pos[v as usize])?;
    }
    let mut cur: Option<i8> = None;
    for (n, &(i, _)) in sorted.iter().enumerate() {
        if cur != Some(i) {
            write_line(w, &format!("g index4_{i:+}"))?;
            cur = Some(i);
        }
        write_line(w, &format!("p {}", n + 1))?;
    }
    Ok(())
}

/// `cut.obj`: the mesh (`g mesh`), every cut arc as a polyline along its side-A halfedges
/// (`g arc_<id>_r<r>`), and the cut-graph nodes as points (`g nodes`).
pub fn dump_cut<W: Write>(w: &mut W, m: &TriMesh, cut: &CutGraph) -> Result<(), AqError> {
    write_line(w, "# aq_io cut dump")?;
    write_mesh_body(w, m, "mesh")?;
    for (a, arc) in cut.arcs.iter().enumerate() {
        write_line(w, &format!("g arc_{a}_r{}", arc.r))?;
        for pl in he_polylines(m, &arc.hes) {
            write_l(w, &pl)?;
        }
    }
    write_line(w, "g nodes")?;
    for &v in &cut.node_verts {
        if (v as usize) < m.pos.len() {
            write_line(w, &format!("p {}", v as u64 + 1))?;
        }
    }
    Ok(())
}

/// `param.obj`: the mesh with one `vt` per face corner (corner `3f + k` is `vt` number
/// `3f + k + 1`), faces as `f v/vt`; folded faces go to `g folded`, the others to `g param`.
pub fn dump_param<W: Write>(w: &mut W, m: &TriMesh, param: &SeamlessParam) -> Result<(), AqError> {
    write_line(w, "# aq_io param dump: vt per corner, quad units")?;
    for &p in &m.pos {
        write_v(w, p)?;
    }
    let nf = m.tris.len().min(param.uv.len());
    for f in 0..nf {
        for k in 0..3 {
            let uv = param.uv[f][k];
            write_line(w, &format!("vt {} {}", uv[0], uv[1]))?;
        }
    }
    for pass in 0..2 {
        let folded_group = pass == 1;
        write_line(w, if folded_group { "g folded" } else { "g param" })?;
        for f in 0..nf {
            let folded = param.folded.get(f).copied().unwrap_or(false);
            if folded != folded_group || !tri_ok(m, f) {
                continue;
            }
            let t = m.tris[f];
            let c = 3 * f as u64 + 1;
            write_line(
                w,
                &format!("f {}/{} {}/{} {}/{}", t[0] as u64 + 1, c, t[1] as u64 + 1, c + 1, t[2] as u64 + 1, c + 2),
            )?;
        }
    }
    Ok(())
}

/// `layout.obj`: the refined mesh W' with one group per patch (`g patch_<p>`, faces of no
/// patch in `g unassigned`), the arcs as polylines (`g arc_<a>_<Kind>_n<len>`) and the
/// T-mesh nodes as points (`g node_<Kind>`).
pub fn dump_layout<W: Write>(w: &mut W, l: &Layout) -> Result<(), AqError> {
    write_line(w, "# aq_io layout dump")?;
    let m = &l.wmesh;
    for &p in &m.pos {
        write_v(w, p)?;
    }
    let nf = m.tris.len();
    let mut owned = vec![false; nf];
    for (p, fs) in l.patch_faces.iter().enumerate() {
        write_line(w, &format!("g patch_{p}"))?;
        for &f in fs {
            let f = f as usize;
            if f < nf && !owned[f] && tri_ok(m, f) {
                owned[f] = true;
                write_f(w, &m.tris[f])?;
            }
        }
    }
    if owned.iter().any(|o| !o) {
        write_line(w, "g unassigned")?;
        for (f, _) in owned.iter().enumerate().filter(|(_, o)| !**o) {
            if tri_ok(m, f) {
                write_f(w, &m.tris[f])?;
            }
        }
    }
    for (a, vs) in l.arc_verts.iter().enumerate() {
        let (kind, len) = l.tmesh.arcs.get(a).map(|t| (format!("{:?}", t.kind), t.len)).unwrap_or(("Unknown".into(), 0.0));
        write_line(w, &format!("g arc_{a}_{kind}_n{len:.3}"))?;
        let ok: Vec<u32> = vs.iter().copied().filter(|&v| (v as usize) < m.pos.len()).collect();
        write_l(w, &ok)?;
    }
    let mut cur = String::new();
    let mut nodes: Vec<(String, u32)> = l
        .node_vert
        .iter()
        .enumerate()
        .filter(|&(_, &v)| (v as usize) < m.pos.len())
        .map(|(i, &v)| (l.tmesh.nodes.get(i).map(|n| format!("{:?}", n.kind)).unwrap_or_default(), v))
        .collect();
    nodes.sort();
    for (k, v) in nodes {
        if k != cur {
            write_line(w, &format!("g node_{k}"))?;
            cur = k;
        }
        write_line(w, &format!("p {}", v as u64 + 1))?;
    }
    Ok(())
}

#[derive(Serialize)]
struct QuantArcJson {
    id: u32,
    ends: [u32; 2],
    kind: String,
    len: f64,
    lb: u32,
    weight: f64,
    wrap: bool,
    chain: Option<u32>,
    n: Option<u32>,
}

#[derive(Serialize)]
struct QuantPatchJson {
    id: u32,
    sides: Vec<Vec<(u32, bool)>>,
    dims: Option<[u32; 2]>,
}

#[derive(Serialize)]
struct QuantJson {
    chi: i32,
    nodes: usize,
    arcs: Vec<QuantArcJson>,
    patches: Vec<QuantPatchJson>,
    rings: Vec<Vec<(u32, bool)>>,
    n_quads: Option<usize>,
    n_verts: Option<u32>,
    sum_abs_dev: f64,
}

/// `quant.json`: per arc its T-mesh data and quantized length `n` (when given, one per arc),
/// per patch its sides and quantized dims (from the `QComb` when given), and summary counts.
/// `sum_abs_dev` is `sum |n_a - len_a|` over the arcs that have an n.
pub fn dump_quant<W: Write>(w: &mut W, t: &TMesh, n: Option<&[u32]>, qc: Option<&QComb>) -> Result<(), AqError> {
    let mut dev = 0.0;
    let arcs = t
        .arcs
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let na = n.and_then(|n| n.get(i).copied());
            if let Some(x) = na {
                dev += (x as f64 - a.len).abs();
            }
            QuantArcJson {
                id: i as u32,
                ends: a.ends,
                kind: format!("{:?}", a.kind),
                len: a.len,
                lb: a.lb,
                weight: a.weight,
                wrap: a.wrap,
                chain: (a.chain != NONE).then_some(a.chain),
                n: na,
            }
        })
        .collect();
    let patches = t
        .patches
        .iter()
        .enumerate()
        .map(|(i, p)| QuantPatchJson {
            id: i as u32,
            sides: p.sides.to_vec(),
            dims: qc.and_then(|q| q.patch_dims.get(i).copied()),
        })
        .collect();
    let j = QuantJson {
        chi: t.chi,
        nodes: t.nodes.len(),
        arcs,
        patches,
        rings: t.rings.clone(),
        n_quads: qc.map(|q| q.quads.len()),
        n_verts: qc.map(|q| q.n_verts),
        sum_abs_dev: dev,
    };
    serde_json::to_writer_pretty(&mut *w, &j)
        .map_err(|e| AqError::new(aq_core::Stage::Engine, aq_core::ErrKind::Internal, format!("quant.json: {e}")))?;
    w.write_all(b"\n").map_err(map_io)
}

/// `patches.obj`: the quad mesh with one group per patch (`g patch_<p>`, from `quad_patch`,
/// one entry per face; faces beyond it go to `g unassigned`).
pub fn dump_patches<W: Write>(w: &mut W, q: &QuadMesh, quad_patch: &[u32]) -> Result<(), AqError> {
    write_line(w, "# aq_io patches dump")?;
    for &p in &q.pos {
        write_v(w, p)?;
    }
    let nf = q.n_faces();
    let mut order: Vec<(u32, usize)> = (0..nf).map(|f| (quad_patch.get(f).copied().unwrap_or(NONE), f)).collect();
    order.sort_unstable();
    let mut cur: Option<u32> = None;
    for (p, f) in order {
        if cur != Some(p) {
            if p == NONE {
                write_line(w, "g unassigned")?;
            } else {
                write_line(w, &format!("g patch_{p}"))?;
            }
            cur = Some(p);
        }
        let (a, b) = (q.face_off[f] as usize, q.face_off[f + 1] as usize);
        if let Some(fv) = q.face_verts.get(a..b)
            && fv.iter().all(|&v| (v as usize) < q.pos.len())
        {
            write_f(w, fv)?;
        }
    }
    Ok(())
}

/// A dump directory: each method writes its standard file name into `dir` (created on
/// first use).
#[derive(Clone, Debug)]
pub struct DumpDir {
    pub dir: PathBuf,
}

impl DumpDir {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        DumpDir { dir: dir.into() }
    }

    /// Path of a file in the directory.
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn file<F>(&self, name: &str, body: F) -> Result<PathBuf, AqError>
    where
        F: FnOnce(&mut std::io::BufWriter<fs::File>) -> Result<(), AqError>,
    {
        let p = self.path(name);
        write_file(&p, body)?;
        Ok(p)
    }

    pub fn proxy(&self, m: &TriMesh, feats: Option<&FeatureSet>) -> Result<PathBuf, AqError> {
        self.file("proxy.obj", |w| dump_proxy(w, m, feats))
    }
    pub fn field(&self, m: &TriMesh, fr: &Frames, theta: &[f64], arm: f64) -> Result<PathBuf, AqError> {
        self.file("field.obj", |w| dump_field(w, m, fr, theta, arm))
    }
    pub fn cones(&self, pos: &[V3], cones: &[(u32, i8)]) -> Result<PathBuf, AqError> {
        self.file("cones.obj", |w| dump_cones(w, pos, cones))
    }
    pub fn cut(&self, m: &TriMesh, cut: &CutGraph) -> Result<PathBuf, AqError> {
        self.file("cut.obj", |w| dump_cut(w, m, cut))
    }
    pub fn param(&self, m: &TriMesh, p: &SeamlessParam) -> Result<PathBuf, AqError> {
        self.file("param.obj", |w| dump_param(w, m, p))
    }
    pub fn layout(&self, l: &Layout) -> Result<PathBuf, AqError> {
        self.file("layout.obj", |w| dump_layout(w, l))
    }
    pub fn quant(&self, t: &TMesh, n: Option<&[u32]>, qc: Option<&QComb>) -> Result<PathBuf, AqError> {
        self.file("quant.json", |w| dump_quant(w, t, n, qc))
    }
    pub fn patches(&self, q: &QuadMesh, quad_patch: &[u32]) -> Result<PathBuf, AqError> {
        self.file("patches.obj", |w| dump_patches(w, q, quad_patch))
    }
    /// Any quad mesh under a chosen name (for example `post.obj`).
    pub fn quads(&self, name: &str, q: &QuadMesh) -> Result<PathBuf, AqError> {
        self.file(name, |w| write_quad_obj(w, q))
    }
}

//! Wavefront OBJ reading and writing (DESIGN 6.15).
//!
//! Reading understands `v x y z [w ...]` (extra columns such as vertex colors are ignored)
//! and `f` with any `/` suffixes (`f 1/2/3 ...`, `f 1//3 ...`), positive and negative
//! (relative) indices. Every other statement (`vt`, `vn`, `l`, `p`, `g`, `o`, `s`, `usemtl`,
//! comments) is skipped. Faces with fewer than 3 vertices are skipped and counted.
//!
//! The reader does not reject non-finite coordinates or degenerate faces: it returns what
//! the file says, so that the engine's own ingest checks (and the fuzz suite) see the data
//! unchanged. Index errors (0, or out of range) are `ErrKind::InvalidInput`.
//!
//! Writing prints coordinates with Rust's shortest round-trip formatting, so a written mesh
//! reads back bit for bit.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use aq_core::{AqError, ErrKind, QuadMesh, Stage, TriMesh, V3};

/// Raw polygon data of an OBJ file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjData {
    pub pos: Vec<V3>,
    /// n_faces + 1 offsets into `face_verts` (CSR), 0-based vertex ids.
    pub face_off: Vec<u32>,
    pub face_verts: Vec<u32>,
    /// Faces with fewer than 3 vertices that were dropped.
    pub skipped_faces: u32,
}

impl ObjData {
    pub fn n_faces(&self) -> usize {
        self.face_off.len().saturating_sub(1)
    }

    pub fn face(&self, f: usize) -> &[u32] {
        let a = self.face_off[f] as usize;
        let b = self.face_off[f + 1] as usize;
        &self.face_verts[a..b]
    }

    /// Polygons kept as they are (`vflag` and `face_comp` are 0).
    pub fn to_quad_mesh(&self) -> QuadMesh {
        QuadMesh {
            pos: self.pos.clone(),
            face_off: if self.face_off.is_empty() { vec![0] } else { self.face_off.clone() },
            face_verts: self.face_verts.clone(),
            vflag: vec![0; self.pos.len()],
            face_comp: vec![0; self.n_faces()],
        }
    }

    /// Polygons fan-triangulated from their first vertex: (v0, vi, vi+1).
    pub fn to_tri_mesh(&self) -> TriMesh {
        let mut tris = Vec::with_capacity(self.face_verts.len().saturating_sub(2 * self.n_faces()));
        for f in 0..self.n_faces() {
            let fv = self.face(f);
            for i in 1..fv.len().saturating_sub(1) {
                tris.push([fv[0], fv[i], fv[i + 1]]);
            }
        }
        TriMesh { pos: self.pos.clone(), tris }
    }
}

fn io_err(kind: ErrKind, msg: String) -> AqError {
    AqError::new(Stage::Ingest, kind, msg)
}

fn parse_index(tok: &str, n_verts: usize, line_no: usize) -> Result<u32, AqError> {
    let head = tok.split('/').next().unwrap_or("");
    let i: i64 = head
        .parse()
        .map_err(|_| io_err(ErrKind::InvalidInput, format!("OBJ line {line_no}: bad face index '{tok}'")))?;
    let idx = if i > 0 {
        i - 1
    } else if i < 0 {
        n_verts as i64 + i
    } else {
        -1
    };
    if idx < 0 || idx >= n_verts as i64 || idx >= u32::MAX as i64 {
        return Err(io_err(
            ErrKind::InvalidInput,
            format!("OBJ line {line_no}: face index {i} out of range (1..={n_verts})"),
        ));
    }
    Ok(idx as u32)
}

/// Parses OBJ text.
pub fn parse_obj(text: &str) -> Result<ObjData, AqError> {
    let mut d = ObjData { face_off: vec![0], ..Default::default() };
    let mut poly: Vec<u32> = Vec::with_capacity(8);
    for (ln, raw) in text.lines().enumerate() {
        let line_no = ln + 1;
        let line = raw.trim();
        let mut it = line.split_ascii_whitespace();
        match it.next() {
            Some("v") => {
                let mut p = [0.0f64; 3];
                for (k, c) in p.iter_mut().enumerate() {
                    let tok = it.next().ok_or_else(|| {
                        io_err(ErrKind::InvalidInput, format!("OBJ line {line_no}: vertex has {k} coordinates"))
                    })?;
                    *c = tok.parse().map_err(|_| {
                        io_err(ErrKind::InvalidInput, format!("OBJ line {line_no}: bad coordinate '{tok}'"))
                    })?;
                }
                d.pos.push(p);
            }
            Some("f") => {
                poly.clear();
                for tok in it {
                    poly.push(parse_index(tok, d.pos.len(), line_no)?);
                }
                if poly.len() < 3 {
                    d.skipped_faces += 1;
                    continue;
                }
                d.face_verts.extend_from_slice(&poly);
                if d.face_verts.len() > u32::MAX as usize {
                    return Err(io_err(ErrKind::Budget, "OBJ: too many face vertices".into()));
                }
                d.face_off.push(d.face_verts.len() as u32);
            }
            _ => {}
        }
    }
    Ok(d)
}

/// Reads an OBJ file (invalid UTF-8 is replaced, not rejected).
pub fn read_obj(path: &Path) -> Result<ObjData, AqError> {
    let bytes = fs::read(path)
        .map_err(|e| io_err(ErrKind::InvalidInput, format!("cannot read {}: {e}", path.display())))?;
    parse_obj(&String::from_utf8_lossy(&bytes))
}

/// Reads an OBJ file as a triangle mesh (polygons fan-triangulated).
pub fn read_obj_tri(path: &Path) -> Result<TriMesh, AqError> {
    Ok(read_obj(path)?.to_tri_mesh())
}

/// Reads an OBJ file as a polygon mesh (faces kept as they are).
pub fn read_obj_poly(path: &Path) -> Result<QuadMesh, AqError> {
    Ok(read_obj(path)?.to_quad_mesh())
}

fn wr(e: std::io::Error) -> AqError {
    AqError::new(Stage::Engine, ErrKind::Internal, format!("write failed: {e}"))
}

pub(crate) fn write_v<W: Write>(w: &mut W, p: V3) -> Result<(), AqError> {
    writeln!(w, "v {} {} {}", p[0], p[1], p[2]).map_err(wr)
}

pub(crate) fn write_f<W: Write>(w: &mut W, verts: &[u32]) -> Result<(), AqError> {
    w.write_all(b"f").map_err(wr)?;
    for &v in verts {
        write!(w, " {}", v as u64 + 1).map_err(wr)?;
    }
    w.write_all(b"\n").map_err(wr)
}

pub(crate) fn write_line<W: Write>(w: &mut W, s: &str) -> Result<(), AqError> {
    w.write_all(s.as_bytes()).map_err(wr)?;
    w.write_all(b"\n").map_err(wr)
}

pub(crate) fn map_io(e: std::io::Error) -> AqError {
    wr(e)
}

/// Writes a triangle mesh.
pub fn write_tri_obj<W: Write>(w: &mut W, m: &TriMesh) -> Result<(), AqError> {
    for &p in &m.pos {
        write_v(w, p)?;
    }
    for t in &m.tris {
        write_f(w, t)?;
    }
    Ok(())
}

/// Writes a polygon mesh (quads as 4-vertex polygons). Faces whose CSR range is invalid are
/// an `InvalidInput` error.
pub fn write_quad_obj<W: Write>(w: &mut W, q: &QuadMesh) -> Result<(), AqError> {
    for &p in &q.pos {
        write_v(w, p)?;
    }
    for f in 0..q.n_faces() {
        let (a, b) = (q.face_off[f] as usize, q.face_off[f + 1] as usize);
        let fv = q.face_verts.get(a..b).ok_or_else(|| {
            AqError::new(Stage::Engine, ErrKind::InvalidInput, format!("QuadMesh face {f} has a bad CSR range"))
        })?;
        write_f(w, fv)?;
    }
    Ok(())
}

/// Creates (or truncates) a file, creating its parent directories, and writes it with `body`.
pub fn write_file<F>(path: &Path, body: F) -> Result<(), AqError>
where
    F: FnOnce(&mut BufWriter<fs::File>) -> Result<(), AqError>,
{
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        fs::create_dir_all(dir).map_err(wr)?;
    }
    let f = fs::File::create(path)
        .map_err(|e| AqError::new(Stage::Engine, ErrKind::Internal, format!("cannot create {}: {e}", path.display())))?;
    let mut w = BufWriter::new(f);
    body(&mut w)?;
    w.flush().map_err(wr)
}

/// Writes a triangle mesh to a file.
pub fn write_obj_tri(path: &Path, m: &TriMesh) -> Result<(), AqError> {
    write_file(path, |w| write_tri_obj(w, m))
}

/// Writes a polygon mesh to a file.
pub fn write_obj_quad(path: &Path, q: &QuadMesh) -> Result<(), AqError> {
    write_file(path, |w| write_quad_obj(w, q))
}

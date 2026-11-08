//! Minimal OBJ loader for the testdata meshes (`v` and `f` records, any `/` suffix, negative
//! indices, polygons fan-triangulated). Everything else is ignored.

use std::path::{Path, PathBuf};

use aq_core::{AqError, TriMesh};

use crate::util::bad;

/// `engine/testdata`.
pub fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("testdata")
}

/// The 12 testdata mesh names (without `.obj`), sorted.
pub fn testdata_names() -> Vec<&'static str> {
    vec![
        "basemesh_female",
        "cube_sharp",
        "cylinder_open",
        "double_torus",
        "noisy_blob",
        "retopo_a",
        "retopo_b",
        "retopo_c",
        "sculpt_cylinder",
        "sculpt_sphere_mirror",
        "suzanne_sub2",
        "torus",
    ]
}

/// Loads `engine/testdata/<name>.obj`.
pub fn load_testdata(name: &str) -> Result<TriMesh, AqError> {
    load_obj(&testdata_dir().join(format!("{name}.obj")))
}

/// Reads and parses an OBJ file.
pub fn load_obj(path: &Path) -> Result<TriMesh, AqError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| bad(format!("cannot read {}: {e}", path.display())))?;
    parse_obj(&text)
}

/// Parses OBJ text.
pub fn parse_obj(text: &str) -> Result<TriMesh, AqError> {
    let mut m = TriMesh::default();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let mut p = [0.0; 3];
                for c in p.iter_mut() {
                    let s = it.next().ok_or_else(|| bad(format!("line {}: short vertex", ln + 1)))?;
                    *c = s.parse().map_err(|_| bad(format!("line {}: bad number {s}", ln + 1)))?;
                }
                m.pos.push(p);
            }
            Some("f") => {
                let mut ids = Vec::new();
                for tok in it {
                    let first = tok.split('/').next().unwrap_or("");
                    let i: i64 = first.parse().map_err(|_| bad(format!("line {}: bad index {tok}", ln + 1)))?;
                    let n = m.pos.len() as i64;
                    let idx = if i < 0 { n + i } else { i - 1 };
                    if idx < 0 || idx >= n {
                        return Err(bad(format!("line {}: index {i} out of range", ln + 1)));
                    }
                    ids.push(idx as u32);
                }
                if ids.len() < 3 {
                    return Err(bad(format!("line {}: face with fewer than 3 vertices", ln + 1)));
                }
                for k in 1..ids.len() - 1 {
                    m.tris.push([ids[0], ids[k], ids[k + 1]]);
                }
            }
            _ => {}
        }
    }
    Ok(m)
}

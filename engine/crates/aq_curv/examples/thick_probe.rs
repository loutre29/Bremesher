//! WP-20 diagnostic: shape diameter statistics of a mesh. Usage:
//! `thick_probe <mesh.obj> <target> [ring_min]`. Scales the mesh to quad units (h0 = 1 for the
//! target), computes d(v) on the input triangles and prints, per connected component, the area
//! with a thickness cap below 1, 0.75 and 0.5 h, and the floors for a few budgets.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use aq_core::{CtlState, math};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut m = aq_io::read_obj_tri(std::path::Path::new(&args[1])).unwrap();
    let target: f64 = args[2].parse().unwrap();
    let ring: f64 = args.get(3).map_or(8.0, |x| x.parse().unwrap());
    let fa: Vec<f64> = (0..m.tris.len() as u32).map(|f| m.face_area(f)).collect();
    let h0 = (fa.iter().sum::<f64>() / target).sqrt();
    for p in m.pos.iter_mut() {
        *p = math::scale(*p, 1.0 / h0);
    }
    let fa: Vec<f64> = (0..m.tris.len() as u32).map(|f| m.face_area(f)).collect();
    let fnrm: Vec<[f64; 3]> = (0..m.tris.len() as u32)
        .map(|f| math::normalize(m.face_cross(f)).unwrap_or([0.0; 3]))
        .collect();
    let st = CtlState::new();
    let ctl = st.unbounded();
    let t = std::time::Instant::now();
    let d = aq_curv::shape_diameter_mesh(&m, &fnrm, &fa, &ctl).unwrap();
    println!("h0 {h0:.5}, {} verts, diameter in {:.1} ms", m.pos.len(), t.elapsed().as_secs_f64() * 1e3);
    let mut av = vec![0.0; m.pos.len()];
    for (f, t) in m.tris.iter().enumerate() {
        for &v in t {
            av[v as usize] += fa[f] / 3.0;
        }
    }
    let raw: Vec<f64> = d.iter().map(|x| std::f64::consts::PI * x / ring).collect();
    let tot: f64 = av.iter().sum();
    for lim in [1.0, 0.75, 0.5, 0.35] {
        let a: f64 = (0..av.len()).filter(|&v| raw[v] < lim).map(|v| av[v]).sum();
        println!("  area with cap < {lim}: {:.1} h^2 ({:.1}%)", a, 100.0 * a / tot);
    }
    if let Some(out) = args.get(4) {
        // Faces whose three vertices have a cap below 1 h, in input units.
        let mut txt = String::new();
        for p in &m.pos {
            txt += &format!("v {} {} {}\n", p[0] * h0, p[1] * h0, p[2] * h0);
        }
        for t in &m.tris {
            if t.iter().all(|&v| raw[v as usize] < 1.0) {
                txt += &format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1);
            }
        }
        std::fs::write(out, txt).unwrap();
    }
    let base = vec![1.0; av.len()];
    for b in [0.02, 0.05, 0.1, 0.2] {
        let f = aq_curv::thickness_floor(&av, &base, &[], &raw, target, b, &ctl).unwrap();
        println!("  budget {b}: floor {f:.3}");
    }
}

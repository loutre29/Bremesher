//! WP-16 diagnostic: where the input is far from the output. Usage:
//! `hd_locate <input.obj> <quads.obj> <target>`. Prints the input vertices farther than 0.7 h
//! from the output, clustered greedily (one line per 1 h cluster), plus the q->in maximum.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use aq_core::{Bvh, TriMesh};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inp = aq_io::read_obj_tri(std::path::Path::new(&args[1])).unwrap();
    let q = aq_io::read_obj_poly(std::path::Path::new(&args[2])).unwrap();
    let target: f64 = args[3].parse().unwrap();
    let qt: TriMesh = aq_eval::fan_triangulate(&q);
    let area: f64 = (0..inp.tris.len() as u32).map(|f| inp.face_area(f)).sum();
    let h = (area / target).sqrt();
    let bq = Bvh::try_build(&qt).unwrap();
    let bi = Bvh::try_build(&inp).unwrap();
    let mut far: Vec<(f64, [f64; 3])> = inp
        .pos
        .iter()
        .map(|&p| (bq.closest(p).2 / h, p))
        .filter(|x| x.0 > 0.7)
        .collect();
    far.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("h = {h:.5}, {} input vertices beyond 0.7 h", far.len());
    let mut centers: Vec<(f64, [f64; 3], u32)> = Vec::new();
    for (d, p) in far {
        if let Some(c) = centers
            .iter_mut()
            .find(|c| aq_core::math::norm(aq_core::math::sub(c.1, p)) < 2.0 * h)
        {
            c.2 += 1;
        } else {
            centers.push((d, p, 1));
        }
    }
    for (d, p, n) in centers.iter().take(15) {
        println!(
            "  in->q {d:.2} h at ({:.4}, {:.4}, {:.4}), {n} verts",
            p[0], p[1], p[2]
        );
    }
    let mut qmax = (0.0, [0.0; 3]);
    for &p in &q.pos {
        let d = bi.closest(p).2 / h;
        if d > qmax.0 {
            qmax = (d, p);
        }
    }
    println!(
        "  q->in max {:.2} h at ({:.4}, {:.4}, {:.4})",
        qmax.0, qmax.1[0], qmax.1[1], qmax.1[2]
    );
}

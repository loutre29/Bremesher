//! DESIGN 6.8 tests for aq_cut.
#![allow(clippy::needless_range_loop)]

use aq_core::{CrossField, CutGraph, ErrKind, NONE};
use aq_cut::{ProxyRef, cut_and_comb, cut_mesh_euler, homology_generators};
use aq_testkit::{FieldKind, Surface, analytic_field};

fn px(s: &Surface) -> ProxyRef<'_> {
    ProxyRef::new(&s.mesh, &s.topo, &s.frames, &s.feats)
}

/// Every DESIGN 6.8 property that does not depend on the surface kind.
fn check_cut(s: &Surface, f: &CrossField, cg: &CutGraph) {
    let (m, t) = (&s.mesh, &s.topo);
    cg.validate(m, t, Some(f)).unwrap();
    // chi(cut) = 1 per component (independent recount with the testkit helper as well).
    for (c, &e) in cut_mesh_euler(&px(s), &cg.is_cut).iter().enumerate() {
        assert_eq!(e, 1, "component {c}: cut mesh V - E + F = {e}");
        assert_eq!(aq_testkit::cut::cut_mesh_euler(s, &cg.is_cut, c as u32), 1);
    }
    // Every cone lies on the cut (interior cones) or on the boundary.
    for (v, &i) in f.index4.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let on_cut = t.fan_out(v as u32).any(|h| cg.is_cut[h as usize]);
        assert!(on_cut || t.is_bnd_vert[v], "cone {v} (index4 {i}) is not on the cut");
        if on_cut {
            assert!(cg.node_verts.binary_search(&(v as u32)).is_ok(), "cone {v} on the cut is not a node");
        }
    }
    // Combed r = 0 on every non-cut interior halfedge; r = (p + c_g - c_f) mod 4 everywhere.
    for h in 0..3 * m.tris.len() {
        let tw = t.twin[h];
        if tw == NONE {
            continue;
        }
        if !cg.is_cut[h] {
            assert_eq!(cg.r[h], 0, "r != 0 off the cut at {h}");
        }
        let (cf, cg_) = (cg.comb_offset[h / 3], cg.comb_offset[tw as usize / 3]);
        assert_eq!(cg.r[h], (f.p[h] + cg_ + 4 - cf) % 4);
    }
    // r constant along each arc, arcs cover the cut.
    let mut covered = 0usize;
    for a in &cg.arcs {
        for &h in &a.hes {
            assert_eq!(cg.r[h as usize], a.r);
        }
        covered += 2 * a.hes.len();
    }
    assert_eq!(covered, cg.is_cut.iter().filter(|&&c| c).count());
}

fn run(s: &Surface, kind: &FieldKind) -> (CrossField, CutGraph) {
    let f = analytic_field(s, kind).unwrap();
    let cg = cut_and_comb(&px(s), &f, 0).unwrap();
    check_cut(s, &f, &cg);
    (f, cg)
}

fn generators() -> Vec<(&'static str, Surface)> {
    use aq_testkit::*;
    vec![
        ("plane", Surface::new(plane_grid(12, 9, 0.3, 0.4, 0.2, 1).unwrap()).unwrap()),
        ("square_disk", Surface::new(square_disk(10, 3.0, 0.2, 2).unwrap()).unwrap()),
        ("uv_sphere", Surface::new(uv_sphere(12, 20, 2.0).unwrap()).unwrap()),
        ("icosphere", Surface::new(icosphere(3, 2.0).unwrap()).unwrap()),
        ("octasphere", Surface::new(octasphere(6, 2.0).unwrap()).unwrap()),
        ("cylinder_open", Surface::new(cylinder_open(1.0, 4.0, 24, 12, 0.2, 3).unwrap()).unwrap()),
        ("cylinder_closed", Surface::new(cylinder_closed(1.0, 4.0, 24, 12, 0.2, 4).unwrap()).unwrap()),
        ("torus", Surface::new(torus(3.0, 1.0, 40, 16, 0.2, 5).unwrap()).unwrap()),
        ("double_torus", Surface::new(double_torus().unwrap()).unwrap()),
        ("cube_sharp", Surface::from_synth(&cube_sharp(6, 2.0, 0.1, 6).unwrap()).unwrap()),
        ("hemisphere", Surface::from_synth(&hemisphere(6, 2.0).unwrap()).unwrap()),
        (
            "two_components",
            Surface::new(merge(&[icosphere(2, 1.0).unwrap(), torus(3.0, 1.0, 30, 12, 0.1, 7).unwrap()])).unwrap(),
        ),
    ]
}

#[test]
fn chi_one_on_every_generator() {
    for (name, s) in generators() {
        for kind in [FieldKind::Constant { angle: 0.3 }, FieldKind::Theta(vec![0.0; s.n_faces()])] {
            let f = analytic_field(&s, &kind).unwrap();
            let cg = cut_and_comb(&px(&s), &f, 0).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            check_cut(&s, &f, &cg);
        }
    }
}

#[test]
fn specific_fields() {
    use aq_testkit::*;
    let s = Surface::new(torus(3.0, 1.0, 48, 18, 0.2, 9).unwrap()).unwrap();
    let (f, cg) = run(&s, &FieldKind::TorusCircumferential);
    assert!(f.cones().is_empty());
    assert!(!cg.arcs.is_empty());
    let s = Surface::from_synth(&cube_sharp(5, 2.0, 0.0, 1).unwrap()).unwrap();
    let (f, _) = run(&s, &FieldKind::CubeAligned);
    assert_eq!(f.cones().len(), 8);
    let s = Surface::new(cylinder_open(1.0, 5.0, 30, 15, 0.2, 2).unwrap()).unwrap();
    let (f, cg) = run(&s, &FieldKind::CylinderCircumferential);
    assert!(f.cones().is_empty());
    // An open cylinder is cut along one path from rim to rim.
    assert!(!cg.arcs.is_empty());
    let s = Surface::new(square_disk(8, 2.0, 0.1, 3).unwrap()).unwrap();
    let (_, cg) = run(&s, &FieldKind::Constant { angle: 0.0 });
    assert!(cg.is_cut.iter().all(|&c| !c), "a disk without interior cones needs no cut");
}

#[test]
fn testdata_meshes() {
    let mut done = 0;
    for name in aq_testkit::testdata_names() {
        let m = split_bowties(aq_testkit::load_testdata(name).unwrap());
        let s = match Surface::new(m) {
            Ok(s) => s,
            Err(e) => {
                // Not manifold without aq_prep's ingest repairs.
                eprintln!("testdata {name}: skipped ({e:?})");
                continue;
            }
        };
        for kind in [FieldKind::Constant { angle: 0.7 }, FieldKind::Theta(vec![0.0; s.n_faces()])] {
            let f = match analytic_field(&s, &kind) {
                Ok(f) => f,
                Err(e) => {
                    // A face with two non-orthogonal boundary edges cannot be aligned (aq_prep
                    // splits those); cut a field with rounded indices instead.
                    eprintln!("testdata {name}: analytic field failed ({e:?}), using rounded indices");
                    rough_field(&s)
                }
            };
            let cg = cut_and_comb(&px(&s), &f, 0).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            check_cut(&s, &f, &cg);
            done += 1;
        }
    }
    eprintln!("testdata cut cases: {done}");
    assert_eq!(done, 24, "every testdata mesh must be cut with both fields");
}

/// Duplicates every vertex whose incident faces form several edge-connected fans (a local
/// stand-in for aq_prep's ingest step 4, which this crate cannot use).
fn split_bowties(mut m: aq_core::TriMesh) -> aq_core::TriMesh {
    let nv = m.pos.len();
    let mut inc: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
    for (f, t) in m.tris.iter().enumerate() {
        for k in 0..3 {
            inc[t[k] as usize].push((f, k));
        }
    }
    for v in 0..nv {
        let c = &inc[v];
        let mut group: Vec<usize> = (0..c.len()).collect();
        fn find(g: &mut [usize], mut x: usize) -> usize {
            while g[x] != x {
                x = g[x];
            }
            x
        }
        for i in 0..c.len() {
            for j in i + 1..c.len() {
                let (ti, tj) = (m.tris[c[i].0], m.tris[c[j].0]);
                let shared = ti.iter().filter(|&&a| a as usize != v && tj.contains(&a)).count();
                if shared > 0 {
                    let (a, b) = (find(&mut group, i), find(&mut group, j));
                    group[a.max(b)] = a.min(b);
                }
            }
        }
        let mut reps: Vec<usize> = Vec::new();
        for i in 0..c.len() {
            let r = find(&mut group, i);
            if !reps.contains(&r) {
                reps.push(r);
            }
        }
        if reps.len() > 1 {
            let mut ids = vec![v as u32];
            for _ in 1..reps.len() {
                ids.push(m.pos.len() as u32);
                m.pos.push(m.pos[v]);
            }
            for i in 0..c.len() {
                let r = find(&mut group, i);
                let gi = reps.iter().position(|&x| x == r).unwrap();
                let (f, k) = c[i];
                m.tris[f][k] = ids[gi];
            }
        }
    }
    m
}

/// theta = 0 with the rounded matching and indices rounded without the residual check (only
/// `index4 != 0` matters to the cut).
fn rough_field(s: &Surface) -> CrossField {
    use std::f64::consts::FRAC_PI_2;
    let theta = vec![0.0; s.n_faces()];
    let (p, delta) = aq_core::matching_from_theta(&s.mesh, &s.topo, &s.frames, &theta);
    let index4 = (0..s.n_verts() as u32)
        .map(|v| {
            let sd: f64 = s.topo.incoming(&s.mesh, v).map(|h| delta[h as usize]).sum();
            if s.topo.vert_he[v as usize] == NONE {
                0
            } else if s.topo.is_bnd_vert[v as usize] {
                2 - ((s.frames.angle_sum(&s.topo, v) - sd) / FRAC_PI_2).round() as i8
            } else {
                ((s.frames.angle_defect(&s.topo, v) + sd) / FRAC_PI_2).round() as i8
            }
        })
        .collect();
    CrossField { theta, p, delta, index4, hard: vec![false; s.n_faces()] }
}

fn check_generator_cycles(s: &Surface, gens: &[Vec<u32>]) {
    for g in gens {
        assert!(!g.is_empty());
        let start = g[0] / 3;
        let mut face = start;
        for &h in g {
            assert_eq!(h / 3, face, "generator dual path broken");
            let tw = s.topo.twin[h as usize];
            assert_ne!(tw, NONE);
            face = tw / 3;
        }
        assert_eq!(face, start, "generator does not close");
    }
}

#[test]
fn homology_counts() {
    use aq_testkit::*;
    let s = Surface::new(torus(3.0, 1.0, 40, 16, 0.2, 5).unwrap()).unwrap();
    for kind in [FieldKind::TorusCircumferential, FieldKind::Constant { angle: 0.2 }] {
        let (_, cg) = run(&s, &kind);
        let gens = homology_generators(&px(&s), &cg);
        assert_eq!(gens.len(), 2, "torus");
        check_generator_cycles(&s, &gens);
    }
    let s = Surface::new(double_torus().unwrap()).unwrap();
    let (_, cg) = run(&s, &FieldKind::Constant { angle: 0.2 });
    let gens = homology_generators(&px(&s), &cg);
    assert_eq!(gens.len(), 4, "double torus");
    check_generator_cycles(&s, &gens);
    // Genus 0: none; open cylinder: 1 (2g + b - 1).
    let s = Surface::new(icosphere(3, 1.0).unwrap()).unwrap();
    let (_, cg) = run(&s, &FieldKind::Constant { angle: 0.2 });
    assert_eq!(homology_generators(&px(&s), &cg).len(), 0);
    let s = Surface::new(cylinder_open(1.0, 3.0, 20, 10, 0.2, 1).unwrap()).unwrap();
    let (_, cg) = run(&s, &FieldKind::CylinderCircumferential);
    let gens = homology_generators(&px(&s), &cg);
    assert_eq!(gens.len(), 1);
    check_generator_cycles(&s, &gens);
}

#[test]
fn genus0_without_cones_is_invariant() {
    use aq_testkit::*;
    let s = Surface::new(icosphere(2, 1.0).unwrap()).unwrap();
    let mut f = analytic_field(&s, &FieldKind::Constant { angle: 0.0 }).unwrap();
    f.index4.iter_mut().for_each(|i| *i = 0);
    let e = cut_and_comb(&px(&s), &f, 0).unwrap_err();
    assert_eq!(e.kind, ErrKind::Invariant);
}

#[test]
fn deterministic() {
    use aq_testkit::*;
    // A regular grid has many faces of exactly the largest area: the seed decides the root.
    let s = Surface::new(torus(3.0, 1.0, 40, 16, 0.0, 0).unwrap()).unwrap();
    let f = analytic_field(&s, &FieldKind::Constant { angle: 0.4 }).unwrap();
    let a = cut_and_comb(&px(&s), &f, 7).unwrap();
    let b = cut_and_comb(&px(&s), &f, 7).unwrap();
    assert_eq!(a, b);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    let c = pool.install(|| cut_and_comb(&px(&s), &f, 7).unwrap());
    assert_eq!(a, c);
    let d = cut_and_comb(&px(&s), &f, 8).unwrap();
    check_cut(&s, &f, &d);
}

#[test]
fn cut_mesh_euler_rejects_wrong_sizes() {
    use aq_testkit::*;
    let s = Surface::new(torus(3.0, 1.0, 20, 8, 0.2, 1).unwrap()).unwrap();
    let nh = 3 * s.n_faces();
    assert!(cut_mesh_euler(&px(&s), &vec![false; nh - 1]).is_empty());
    assert!(cut_mesh_euler(&px(&s), &[]).is_empty());
    assert!(cut_mesh_euler(&px(&s), &vec![false; nh + 3]).is_empty());
    assert_eq!(cut_mesh_euler(&px(&s), &vec![false; nh]).len(), 1);
}

#[test]
fn performance_100k_faces() {
    use aq_testkit::*;
    let s = Surface::new(torus(6.0, 2.0, 250, 200, 0.2, 11).unwrap()).unwrap();
    assert_eq!(s.n_faces(), 100_000);
    let f = analytic_field(&s, &FieldKind::Constant { angle: 0.4 }).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let cg = cut_and_comb(&px(&s), &f, 0).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        std::hint::black_box(cg);
    }
    eprintln!("cut_and_comb 100k faces: {best:.3} s ({} cones)", f.cones().len());
    if !cfg!(debug_assertions) {
        assert!(best < 0.3, "cut_and_comb took {best:.3} s");
    }
}

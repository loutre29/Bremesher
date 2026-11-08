//! WP-16 diagnostic: cone counts of S4 over a grid of field parameters, one proxy per
//! component. Usage: `cone_sweep <mesh.obj> <target> [scales] [aligns] [cancels]` where each list
//! is comma separated (defaults "1", "1,0.7,0.5,0.35,0", "1.2").
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicI32, AtomicU32};
use std::time::{Duration, Instant};

use aq_core::{Ctl, Projector};

fn list(s: Option<&String>, d: &str) -> Vec<f64> {
    s.map(|s| s.as_str())
        .unwrap_or(d)
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let target: f64 = args[2].parse().unwrap();
    let scales = list(args.get(3), "1");
    let aligns = list(args.get(4), "1,0.7,0.5,0.35,0");
    let cancels = list(args.get(5), "1.2");
    let cancel = AtomicI32::new(0);
    let prog = AtomicU32::new(0);
    let stage = AtomicU32::new(0);
    let ctl = Ctl::new(
        &cancel,
        &prog,
        &stage,
        Instant::now() + Duration::from_secs(3600),
    );
    let mut m = aq_io::read_obj_tri(std::path::Path::new(path)).unwrap();
    aq_engine::host::normalize(&mut m);
    let ing = aq_prep::ingest(m, &[], &[], &ctl).unwrap();
    let comps = aq_prep::split_components(&ing);
    let a_tot: f64 = comps.iter().map(|c| c.area).sum();
    // (scale, align, cancel) -> (total cones, interior cones)
    type Key = (usize, usize, usize);
    let mut tot: std::collections::BTreeMap<Key, [u32; 4]> = Default::default();
    for c0 in comps.iter() {
        let n_c = target * c0.area / a_tot;
        if n_c < 24.0 {
            continue;
        }
        let comp = c0.scaled((n_c / c0.area).sqrt());
        let (fr, bvh) = comp.source_geometry().unwrap();
        let proj = Projector::new(&comp.mesh, &comp.topo, &fr, &bvh);
        let pp = aq_prep::ProxyParams {
            max_tris: (150000.0 * c0.area / a_tot) as u32,
            ..Default::default()
        };
        let mut px = aq_prep::build_proxy(&comp, &proj, &|_| 1.0 / 3.0, &pp, &ctl).unwrap();
        aq_prep::isolate_features_with(&mut px, Some(&proj)).unwrap();
        let ones = vec![1.0; px.mesh.pos.len()];
        aq_prep::classify_chains(&mut px, &ones, &pp);
        let cv = aq_curv::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        for (si, &sc) in scales.iter().enumerate() {
            let (curv, size, _) = aq_curv::curvature_and_sizing(
                &cv,
                None,
                &aq_curv::CurvParams {
                    scale: sc,
                    ..Default::default()
                },
                &aq_curv::SizingParams::new(n_c),
                &ctl,
            )
            .unwrap();
            let fv = aq_field::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
            for (ai, &al) in aligns.iter().enumerate() {
                for (ki, &ca) in cancels.iter().enumerate() {
                    let fpp = aq_field::FieldParams {
                        align: al,
                        cancel_dist: ca,
                        ..Default::default()
                    };
                    let (field, frep) =
                        aq_field::solve_field(&fv, &curv, &size, &fpp, &ctl).unwrap();
                    let interior = (0..px.mesh.pos.len())
                        .filter(|&v| field.index4[v] != 0 && !px.topo.is_bnd_vert[v])
                        .count() as u32;
                    let mut adj_same = 0;
                    let mut adj_opp = 0;
                    for tri in &px.mesh.tris {
                        for k in 0..3 {
                            let (a, b) = (tri[k] as usize, tri[(k + 1) % 3] as usize);
                            if a < b && field.index4[a] != 0 && field.index4[b] != 0 {
                                if field.index4[a] == field.index4[b] {
                                    adj_same += 1
                                } else {
                                    adj_opp += 1
                                }
                            }
                        }
                    }
                    println!(
                        "  comp: cones {} split {} adjacent same-sign edges {adj_same} opposite {adj_opp}",
                        frep.cones_final, frep.split
                    );
                    let pos = field.index4.iter().filter(|&&i| i > 0).count() as u32;
                    let neg = field.index4.iter().filter(|&&i| i < 0).count() as u32;
                    let e = tot.entry((si, ai, ki)).or_default();
                    e[0] += frep.cones_final;
                    e[1] += interior;
                    e[2] += pos;
                    e[3] += neg;
                }
            }
        }
    }
    let name = std::path::Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    for ((si, ai, ki), [t, i, ps, ng]) in tot {
        println!(
            "{name} scale {} align {} cancel {}: cones {t} interior {i} pos {ps} neg {ng}",
            scales[si], aligns[ai], cancels[ki]
        );
    }
}

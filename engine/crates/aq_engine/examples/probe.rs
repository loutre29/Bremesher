//! WP-16 diagnostic probe: runs S0 to S7 on one test mesh and prints where the folds left by
//! S7 sit (cones, chains, boundary). Usage: `probe <mesh.obj> <target> [hard_angle_deg]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicI32, AtomicU32};
use std::time::{Duration, Instant};

use aq_core::{Ctl, FeatureKind, NONE, Projector};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let target: f64 = args[2].parse().unwrap();
    let cancel = AtomicI32::new(0);
    let prog = AtomicU32::new(0);
    let stage = AtomicU32::new(0);
    let ctl = Ctl::new(
        &cancel,
        &prog,
        &stage,
        Instant::now() + Duration::from_secs(600),
    );
    let mut m = aq_io::read_obj_tri(std::path::Path::new(path)).unwrap();
    aq_engine::host::normalize(&mut m);
    let pairs: Vec<[u32; 2]> = match args.get(3) {
        Some(a) => aq_engine::host::auto_features(&m, a.parse().unwrap()),
        None => Vec::new(),
    };
    let kinds = vec![1u8; pairs.len()];
    let ing = aq_prep::ingest(m, &pairs, &kinds, &ctl).unwrap();
    let comps = aq_prep::split_components(&ing);
    let a_tot: f64 = comps.iter().map(|c| c.area).sum();
    for (ci, c0) in comps.iter().enumerate() {
        let n_c = (target * c0.area / a_tot).max(24.0);
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
        let (curv, size, _) = aq_curv::curvature_and_sizing(
            &cv,
            None,
            &aq_curv::CurvParams {
                scale: std::env::var("AQ_SMOOTH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0),
                ..Default::default()
            },
            &aq_curv::SizingParams::new(n_c),
            &ctl,
        )
        .unwrap();
        let fv = aq_field::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        let envf = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let fpp = aq_field::FieldParams {
            align: envf("AQ_ALIGN", 1.0),
            cancel_dist: envf("AQ_CANCEL", 1.2),
            ..Default::default()
        };
        let (field, frep) = aq_field::solve_field(&fv, &curv, &size, &fpp, &ctl).unwrap();
        println!(
            "  cones initial {} final {} cancelled {}",
            frep.cones_initial, frep.cones_final, frep.cancelled
        );
        let lv = aq_loops::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        let mut loops = aq_loops::find_loops(
            &lv,
            &field,
            &curv,
            &size,
            &aq_loops::LoopParams::default(),
            &ctl,
        )
        .unwrap();
        if std::env::var("AQ_NOLOOPS").is_ok() {
            loops.clear();
        }
        let bb = px
            .mesh
            .pos
            .iter()
            .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
                (
                    [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                    [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
                )
            });
        println!("  bbox {:?}", bb);
        let kv = aq_cut::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        let cut = aq_cut::cut_and_comb(&kv, &field, 0).unwrap();
        let pv = aq_param::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        let si: u32 = std::env::var("AQ_STIFF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        let prp = aq_param::ParamParams {
            n_target: n_c,
            stiff_iters: si,
            ..Default::default()
        };
        let (prm, diag) =
            aq_param::seamless_detailed(&pv, &field, &cut, &size, &loops, &prp, &ctl).unwrap();
        println!(
            "comp {ci}: N_c {n_c:.0}, proxy {} faces, cones {}, loops {}, folds {:?} repair {:?}",
            px.mesh.tris.len(),
            frep.cones_final,
            loops.len(),
            diag.fold_history,
            (
                diag.repair.regions,
                diag.repair.repaired,
                diag.repair.skipped_cone,
                diag.repair.failed,
                diag.repair.untangled
            )
        );
        {
            let mut wrong = Vec::new();
            for v in 0..px.mesh.pos.len() {
                let i4 = field.index4[v];
                if i4 == 0 || px.topo.is_bnd_vert[v] {
                    continue;
                }
                let mut su = 0.0;
                for h in px.topo.fan_out(v as u32) {
                    let ff = (h / 3) as usize;
                    let k = (h % 3) as usize;
                    let a = prm.uv[ff][k];
                    let b = prm.uv[ff][(k + 1) % 3];
                    let c = prm.uv[ff][(k + 2) % 3];
                    let (u1, u2) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
                    su += (u1[0] * u2[1] - u1[1] * u2[0]).atan2(u1[0] * u2[0] + u1[1] * u2[1]);
                }
                let exp = 360.0 - 90.0 * i4 as f64;
                if (su.to_degrees() - exp).abs() > 1.0 {
                    wrong.push((v, i4, su.to_degrees().round()));
                }
            }
            println!("  cones with wrong uv angle sum: {:?}", wrong);
        }
        let mut hist = std::collections::BTreeMap::new();
        for v in 0..px.mesh.pos.len() {
            if px.topo.is_bnd_vert[v] {
                let om: f64 = px
                    .topo
                    .fan_out(v as u32)
                    .map(|h| px.frames.corner_angle[h as usize])
                    .sum();
                let i4 = field.index4[v];
                *hist.entry(i4).or_insert(0u32) += 1;
                if i4.abs() >= 2 {
                    println!(
                        "  bnd cone v{v} index4 {i4} omega {:.1} deg corner {}",
                        om.to_degrees(),
                        px.feats.vert_corner[v]
                    );
                }
            }
        }
        println!("  boundary index4 histogram {hist:?}");
        let chain_v: Vec<bool> = {
            let mut c = vec![false; px.mesh.pos.len()];
            for (h, k) in px.feats.he_kind.iter().enumerate() {
                if *k != FeatureKind::None {
                    let t = px.mesh.tris[h / 3];
                    c[t[h % 3] as usize] = true;
                    c[t[(h % 3 + 1) % 3] as usize] = true;
                }
            }
            c
        };
        for (f, &fd) in prm.folded.iter().enumerate() {
            if !fd {
                continue;
            }
            let t = px.mesh.tris[f];
            let desc: Vec<String> = t
                .iter()
                .map(|&v| {
                    let mut s = format!("v{v}");
                    if field.index4[v as usize] != 0 {
                        s += &format!("[cone {}]", field.index4[v as usize]);
                    }
                    if px.topo.is_bnd_vert[v as usize] {
                        s += "[bnd]";
                    }
                    if chain_v[v as usize] {
                        s += "[chain]";
                    }
                    // Distance in rings to the nearest cone.
                    s
                })
                .collect();
            let near_cone = t.iter().any(|&v| {
                px.topo.fan_out(v).any(|h| {
                    let w = px.topo.head(&px.mesh, h);
                    w != NONE && field.index4[w as usize] != 0
                })
            });
            let cut_e = (0..3).filter(|&k| cut.is_cut[3 * f + k]).count();
            println!(
                "  folded face {f}: {} near_cone {near_cone} cut_edges {cut_e}",
                desc.join(" ")
            );
            let c0 = px.mesh.pos[t[0] as usize];
            let mut near: Vec<(f64, usize, i8)> = (0..px.mesh.pos.len())
                .filter(|&v| field.index4[v] != 0)
                .map(|v| (aq_core::math::dist(px.mesh.pos[v], c0), v, field.index4[v]))
                .filter(|x| x.0 < 3.0)
                .collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            println!(
                "    at {:?} h {:.2} cones within 3h: {:?}",
                c0, size.h_vert[t[0] as usize], near
            );
            // uv angle sum (degrees) and 3D angle sum at each corner vertex and the nearest cone.
            let mut vs: Vec<u32> = t.to_vec();
            if let Some(n) = near.first() {
                vs.push(n.1 as u32);
            }
            for &v in &vs {
                let mut su = 0.0;
                let mut s3 = 0.0;
                let mut dets = Vec::new();
                for h in px.topo.fan_out(v) {
                    let ff = (h / 3) as usize;
                    let k = (h % 3) as usize;
                    let a = prm.uv[ff][k];
                    let b = prm.uv[ff][(k + 1) % 3];
                    let c = prm.uv[ff][(k + 2) % 3];
                    let (u1, u2) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
                    let cr = u1[0] * u2[1] - u1[1] * u2[0];
                    let dt = u1[0] * u2[0] + u1[1] * u2[1];
                    su += cr.atan2(dt);
                    s3 += px.frames.corner_angle[h as usize];
                    dets.push(format!("{:.3}", cr));
                }
                println!(
                    "    v{v} index4 {} uv sum {:.1} 3d sum {:.1} dets {}",
                    field.index4[v as usize],
                    su.to_degrees(),
                    s3.to_degrees(),
                    dets.join(",")
                );
            }
        }
    }
}

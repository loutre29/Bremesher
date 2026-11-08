//! WP-23 diagnostic: S2 to S4 exactly as the R0 rung runs them (refined proxy included), then
//! the cone statistics of S4 for one or more field parameter sets.
//!
//! Usage: `pole_probe <mesh.obj> <target> [config ...]`, where each config is a comma separated
//! list of `key=value` overrides of `aq_field::FieldParams` (align, cancel, cancel_low, rounds,
//! repel, cap, merge). The default config is the pipeline's R0 field parameters. Env
//! `AQ_CONES=1` prints every cone of the last config.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicI32, AtomicU32};
use std::time::{Duration, Instant};

use aq_core::{Bvh, Ctl, Projector};

const PROXY_LEN: f64 = 1.0 / 3.0;
const REFINE_H: f64 = 0.7;
const REFINE_AREA: f64 = 0.01;
const REFINE_MIN_H: f64 = 0.25;

fn make_proxy(
    comp: &aq_prep::ComponentMesh,
    proj: &Projector,
    len: &dyn Fn([f64; 3]) -> f64,
    pp: &aq_prep::ProxyParams,
    ctl: &Ctl,
) -> aq_prep::Proxy {
    let mut px = aq_prep::build_proxy(comp, proj, len, pp, ctl).unwrap();
    aq_prep::isolate_features_with(&mut px, Some(proj)).unwrap();
    let ones = vec![1.0; px.mesh.pos.len()];
    aq_prep::classify_chains(&mut px, &ones, pp);
    px
}

fn parse_cfg(s: &str, base: &aq_field::FieldParams) -> aq_field::FieldParams {
    let mut p = base.clone();
    for kv in s.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').expect("key=value");
        let x: f64 = v.parse().unwrap();
        match k {
            "align" => p.align = x,
            "cancel" => p.cancel_dist = x,
            "cancel_low" => p.cancel_dist_lowconf = x,
            "rounds" => p.max_cancel_rounds = x as u32,
            "cancel_low_conf" => p.cancel_dist_lowconf = x,
            "far" => p.far_cancel = x,
            "far_e" => p.far_cancel_energy = x,
            "merge" => p.merge_neg = x,
            _ => panic!("unknown key {k}"),
        }
    }
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let target: u32 = args[2].parse().unwrap();
    let cfgs: Vec<String> = if args.len() > 3 { args[3..].to_vec() } else { vec![String::new()] };
    let cancel = AtomicI32::new(0);
    let prog = AtomicU32::new(0);
    let stage = AtomicU32::new(0);
    let ctl = Ctl::new(&cancel, &prog, &stage, Instant::now() + Duration::from_secs(3600));
    let mut m = aq_io::read_obj_tri(std::path::Path::new(path)).unwrap();
    aq_engine::host::normalize(&mut m);
    let ing = aq_prep::ingest(m, &[], &[], &ctl).unwrap();
    let comps = aq_prep::split_components(&ing);
    let areas: Vec<f64> = comps.iter().map(|c| c.area).collect();
    let closed: Vec<bool> = comps.iter().map(|c| c.closed).collect();
    let plan = aq_engine::budget::plan_targets(&areas, &closed, target);
    let shares = aq_engine::budget::proxy_shares(&plan, 150_000);
    let base = aq_field::FieldParams { align: 0.5, ..Default::default() };
    let mut totals = vec![[0u32; 6]; cfgs.len()];
    for (ci, c0) in comps.iter().enumerate() {
        if plan[ci].debris {
            continue;
        }
        let n_c = plan[ci].target as f64;
        let comp = c0.scaled((n_c / c0.area).sqrt());
        let t = Instant::now();
        let (fr, bvh) = comp.source_geometry().unwrap();
        let proj = Projector::new(&comp.mesh, &comp.topo, &fr, &bvh);
        let pp = aq_prep::ProxyParams { max_tris: shares[ci], ..Default::default() };
        let mut px = make_proxy(&comp, &proj, &|_| PROXY_LEN, &pp, &ctl);
        let cp = aq_curv::CurvParams { scale: std::env::var("AQ_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0), ..Default::default() };
        let sp = aq_curv::SizingParams::new(n_c);
        let cview = aq_curv::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        let (mut curv, mut size, _) = aq_curv::curvature_and_sizing(&cview, None, &cp, &sp, &ctl).unwrap();
        let small: f64 = (0..px.mesh.tris.len()).filter(|&f| size.h_face[f] < REFINE_H).map(|f| px.frames.area[f]).sum();
        let area: f64 = px.frames.area.iter().sum();
        if small > REFINE_AREA * area {
            let bvh1 = Bvh::try_build(&px.mesh).unwrap();
            let hf = size.h_face.clone();
            let len_at = |q: [f64; 3]| {
                let (sp, _, _) = bvh1.closest(q);
                PROXY_LEN * hf.get(sp.face as usize).copied().unwrap_or(1.0).clamp(REFINE_MIN_H, 1.0)
            };
            px = make_proxy(&comp, &proj, &len_at, &pp, &ctl);
            let cview = aq_curv::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
            (curv, size, _) = aq_curv::curvature_and_sizing(&cview, None, &cp, &sp, &ctl).unwrap();
        }
        println!("comp {ci}: N_c {n_c}, proxy {} faces, S2+S3 {:.2} s", px.mesh.tris.len(), t.elapsed().as_secs_f64());
        let fv = aq_field::ProxyRef::new(&px.mesh, &px.topo, &px.frames, &px.feats);
        for (k, cfg) in cfgs.iter().enumerate() {
            let fp = parse_cfg(cfg, &base);
            let t = Instant::now();
            let (field, rep) = aq_field::solve_field(&fv, &curv, &size, &fp, &ctl).unwrap();
            let dt = t.elapsed().as_secs_f64();
            let nv = px.mesh.pos.len();
            let (mut ip, mut ineg, mut bnd) = (0u32, 0u32, 0u32);
            for v in 0..nv {
                let i = field.index4[v];
                if i == 0 {
                    continue;
                }
                if px.topo.is_bnd_vert[v] || px.feats.vert_corner[v] {
                    bnd += 1;
                } else if i > 0 {
                    ip += 1;
                } else {
                    ineg += 1;
                }
            }
            let tt = &mut totals[k];
            tt[0] += ip;
            tt[1] += ineg;
            tt[2] += bnd;
            tt[3] += rep.cancelled;
            tt[4] += rep.reverted;
            tt[5] += rep.pushed;
            println!(
                "  [{cfg}] init {} final {} (+{ip} -{ineg} bnd {bnd}) cancelled {} reverted {} pushed {} split {} max|d| {:.3} excess {} {dt:.2} s",
                rep.cones_initial, rep.cones_final, rep.cancelled, rep.reverted, rep.pushed, rep.split, rep.max_abs_delta, rep.excess_poles
            );
            if std::env::var("AQ_CONES").is_ok() && k + 1 == cfgs.len() {
                for v in 0..nv {
                    if field.index4[v] != 0 {
                        let p = px.mesh.pos[v];
                        let om: f64 = px.topo.fan_out(v as u32).map(|h| px.frames.corner_angle[h as usize]).sum();
                        let tag = if px.topo.is_bnd_vert[v] {
                            format!(" bnd omega {:.0} ideal {}", om.to_degrees(), 2 - (om / std::f64::consts::FRAC_PI_2).round().max(1.0) as i32)
                        } else {
                            String::new()
                        };
                        let r = 1.5 * size.h_vert[v];
                        let kq: f64 = (0..nv)
                            .filter(|&u| !px.topo.is_bnd_vert[u] && aq_core::math::dist(px.mesh.pos[u], p) < r)
                            .map(|u| 2.0 * std::f64::consts::PI - px.frames.angle_sum(&px.topo, u as u32))
                            .sum::<f64>()
                            / std::f64::consts::FRAC_PI_2;
                        println!("    cone v{v} i{} at [{:.2}, {:.2}, {:.2}] h {:.2} K1.5 {kq:.2}{tag}", field.index4[v], p[0], p[1], p[2], size.h_vert[v]);
                    }
                }
            }
        }
    }
    let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
    for (k, cfg) in cfgs.iter().enumerate() {
        let t = totals[k];
        println!(
            "{name} [{cfg}]: interior {} (+{} -{}) bnd/corner {} cancelled {} reverted {} pushed {}",
            t[0] + t[1], t[0], t[1], t[2], t[3], t[4], t[5]
        );
    }
}

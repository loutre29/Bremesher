//! DESIGN 6.1: transitions, BVH, projector; plus Ctl and rng behaviour.

mod common;
use common::*;

use std::time::Duration;

use aq_core::math;
use aq_core::tol::{DYADIC_QUANTUM, is_dyadic, snap_dyadic};
use aq_core::{Bvh, CtlState, ErrKind, Projector, SplitMix64, Stage, Transition, TriMesh, rot};

fn dyadic(rng: &mut SplitMix64, lim: f64) -> f64 {
    snap_dyadic(rng.range_f64(-lim, lim))
}

#[test]
fn rotations() {
    assert_eq!(rot(1, [1.0, 2.0]), [-2.0, 1.0]);
    assert_eq!(rot(2, [1.0, 2.0]), [-1.0, -2.0]);
    assert_eq!(rot(3, [1.0, 2.0]), [2.0, -1.0]);
    assert_eq!(rot(4, [1.0, 2.0]), [1.0, 2.0]);
    for k in 0..4u8 {
        let a = [0.25, -3.5];
        assert_eq!(rot((4 - k) % 4, rot(k, a)), a);
    }
}

#[test]
fn transitions_exact() {
    let mut rng = SplitMix64::new(42);
    for _ in 0..10_000 {
        let t = Transition { r: rng.below(4) as u8, t: [dyadic(&mut rng, 1e6), dyadic(&mut rng, 1e6)] };
        let id = t.compose(&t.inverse());
        assert_eq!(id.r, 0);
        assert_eq!(id.t[0].to_bits(), 0.0f64.to_bits(), "compose(T, T^-1) not identity: {id:?}");
        assert_eq!(id.t[1].to_bits(), 0.0f64.to_bits());
        let id2 = t.inverse().compose(&t);
        assert!(id2.is_identity());
        // Twin formula round-trips exactly.
        let back = t.inverse().inverse();
        assert_eq!(back.r, t.r);
        assert_eq!(back.t[0].to_bits(), (t.t[0] + 0.0).to_bits());
        assert_eq!(back.t[1].to_bits(), (t.t[1] + 0.0).to_bits());
        // apply and inverse apply round-trip on dyadic points.
        let p = [dyadic(&mut rng, 1e6), dyadic(&mut rng, 1e6)];
        let q = t.apply(p);
        assert!(is_dyadic(q[0]) && is_dyadic(q[1]));
        assert_eq!(t.inverse().apply(q), p);
        // C10 twin formula: r' = (4 - r) % 4, t' = -R(r) t.
        let rt = rot(t.r, t.t);
        assert_eq!(t.inverse(), Transition { r: (4 - t.r) % 4, t: [-rt[0], -rt[1]] });
        // compose == sequential application.
        let s = Transition { r: rng.below(4) as u8, t: [dyadic(&mut rng, 1e5), dyadic(&mut rng, 1e5)] };
        assert_eq!(t.compose(&s).apply(p), s.apply(t.apply(p)));
    }
    // phi_g = R(-r) phi_f + t: r = 1 rotates by -90 degrees.
    let t = Transition { r: 1, t: [10.0, 0.0] };
    assert_eq!(t.apply([1.0, 0.0]), [10.0, -1.0]);
    assert_eq!(DYADIC_QUANTUM, 2f64.powi(-24));
}

#[test]
fn bvh_matches_brute_force() {
    let mut m = icosphere(4, 1.0);
    add_radial_noise(&mut m, 0.05, 3);
    let bvh = Bvh::build(&m);
    let all: Vec<u32> = (0..m.tris.len() as u32).collect();
    let mut rng = SplitMix64::new(9);
    for _ in 0..10_000 {
        let p = [rng.range_f64(-1.5, 1.5), rng.range_f64(-1.5, 1.5), rng.range_f64(-1.5, 1.5)];
        let (sp, q, d) = bvh.closest(p);
        let (bsp, bq, bd) = bvh.closest_among(p, &all).unwrap();
        assert!((d - bd).abs() < 1e-12, "bvh {d} vs brute {bd}");
        assert!(math::dist(q, bq) < 1e-9 || sp.face != bsp.face);
        assert!((math::dist(p, q) - d).abs() < 1e-12);
        // Barycentrics reproduce the point.
        let pp = sp.position(&m);
        assert!(math::dist(pp, q) < 1e-12);
        sp.validate(m.tris.len()).unwrap();
    }
    let empty = Bvh::build(&TriMesh::default());
    assert_eq!(empty.closest([0.0; 3]).2, f64::INFINITY);
}

#[test]
fn projector_sphere_agrees_with_bvh() {
    let c = ctx(icosphere(4, 1.0));
    let bvh = Bvh::build(&c.m);
    let pr = Projector::new(&c.m, &c.t, &c.fr, &bvh);
    let mut rng = SplitMix64::new(5);
    let n = 20_000;
    let mut agree = 0;
    for _ in 0..n {
        let f = rng.below(c.m.tris.len() as u64) as u32;
        let (mut a, mut b) = (rng.next_f64(), rng.next_f64());
        if a + b > 1.0 {
            a = 1.0 - a;
            b = 1.0 - b;
        }
        let [p0, p1, p2] = c.m.tri_pos(f);
        let s = math::bary_point(p0, p1, p2, [1.0 - a - b, a, b]);
        let nrm = math::normalize(s).unwrap();
        let p = math::add(s, math::scale(nrm, rng.range_f64(-0.05, 0.05)));
        let (gsp, gq, _) = bvh.closest(p);
        if let Some((sp, q)) = pr.project(p, f, nrm) {
            if sp.face == gsp.face || math::dist(q, gq) < 1e-9 {
                agree += 1;
            }
        }
    }
    let frac = agree as f64 / n as f64;
    assert!(frac >= 0.999, "projector agrees with the BVH on {frac}");
}

#[test]
fn projector_two_parallel_planes() {
    // Plane 1 at z = 0, plane 2 at z = 0.05, same orientation, separate components.
    let p1 = plane_grid(0.0, 1.0, 0.0, 1.0, 20, 20, 0.0, 0);
    let mut m = p1.clone();
    let off = m.pos.len() as u32;
    m.pos.extend(p1.pos.iter().map(|p| [p[0], p[1], 0.05]));
    m.tris.extend(p1.tris.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    let n1 = p1.tris.len() as u32;
    let c = ctx(m);
    let bvh = Bvh::build(&c.m);
    let pr = Projector::new(&c.m, &c.t, &c.fr, &bvh);
    let mut rng = SplitMix64::new(77);
    let mut on2 = 0;
    let mut none = 0;
    for _ in 0..5000 {
        let (x, y) = (rng.range_f64(0.1, 0.9), rng.range_f64(0.1, 0.9));
        let p = [x, y, 0.03];
        // Hint: the plane-1 face under (x, y).
        let hint = (0..n1)
            .find(|&f| {
                let [a, b, cc] = c.m.tri_pos(f);
                let o = |u: [f64; 3], v: [f64; 3]| (v[0] - u[0]) * (y - u[1]) - (v[1] - u[1]) * (x - u[0]);
                o(a, b) >= 0.0 && o(b, cc) >= 0.0 && o(cc, a) >= 0.0
            })
            .unwrap();
        // The global closest point is on plane 2.
        assert!(bvh.closest(p).0.face >= n1);
        match pr.project(p, hint, [0.0, 0.0, 1.0]) {
            Some((sp, q)) => {
                if sp.face >= n1 {
                    on2 += 1;
                }
                assert!(q[2].abs() < 1e-12);
            }
            None => none += 1,
        }
    }
    assert_eq!(on2, 0);
    assert_eq!(none, 0);
    // A wrong normal fails both the walk and the BVH: None.
    assert!(pr.project([0.5, 0.5, 0.03], 0, [0.0, 0.0, -1.0]).is_none());
}

#[test]
fn ctl_behaviour() {
    let st = CtlState::new();
    let ctl = st.unbounded();
    ctl.check().unwrap();
    ctl.progress(Stage::Field, 0.25);
    assert_eq!(st.progress_frac(), 0.25);
    assert_eq!(st.stage.load(std::sync::atomic::Ordering::Relaxed), Stage::Field as u32);
    ctl.progress(Stage::Field, f32::NAN);
    assert_eq!(st.progress_frac(), 0.0);
    ctl.progress(Stage::Param, 7.0);
    assert_eq!(st.progress_frac(), 1.0);
    st.cancel();
    let e = ctl.check().unwrap_err();
    assert_eq!(e.kind, ErrKind::Cancelled);
    assert_eq!(e.stage, Stage::Param);
    let st = CtlState::new();
    let ctl = st.ctl_for(Duration::from_secs(0));
    assert_eq!(ctl.check().unwrap_err().kind, ErrKind::Timeout);
    let ctl = st.ctl_for(Duration::from_secs(1000));
    assert!(ctl.remaining() > Duration::from_secs(900));
    let sub = ctl.sub(std::time::Instant::now());
    assert_eq!(sub.check().unwrap_err().kind, ErrKind::Timeout);
    for s in Stage::ALL {
        assert_eq!(Stage::from_u32(s as u32), Some(s));
    }
}

#[test]
fn rng_deterministic() {
    let mut a = SplitMix64::new(0);
    // Reference values of SplitMix64 with seed 0.
    assert_eq!(a.next_u64(), 0xE220_A839_7B1D_CDAF);
    assert_eq!(a.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    let mut b = SplitMix64::new(123);
    let mut c = SplitMix64::new(123);
    let mut v1: Vec<u32> = (0..100).collect();
    let mut v2 = v1.clone();
    b.shuffle(&mut v1);
    c.shuffle(&mut v2);
    assert_eq!(v1, v2);
    for _ in 0..1000 {
        let x = b.below(7);
        assert!(x < 7);
        let f = b.next_f64();
        assert!((0.0..1.0).contains(&f));
    }
}

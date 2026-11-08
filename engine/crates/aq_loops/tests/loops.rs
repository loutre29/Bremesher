//! DESIGN 6.7 tests for aq_loops.

use std::f64::consts::{PI, TAU};

use aq_core::math;
use aq_core::{CrossField, CtlState, CurvField, FeatureKind, NONE, SizeField, SurfacePoint, Topology, V3};
use aq_loops::{
    CycleKind, LoopInfo, LoopParams, LoopReport, MAX_TURNING_EXCESS, ProxyRef, StopReason, find_loops,
    find_loops_report, loop_turning, trace_streamline, turning_check,
};
use aq_testkit::{CurvKind, FieldKind, PlantChart, Surface, analytic_curv, analytic_field, uniform_size};

fn px(s: &Surface) -> ProxyRef<'_> {
    ProxyRef::new(&s.mesh, &s.topo, &s.frames, &s.feats)
}

struct Case {
    s: Surface,
    f: CrossField,
    c: CurvField,
    z: SizeField,
}

fn case(s: Surface, fk: FieldKind, ck: CurvKind) -> Case {
    let f = analytic_field(&s, &fk).unwrap();
    let c = analytic_curv(&s, &ck).unwrap();
    let z = uniform_size(&s, 1.0).unwrap();
    Case { s, f, c, z }
}

fn run(cs: &Case, p: &LoopParams) -> (Vec<LoopInfo>, LoopReport) {
    let st = CtlState::new();
    let (loops, rep) = find_loops_report(&px(&cs.s), &cs.f, &cs.c, &cs.z, p, &st.unbounded()).unwrap();
    for l in &loops {
        l.cand.validate(&cs.s.mesh, &cs.s.topo).unwrap();
        assert!(l.closure_err < p.close_tol);
        assert!(l.turning_excess <= MAX_TURNING_EXCESS, "turning excess {}", l.turning_excess);
        let (tt, _) = loop_turning(&px(&cs.s), &cs.f, &l.cand.hes);
        assert_eq!(tt, l.turning);
        let tc = turning_check(&px(&cs.s), &cs.f, &l.cand.hes, &st.unbounded()).unwrap();
        assert_eq!(tc.kind, l.cycle);
        assert_eq!(tc.excess, l.turning_excess);
        assert_eq!(tc.expected, l.expected_turning);
        match l.cycle {
            // The excess is a Poincare-Hopf residual of the enclosed side (a multiple of 90
            // degrees), so an accepted separating loop matches Gauss-Bonnet exactly.
            CycleKind::Separating => assert!(l.turning_excess < 1e-6, "excess {}", l.turning_excess),
            CycleKind::NonSeparating => assert_eq!(l.turning_excess, 0.0),
            CycleKind::Overlapping => panic!("an overlapping cycle was accepted"),
        }
        // The cycle returns to its branch.
        let k = l.cand.hes.iter().fold(l.cand.branch, |k, &h| aq_core::cross::map_branch(k, cs.f.p[h as usize]));
        assert_eq!(k, l.cand.branch);
    }
    check_clustering(&loops, p.cluster);
    (loops, rep)
}

/// DESIGN 6.7 clustering test: no two accepted loops of the same family within `cluster` over
/// 80% of the length of either one (brute force over the samples).
fn check_clustering(loops: &[LoopInfo], cluster: f64) {
    // Fraction of a's length whose nearest point on b lies within `cluster` with a parallel
    // tangent (same family).
    let near_frac = |a: &LoopInfo, b: &LoopInfo| -> f64 {
        let total: f64 = a.samples.iter().map(|x| x.2).sum();
        let near: f64 = a
            .samples
            .iter()
            .filter(|(q, t, _)| {
                let nb = b
                    .samples
                    .iter()
                    .min_by(|x, y| math::dist(*q, x.0).total_cmp(&math::dist(*q, y.0)))
                    .unwrap();
                math::dist(*q, nb.0) <= cluster && math::dot(*t, nb.1).abs() >= std::f64::consts::FRAC_1_SQRT_2
            })
            .map(|x| x.2)
            .sum();
        near / total
    };
    for i in 0..loops.len() {
        for j in 0..i {
            // The later loop (j accepted before i) must not be covered by the earlier one.
            let fr = near_frac(&loops[i], &loops[j]);
            assert!(fr < 0.8, "loops {j} and {i} cluster ({fr:.2} of the length within {cluster})");
        }
    }
}

/// Mean |cos| between the loop tangent and the circumferential direction about z.
fn circumferential(l: &LoopInfo) -> f64 {
    let mut w = 0.0;
    let mut c = 0.0;
    for (q, t, len) in &l.samples {
        let phi = math::normalize([-q[1], q[0], 0.0]).unwrap_or([1.0, 0.0, 0.0]);
        c += len * math::dot(*t, phi).abs();
        w += len;
    }
    c / w
}

fn loop_faces_touch_cone(cs: &Case, l: &LoopInfo) -> bool {
    let mut face = l.cand.start_face;
    let mut faces = vec![face];
    for &h in &l.cand.hes {
        face = cs.s.topo.twin[h as usize] / 3;
        faces.push(face);
    }
    faces.iter().any(|&f| cs.s.mesh.tris[f as usize].iter().any(|&v| cs.f.index4[v as usize] != 0))
}

#[test]
fn torus_circumferential() {
    let (big_r, r) = (4.0, 1.5);
    let nu = (3.0 * TAU * big_r).round() as u32;
    let nv = (3.0 * TAU * r).round() as u32;
    let s = Surface::new(aq_testkit::torus(big_r, r, nu, nv, 0.2, 3).unwrap()).unwrap();
    let cs = case(s, FieldKind::TorusCircumferential, CurvKind::Torus { big_r, r });
    let (loops, rep) = run(&cs, &LoopParams::default());
    eprintln!("torus: {rep:?}");
    let mut merid = 0;
    let mut longi = 0;
    for l in &loops {
        assert!(l.closure_err < 0.05, "closure error {} h", l.closure_err);
        // Meridians and big circles do not bound anything: no Gauss-Bonnet prediction.
        assert_eq!(l.cycle, CycleKind::NonSeparating);
        let c = circumferential(l);
        if c > 0.9 {
            longi += 1;
            assert!(!l.cand.ring, "a big circle is not a ring");
        } else {
            assert!(c < 0.1, "loop is neither family ({c})");
            merid += 1;
            assert!(l.cand.ring, "meridians are rings");
            assert!((l.cand.length - TAU * r).abs() < 0.05 * TAU * r, "meridian length {}", l.cand.length);
        }
    }
    eprintln!("torus: {merid} meridians, {longi} longitudes");
    // At least one loop per family per 6h of extent.
    assert!(merid >= (TAU * big_r / 6.0).floor() as usize, "{merid} meridians");
    assert!(longi >= (TAU * r / 6.0).floor().max(1.0) as usize, "{longi} longitudes");
}

#[test]
fn cylinders_rings() {
    let (rad, len) = (1.5, 18.0);
    let na = (3.0 * TAU * rad).round() as u32;
    let nl = (3.0 * len) as u32;
    for closed in [false, true] {
        let m = if closed {
            aq_testkit::cylinder_closed(rad, len, na, nl, 0.2, 5).unwrap()
        } else {
            aq_testkit::cylinder_open(rad, len, na, nl, 0.2, 5).unwrap()
        };
        let cs = case(Surface::new(m).unwrap(), FieldKind::CylinderCircumferential, CurvKind::Cylinder { radius: rad });
        let (loops, rep) = run(&cs, &LoopParams::default());
        // Rings of the side. The flat caps of the closed cylinder may also carry concentric
        // (eye-like) loops: those are separating with turning 2pi minus the enclosed curvature.
        let on_side = |l: &LoopInfo| l.samples.iter().all(|x| (x.0[0].hypot(x.0[1]) - rad).abs() < 0.1);
        let rings: Vec<_> = loops.iter().filter(|l| l.cand.ring && circumferential(l) > 0.9 && on_side(l)).collect();
        for l in loops.iter().filter(|l| !on_side(l)) {
            eprintln!(
                "cylinder closed={closed}: cap loop, length {:.2}, turning {:.4}, Gauss-Bonnet {:.4}",
                l.cand.length, l.turning, l.expected_turning
            );
            assert!(closed, "an open cylinder has no loop off its side");
            assert_eq!(l.cycle, CycleKind::Separating);
        }
        eprintln!("cylinder closed={closed}: {rep:?}, {} rings", rings.len());
        // One circumferential loop per 6h of length. On the closed cylinder a cap loop near the
        // rim is shorter, so it is accepted first and clusters out the side ring next to it.
        let circ = loops.iter().filter(|l| circumferential(l) > 0.9).count();
        assert!(circ >= (len / 6.0).floor() as usize, "closed={closed}: {circ} circumferential loops");
        let min_rings = if closed { 1 } else { (len / 6.0).floor() as usize };
        assert!(rings.len() >= min_rings, "closed={closed}: {} rings", rings.len());
        for l in &rings {
            assert!((l.cand.length - TAU * rad).abs() < 0.05 * TAU * rad);
            assert!(!loop_faces_touch_cone(&cs, l));
            // A ring separates the two ends; a cylinder ring is a geodesic (up to the jitter,
            // which makes the proxy slightly non-developable).
            assert_eq!(l.cycle, CycleKind::Separating);
            assert!(l.turning.abs() < 0.02, "ring turning {}", l.turning);
        }
    }
}

#[test]
fn twisted_cylinder() {
    let (rad, len) = (1.5, 18.0);
    let na = (3.0 * TAU * rad).round() as u32;
    let nl = (3.0 * len) as u32;
    let m = aq_testkit::cylinder_open(rad, len, na, nl, 0.2, 7).unwrap();
    let s = Surface::new(m).unwrap();
    let cs = case(s.clone(), FieldKind::CylinderTwisted { pitch: 2.0 }, CurvKind::Cylinder { radius: rad });
    let (loops, rep) = run(&cs, &LoopParams::default());
    eprintln!("twisted pitch 2h: {rep:?}");
    assert!(loops.is_empty(), "pitch 2h: {} loops accepted", loops.len());
    let cs = case(s, FieldKind::CylinderTwisted { pitch: 0.2 }, CurvKind::Cylinder { radius: rad });
    let (loops, rep) = run(&cs, &LoopParams::default());
    eprintln!("twisted pitch 0.2h: {rep:?}");
    assert!(!loops.is_empty(), "pitch 0.2h: no loop accepted");
}

#[test]
fn sphere_with_8_cones() {
    let pc = aq_testkit::cube_sphere_param_with(6, aq_testkit::CubeSphereOpts { res: 3, jitter: 0.2, seed: 3, radius: None })
        .unwrap();
    let s = pc.surface.clone();
    let f = pc.field.clone();
    assert_eq!(f.cones().len(), 8);
    let rad = s.mesh.pos.iter().map(|p| math::norm(*p)).sum::<f64>() / s.n_verts() as f64;
    let mut c = analytic_curv(&s, &CurvKind::Sphere { center: [0.0; 3], radius: rad }).unwrap();
    // A sphere has no confident direction; force seeds everywhere to exercise the tracer.
    c.conf.iter_mut().for_each(|x| *x = 1.0);
    let z = uniform_size(&s, 1.0).unwrap();
    let cs = Case { s, f, c, z };
    let (loops, rep) = run(&cs, &LoopParams::default());
    eprintln!("cube sphere: {rep:?}");
    assert!(rep.seeds > 10);
    assert!(!loops.is_empty());
    for l in &loops {
        assert!(!loop_faces_touch_cone(&cs, l), "a loop passes through a cone face");
        // DESIGN 6.7 "turning < 60 degrees", measured against Gauss-Bonnet: every closed field
        // line on a sphere separates it.
        assert_eq!(l.cycle, CycleKind::Separating);
        assert!(l.turning_excess < MAX_TURNING_EXCESS);
        // Independent check of the real turning: T = 2pi - area(left side) / R^2, with the
        // area from the loop's own samples (signed spherical triangles fanned from a point off
        // the loop, modulo 4pi).
        let q: Vec<V3> = l.samples.iter().map(|x| math::normalize(x.0).unwrap()).collect();
        let z = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
            .into_iter()
            .max_by(|a: &V3, b: &V3| {
                let da = q.iter().map(|p| math::dist(*p, *a)).fold(f64::INFINITY, f64::min);
                let db = q.iter().map(|p| math::dist(*p, *b)).fold(f64::INFINITY, f64::min);
                da.total_cmp(&db)
            })
            .unwrap();
        let mut area = 0.0;
        for i in 0..q.len() {
            let (a, b) = (q[i], q[(i + 1) % q.len()]);
            let num = math::dot(z, math::cross(a, b));
            let den = 1.0 + math::dot(z, a) + math::dot(a, b) + math::dot(b, z);
            area += 2.0 * num.atan2(den);
        }
        let area = area.rem_euclid(2.0 * TAU);
        let t_sphere = TAU - area;
        eprintln!(
            "cube sphere loop: turning {:.4}, Gauss-Bonnet {:.4}, sphere area {:.4}, length {:.2}",
            l.turning, l.expected_turning, t_sphere, l.cand.length
        );
        assert!((l.turning - t_sphere).abs() < 0.3, "turning {} vs sphere {}", l.turning, t_sphere);
    }
    // With the real (zero) confidence there are no seeds at all.
    let c0 = analytic_curv(&cs.s, &CurvKind::Sphere { center: [0.0; 3], radius: rad }).unwrap();
    let st = CtlState::new();
    let none = find_loops(&px(&cs.s), &cs.f, &c0, &cs.z, &LoopParams::default(), &st.unbounded()).unwrap();
    assert!(none.is_empty());
}

#[test]
fn noisy_cylinder() {
    let (rad, len) = (1.5, 18.0);
    let na = (3.0 * TAU * rad).round() as u32;
    let nl = (3.0 * len) as u32;
    let mut m = aq_testkit::cylinder_open(rad, len, na, nl, 0.2, 9).unwrap();
    aq_testkit::add_noise(&mut m, 0.05, 1.0, 4).unwrap();
    let cs = case(Surface::new(m).unwrap(), FieldKind::CylinderCircumferential, CurvKind::Cylinder { radius: rad });
    let (loops, rep) = run(&cs, &LoopParams::default());
    let rings = loops.iter().filter(|l| l.cand.ring && circumferential(l) > 0.8).count();
    eprintln!("noisy cylinder: {rep:?}, {rings} rings");
    assert!(rings >= (len / 6.0).floor() as usize, "{rings} rings");
}

#[test]
fn deterministic() {
    let (big_r, r) = (4.0, 1.5);
    let s = Surface::new(aq_testkit::torus(big_r, r, 75, 28, 0.2, 3).unwrap()).unwrap();
    let cs = case(s, FieldKind::TorusCircumferential, CurvKind::Torus { big_r, r });
    let p = LoopParams { seed: 17, ..Default::default() };
    let st = CtlState::new();
    let go = || find_loops(&px(&cs.s), &cs.f, &cs.c, &cs.z, &p, &st.unbounded()).unwrap();
    let a = go();
    assert!(!a.is_empty());
    assert_eq!(a, go());
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        assert_eq!(a, pool.install(go), "{threads} threads");
    }
}

#[test]
fn trace_on_plane() {
    // Constant field at 30 degrees on a jittered plane: a straight line to the boundary.
    let ang = 30f64.to_radians();
    let s = Surface::new(aq_testkit::plane_grid(30, 30, 0.33, 0.0, 0.2, 2).unwrap()).unwrap();
    let f = analytic_field(&s, &FieldKind::Constant { angle: ang }).unwrap();
    let start = (0..s.n_faces() as u32)
        .find(|&fc| {
            let [a, b, c] = s.mesh.tri_pos(fc);
            let cen = math::scale(math::add(math::add(a, b), c), 1.0 / 3.0);
            math::dist(cen, [3.0, 3.0, 0.0]) < 0.2 && !f.hard[fc as usize]
        })
        .unwrap();
    let from = SurfacePoint { face: start, bary: [1.0 / 3.0; 3] };
    let p0 = from.position(&s.mesh);
    let sl = trace_streamline(&px(&s), &f, from, 0, 100.0);
    assert_eq!(sl.stop, StopReason::Boundary);
    assert_eq!(sl.points.len(), sl.faces.len() + 1);
    assert_eq!(sl.hes.len() + 1, sl.faces.len());
    let end = *sl.points.last().unwrap();
    let d = math::sub(end, p0);
    // Straight except inside the boundary-snapped faces (one row, about 0.33 wide).
    let dir = [ang.cos(), ang.sin(), 0.0];
    let off = math::norm(math::sub(d, math::scale(dir, math::dot(d, dir))));
    assert!(off < 0.35, "off-line by {off}");
    assert!((sl.length - math::norm(d)).abs() < 0.1);
    for (i, &h) in sl.hes.iter().enumerate() {
        assert_eq!(h / 3, sl.faces[i]);
        assert_eq!(s.topo.twin[h as usize] / 3, sl.faces[i + 1]);
        assert_eq!(sl.branches[i + 1], aq_core::cross::map_branch(sl.branches[i], f.p[h as usize]));
    }
    // Max length stops exactly.
    let sl = trace_streamline(&px(&s), &f, from, 1, 1.25);
    assert_eq!(sl.stop, StopReason::MaxLen);
    assert!((sl.length - 1.25).abs() < 1e-9);
    // Invalid starts never panic.
    let bad = trace_streamline(&px(&s), &f, SurfacePoint { face: NONE, bary: [1.0 / 3.0; 3] }, 0, 1.0);
    assert_eq!(bad.stop, StopReason::Invalid);
}

#[test]
fn trace_through_vertices() {
    // Unjittered grid, diagonal direction: the ray passes exactly through grid vertices and
    // must be nudged off them.
    let s = Surface::new(aq_testkit::plane_grid(20, 20, 0.5, 0.0, 0.0, 0).unwrap()).unwrap();
    let ang = PI / 4.0;
    let f = analytic_field(&s, &FieldKind::Constant { angle: ang }).unwrap();
    // A point on the 45 degree line y = x + 3.5, which runs corner to corner through cells
    // whose diagonal is the other one, so it hits grid vertices exactly (and ends on the top
    // side, away from the corner cones).
    let p: V3 = [2.3, 5.8, 0.0];
    let mut from = None;
    for fc in 0..s.n_faces() as u32 {
        let [a, b, c] = s.mesh.tri_pos(fc);
        let ar = |u: V3, v: V3, w: V3| (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
        let t = ar(a, b, c);
        let l = [ar(p, b, c) / t, ar(a, p, c) / t, ar(a, b, p) / t];
        if l.iter().all(|&x| x > 1e-6) {
            from = Some(SurfacePoint { face: fc, bary: l });
        }
    }
    let from = from.unwrap();
    let sl = trace_streamline(&px(&s), &f, from, 0, 100.0);
    // The 45 degree field snapped to the boundary rows plants boundary cones (the snap is a
    // tie), so the trace ends at the boundary or just before a boundary cone face.
    assert!(matches!(sl.stop, StopReason::Boundary | StopReason::ConeFace), "{:?}", sl.stop);
    let end = *sl.points.last().unwrap();
    assert!(end[1] >= 9.0 - 1e-6, "stopped early at {end:?}");
    // It passed through 7 grid vertices, each nudged by a few 1e-9 to the left of the ray.
    for q in &sl.points[1..] {
        let off = (q[1] - q[0] - 3.5) / 2f64.sqrt();
        assert!(off.abs() < 1e-8, "off the line by {off}");
    }
}

#[test]
fn performance_100k_faces() {
    // A 100k-face proxy at about h/3: torus with R = 31.8 h, r = 4.46 h.
    let (nu, nv) = (600u32, 84u32);
    let big_r = nu as f64 / (3.0 * TAU);
    let r = nv as f64 / (3.0 * TAU);
    let s = Surface::new(aq_testkit::torus(big_r, r, nu, nv, 0.2, 21).unwrap()).unwrap();
    assert_eq!(s.n_faces(), 100_800);
    let cs = case(s, FieldKind::TorusCircumferential, CurvKind::Torus { big_r, r });
    let st = CtlState::new();
    let mut best = f64::INFINITY;
    let mut rep = LoopReport::default();
    for _ in 0..2 {
        let t0 = std::time::Instant::now();
        let (l, rp) = find_loops_report(&px(&cs.s), &cs.f, &cs.c, &cs.z, &LoopParams::default(), &st.unbounded()).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        rep = rp;
        std::hint::black_box(l);
    }
    eprintln!("find_loops 100k faces: {best:.3} s, {rep:?}");
    assert!(rep.accepted > 0);
    if !cfg!(debug_assertions) {
        assert!(best < 0.5, "find_loops took {best:.3} s");
    }
}

/// Face containing the 2D point (x, y) of a planar mesh in z = 0, with its barycentric
/// coordinates.
fn locate(s: &Surface, x: f64, y: f64) -> SurfacePoint {
    let p: V3 = [x, y, 0.0];
    for fc in 0..s.n_faces() as u32 {
        let [a, b, c] = s.mesh.tri_pos(fc);
        let ar = |u: V3, v: V3, w: V3| (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
        let t = ar(a, b, c);
        let l = [ar(p, b, c) / t, ar(a, p, c) / t, ar(a, b, p) / t];
        if l.iter().all(|&x| x > 1e-6) {
            return SurfacePoint { face: fc, bary: l };
        }
    }
    panic!("no face at ({x}, {y})");
}

#[test]
fn planar_concentric_eye_loops() {
    // Eye-like loops: a flat disk with the concentric field around one index-1 vertex (branch 0
    // radial, branch 1 along CCW circles). Every circle turns by 2pi, which is exactly what
    // Gauss-Bonnet asks of a closed curve around a flat disk, so the loops must be accepted.
    let n = 60u32;
    let cell = 1.0 / 3.0;
    let s = Surface::new(aq_testkit::plane_grid(n, n, cell, 0.0, 0.2, 4).unwrap()).unwrap();
    let cv = (n / 2) * (n + 1) + n / 2;
    let kind = FieldKind::Planted {
        base: Box::new(FieldKind::Constant { angle: 0.0 }),
        sings: vec![(cv, 4)],
        chart: PlantChart::Plane,
    };
    let f = analytic_field(&s, &kind).unwrap();
    // A valence-6 vertex cannot hold the whole index 1 (4 quarter turns over 6 faces), so it
    // splits into cones next to it; together they hold index4 4, and nothing else is inside.
    let c0 = s.mesh.pos[cv as usize];
    let near: i32 = (0..s.n_verts())
        .filter(|&v| math::dist(s.mesh.pos[v], c0) < 1.0)
        .map(|v| f.index4[v] as i32)
        .sum();
    assert_eq!(near, 4);
    assert!((0..s.n_verts()).all(|v| {
        let d = math::dist(s.mesh.pos[v], c0);
        d < 1.0 || d > 8.5 || f.index4[v] == 0
    }));
    let mut c = analytic_curv(&s, &CurvKind::Plane).unwrap();
    c.conf.iter_mut().for_each(|x| *x = 1.0);
    let z = uniform_size(&s, 1.0).unwrap();
    let center = s.mesh.pos[cv as usize];
    let cs = Case { s, f, c, z };
    let (loops, rep) = run(&cs, &LoopParams::default());
    eprintln!("concentric plane: {rep:?}, {} loops", loops.len());
    assert!(!loops.is_empty(), "no eye-like loop accepted");
    assert_eq!(rep.rejected_turning, 0);
    for l in &loops {
        // A circle about the centre.
        let rs: Vec<f64> = l.samples.iter().map(|x| math::dist(x.0, center)).collect();
        let (rmin, rmax) = rs.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &r| (a.min(r), b.max(r)));
        assert!(rmax - rmin < 0.1 * rmax, "not a circle: radius {rmin} to {rmax}");
        assert_eq!(l.cycle, CycleKind::Separating);
        // Real turning 2pi (CCW), predicted 2pi chi - K = 2pi; the old 180 degree cap on |T|
        // would have rejected it.
        assert!((l.turning - TAU).abs() < 1e-6, "turning {}", l.turning);
        assert!((l.expected_turning - TAU).abs() < 1e-6, "expected {}", l.expected_turning);
        assert!(l.turning.abs() > PI);
    }
}

/// Dual cycle CCW around vertex v: its incoming halfedges in order (C3).
fn cycle_around(s: &Surface, v: u32) -> Vec<u32> {
    let h0 = Topology::prev(s.topo.vert_he[v as usize]);
    let mut out = vec![h0];
    let mut h = h0;
    loop {
        h = Topology::prev(s.topo.twin[h as usize]);
        if h == h0 {
            break;
        }
        out.push(h);
    }
    for (i, &h) in out.iter().enumerate() {
        assert_eq!(s.topo.head(&s.mesh, h), v);
        assert_eq!(s.topo.twin[h as usize] / 3, out[(i + 1) % out.len()] / 3);
    }
    out
}

#[test]
fn turning_check_rejects_cycles_gauss_bonnet_forbids() {
    let st = CtlState::new();
    let ctl = st.unbounded();
    // Plane, constant field: a tight cycle around a regular vertex turns by 0, but a closed
    // curve around a flat disk must turn by 2pi: excess 2pi.
    let s = Surface::new(aq_testkit::plane_grid(10, 10, 1.0, 0.0, 0.2, 1).unwrap()).unwrap();
    let f = analytic_field(&s, &FieldKind::Constant { angle: 0.3 }).unwrap();
    let v = 5 * 11 + 5;
    assert_eq!(f.index4[v as usize], 0);
    let cyc = cycle_around(&s, v);
    let tc = turning_check(&px(&s), &f, &cyc, &ctl).unwrap();
    assert_eq!(tc.kind, CycleKind::Separating);
    assert!(tc.side_left && tc.side_chi == 1, "{tc:?}");
    assert!(tc.turning.abs() < 1e-9 && (tc.excess - TAU).abs() < 1e-9, "{tc:?}");
    assert!(tc.excess > MAX_TURNING_EXCESS);
    // The same cycle backwards: v is on the right, the prediction flips sign.
    let rev: Vec<u32> = cyc.iter().rev().map(|&h| s.topo.twin[h as usize]).collect();
    let tr = turning_check(&px(&s), &f, &rev, &ctl).unwrap();
    assert_eq!(tr.kind, CycleKind::Separating);
    assert!((tr.expected + TAU).abs() < 1e-9 && (tr.excess - TAU).abs() < 1e-9, "{tr:?}");
    // Winding twice: rejected whatever its turning.
    let twice: Vec<u32> = cyc.iter().chain(cyc.iter()).copied().collect();
    assert_eq!(turning_check(&px(&s), &f, &twice, &ctl).unwrap().kind, CycleKind::Overlapping);
    // A cycle that goes out and back crosses nothing net.
    let back = vec![cyc[0], s.topo.twin[cyc[0] as usize]];
    assert_eq!(turning_check(&px(&s), &f, &back, &ctl).unwrap().kind, CycleKind::Overlapping);

    // Cube sphere: a tight cycle around a +1 cone (the "spiral around a cone" case): turning
    // (pi/2) - K(v), prediction 2pi - K(v), excess 3pi/2.
    let pc = aq_testkit::cube_sphere_param(4).unwrap();
    let (s, f) = (&pc.surface, &pc.field);
    let (cone, idx) = f.cones()[0];
    assert_eq!(idx, 1);
    let tc = turning_check(&px(s), f, &cycle_around(s, cone), &ctl).unwrap();
    assert_eq!(tc.kind, CycleKind::Separating);
    assert!((tc.excess - 1.5 * PI).abs() < 1e-6, "{tc:?}");

    // Torus: the accepted loops (meridians and big circles) are non-separating.
    let s = Surface::new(aq_testkit::torus(3.0, 1.0, 57, 19, 0.0, 0).unwrap()).unwrap();
    let cs = case(s, FieldKind::TorusCircumferential, CurvKind::Torus { big_r: 3.0, r: 1.0 });
    let (loops, _) = run(&cs, &LoopParams::default());
    assert!(!loops.is_empty() && loops.iter().all(|l| l.cycle == CycleKind::NonSeparating));

    // Errors, not panics, on a cycle with a boundary or out-of-range halfedge.
    let s = Surface::new(aq_testkit::plane_grid(4, 4, 1.0, 0.0, 0.0, 0).unwrap()).unwrap();
    let f = analytic_field(&s, &FieldKind::Constant { angle: 0.0 }).unwrap();
    let bnd = (0..3 * s.n_faces() as u32).find(|&h| s.topo.twin[h as usize] == NONE).unwrap();
    assert!(turning_check(&px(&s), &f, &[bnd], &ctl).is_err());
    assert!(turning_check(&px(&s), &f, &[u32::MAX - 1], &ctl).is_err());
}

#[test]
fn trace_stops_at_hard_chain() {
    // Constant field along +x on a plane crossed by a vertical chain at x = 20/3.
    let (n, cell) = (30u32, 1.0 / 3.0);
    let col = 20u32;
    for (kind, stops) in [(FeatureKind::Hard, true), (FeatureKind::GuideHard, true), (FeatureKind::Guide, false)] {
        let m = aq_testkit::plane_grid(n, n, cell, 0.0, 0.0, 0).unwrap();
        let edges: Vec<([u32; 2], FeatureKind)> =
            (0..n).map(|j| ([j * (n + 1) + col, (j + 1) * (n + 1) + col], kind)).collect();
        let s = Surface::with_features(m, &edges, aq_testkit::surface::CORNER_DEG, aq_testkit::surface::SYM_CORNER_DEG)
            .unwrap();
        let f = analytic_field(&s, &FieldKind::Constant { angle: 0.0 }).unwrap();
        let from = locate(&s, 3.05, 5.1);
        let sl = trace_streamline(&px(&s), &f, from, 0, 100.0);
        let end = *sl.points.last().unwrap();
        if stops {
            assert_eq!(sl.stop, StopReason::HardChain, "{kind:?}");
            let h = sl.stop_he;
            assert_ne!(h, NONE);
            assert_eq!(h / 3, *sl.faces.last().unwrap());
            assert_eq!(s.feats.he_kind[h as usize], kind);
            // The stop halfedge lies on the chain, and the trace ends on it.
            for v in [s.topo.tail(&s.mesh, h), s.topo.head(&s.mesh, h)] {
                assert!((s.mesh.pos[v as usize][0] - col as f64 * cell).abs() < 1e-12);
            }
            assert!((end[0] - col as f64 * cell).abs() < 1e-9 && (end[1] - 5.1).abs() < 1e-9, "{end:?}");
        } else {
            // A soft guide does not stop the trace.
            assert_eq!(sl.stop, StopReason::Boundary, "{kind:?}");
            assert!((end[0] - n as f64 * cell).abs() < 1e-9, "{end:?}");
        }
    }
}

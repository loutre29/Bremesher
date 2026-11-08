//! T-mesh generators: validated, expected structure, feasibility of the random instances.

use aq_core::{ArcKind, NodeKind, TMesh};
use aq_testkit::*;

/// Sum of sigma * n over the occurrences on each (patch, axis) node; true when all are 0.
fn balanced(t: &TMesh, n: &[i64]) -> bool {
    t.patches.iter().all(|p| {
        (0..2).all(|axis| {
            let plus: i64 = p.sides[axis].iter().map(|&(a, _)| n[a as usize]).sum();
            let minus: i64 = p.sides[axis + 2].iter().map(|&(a, _)| n[a as usize]).sum();
            plus == minus
        })
    })
}

fn side_len(t: &TMesh, s: &[(u32, bool)]) -> f64 {
    s.iter().map(|&(a, _)| t.arcs[a as usize].len).sum()
}

#[test]
fn rect() {
    let t = tmesh_rect(3.4, 5.6).unwrap();
    t.validate().unwrap();
    assert_eq!((t.nodes.len(), t.arcs.len(), t.patches.len(), t.chi), (4, 4, 1, 1));
    let p = &t.patches[0];
    let l: Vec<f64> = p.sides.iter().map(|s| side_len(&t, s)).collect();
    assert_eq!(l, vec![3.4, 5.6, 3.4, 5.6]);
    assert!(t.nodes.iter().all(|n| n.kind == NodeKind::BoundaryCorner && n.index4 == 1));
    assert!(t.arcs.iter().all(|a| a.kind == ArcKind::Boundary && a.weight == 4.0 && a.lb == 1));
    assert_eq!(t.rings.len(), 1);
    assert_eq!(t.rings[0].len(), 4);
    assert!(balanced(&t, &[3, 6, 3, 6]));
}

#[test]
fn grid_with_t_junctions() {
    let t = tmesh_grid(2, 2, &[0.0, 0.3]).unwrap();
    t.validate().unwrap();
    assert_eq!(t.patches.len(), 4);
    // The two T-nodes on the middle line are joined by an arc of length 0.3.
    let short: Vec<_> = t.arcs.iter().filter(|a| (a.len - 0.3).abs() < 1e-12).collect();
    assert_eq!(short.len(), 1);
    assert_eq!(short[0].kind, ArcKind::Separatrix);
    assert_eq!(t.nodes.len() as i64 - t.arcs.len() as i64 + t.patches.len() as i64, 1);
    // Every interior arc occurs twice, every boundary arc once.
    let occ = t.arc_occurrences();
    for (a, o) in t.arcs.iter().zip(&occ) {
        assert_eq!(*o, if a.kind == ArcKind::Boundary { 1 } else { 2 });
    }
    // A regular grid (equal offsets) has no T arcs.
    let g = tmesh_grid(3, 4, &[]).unwrap();
    assert_eq!((g.nodes.len(), g.arcs.len(), g.patches.len()), (20, 31, 12));
    let big = tmesh_grid_sized(20, 25, 3.0, 2.5, &[0.0, 0.7, 0.0, 1.1]).unwrap();
    assert_eq!(big.patches.len(), 500);
    assert!(tmesh_grid(2, 2, &[0.0, 1.5]).is_err());
}

#[test]
fn cube() {
    let t = tmesh_cube().unwrap();
    t.validate().unwrap();
    assert_eq!((t.nodes.len(), t.arcs.len(), t.patches.len(), t.chi), (8, 12, 6, 2));
    assert!(t.arc_occurrences().iter().all(|&o| o == 2));
    assert!(t.nodes.iter().all(|n| n.kind == NodeKind::Cone && n.index4 == 1));
    // Each node has valence 3.
    let mut val = [0u32; 8];
    for a in &t.arcs {
        val[a.ends[0] as usize] += 1;
        val[a.ends[1] as usize] += 1;
    }
    assert!(val.iter().all(|&v| v == 3));
    assert!(balanced(&t, &[1; 12]));
    let b = tmesh_box(2.0, 3.0, 4.0).unwrap();
    for p in &b.patches {
        assert_eq!(side_len(&b, &p.sides[0]), side_len(&b, &p.sides[2]));
        assert_eq!(side_len(&b, &p.sides[1]), side_len(&b, &p.sides[3]));
    }
}

#[test]
fn helix() {
    let t = tmesh_helix(8.0, 6.0, 0.6, 5).unwrap();
    t.validate().unwrap();
    assert_eq!(t.chi, 0);
    assert_eq!(t.patches.len(), 6);
    let wraps: Vec<_> = t.arcs.iter().filter(|a| a.wrap).collect();
    assert_eq!(wraps.len(), 5);
    assert!(wraps.iter().all(|a| a.len == 0.6 && a.lb == 0 && a.weight == 8.0));
    assert_eq!(t.rings.len(), 2);
    // Wrap arcs at 0 and the turns at 8 balance every patch.
    let mut n: Vec<i64> = t.arcs.iter().map(|a| a.len.round() as i64).collect();
    for (i, a) in t.arcs.iter().enumerate() {
        if a.wrap {
            n[i] = 0;
        }
    }
    assert!(balanced(&t, &n));
    // Any nonzero wrap arc is infeasible for the bottom region (it cancels everything else).
    n[4] = 1;
    assert!(!balanced(&t, &n));
    assert!(tmesh_helix(8.0, 2.0, 0.6, 5).is_err());
}

#[test]
fn random_bidirected_instances() {
    let (mut pp, mut mm, mut pm) = (0, 0, 0);
    for n_arcs in 2..=8u32 {
        for seed in 0..40u64 {
            let t = tmesh_random_bidirected(n_arcs, seed).unwrap();
            t.validate().unwrap();
            assert_eq!(t.arcs.len(), n_arcs as usize);
            // The planted solution is round(len): feasible and above lb.
            let n: Vec<i64> = t.arcs.iter().map(|a| a.len.round() as i64).collect();
            assert!(balanced(&t, &n), "n_arcs {n_arcs} seed {seed}");
            assert!(t.arcs.iter().zip(&n).all(|(a, &x)| x >= a.lb as i64 && x >= 1));
            // Edge sign classes.
            for a in 0..t.arcs.len() as u32 {
                let mut signs = Vec::new();
                for p in &t.patches {
                    for (k, s) in p.sides.iter().enumerate() {
                        if s.iter().any(|&(x, _)| x == a) {
                            signs.push(k < 2);
                        }
                    }
                }
                if signs.len() == 2 {
                    match (signs[0], signs[1]) {
                        (true, true) => pp += 1,
                        (false, false) => mm += 1,
                        _ => pm += 1,
                    }
                }
            }
            assert_eq!(t, tmesh_random_bidirected(n_arcs, seed).unwrap());
        }
    }
    assert!(pp > 0 && mm > 0 && pm > 0, "{pp} {mm} {pm}");
    assert!(tmesh_random_bidirected(1, 0).is_err());
}

#[test]
fn random_dissection() {
    for seed in 0..5 {
        let t = tmesh_random_dissection(50, seed).unwrap();
        t.validate().unwrap();
        assert_eq!(t.patches.len(), 50);
        assert_eq!(t.chi, 1);
        assert_eq!(t, tmesh_random_dissection(50, seed).unwrap());
    }
    let t = tmesh_random_dissection(500, 3).unwrap();
    assert_eq!(t.patches.len(), 500);
    // Integer coordinates before scaling: some T-junctions exist.
    let deg3 = {
        let mut val = vec![0u32; t.nodes.len()];
        for a in &t.arcs {
            val[a.ends[0] as usize] += 1;
            val[a.ends[1] as usize] += 1;
        }
        val.iter().filter(|&&v| v == 3).count()
    };
    assert!(deg3 > 10);
}

#[test]
fn rect_dissection_rejects_gaps() {
    assert!(tmesh_from_rects(&[[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 3.0, 1.0]], &[]).is_err());
    let t = tmesh_from_rects(&[[0.0, 0.0, 2.0, 1.0]], &[[1.0, 0.0]]).unwrap();
    assert_eq!(t.arcs.len(), 5);
    assert_eq!(t.patches[0].sides[0].len(), 2);
}

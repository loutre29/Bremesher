//! Timing test (DESIGN 6.3): Laplacian plus mass on 200k vertices, analyze and factor < 1.5 s
//! on 8 threads; a numeric refactor with the same symbolic < 60% of the full time.
//!
//! Each time is the best of 3 runs, so that a machine busy with other builds does not make
//! the test fail spuriously. The test runs in release builds only (`cargo test --release`).
//!
//! The 60% assertion is the DESIGN criterion on `Chol::analyze` (Ordering::Auto). Its
//! denominator includes the ordering search, part of which is discarded (on the torus the
//! nested dissection loses to AMD). The ratio against the kept ordering alone is printed as
//! "kept-only" for information: with AMD it is about 75 to 80%, see CHANGES.md (WP-02 item 4
//! and the review follow-up).

mod common;

use std::time::Instant;

use aq_linalg::{Chol, Csc, Ordering, Triplets, cotan_laplacian, lumped_mass};
use common::*;

fn rel_residual(a: &Csc, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; a.n()];
    a.mul_vec(x, &mut ax);
    ax.iter()
        .zip(b)
        .map(|(p, q)| (p - q).powi(2))
        .sum::<f64>()
        .sqrt()
        / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn time_case(name: &str, m: aq_core::TriMesh) {
    let c = ctx(m);
    let mut a = cotan_laplacian(&c.m, &c.t, &c.fr, None);
    assert!(a.n() >= 160_000);
    a.add_diag_vec(&lumped_mass(&c.m, &c.t, &c.fr)).unwrap();
    // The stiffened system: same pattern, other values.
    let mut a2 = a.clone();
    a2.scale(1.5);
    a2.add_diag(1e-3);
    let b: Vec<f64> = (0..a.n()).map(|i| ((i % 101) as f64).sin()).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let (full, refac, fac, kept) = pool.install(|| {
        let (mut full, mut refac, mut fac) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut kept = Ordering::Amd;
        for _ in 0..3 {
            let t0 = Instant::now();
            let sym = Chol::analyze(&a).unwrap();
            let t1 = Instant::now();
            let mut ch = Chol::factor(&sym, &a).unwrap();
            let t2 = Instant::now();
            let mut x = b.clone();
            ch.solve(&mut x);
            assert!(rel_residual(&a, &x, &b) < 1e-10);
            let t3 = Instant::now();
            ch.refactor(&a2).unwrap();
            let t4 = Instant::now();
            let mut x = b.clone();
            ch.solve(&mut x);
            assert!(rel_residual(&a2, &x, &b) < 1e-10);
            println!(
                "{name}: analyze {:.3} s ({:?}, flops {:.2e}, nnz(L) {}), factor {:.3} s, refactor {:.3} s",
                (t1 - t0).as_secs_f64(),
                sym.ordering(),
                sym.flops(),
                sym.factor_len(),
                (t2 - t1).as_secs_f64(),
                (t4 - t3).as_secs_f64()
            );
            full = full.min((t2 - t0).as_secs_f64());
            fac = fac.min((t2 - t1).as_secs_f64());
            refac = refac.min((t4 - t3).as_secs_f64());
            kept = sym.ordering();
        }
        (full, refac, fac, kept)
    });
    // Analysis with the kept ordering only (no discarded candidate), for the honest ratio.
    let ana_kept = pool.install(|| {
        (0..3)
            .map(|_| {
                let t0 = Instant::now();
                let _ = Chol::analyze_with(&a, kept).unwrap();
                t0.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    });
    println!(
        "{name} (n = {}, nnz(A) = {}): analyze + factor {full:.3} s, refactor {refac:.3} s ({:.0}%); \
         kept-only ({kept:?}) analyze {ana_kept:.3} s + factor {fac:.3} s, refactor {:.0}% of that",
        a.n(),
        a.nnz(),
        100.0 * refac / full,
        100.0 * refac / (ana_kept + fac)
    );
    assert!(full < 1.5, "{name}: analyze + factor took {full:.3} s");
    assert!(
        refac < 0.6 * full,
        "{name}: refactor {refac:.3} s is not < 60% of {full:.3} s"
    );
}

#[test]
fn laplacian_200k_timing() {
    if cfg!(debug_assertions) {
        eprintln!("laplacian_200k_timing: timings are only meaningful in release builds, skipped");
        return;
    }
    // Closed torus, 640 x 320 = 204800 vertices (regular, periodic in both directions).
    time_case("torus 200k", torus(3.0, 1.0, 640, 320));
    // Irregular closed surface: noisy icosphere level 7 (163842 vertices) plus a jittered
    // open grid with 203401 vertices.
    let mut ico = icosphere(7, 1.0);
    add_radial_noise(&mut ico, 0.005, 7);
    time_case("noisy icosphere 164k", ico);
    time_case(
        "jittered plane 203k",
        plane_grid(0.0, 1.0, 0.0, 1.0, 450, 450, 0.4, 3),
    );
}

/// Best of 3 wall times of `Chol::analyze_with(a, ord)`, with the ordering that was kept.
fn analyze_best(a: &Csc, ord: Ordering) -> (f64, Ordering) {
    let mut best = f64::INFINITY;
    let mut used = ord;
    for _ in 0..3 {
        let t0 = Instant::now();
        let sym = Chol::analyze_with(a, ord).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        used = sym.ordering();
    }
    (best, used)
}

/// Diagonally dominant SPD matrix on the given off-diagonal pattern.
fn spd_from_edges(n: usize, edges: &[(usize, usize)]) -> Csc {
    let mut t = Triplets::new(n);
    for i in 0..n {
        t.push(i, i, 5.0);
    }
    for &(i, j) in edges {
        t.push(i, j, -1.0);
    }
    t.to_csc().unwrap()
}

/// Review regression: the nested dissection used to peel one connected component per pass
/// over the whole remainder, which is quadratic on fragmented patterns (100k isolated
/// unknowns took 13 s). Auto analysis must stay within a small factor of AMD alone there.
#[test]
fn auto_ordering_overhead_on_fragmented_patterns() {
    if cfg!(debug_assertions) {
        eprintln!("auto_ordering_overhead_on_fragmented_patterns: release builds only, skipped");
        return;
    }
    let mut cases: Vec<(&str, Csc)> = Vec::new();
    // 200k isolated unknowns.
    cases.push(("200k isolated", spd_from_edges(200_000, &[])));
    // 100k 2x2 blocks (200k unknowns), the two unknowns of a block far apart in the index.
    let nb = 100_000;
    let e: Vec<(usize, usize)> = (0..nb).map(|k| (k, k + nb)).collect();
    cases.push(("100k 2x2 blocks", spd_from_edges(2 * nb, &e)));
    // 300 x 300 grid with 30% of the grid rows pinned by identity rows (Dirichlet rows: the
    // pinned unknowns keep only their diagonal).
    let m = 300;
    let id = |i: usize, j: usize| j * m + i;
    let pinned = |j: usize| j % 10 < 3;
    let mut e = Vec::new();
    for j in 0..m {
        for i in 0..m {
            if pinned(j) {
                continue;
            }
            if i + 1 < m {
                e.push((id(i, j), id(i + 1, j)));
            }
            if j + 1 < m && !pinned(j + 1) {
                e.push((id(i, j), id(i, j + 1)));
            }
        }
    }
    cases.push(("300x300 grid, 30% identity rows", spd_from_edges(m * m, &e)));
    // Mixed: 150k isolated unknowns interleaved with a connected 250 x 250 grid.
    let (g, iso) = (250usize, 150_000usize);
    let n = g * g + iso;
    let gid = |i: usize, j: usize| 3 * (j * g + i); // grid unknowns spread among the isolated ones
    let mut e = Vec::new();
    for j in 0..g {
        for i in 0..g {
            if i + 1 < g {
                e.push((gid(i, j), gid(i + 1, j)));
            }
            if j + 1 < g {
                e.push((gid(i, j), gid(i, j + 1)));
            }
        }
    }
    assert!(gid(g - 1, g - 1) < n);
    cases.push(("250x250 grid + 150k isolated", spd_from_edges(n, &e)));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        for (name, a) in &cases {
            let (t_amd, _) = analyze_best(a, Ordering::Amd);
            let (t_auto, used) = analyze_best(a, Ordering::Auto);
            let (t_nd, _) = analyze_best(a, Ordering::NestedDissection);
            println!(
                "{name} (n = {}): analyze AMD {t_amd:.3} s, Auto {t_auto:.3} s (kept {used:?}), ND alone {t_nd:.3} s",
                a.n()
            );
            assert!(t_auto < 3.0 * t_amd + 0.05, "{name}: Auto analyze {t_auto:.3} s vs AMD {t_amd:.3} s");
            // The ordering is valid and the factor solves.
            let ch = Chol::new(a).unwrap();
            let b: Vec<f64> = (0..a.n()).map(|i| ((i % 37) as f64).cos()).collect();
            let mut x = b.clone();
            ch.solve(&mut x);
            assert!(rel_residual(a, &x, &b) < 1e-12);
        }
    });
}

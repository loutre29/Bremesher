//! Cholesky, sparse storage and hermitian_to_real tests (DESIGN 6.3).

mod common;

use aq_core::{ErrKind, SplitMix64};
use aq_linalg::{
    CTriplets, Chol, Csc, Ordering, Triplets, complex_to_real, cotan_laplacian, hermitian_to_real,
    lumped_mass, real_to_complex,
};
use common::*;

/// Random sparse SPD matrix: banded random couplings plus a few long-range ones, made
/// strictly diagonally dominant with a positive diagonal.
fn random_spd(n: usize, seed: u64) -> Csc {
    let mut rng = SplitMix64::new(seed);
    let mut t = Triplets::new(n);
    let mut rowsum = vec![0.0f64; n];
    let mut add = |t: &mut Triplets, i: usize, j: usize, v: f64| {
        t.push(i, j, v);
        rowsum[i] += v.abs();
        rowsum[j] += v.abs();
    };
    for i in 0..n {
        for _ in 0..4 {
            let j = (i + 1 + rng.below(40) as usize) % n;
            if j != i {
                let v = rng.range_f64(-1.0, 1.0);
                add(&mut t, i, j, v);
            }
        }
    }
    for _ in 0..n / 100 {
        let i = rng.below(n as u64) as usize;
        let j = rng.below(n as u64) as usize;
        if i != j {
            add(&mut t, j, i, rng.range_f64(-1.0, 1.0));
        }
    }
    for (i, s) in rowsum.iter().enumerate() {
        t.push(i, i, s + rng.range_f64(0.01, 1.0));
    }
    t.to_csc().unwrap()
}

fn rel_residual(a: &Csc, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.mul_vec(x, &mut ax);
    let r: f64 = ax
        .iter()
        .zip(b)
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    r / nb
}

#[test]
fn triplets_merge_and_pattern() {
    let mut t = Triplets::new(3);
    t.push(0, 1, 2.0);
    t.push(1, 0, 3.0); // same entry, other orientation: summed
    t.push(2, 2, 4.0);
    t.push(2, 0, 0.0); // explicit zero kept in the pattern
    let a = t.to_csc().unwrap();
    a.validate().unwrap();
    assert_eq!(a.col_ptr(), &[0, 3, 4, 5]);
    assert_eq!(a.row_idx(), &[0, 1, 2, 1, 2]);
    assert_eq!(a.values(), &[0.0, 5.0, 0.0, 0.0, 4.0]);
    assert_eq!(a.get(0, 1), 5.0);
    assert_eq!(a.get(1, 0), 5.0);
    let d = a.to_dense().unwrap();
    assert_eq!(d, vec![0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
    // Out of range and non-finite inputs are errors, not panics.
    let mut t = Triplets::new(2);
    t.push(2, 0, 1.0);
    assert_eq!(t.to_csc().unwrap_err().kind, ErrKind::InvalidInput);
    let mut t = Triplets::new(2);
    t.push(1, 0, f64::NAN);
    assert_eq!(t.to_csc().unwrap_err().kind, ErrKind::Numerical);
    // mul_vec matches the dense product.
    let a = random_spd(50, 3);
    let d = a.to_dense().unwrap();
    let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.37).sin()).collect();
    let mut y = vec![0.0; 50];
    a.mul_vec(&x, &mut y);
    for i in 0..50 {
        let r: f64 = (0..50).map(|j| d[i * 50 + j] * x[j]).sum();
        assert!((r - y[i]).abs() < 1e-12);
    }
    // lin_comb on the union pattern.
    let i3 = Csc::identity(50).unwrap();
    let c = Csc::lin_comb(&a, 2.0, &i3, -1.0).unwrap();
    assert!(c.same_pattern(&a));
    for i in 0..50 {
        assert_eq!(c.get(i, i), 2.0 * a.get(i, i) - 1.0);
    }
}

#[test]
fn random_spd_10k() {
    let n = 10_000;
    let a = random_spd(n, 11);
    let mut rng = SplitMix64::new(5);
    let b: Vec<f64> = (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect();
    let ch = Chol::new(&a).unwrap();
    let mut x = b.clone();
    ch.solve(&mut x);
    let r = rel_residual(&a, &x, &b);
    println!(
        "random SPD 10k: relative residual {r:e}, factor nnz {}",
        ch.symbolic().factor_len()
    );
    assert!(r < 1e-10, "residual {r}");
}

#[test]
fn refactor_with_same_symbolic() {
    let n = 2000;
    let mut a = random_spd(n, 21);
    let sym = Chol::analyze(&a).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
    for s in [1.0, 3.0, 0.25] {
        a.scale(s);
        a.add_diag(0.5);
        let ch = Chol::factor(&sym, &a).unwrap();
        let mut x = b.clone();
        ch.solve(&mut x);
        assert!(rel_residual(&a, &x, &b) < 1e-12);
    }
    // A different pattern is rejected.
    let other = random_spd(n, 22);
    assert_eq!(
        Chol::factor(&sym, &other).unwrap_err().kind,
        ErrKind::InvalidInput
    );
}

#[test]
fn solve_multi_matches_single() {
    let n = 3000;
    let a = random_spd(n, 31);
    let ch = Chol::new(&a).unwrap();
    let mut bs: Vec<Vec<f64>> = (0..3)
        .map(|k| (0..n).map(|i| ((i * (k + 1)) as f64).sin()).collect())
        .collect();
    let singles: Vec<Vec<f64>> = bs
        .iter()
        .map(|b| {
            let mut x = b.clone();
            ch.solve(&mut x);
            x
        })
        .collect();
    let orig = bs.clone();
    ch.solve_multi(&mut bs);
    for k in 0..3 {
        assert!(rel_residual(&a, &bs[k], &orig[k]) < 1e-12);
        for i in 0..n {
            assert!((bs[k][i] - singles[k][i]).abs() < 1e-12);
        }
    }
    // A wrong length does not panic.
    let mut short = vec![1.0; 5];
    ch.solve(&mut short);
    assert!(short.iter().all(|v| v.is_nan()));
    assert_eq!(
        ch.try_solve(&mut [0.0; 3]).unwrap_err().kind,
        ErrKind::InvalidInput
    );
}

#[test]
fn non_spd_reports_column() {
    let n = 3000;
    let bad_col = 1234;
    let mut t = Triplets::new(n);
    for i in 0..n {
        t.push(i, i, if i == bad_col { -5.0 } else { 4.0 });
        if i + 1 < n {
            t.push(i + 1, i, -1.0);
        }
        if i + 7 < n {
            t.push(i + 7, i, -0.5);
        }
    }
    let a = t.to_csc().unwrap();
    println!(
        "band matrix: supernodal {}",
        Chol::analyze(&a).unwrap().is_supernodal()
    );
    let e = Chol::new(&a).unwrap_err();
    println!("{e}");
    assert_eq!(e.kind, ErrKind::Numerical);
    assert!(e.msg.contains(&format!("column {bad_col})")), "{}", e.msg);
    // Same on a 2D grid pattern (faer stays simplicial here, see the 3D grid below).
    let m = 150;
    let id = |i: usize, j: usize| j * m + i;
    for bad in [0, 777, 11_111, m * m - 1] {
        let mut t = Triplets::new(m * m);
        for j in 0..m {
            for i in 0..m {
                t.push(id(i, j), id(i, j), if id(i, j) == bad { -1.0 } else { 4.5 });
                if i + 1 < m {
                    t.push(id(i + 1, j), id(i, j), -1.0);
                }
                if j + 1 < m {
                    t.push(id(i, j + 1), id(i, j), -1.0);
                }
            }
        }
        let a = t.to_csc().unwrap();
        let sym = Chol::analyze(&a).unwrap();
        let e = Chol::factor(&sym, &a).unwrap_err();
        println!("grid (supernodal {}): {e}", sym.is_supernodal());
        assert_eq!(e.kind, ErrKind::Numerical);
        assert!(e.msg.contains(&format!("column {bad})")), "{}", e.msg);
    }
    // 3D 7-point grid, 20^3 unknowns: faer takes the supernodal path here (asserted), so the
    // 0-based pivot index of that path is tested too.
    let m = 20;
    let id = |i: usize, j: usize, k: usize| (k * m + j) * m + i;
    for bad in [0, 4321, m * m * m - 1] {
        let mut t = Triplets::new(m * m * m);
        for k in 0..m {
            for j in 0..m {
                for i in 0..m {
                    let v = id(i, j, k);
                    t.push(v, v, if v == bad { -1.0 } else { 6.5 });
                    if i + 1 < m {
                        t.push(id(i + 1, j, k), v, -1.0);
                    }
                    if j + 1 < m {
                        t.push(id(i, j + 1, k), v, -1.0);
                    }
                    if k + 1 < m {
                        t.push(id(i, j, k + 1), v, -1.0);
                    }
                }
            }
        }
        let a = t.to_csc().unwrap();
        let sym = Chol::analyze(&a).unwrap();
        assert!(
            sym.is_supernodal(),
            "the 3D grid should take the supernodal path"
        );
        let e = Chol::factor(&sym, &a).unwrap_err();
        println!("3D grid (supernodal {}): {e}", sym.is_supernodal());
        assert_eq!(e.kind, ErrKind::Numerical);
        assert!(e.msg.contains(&format!("column {bad})")), "{}", e.msg);
    }
    // An indefinite matrix with a positive diagonal.
    let mut t = Triplets::new(2);
    t.push(0, 0, 1.0);
    t.push(1, 1, 1.0);
    t.push(1, 0, 2.0);
    assert_eq!(
        Chol::new(&t.to_csc().unwrap()).unwrap_err().kind,
        ErrKind::Numerical
    );
    // Non-finite values.
    let mut a = random_spd(10, 1);
    a.values_mut()[0] = f64::INFINITY;
    assert_eq!(Chol::new(&a).unwrap_err().kind, ErrKind::Numerical);
}

#[test]
fn empty_and_tiny() {
    let a = Triplets::new(0).to_csc().unwrap();
    let ch = Chol::new(&a).unwrap();
    let mut b: Vec<f64> = vec![];
    ch.try_solve(&mut b).unwrap();
    let mut t = Triplets::new(1);
    t.push(0, 0, 4.0);
    let ch = Chol::new(&t.to_csc().unwrap()).unwrap();
    let mut b = vec![2.0];
    ch.solve(&mut b);
    assert_eq!(b, vec![0.5]);
}

#[test]
fn chol_deterministic_across_threads() {
    // A banded random matrix (simplicial path) and a mesh Laplacian large enough for the
    // supernodal path and the nested dissection.
    let mut ico = icosphere(6, 1.0);
    add_radial_noise(&mut ico, 0.01, 4);
    let c = ctx(ico);
    let mut lap = cotan_laplacian(&c.m, &c.t, &c.fr, None);
    lap.add_diag_vec(&lumped_mass(&c.m, &c.t, &c.fr)).unwrap();
    for a in [random_spd(40_000, 41), lap] {
        let n = a.n();
        let b: Vec<f64> = (0..n).map(|i| ((i % 97) as f64 - 48.0) / 7.0).collect();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ch = Chol::new(&a).unwrap();
                let mut x = b.clone();
                ch.solve(&mut x);
                (x, ch.symbolic().is_supernodal(), ch.symbolic().ordering())
            })
        };
        let (x1, sup, ord) = run(1);
        println!("n = {n}: supernodal {sup}, ordering {ord:?}");
        for th in [2, 3, 8] {
            let (xt, _, _) = run(th);
            assert!(
                x1.iter().zip(&xt).all(|(p, q)| p.to_bits() == q.to_bits()),
                "differs with {th} threads"
            );
        }
    }
}

#[test]
fn orderings_agree() {
    // Every ordering gives a valid factor; Auto keeps the cheaper one.
    let mut ico = icosphere(6, 1.0);
    add_radial_noise(&mut ico, 0.01, 5);
    let c = ctx(ico);
    let mut a = cotan_laplacian(&c.m, &c.t, &c.fr, None);
    a.add_diag_vec(&lumped_mass(&c.m, &c.t, &c.fr)).unwrap();
    let b: Vec<f64> = (0..a.n()).map(|i| (i as f64 * 0.01).cos()).collect();
    let mut flops = Vec::new();
    for ord in [Ordering::Amd, Ordering::NestedDissection, Ordering::Auto] {
        let sym = Chol::analyze_with(&a, ord).unwrap();
        let mut p = sym.perm();
        p.sort_unstable();
        assert!(
            p.iter().enumerate().all(|(k, &v)| k == v),
            "{ord:?}: not a permutation"
        );
        let ch = Chol::factor(&sym, &a).unwrap();
        let mut x = b.clone();
        ch.solve(&mut x);
        let r = rel_residual(&a, &x, &b);
        println!(
            "{ord:?} -> {:?}: flops {:.3e}, nnz(L) {}, residual {r:e}",
            sym.ordering(),
            sym.flops(),
            sym.factor_len()
        );
        assert!(r < 1e-12);
        flops.push(sym.flops());
    }
    assert_eq!(flops[2], flops[0].min(flops[1]));
}

/// Nested dissection on fragmented patterns (isolated unknowns, 2x2 blocks, identity rows,
/// components above and below the leaf size, interleaved in the index): a valid permutation
/// and an accurate solve, for the forced ordering and for Auto.
#[test]
fn nested_dissection_on_fragmented_patterns() {
    let mut rng = SplitMix64::new(11);
    // Pieces: grids of side 10 to 40 (100 to 1600 unknowns, so some are above the leaf size
    // of 512), single unknowns and 2x2 blocks, in random order, then a random relabeling.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut n = 0usize;
    while n < 24_000 {
        match rng.below(3) {
            0 => {
                let s = 10 + rng.below(31) as usize;
                for j in 0..s {
                    for i in 0..s {
                        let v = n + j * s + i;
                        if i + 1 < s {
                            edges.push((v, v + 1));
                        }
                        if j + 1 < s {
                            edges.push((v, v + s));
                        }
                    }
                }
                n += s * s;
            }
            1 => n += 1,
            _ => {
                edges.push((n, n + 1));
                n += 2;
            }
        }
    }
    let mut relabel: Vec<usize> = (0..n).collect();
    for k in (1..n).rev() {
        relabel.swap(k, rng.below(k as u64 + 1) as usize);
    }
    let mut t = Triplets::new(n);
    for i in 0..n {
        t.push(i, i, 4.5);
    }
    for &(i, j) in &edges {
        t.push(relabel[i], relabel[j], -1.0);
    }
    let frag = t.to_csc().unwrap();
    // A 160 x 160 grid with 30% of its grid rows pinned by identity rows.
    let m = 160;
    let pinned = |j: usize| j % 10 < 3;
    let mut t = Triplets::new(m * m);
    for j in 0..m {
        for i in 0..m {
            let v = j * m + i;
            t.push(v, v, if pinned(j) { 1.0 } else { 4.5 });
            if pinned(j) {
                continue;
            }
            if i + 1 < m {
                t.push(v, v + 1, -1.0);
            }
            if j + 1 < m && !pinned(j + 1) {
                t.push(v, v + m, -1.0);
            }
        }
    }
    let pinned_grid = t.to_csc().unwrap();
    for (name, a) in [("fragments", &frag), ("pinned grid", &pinned_grid)] {
        assert!(a.n() >= aq_linalg::ND_MIN_N);
        let b: Vec<f64> = (0..a.n()).map(|i| (i as f64 * 0.37).sin()).collect();
        for ord in [Ordering::NestedDissection, Ordering::Auto] {
            let sym = Chol::analyze_with(a, ord).unwrap();
            let mut p = sym.perm();
            p.sort_unstable();
            assert!(
                p.iter().enumerate().all(|(k, &v)| k == v),
                "{name} {ord:?}: not a permutation"
            );
            let ch = Chol::factor(&sym, a).unwrap();
            let mut x = b.clone();
            ch.solve(&mut x);
            let r = rel_residual(a, &x, &b);
            println!(
                "{name} (n = {}) {ord:?} -> {:?}: flops {:.3e}, residual {r:e}",
                a.n(),
                sym.ordering(),
                sym.flops()
            );
            assert!(r < 1e-12);
        }
    }
}

/// `analyze_ctl` returns the caller's cancellation and timeout.
#[test]
fn analyze_ctl_reports_cancel_and_timeout() {
    use std::time::{Duration, Instant};
    let a = random_spd(30_000, 3);
    let st = aq_core::CtlState::new();
    let live = st.ctl(Instant::now() + Duration::from_secs(600));
    let sym = Chol::analyze_ctl(&a, Ordering::Auto, Some(&live)).unwrap();
    assert_eq!(sym.flops(), Chol::analyze(&a).unwrap().flops());
    let past = st.ctl(Instant::now());
    assert_eq!(
        Chol::analyze_ctl(&a, Ordering::NestedDissection, Some(&past))
            .unwrap_err()
            .kind,
        ErrKind::Timeout
    );
    st.cancel();
    let c = st.ctl(Instant::now() + Duration::from_secs(600));
    assert_eq!(
        Chol::analyze_ctl(&a, Ordering::Auto, Some(&c))
            .unwrap_err()
            .kind,
        ErrKind::Cancelled
    );
}

#[test]
fn refactor_in_place() {
    let n = 3000;
    let mut a = random_spd(n, 51);
    let mut ch = Chol::new(&a).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    a.scale(2.0);
    a.add_diag(1.0);
    ch.refactor(&a).unwrap();
    let mut x = b.clone();
    ch.solve(&mut x);
    assert!(rel_residual(&a, &x, &b) < 1e-12);
    // Same bits as a fresh factorization with the same symbolic.
    let fresh = Chol::factor(ch.symbolic(), &a).unwrap();
    let mut y = b.clone();
    fresh.solve(&mut y);
    assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()));
    // A failed refactor invalidates the factor until the next success.
    let mut bad = a.clone();
    bad.values_mut()[0] = -1e6;
    assert_eq!(ch.refactor(&bad).unwrap_err().kind, ErrKind::Numerical);
    assert_eq!(
        ch.try_solve(&mut b.clone()).unwrap_err().kind,
        ErrKind::Numerical
    );
    ch.refactor(&a).unwrap();
    ch.try_solve(&mut b.clone()).unwrap();
}

// ---- complex Hermitian -> real ----

type C = [f64; 2];
fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn cdiv(a: C, b: C) -> C {
    let d = b[0] * b[0] + b[1] * b[1];
    [
        (a[0] * b[0] + a[1] * b[1]) / d,
        (a[1] * b[0] - a[0] * b[1]) / d,
    ]
}
fn csub(a: C, b: C) -> C {
    [a[0] - b[0], a[1] - b[1]]
}
fn cabs(a: C) -> f64 {
    a[0].hypot(a[1])
}

/// Dense complex Gaussian elimination with partial pivoting (reference).
fn dense_complex_solve(n: usize, mut h: Vec<C>, mut c: Vec<C>) -> Vec<C> {
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| cabs(h[i * n + k]).total_cmp(&cabs(h[j * n + k])))
            .unwrap();
        if p != k {
            for j in 0..n {
                h.swap(k * n + j, p * n + j);
            }
            c.swap(k, p);
        }
        for i in k + 1..n {
            let f = cdiv(h[i * n + k], h[k * n + k]);
            for j in k..n {
                h[i * n + j] = csub(h[i * n + j], cmul(f, h[k * n + j]));
            }
            c[i] = csub(c[i], cmul(f, c[k]));
        }
    }
    let mut z = vec![[0.0; 2]; n];
    for i in (0..n).rev() {
        let mut s = c[i];
        for j in i + 1..n {
            s = csub(s, cmul(h[i * n + j], z[j]));
        }
        z[i] = cdiv(s, h[i * n + i]);
    }
    z
}

#[test]
fn hermitian_to_real_matches_dense_complex() {
    let n = 200;
    let mut rng = SplitMix64::new(77);
    // H = B^H B + I with a sparse-ish random B (about 10 entries per row).
    let mut bm = vec![[0.0f64; 2]; n * n];
    for i in 0..n {
        for _ in 0..10 {
            let j = rng.below(n as u64) as usize;
            bm[i * n + j] = [rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0)];
        }
    }
    let mut h = vec![[0.0f64; 2]; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = [0.0, 0.0];
            for k in 0..n {
                let bki = bm[k * n + i];
                let conj = [bki[0], -bki[1]];
                let p = cmul(conj, bm[k * n + j]);
                s = [s[0] + p[0], s[1] + p[1]];
            }
            h[i * n + j] = s;
        }
        h[i * n + i] = [h[i * n + i][0] + 1.0, 0.0];
    }
    // Feed the lower triangle (and some entries via the upper orientation).
    let mut ct = CTriplets::new(n);
    for i in 0..n {
        for j in 0..=i {
            let z = h[i * n + j];
            if z == [0.0, 0.0] && i != j {
                continue;
            }
            if (i + j) % 3 == 0 && i != j {
                ct.push(j, i, [z[0], -z[1]]); // H[j][i] = conj(H[i][j])
            } else {
                ct.push(i, j, z);
            }
        }
    }
    let r = hermitian_to_real(&ct);
    assert_eq!(r.n(), 2 * n);
    let c: Vec<C> = (0..n)
        .map(|_| [rng.range_f64(-1.0, 1.0), rng.range_f64(-1.0, 1.0)])
        .collect();
    let zref = dense_complex_solve(n, h.clone(), c.clone());
    let ch = Chol::new(&r).unwrap();
    let mut x = complex_to_real(&c);
    ch.solve(&mut x);
    let z = real_to_complex(&x);
    let mut maxerr = 0.0f64;
    for i in 0..n {
        maxerr = maxerr.max(cabs(csub(z[i], zref[i])));
    }
    println!("hermitian_to_real n=200: max |z - z_ref| = {maxerr:e}");
    assert!(maxerr < 1e-10);
    // The real matrix applied to [x; y] equals H z.
    let mut y = vec![0.0; 2 * n];
    r.mul_vec(&complex_to_real(&zref), &mut y);
    for i in 0..n {
        let mut s = [0.0, 0.0];
        for j in 0..n {
            let p = cmul(h[i * n + j], zref[j]);
            s = [s[0] + p[0], s[1] + p[1]];
        }
        assert!((s[0] - y[2 * i]).abs() < 1e-9 && (s[1] - y[2 * i + 1]).abs() < 1e-9);
    }
}

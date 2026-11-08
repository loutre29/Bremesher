//! Elim tests (DESIGN 6.3).

use aq_core::SplitMix64;
use aq_linalg::{Elim, RowOutcome};

fn no_pref(_: u32) -> i32 {
    0
}

/// `sum c_i expr(x_i)` must be identically zero for an accepted row.
fn row_holds(e: &Elim, row: &[(u32, i64)]) -> bool {
    e.substitute(row).is_some_and(|t| t.is_empty())
}

#[test]
fn hand_chain() {
    // x1 = x2, x2 = -x3, x3 + x4 - x5 = 0.
    let mut e = Elim::new(6);
    let r1 = [(1, 1), (2, -1)];
    let r2 = [(2, 1), (3, 1)];
    let r3 = [(3, 1), (4, 1), (5, -1)];
    assert_eq!(
        e.add_row(&r1, &no_pref),
        RowOutcome::Eliminated { pivot: 1 }
    );
    assert_eq!(e.expr(1), &[(2, 1)]);
    assert_eq!(
        e.add_row(&r2, &no_pref),
        RowOutcome::Eliminated { pivot: 2 }
    );
    assert_eq!(e.expr(2), &[(3, -1)]);
    assert_eq!(e.expr(1), &[(3, -1)]);
    assert_eq!(
        e.add_row(&r3, &no_pref),
        RowOutcome::Eliminated { pivot: 3 }
    );
    // Hand-computed: x3 = -x4 + x5, x2 = -x3 = x4 - x5, x1 = x2 = x4 - x5.
    assert_eq!(e.expr(3), &[(4, -1), (5, 1)]);
    assert_eq!(e.expr(2), &[(4, 1), (5, -1)]);
    assert_eq!(e.expr(1), &[(4, 1), (5, -1)]);
    assert_eq!(e.expr(0), &[(0, 1)]);
    assert_eq!(e.expr(4), &[(4, 1)]);
    assert_eq!(e.expr(5), &[(5, 1)]);
    assert_eq!(e.n_free(), 3);
    assert_eq!(e.free_vars(), vec![0, 4, 5]);
    assert_eq!(e.free_index(0), Some(0));
    assert_eq!(e.free_index(4), Some(1));
    assert_eq!(e.free_index(5), Some(2));
    assert_eq!(e.free_index(1), None);
    for r in [&r1[..], &r2[..], &r3[..]] {
        assert!(row_holds(&e, r));
    }
    // Numeric reconstruction.
    let free = [7.0, 2.0, 5.0];
    assert_eq!(e.eval(1, &free), -3.0);
    assert_eq!(e.eval(3, &free), 3.0);
}

#[test]
fn duplicate_row_is_redundant() {
    let mut e = Elim::new(4);
    let r = [(0, 1), (2, 1), (3, -1)];
    assert!(matches!(
        e.add_row(&r, &no_pref),
        RowOutcome::Eliminated { .. }
    ));
    assert_eq!(e.add_row(&r, &no_pref), RowOutcome::Redundant);
    // A negated duplicate and a permuted one too.
    assert_eq!(
        e.add_row(&[(3, 1), (0, -1), (2, -1)], &no_pref),
        RowOutcome::Redundant
    );
    assert_eq!(e.n_free(), 3);
}

#[test]
fn non_unit_row_has_no_pivot() {
    let mut e = Elim::new(3);
    assert_eq!(
        e.add_row(&[(1, 2), (2, 2)], &no_pref),
        RowOutcome::NoUnitPivot
    );
    assert_eq!(e.n_free(), 3);
    assert_eq!(e.expr(1), &[(1, 1)]);
    // After substitution a row can become non-unit: x0 = x1, then x0 + x1 = 0 is 2 x1 = 0.
    assert!(matches!(
        e.add_row(&[(0, 1), (1, -1)], &no_pref),
        RowOutcome::Eliminated { pivot: 0 }
    ));
    assert_eq!(
        e.add_row(&[(0, 1), (1, 1)], &no_pref),
        RowOutcome::NoUnitPivot
    );
    // Zero coefficients and merged duplicates: x2 - x2 = 0 is redundant.
    assert_eq!(
        e.add_row(&[(2, 1), (2, -1)], &no_pref),
        RowOutcome::Redundant
    );
    assert_eq!(e.add_row(&[(2, 0)], &no_pref), RowOutcome::Redundant);
}

#[test]
fn pivot_preference_and_ties() {
    // Prefer the highest id ("newest variable").
    let mut e = Elim::new(5);
    let newest = |v: u32| v as i32;
    assert_eq!(
        e.add_row(&[(1, 1), (3, -1), (2, 1)], &newest),
        RowOutcome::Eliminated { pivot: 3 }
    );
    assert_eq!(e.expr(3), &[(1, 1), (2, 1)]);
    // Ties go to the lowest id; non-unit coefficients are never pivots.
    let mut e = Elim::new(5);
    let flat = |v: u32| if v >= 2 { 1 } else { 0 };
    assert_eq!(
        e.add_row(&[(0, 1), (2, 2), (3, -1), (4, 1)], &flat),
        RowOutcome::Eliminated { pivot: 3 }
    );
    assert_eq!(e.expr(3), &[(0, 1), (2, 2), (4, 1)]);
}

#[test]
fn pin_gauge() {
    let mut e = Elim::new(4);
    assert!(matches!(
        e.add_row(&[(0, 1), (1, -1), (2, 1)], &no_pref),
        RowOutcome::Eliminated { pivot: 0 }
    ));
    // Pin x1 = 0: it disappears from x0's expression.
    assert_eq!(e.pin(1, true), RowOutcome::Eliminated { pivot: 1 });
    assert_eq!(e.expr(0), &[(2, -1)]);
    assert_eq!(e.expr(1), &[] as &[(u32, i64)]);
    assert_eq!(e.n_free(), 2);
    assert_eq!(e.pin(1, true), RowOutcome::Redundant);
    // Pinning an eliminated variable to zero is the row x0 = 0, which eliminates x2.
    assert_eq!(e.pin(0, true), RowOutcome::Eliminated { pivot: 2 });
    assert_eq!(e.expr(0), &[] as &[(u32, i64)]);
    assert_eq!(e.n_free(), 1);
    // Rows only on zero-pinned variables reduce to 0 = 0.
    assert_eq!(
        e.add_row(&[(0, 1), (1, 3)], &no_pref),
        RowOutcome::Redundant
    );
    // A value-pinned variable stays as a term; a row on it alone is Inconsistent.
    assert_eq!(e.pin(3, false), RowOutcome::Eliminated { pivot: 3 });
    assert!(e.is_value_pinned(3));
    assert_eq!(e.n_free(), 0);
    assert_eq!(e.add_row(&[(3, 1)], &no_pref), RowOutcome::Inconsistent);
    let mut e = Elim::new(3);
    e.pin(2, false);
    assert_eq!(
        e.add_row(&[(0, 1), (2, -1)], &no_pref),
        RowOutcome::Eliminated { pivot: 0 }
    );
    assert_eq!(e.expr(0), &[(2, 1)]);
    assert_eq!(
        e.eval_with(0, &[0.5], &|v| if v == 2 { 4.0 } else { f64::NAN }),
        4.0
    );
    // Out of range.
    assert_eq!(e.add_row(&[(9, 1)], &no_pref), RowOutcome::Inconsistent);
    assert_eq!(e.pin(9, true), RowOutcome::Inconsistent);
}

/// Runs random +-1 rows of 2 to 4 terms. `window` bounds the spread of a row's variables
/// (mesh-like locality); `None` draws them uniformly.
fn random_rows(n_vars: usize, n_rows: usize, window: Option<u64>, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let mut e = Elim::new(n_vars);
    let prefer = |v: u32| (v % 7) as i32;
    let mut accepted: Vec<Vec<(u32, i64)>> = Vec::new();
    let (mut elim, mut red, mut nup, mut inc) = (0, 0, 0, 0);
    for _ in 0..n_rows {
        let k = 2 + rng.below(3) as usize;
        let mut row = Vec::new();
        let base = rng.below(n_vars as u64);
        for _ in 0..k {
            let v = match window {
                Some(w) => (base + rng.below(w)) % n_vars as u64,
                None => rng.below(n_vars as u64),
            } as u32;
            let c = if rng.below(2) == 0 { 1 } else { -1 };
            row.push((v, c));
        }
        match e.add_row(&row, &prefer) {
            RowOutcome::Eliminated { pivot } => {
                assert!(!e.is_free(pivot));
                elim += 1;
                accepted.push(row);
            }
            RowOutcome::Redundant => {
                red += 1;
                accepted.push(row);
            }
            RowOutcome::NoUnitPivot => nup += 1,
            RowOutcome::Inconsistent => inc += 1,
        }
    }
    let max = e.max_abs_coeff();
    println!(
        "n_vars {n_vars} rows {n_rows} window {window:?}: eliminated {elim} redundant {red} no-unit {nup} \
         inconsistent {inc}, n_free {}, max |coeff| {max}",
        e.n_free()
    );
    assert_eq!(inc, 0);
    assert_eq!(e.n_free(), n_vars - elim);
    assert!(max < 1 << 20, "max |coeff| = {max}");
    // Every expression is over free variables only.
    for v in 0..n_vars as u32 {
        for &(w, c) in e.expr(v) {
            assert!(e.is_free(w), "expr({v}) references non-free {w}");
            assert_ne!(c, 0);
        }
    }
    // Every accepted row holds exactly (symbolically, in i64).
    for r in &accepted {
        assert!(row_holds(&e, r), "row {r:?} violated");
    }
    // And numerically with random integer free values, in i64 arithmetic.
    let vals: Vec<i64> = (0..e.n_free())
        .map(|_| rng.below(2001) as i64 - 1000)
        .collect();
    let value = |v: u32| -> i64 {
        e.expr(v)
            .iter()
            .map(|&(w, c)| c * vals[e.free_index(w).unwrap() as usize])
            .sum()
    };
    for r in &accepted {
        let s: i64 = r.iter().map(|&(v, c)| c * value(v)).sum();
        assert_eq!(s, 0);
    }
}

#[test]
fn random_rows_local_10k() {
    random_rows(20_000, 10_000, Some(60), 1);
}

#[test]
fn random_rows_global() {
    random_rows(3_000, 2_000, None, 2);
}

#[test]
fn random_rows_dense_overdetermined() {
    // More rows than variables: most become redundant or non-unit, nothing may break.
    random_rows(400, 1_000, Some(30), 3);
}

#[test]
fn deterministic() {
    let run = || {
        let mut rng = SplitMix64::new(9);
        let mut e = Elim::new(500);
        for _ in 0..400 {
            let a = rng.below(500) as u32;
            let b = rng.below(500) as u32;
            let c = rng.below(500) as u32;
            e.add_row(&[(a, 1), (b, -1), (c, 1)], &|v| (v % 3) as i32);
        }
        (0..500).map(|v| e.expr(v).to_vec()).collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn try_new_matches_new() {
    let a = aq_linalg::Elim::try_new(1000).unwrap();
    let b = aq_linalg::Elim::new(1000);
    assert_eq!(a.n_vars(), 1000);
    assert_eq!(a.n_free(), b.n_free());
    for v in [0u32, 1, 999] {
        assert_eq!(a.expr(v), b.expr(v));
        assert_eq!(a.free_index(v), b.free_index(v));
    }
}

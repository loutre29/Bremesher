//! Integer linear expressions and symbolic transitions (DESIGN 6.9 "Corner expressions").
//!
//! A [`Lin`] is a sparse integer combination `sum c_i x_i` of scalar variables, sorted by
//! variable id with no zero coefficient. A [`Lin2`] is a 2-vector of them (u and v). A
//! [`SymTr`] is a chart transition `phi -> R(-r) phi + t` (C10) whose translation is symbolic.
//! Every operation is exact (checked i64 arithmetic).

use aq_core::AqError;

/// Sorted sparse integer combination of scalar variables.
pub(crate) type Lin = Vec<(u32, i64)>;
/// A symbolic 2-vector (u, v).
pub(crate) type Lin2 = [Lin; 2];

fn overflow() -> AqError {
    AqError::numerical("aq_param: integer overflow in a corner expression")
}

/// Sorts by variable and merges equal variables (checked), dropping zero coefficients.
pub(crate) fn normalize(t: &mut Lin) -> Result<(), AqError> {
    t.sort_by_key(|e| e.0);
    let mut out = 0;
    let mut k = 0;
    while k < t.len() {
        let v = t[k].0;
        let mut c = t[k].1;
        k += 1;
        while k < t.len() && t[k].0 == v {
            c = c.checked_add(t[k].1).ok_or_else(overflow)?;
            k += 1;
        }
        if c != 0 {
            t[out] = (v, c);
            out += 1;
        }
    }
    t.truncate(out);
    Ok(())
}

/// `-a`.
pub(crate) fn neg(a: &Lin) -> Result<Lin, AqError> {
    a.iter()
        .map(|&(v, c)| c.checked_neg().map(|n| (v, n)).ok_or_else(overflow))
        .collect()
}

/// `a + b`, normalized.
pub(crate) fn add(a: &Lin, b: &Lin) -> Result<Lin, AqError> {
    let mut r = Vec::with_capacity(a.len() + b.len());
    r.extend_from_slice(a);
    r.extend_from_slice(b);
    normalize(&mut r)?;
    Ok(r)
}

/// `a - b`, normalized.
pub(crate) fn sub(a: &Lin, b: &Lin) -> Result<Lin, AqError> {
    add(a, &neg(b)?)
}

/// C10 `R(k)` applied to a symbolic vector: `R(1)(u, v) = (-v, u)`.
pub(crate) fn rot(k: u8, a: &Lin2) -> Result<Lin2, AqError> {
    Ok(match k % 4 {
        0 => a.clone(),
        1 => [neg(&a[1])?, a[0].clone()],
        2 => [neg(&a[0])?, neg(&a[1])?],
        _ => [a[1].clone(), neg(&a[0])?],
    })
}

/// The unit vector of the two scalar variables `(base, base + 1)`.
pub(crate) fn var2(base: u32) -> Lin2 {
    [vec![(base, 1)], vec![(base + 1, 1)]]
}

/// A chart transition with a symbolic translation: `phi -> R(-r) phi + t`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SymTr {
    pub r: u8,
    pub t: Lin2,
}

impl SymTr {
    pub fn identity() -> Self {
        SymTr {
            r: 0,
            t: [Vec::new(), Vec::new()],
        }
    }

    /// First `self`, then `(r2, t2)`: `r = r1 + r2`, `t = R(-r2) t1 + t2` (as
    /// `aq_core::Transition::compose`).
    pub fn then(&self, r2: u8, t2: &Lin2) -> Result<SymTr, AqError> {
        let a = rot((4 - r2 % 4) % 4, &self.t)?;
        Ok(SymTr {
            r: (self.r % 4 + r2 % 4) % 4,
            t: [add(&a[0], &t2[0])?, add(&a[1], &t2[1])?],
        })
    }

    /// `R(-r) x + t`.
    pub fn apply(&self, x: &Lin2) -> Result<Lin2, AqError> {
        let a = rot((4 - self.r % 4) % 4, x)?;
        Ok([add(&a[0], &self.t[0])?, add(&a[1], &self.t[1])?])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use aq_core::Transition;

    /// Evaluates a symbolic vector at integer values.
    fn ev(a: &Lin2, x: &[f64]) -> [f64; 2] {
        let e = |l: &Lin| {
            l.iter()
                .map(|&(v, c)| c as f64 * x[v as usize])
                .sum::<f64>()
                + 0.0
        };
        [e(&a[0]), e(&a[1])]
    }

    #[test]
    fn symbolic_matches_numeric_transitions() {
        // Variables 0,1 = point; 2,3 = t1; 4,5 = t2.
        let x = [3.0, -5.0, 7.0, 11.0, -13.0, 17.0];
        for r1 in 0..4u8 {
            for r2 in 0..4u8 {
                let s = SymTr::identity()
                    .then(r1, &var2(2))
                    .unwrap()
                    .then(r2, &var2(4))
                    .unwrap();
                let t1 = Transition {
                    r: r1,
                    t: [x[2], x[3]],
                };
                let t2 = Transition {
                    r: r2,
                    t: [x[4], x[5]],
                };
                let c = t1.compose(&t2);
                assert_eq!(s.r, c.r);
                assert_eq!(ev(&s.t, &x), c.t);
                assert_eq!(ev(&s.apply(&var2(0)).unwrap(), &x), c.apply([x[0], x[1]]));
            }
        }
    }

    #[test]
    fn normalize_merges_and_drops() {
        let mut l = vec![(3, 1), (1, 2), (3, -1), (1, 1)];
        normalize(&mut l).unwrap();
        assert_eq!(l, vec![(1, 3)]);
        let mut o = vec![(0, i64::MAX), (0, 1)];
        assert!(normalize(&mut o).is_err());
    }
}

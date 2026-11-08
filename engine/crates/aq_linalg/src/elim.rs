//! Exact linear elimination of homogeneous integer constraints (DESIGN 6.3, used by aq_param).
//!
//! Every variable is either **free**, **eliminated** (an integer expression over free
//! variables), **pinned to zero** (gauge) or **pinned to a value** (a caller-supplied constant
//! that stays as a term in expressions). A row `sum c_i x_i = 0` is processed as follows:
//!
//! 1. Substitute the current expressions, merge equal variables, drop zero coefficients.
//! 2. An empty row is [`RowOutcome::Redundant`].
//! 3. The pivot is a free variable with coefficient +-1 maximizing `prefer`, ties broken by the
//!    lowest id.
//! 4. `x_p = -c_p * sum_{i != p} c_i x_i` is substituted into every expression referencing p
//!    (reverse index).
//! 5. No coefficient +-1 gives [`RowOutcome::NoUnitPivot`] (the caller turns the row into a
//!    penalty). A row that only contains value-pinned variables is
//!    [`RowOutcome::Inconsistent`]: it cannot be enforced, because their values are not part
//!    of the elimination. An integer overflow in step 1 or 4 also gives `NoUnitPivot`, with the
//!    state unchanged.
//!
//! Every coefficient is an `i64` and every operation is checked, so expressions are exact.
//! Expressions are sorted by variable id, which makes everything deterministic.

use std::sync::OnceLock;

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::{AqError, NONE};

/// Result of [`Elim::add_row`] and [`Elim::pin`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowOutcome {
    /// The row was eliminated; `pivot` is no longer free.
    Eliminated { pivot: u32 },
    /// The row already holds (it reduces to 0 = 0).
    Redundant,
    /// The row cannot hold (it only involves pinned values), or it names a variable out of
    /// range. Nothing changed.
    Inconsistent,
    /// No free variable has a coefficient of +-1. Nothing changed.
    NoUnitPivot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Free,
    Expr,
    PinnedZero,
    PinnedValue,
}

/// Exact elimination state. `Send + Sync`.
#[derive(Debug)]
pub struct Elim {
    state: Vec<State>,
    /// Free and value-pinned: `[(v, 1)]`; eliminated: the expression; zero-pinned: empty.
    expr: Vec<Vec<(u32, i64)>>,
    /// For a free or value-pinned variable f: variables whose expression may contain f
    /// (a superset: stale entries are skipped).
    refs: Vec<Vec<u32>>,
    n_free: usize,
    free_idx: OnceLock<Vec<u32>>,
}

impl Clone for Elim {
    fn clone(&self) -> Self {
        Elim {
            state: self.state.clone(),
            expr: self.expr.clone(),
            refs: self.refs.clone(),
            n_free: self.n_free,
            free_idx: OnceLock::new(),
        }
    }
}

/// `a * b` checked.
fn mul(a: i64, b: i64) -> Option<i64> {
    a.checked_mul(b)
}

/// Sorts by variable and merges equal variables with checked sums, dropping zeros.
fn normalize(terms: &mut Vec<(u32, i64)>) -> Option<()> {
    terms.sort_by_key(|t| t.0);
    let mut out = 0;
    let mut k = 0;
    while k < terms.len() {
        let v = terms[k].0;
        let mut c = terms[k].1;
        k += 1;
        while k < terms.len() && terms[k].0 == v {
            c = c.checked_add(terms[k].1)?;
            k += 1;
        }
        if c != 0 {
            terms[out] = (v, c);
            out += 1;
        }
    }
    terms.truncate(out);
    Some(())
}

impl Elim {
    /// `n_vars` free variables `0..n_vars` (`n_vars` is capped at `u32::MAX`).
    /// The state takes about 60 bytes per variable; when it cannot be allocated this aborts
    /// like any `Vec`, so a caller with a large `n_vars` should use [`Elim::try_new`].
    pub fn new(n_vars: usize) -> Self {
        match Elim::try_new(n_vars) {
            Ok(e) => e,
            Err(_) => std::alloc::handle_alloc_error(std::alloc::Layout::new::<(u32, i64)>()),
        }
    }

    /// [`Elim::new`] returning `ErrKind::Budget` when the state cannot be allocated
    /// (DESIGN 3.4: buffers above 16 MB are allocated fallibly, about 300k variables here).
    pub fn try_new(n_vars: usize) -> Result<Self, AqError> {
        let n = n_vars.min(NONE as usize);
        let state = try_vec(n, State::Free)?;
        let mut expr: Vec<Vec<(u32, i64)>> = try_with_capacity(n)?;
        for v in 0..n as u32 {
            let mut e = try_with_capacity(1)?;
            e.push((v, 1));
            expr.push(e);
        }
        let refs = try_vec(n, Vec::new())?;
        Ok(Elim {
            state,
            expr,
            refs,
            n_free: n,
            free_idx: OnceLock::new(),
        })
    }

    pub fn n_vars(&self) -> usize {
        self.state.len()
    }

    /// Number of free variables.
    pub fn n_free(&self) -> usize {
        self.n_free
    }

    pub fn is_free(&self, var: u32) -> bool {
        self.state.get(var as usize) == Some(&State::Free)
    }

    /// True for a variable pinned by [`Elim::pin`] (to zero or to a value).
    pub fn is_pinned(&self, var: u32) -> bool {
        matches!(
            self.state.get(var as usize),
            Some(State::PinnedZero | State::PinnedValue)
        )
    }

    /// True for a variable pinned to a caller-supplied value (`pin(var, false)`).
    pub fn is_value_pinned(&self, var: u32) -> bool {
        self.state.get(var as usize) == Some(&State::PinnedValue)
    }

    /// The expression of `var` as sorted `(variable, coefficient)` terms over free variables
    /// and value-pinned variables: `[(var, 1)]` for a free or value-pinned variable, empty for
    /// a zero-pinned one (and for an id out of range).
    pub fn expr(&self, var: u32) -> &[(u32, i64)] {
        self.expr.get(var as usize).map_or(&[], |e| e.as_slice())
    }

    /// Dense index of a free variable in `0..n_free()`, in increasing variable order. The
    /// table is built on the first call after a change (O(n_vars)).
    pub fn free_index(&self, var: u32) -> Option<u32> {
        let t = self.free_idx.get_or_init(|| {
            let mut k = 0u32;
            self.state
                .iter()
                .map(|s| {
                    if *s == State::Free {
                        k += 1;
                        k - 1
                    } else {
                        NONE
                    }
                })
                .collect()
        });
        t.get(var as usize).copied().filter(|&i| i != NONE)
    }

    /// The free variables in free-index order.
    pub fn free_vars(&self) -> Vec<u32> {
        (0..self.state.len() as u32)
            .filter(|&v| self.state[v as usize] == State::Free)
            .collect()
    }

    /// Largest |coefficient| over all expressions.
    pub fn max_abs_coeff(&self) -> u64 {
        self.expr
            .iter()
            .flatten()
            .map(|t| t.1.unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    /// Value of `var` given the free values (indexed by [`Elim::free_index`]) and the values
    /// of value-pinned variables (`pinned(var)`).
    pub fn eval_with(&self, var: u32, free: &[f64], pinned: &dyn Fn(u32) -> f64) -> f64 {
        let mut s = 0.0;
        for &(v, c) in self.expr(var) {
            let x = match self.state[v as usize] {
                State::PinnedValue => pinned(v),
                _ => self
                    .free_index(v)
                    .and_then(|i| free.get(i as usize).copied())
                    .unwrap_or(f64::NAN),
            };
            s += c as f64 * x;
        }
        s
    }

    /// [`Elim::eval_with`] when no variable is value-pinned (value-pinned terms count as 0).
    pub fn eval(&self, var: u32, free: &[f64]) -> f64 {
        self.eval_with(var, free, &|_| 0.0)
    }

    /// `sum c_i expr(x_i)`, normalized; `None` on overflow or an id out of range.
    pub fn substitute(&self, row: &[(u32, i64)]) -> Option<Vec<(u32, i64)>> {
        let mut acc: Vec<(u32, i64)> = Vec::new();
        for &(v, c) in row {
            let e = self.expr.get(v as usize)?;
            for &(w, d) in e {
                acc.push((w, mul(c, d)?));
            }
        }
        normalize(&mut acc)?;
        Some(acc)
    }

    /// Adds the homogeneous row `sum c_i x_i = 0` (see the module documentation).
    pub fn add_row(&mut self, row: &[(u32, i64)], prefer: &dyn Fn(u32) -> i32) -> RowOutcome {
        if row.iter().any(|&(v, _)| v as usize >= self.state.len()) {
            return RowOutcome::Inconsistent;
        }
        let Some(terms) = self.substitute(row) else {
            return RowOutcome::NoUnitPivot;
        };
        if terms.is_empty() {
            return RowOutcome::Redundant;
        }
        let mut best: Option<(u32, i32, i64)> = None; // (var, prefer, coeff)
        let mut any_free = false;
        for &(v, c) in &terms {
            if self.state[v as usize] != State::Free {
                continue;
            }
            any_free = true;
            if c != 1 && c != -1 {
                continue;
            }
            let pr = prefer(v);
            // Terms are sorted by id, so a strict > keeps the lowest id on ties.
            if best.is_none_or(|b| pr > b.1) {
                best = Some((v, pr, c));
            }
        }
        let Some((p, _, cp)) = best else {
            return if any_free {
                RowOutcome::NoUnitPivot
            } else {
                RowOutcome::Inconsistent
            };
        };
        // x_p = -c_p * sum_{i != p} c_i x_i.
        let mut ep: Vec<(u32, i64)> = Vec::with_capacity(terms.len() - 1);
        for &(v, c) in &terms {
            if v != p {
                let Some(x) = mul(-cp, c) else {
                    return RowOutcome::NoUnitPivot;
                };
                ep.push((v, x));
            }
        }
        match self.eliminate(p, ep) {
            Some(()) => RowOutcome::Eliminated { pivot: p },
            None => RowOutcome::NoUnitPivot,
        }
    }

    /// Replaces free variable `p` by `ep` (sorted, over free or value-pinned variables other
    /// than p) everywhere. `None` (state unchanged) on overflow.
    fn eliminate(&mut self, p: u32, ep: Vec<(u32, i64)>) -> Option<()> {
        let mut users = std::mem::take(&mut self.refs[p as usize]);
        users.sort_unstable();
        users.dedup();
        // Compute every new expression before committing anything.
        let mut updates: Vec<(u32, Vec<(u32, i64)>)> = Vec::new();
        for &q in &users {
            if self.state[q as usize] != State::Expr {
                continue;
            }
            let old = &self.expr[q as usize];
            let Ok(k) = old.binary_search_by_key(&p, |t| t.0) else {
                continue;
            };
            let cq = old[k].1;
            let mut new: Vec<(u32, i64)> = Vec::with_capacity(old.len() + ep.len());
            // Sorted merge of (old without p) and cq * ep.
            let (mut i, mut j) = (0, 0);
            let mut fail = false;
            while i < old.len() || j < ep.len() {
                if i < old.len() && old[i].0 == p {
                    i += 1;
                    continue;
                }
                let a = old.get(i).map_or(u32::MAX, |t| t.0);
                let b = ep.get(j).map_or(u32::MAX, |t| t.0);
                if a < b {
                    new.push(old[i]);
                    i += 1;
                } else {
                    let Some(x) = mul(cq, ep[j].1) else {
                        fail = true;
                        break;
                    };
                    if a == b {
                        let Some(s) = old[i].1.checked_add(x) else {
                            fail = true;
                            break;
                        };
                        if s != 0 {
                            new.push((a, s));
                        }
                        i += 1;
                    } else {
                        new.push((b, x));
                    }
                    j += 1;
                }
            }
            if fail {
                self.refs[p as usize] = users;
                return None;
            }
            updates.push((q, new));
        }
        // Commit.
        for (q, new) in updates {
            let old = std::mem::replace(&mut self.expr[q as usize], new);
            for &(f, _) in &self.expr[q as usize] {
                if old.binary_search_by_key(&f, |t| t.0).is_err() {
                    self.refs[f as usize].push(q);
                }
            }
        }
        for &(f, _) in &ep {
            self.refs[f as usize].push(p);
        }
        self.expr[p as usize] = ep;
        self.state[p as usize] = State::Expr;
        self.n_free -= 1;
        self.free_idx.take();
        Some(())
    }

    /// Gauge: pins `var`. With `value_zero` the variable is 0 and disappears from every
    /// expression; otherwise it keeps a caller-supplied value and stays as a term (see
    /// [`Elim::eval_with`]). Pinning a variable that is already eliminated to zero is the row
    /// `x_var = 0` (lowest-id pivot); pinning it to a value is `Inconsistent` (its value is
    /// determined by the free variables). The return value may be ignored.
    pub fn pin(&mut self, var: u32, value_zero: bool) -> RowOutcome {
        let Some(&st) = self.state.get(var as usize) else {
            return RowOutcome::Inconsistent;
        };
        match (st, value_zero) {
            (State::Free, true) | (State::PinnedValue, true) => {
                self.drop_var(var);
                if st == State::Free {
                    self.n_free -= 1;
                }
                self.state[var as usize] = State::PinnedZero;
                self.expr[var as usize].clear();
                self.free_idx.take();
                RowOutcome::Eliminated { pivot: var }
            }
            (State::Free, false) => {
                self.state[var as usize] = State::PinnedValue;
                self.n_free -= 1;
                self.free_idx.take();
                RowOutcome::Eliminated { pivot: var }
            }
            (State::Expr, true) => self.add_row(&[(var, 1)], &|_| 0),
            (State::Expr, false) => RowOutcome::Inconsistent,
            (State::PinnedZero, true) | (State::PinnedValue, false) => RowOutcome::Redundant,
            (State::PinnedZero, false) => RowOutcome::Inconsistent,
        }
    }

    /// Removes free or value-pinned `f` from every expression (substitutes 0).
    fn drop_var(&mut self, f: u32) {
        let mut users = std::mem::take(&mut self.refs[f as usize]);
        users.sort_unstable();
        users.dedup();
        for q in users {
            if self.state[q as usize] != State::Expr {
                continue;
            }
            let e = &mut self.expr[q as usize];
            if let Ok(k) = e.binary_search_by_key(&f, |t| t.0) {
                e.remove(k);
            }
        }
    }
}

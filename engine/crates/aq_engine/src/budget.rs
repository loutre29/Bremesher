//! S1 budgets (DESIGN 7.1 steps 2 to 5): per-component quad targets with floors, debris
//! detection, proxy triangle shares and time shares.

use std::time::Duration;

/// Quad floor of a closed component (DESIGN 7.1 step 4).
pub const FLOOR_CLOSED: u32 = 24;
/// Quad floor of an open component.
pub const FLOOR_OPEN: u32 = 8;
/// A closed component with an area below this many h^2 (quad units) is debris (7.1 step 3).
pub const DEBRIS_AREA: f64 = 6.0;
/// Smallest time share of a component (7.1 step 5).
pub const MIN_TIME_SHARE: Duration = Duration::from_secs(2);

/// The budget of one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompPlan {
    /// Quad target N_c.
    pub target: u32,
    /// Batched to SF directly, with no proxy (debris).
    pub debris: bool,
}

/// Floor of a component.
pub fn floor_of(closed: bool) -> u32 {
    if closed { FLOOR_CLOSED } else { FLOOR_OPEN }
}

/// Per-component targets (DESIGN 7.1 steps 3 and 4).
///
/// `areas` are in quad units (so that `sum areas` is about `n_target`, but any positive scale
/// works: pass 1 is `N_c = n_target * A_c / A_total`). A component is debris when it is
/// closed and either has an area below [`DEBRIS_AREA`] h^2 (with `h^2 = A_total / n_target`) or
/// a pass-1 N_c below its floor, or when it is open with both (WP-20: an open piece of at least
/// [`DEBRIS_AREA`] h^2 runs the full path at its floor); debris gets its floor. The remaining budget
/// `n_target - sum(debris floors)` is then shared by area among the other components; a
/// component that falls below its floor there is fixed at its floor (not debris) and the rest
/// is shared again, until stable. Integer targets use largest-remainder rounding (ties to the
/// lower index), so the non-floored targets sum exactly to what is left. When the floors alone
/// exceed `n_target`, every component gets its floor and the total exceeds `n_target`.
///
/// A non-finite or non-positive area counts as 0 (the component becomes debris).
pub fn plan_targets(areas: &[f64], closed: &[bool], n_target: u32) -> Vec<CompPlan> {
    let n = areas.len();
    let a: Vec<f64> = areas
        .iter()
        .map(|&x| if x.is_finite() && x > 0.0 { x } else { 0.0 })
        .collect();
    let total: f64 = a.iter().sum();
    let nt = n_target as f64;
    let h2 = if total > 0.0 && nt > 0.0 {
        total / nt
    } else {
        1.0
    };
    let mut plan = vec![
        CompPlan {
            target: 0,
            debris: false
        };
        n
    ];
    // fixed[i]: the target is its floor (debris or floored in a later round).
    let mut fixed = vec![false; n];
    for i in 0..n {
        let fl = floor_of(closed.get(i).copied().unwrap_or(true));
        let pass1 = if total > 0.0 { nt * a[i] / total } else { 0.0 };
        let is_closed = closed.get(i).copied().unwrap_or(true);
        let tiny = a[i] < DEBRIS_AREA * h2;
        // WP-20: an open component below its floor of 8 quads but with at least 6 h^2 (a nose
        // pad, a lens) runs the full path at its floor instead of SF (CHANGES.md, WP-20).
        if (is_closed && (tiny || pass1 < fl as f64)) || (!is_closed && tiny && pass1 < fl as f64) {
            plan[i] = CompPlan {
                target: fl,
                debris: true,
            };
            fixed[i] = true;
        }
    }
    loop {
        let used: u64 = (0..n)
            .filter(|&i| fixed[i])
            .map(|i| plan[i].target as u64)
            .sum();
        let left = (n_target as u64).saturating_sub(used) as f64;
        let rest: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();
        if rest.is_empty() {
            break;
        }
        let a_rest: f64 = rest.iter().map(|&i| a[i]).sum();
        let share = |i: usize| {
            if a_rest > 0.0 {
                left * a[i] / a_rest
            } else {
                0.0
            }
        };
        // Floor the components whose share is below their floor, then share again.
        let mut changed = false;
        for &i in &rest {
            let fl = floor_of(closed.get(i).copied().unwrap_or(true));
            if share(i) < fl as f64 {
                plan[i].target = fl;
                fixed[i] = true;
                changed = true;
            }
        }
        if changed {
            continue;
        }
        // Largest remainder rounding of the shares.
        let mut sum_floor = 0u64;
        let mut rem: Vec<(f64, usize)> = Vec::with_capacity(rest.len());
        for &i in &rest {
            let s = share(i);
            let f = s.floor();
            plan[i].target = f as u32;
            sum_floor += f as u64;
            rem.push((s - f, i));
        }
        let mut extra = (left.round() as u64).saturating_sub(sum_floor);
        rem.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        for &(_, i) in &rem {
            if extra == 0 {
                break;
            }
            plan[i].target += 1;
            extra -= 1;
        }
        break;
    }
    plan
}

/// WP-20: share of the target the global thickness plan may spend on thin parts. Larger than
/// the in-component budget (`aq_curv::THICK_BUDGET`): a component that is thin all over (a
/// spectacle frame) gets nothing from its own rung, only from this plan, and on retopo_a the
/// frame meshes reliably only from about 450 quads up (CHANGES.md, WP-20).
pub const PLAN_THICK_BUDGET: f64 = 0.25;
/// WP-20: lowest floor of the global thickness plan, in h.
pub const PLAN_FLOOR_MIN: f64 = 0.5;

/// WP-20: shares of the target by quad demand under the global thickness cap.
///
/// With two or more components (in quad units, `areas` their areas), the shape diameter of
/// every component of at least [`DEBRIS_AREA`] h^2 is measured on its input triangles and
/// [`aq_curv::thickness_plan`] finds the floor of the caps `pi d / ring_min` that spends at most
/// [`PLAN_THICK_BUDGET`] of the target (and is at least [`PLAN_FLOOR_MIN`]), and each
/// component's demand at that floor. The
/// demands replace the areas in [`plan_targets`] (they are in quads, so the debris rule reads
/// the same), and the floor goes to every rung as a lower bound of its own floor. With one component, or when the plan fails
/// (a warning), the areas are returned with no floor: each rung then plans by itself.
pub(crate) fn thickness_shares(
    comps: &[aq_prep::ComponentMesh],
    areas: &[f64],
    p: &crate::Params,
    ctl: &aq_core::Ctl,
    warnings: &mut Vec<String>,
) -> (Vec<f64>, Option<f64>) {
    if !crate::pipeline::THICKNESS || comps.len() < 2 || p.ring_min == 0 || !crate::pipeline::AVAILABLE {
        return (areas.to_vec(), None);
    }
    let n = p.target_quads as f64;
    let total: f64 = areas.iter().filter(|a| a.is_finite() && **a > 0.0).sum();
    if !(total > 0.0 && n > 0.0) {
        return (areas.to_vec(), None);
    }
    let h2 = total / n;
    let measure: Vec<bool> = areas.iter().map(|&a| a >= DEBRIS_AREA * h2).collect();
    let meshes: Vec<&aq_core::TriMesh> = comps.iter().map(|c| &c.mesh).collect();
    match aq_curv::thickness_plan(&meshes, &measure, p.ring_min, n, PLAN_THICK_BUDGET, PLAN_FLOOR_MIN, ctl) {
        Ok(tp) => (tp.demand, Some(tp.floor)),
        Err(e) => {
            if !matches!(e.kind, aq_core::ErrKind::Cancelled | aq_core::ErrKind::Timeout) {
                warnings.push(format!("thickness plan: {e}; shares by area"));
            }
            (areas.to_vec(), None)
        }
    }
}

/// Proxy triangle budget of each component, proportional to its target (7.1 step 4).
pub fn proxy_shares(plan: &[CompPlan], max_proxy_tris: u32) -> Vec<u32> {
    let sum: f64 = plan.iter().map(|p| p.target as f64).sum();
    plan.iter()
        .map(|p| {
            if sum > 0.0 {
                (max_proxy_tris as f64 * p.target as f64 / sum).round() as u32
            } else {
                0
            }
        })
        .collect()
}

/// Time share of each component: `T_c = remaining * N_c / sum N`, at least
/// [`MIN_TIME_SHARE`] (7.1 step 5).
pub fn time_shares(plan: &[CompPlan], remaining: Duration) -> Vec<Duration> {
    let sum: f64 = plan.iter().map(|p| p.target as f64).sum();
    plan.iter()
        .map(|p| {
            let f = if sum > 0.0 {
                p.target as f64 / sum
            } else {
                0.0
            };
            let t = Duration::try_from_secs_f64(remaining.as_secs_f64() * f).unwrap_or(remaining);
            t.max(MIN_TIME_SHARE)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn total(p: &[CompPlan]) -> u64 {
        p.iter().map(|c| c.target as u64).sum()
    }

    #[test]
    fn single_component_gets_everything() {
        let p = plan_targets(&[5000.0], &[true], 5000);
        assert_eq!(
            p,
            vec![CompPlan {
                target: 5000,
                debris: false
            }]
        );
    }

    #[test]
    fn proportional_split_sums_to_target() {
        let p = plan_targets(&[1.0, 2.0, 3.0], &[true, false, true], 2000);
        assert_eq!(total(&p), 2000);
        assert_eq!(p[0].target, 333);
        assert_eq!(p[1].target, 667);
        assert_eq!(p[2].target, 1000);
        assert!(p.iter().all(|c| !c.debris));
    }

    #[test]
    fn debris_gets_floor_and_rest_is_renormalized() {
        // Component 1 is tiny and closed, component 2 tiny and open.
        let p = plan_targets(&[1000.0, 0.5, 0.5], &[true, true, false], 1000);
        assert!(p[1].debris && p[1].target == FLOOR_CLOSED);
        assert!(p[2].debris && p[2].target == FLOOR_OPEN);
        assert!(!p[0].debris);
        assert_eq!(total(&p), 1000);
        assert_eq!(p[0].target, 1000 - 24 - 8);
    }

    #[test]
    fn closed_small_area_is_debris_even_above_floor() {
        // Pass-1 N_c = 30 >= 24, but the area is below 6 h^2 only when h^2 is large:
        // here h^2 = 1, so area 30 is not debris; area 5 is (and also below the floor).
        let p = plan_targets(&[970.0, 30.0], &[true, true], 1000);
        assert!(!p[1].debris);
        let p = plan_targets(&[995.0, 5.0], &[true, true], 1000);
        assert!(p[1].debris);
    }

    #[test]
    fn open_piece_above_debris_area_runs_at_its_floor() {
        // WP-20: an open component of 7 h^2 (pass-1 N_c 7 < floor 8) is not debris any more; it
        // is floored at 8. One of 5 h^2 stays debris.
        let p = plan_targets(&[993.0, 7.0], &[true, false], 1000);
        assert!(!p[1].debris && p[1].target == FLOOR_OPEN);
        assert_eq!(total(&p), 1000);
        let p = plan_targets(&[995.0, 5.0], &[true, false], 1000);
        assert!(p[1].debris && p[1].target == FLOOR_OPEN);
        // Closed components keep the M1 rule.
        let p = plan_targets(&[993.0, 7.0], &[true, true], 1000);
        assert!(p[1].debris && p[1].target == FLOOR_CLOSED);
    }

    #[test]
    fn renormalization_floors_without_debris() {
        // 30 components of pass-1 N_c 33 (above the floor 24) and 20 debris of floor 24
        // each: after the debris take 480, the rest get (1000 - 480) / 30 = 17.3 < 24, so
        // they are floored too (not debris).
        let mut areas = vec![33.0; 30];
        areas.extend(vec![0.5; 20]);
        let closed = vec![true; 50];
        let p = plan_targets(&areas, &closed, 1000);
        assert!(p[..30].iter().all(|c| !c.debris && c.target == 24));
        assert!(p[30..].iter().all(|c| c.debris && c.target == 24));
    }

    #[test]
    fn floors_exceed_target() {
        let p = plan_targets(&[1.0; 10], &[true; 10], 100);
        assert!(p.iter().all(|c| c.target == 24 && c.debris));
        assert_eq!(total(&p), 240);
    }

    #[test]
    fn bad_areas_become_debris() {
        let p = plan_targets(&[f64::NAN, 100.0, -1.0], &[true, true, false], 100);
        assert!(p[0].debris && p[2].debris);
        assert!(!p[1].debris);
    }

    #[test]
    fn deterministic_rounding() {
        let a = plan_targets(&[1.0, 1.0, 1.0], &[false; 3], 100);
        let b = plan_targets(&[1.0, 1.0, 1.0], &[false; 3], 100);
        assert_eq!(a, b);
        assert_eq!(total(&a), 100);
        assert_eq!(a[0].target, 34);
    }

    #[test]
    fn shares() {
        let p = vec![
            CompPlan {
                target: 300,
                debris: false,
            },
            CompPlan {
                target: 100,
                debris: false,
            },
        ];
        let t = time_shares(&p, Duration::from_secs(40));
        assert_eq!(t[0], Duration::from_secs(30));
        assert_eq!(t[1], Duration::from_secs(10));
        let t = time_shares(&p, Duration::from_secs(1));
        assert!(t.iter().all(|&d| d == MIN_TIME_SHARE));
        assert_eq!(proxy_shares(&p, 150_000), vec![112_500, 37_500]);
    }
}

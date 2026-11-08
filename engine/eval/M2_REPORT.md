# M2 report (integration of WP-20 thickness cap and WP-23 poles)

Engine: `engine` after the merge of WP-20 (`_wip/thin`) and WP-23 (`_wip/poles`), plus the integration fixes in CHANGES.md, section "M2 integration (WP-20 + WP-23)".

**Verdict against the M1 gates: 7 of 13 pass, on seed 0 and on 5 of 5 seeds.** That is the same 7 as M1: double_torus, cube_sharp, cylinder_open, suzanne_sub2, noisy_blob, sculpt_sphere_mirror and the 328k perf input. No mesh got worse on its gates. Irregular vertices went down on 8 meshes. retopo_a now passes vertex distance and Hausdorff, which failed in M1.

## How it was run

```
export CARGO_TARGET_DIR=target/m2
cargo build --release -p aq_cli
target/m2/release/autoquad.exe batch --suite eval/suite.toml --out eval/results/M2 --seeds 0..4 \
    --gate M1 --compare baselines/ --markdown eval/results/M2/summary.md
```

- **Results:** `eval/results/M2/summary.md`, `summary.json`, and `<mesh>_s<seed>.obj` / `.json` for every run.
  - The batch log is `eval/scratch/M2_batch.log`.
  - The merged engine before these fixes is in `eval/scratch/M2_premerge_fix`.
- **Baselines:** `baselines/*_qf*.obj` (QuadriFlow), measured with the `tools/quad_stats.py` rules.
- **Machine:** times are wall clock on the user's machine (Windows 11), one mesh at a time.

## Gate table (seed 0; M1 / M2 / QuadriFlow)

All runs are 100% quads. Count error stays within 3.8% on every seed.

| mesh | faces M2 | irregular M1 / M2 / QF (gate) | edge CV M1 / M2 / QF | angle dev M1 / M2 / QF | vertex dist h M1 / M2 | Hausdorff h M1 / M2 | time s M1 / M2 | components M2 | verdict M1 / M2 (seeds) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 2001 | 0 / 0 / 14 (0) | 0.135 / 0.135 / 0.107 | 1.34 / 1.34 / 6.92 | 0.018 / 0.018 | 0.068 / 0.068 | 2.2 / 2.9 | 1 full | FAIL / FAIL edge_cv (0/5) |
| double_torus | 2041 | 8 / 8 / 36 (<= 12) | 0.123 / 0.123 / 0.148 | 3.28 / 3.28 / 9.19 | 0.049 / 0.049 | 0.283 / 0.283 | 2.9 / 3.4 | 1 full | PASS / PASS (5/5) |
| cube_sharp | 1944 | 8 / 8 / 8 (8 at corners) | 0.000 / 0.000 / 0.005 | 0.00 / 0.00 / 0.21 | 0.000 / 0.000 | 0.000 / 0.000 | 2.0 / 2.2 | 1 full | PASS / PASS (5/5) |
| cylinder_open | 2015 | 0 / 0 / 0 (0) | 0.001 / 0.001 / 0.014 | 0.02 / 0.02 / 0.04 | 0.008 / 0.008 | 0.024 / 0.024 | 2.6 / 2.8 | 1 full | PASS / PASS (5/5) |
| suzanne_sub2 | 1982 | 44 / **36** / 86 (<= 50) | 0.149 / 0.172 / 0.307 | 5.51 / 5.62 / 11.49 | 0.109 / 0.108 | 0.597 / 0.637 | 3.0 / 3.4 | 3 full | PASS / PASS (5/5) |
| retopo_a | 1983 | 48 / 48 / 36 (<= 24) | 0.598 / **0.380** / 0.215 | 11.47 / 10.04 / 11.82 | 0.409 / **0.091** | 1.062 / **0.769** | 5.8 / 4.8 | 6 full, 3 debris_sf | FAIL / FAIL no_sf, edge_cv, irregular (0/5) |
| retopo_b | 1944 | 34 / **26** / 38 (<= 24) | 0.160 / 0.172 / 0.253 | 7.29 / 6.15 / 9.71 | 0.045 / 0.101 | 0.603 / 0.611 | 2.9 / 3.6 | 1 full | FAIL / FAIL irregular (0/5) |
| retopo_c | 1990 | 37 / **29** / 50 (<= 24) | 0.201 / 0.214 / 0.190 | 6.99 / 8.03 / 8.64 | 0.117 / 0.095 | 0.539 / 0.596 | 8.8 / 5.5 | 1 full | FAIL / FAIL edge_cv, irregular, time (0/5) |
| sculpt_cylinder | 2013 | 24 / **23** / 32 (<= 16) | 0.173 / 0.188 / 0.197 | 6.58 / 7.08 / 8.38 | 0.045 / 0.061 | 0.871 / 0.790 | 6.0 / 6.8 | 1 full | FAIL / FAIL irregular, loops, time (0/5) |
| basemesh_female | 5132 | 140 / **128** / 136 (<= 90) | 0.426 / 0.394 / 0.302 | 7.67 / 7.33 / 16.12 | 0.256 / 0.240 | 1.037 / **0.770** | 17.7 / 24.9 | 1 full | FAIL / FAIL edge_cv, irregular, time (0/5) |
| noisy_blob | 5033 | 14 / **10** / 14 (<= 14) | 0.113 / 0.117 / 0.221 | 7.86 / 7.62 / 9.50 | 0.031 / 0.032 | 0.493 / 0.484 | 4.9 / 6.7 | 1 full | PASS / PASS (5/5) |
| sculpt_sphere_mirror | 4824 | 92 / **84** / - | 0.156 / 0.167 / - | 6.77 / 7.25 / - | 0.047 / 0.058 | 0.746 / 0.685 | 3.7 / 4.9 | 1 full | PASS / PASS (5/5) |
| perf_noisy_blob_328k | 4958 | 14 / **10** / 14 | 0.121 / 0.121 / 0.221 | 7.55 / 7.78 / 9.50 | 0.048 / 0.060 | 0.496 / 0.572 | 7.3 / 8.4 (< 30) | 1 full | PASS / PASS (5/5) |

**Seeds 0 to 4:** every mesh gives the same irregular count on all five seeds, with one exception. retopo_a seed 1 gives 55 (vertex distance 0.141 h), because there the targeted retry is not enough and all WP-23 edits are dropped.

**suzanne_sub2 edge CV over the seeds:** 0.171 to 0.179.

### Irregular vertices against each package alone (seeds 0 to 4)

| mesh | M1 | WP-20 alone | WP-23 alone | merged, before fix | M2 |
| --- | --- | --- | --- | --- | --- |
| suzanne_sub2 | 44 | 44 | 34 | 32 (CV 0.217) | 36 (CV 0.172) |
| retopo_a | 48 | 54 | 42 (vertex dist 0.285 h, fails) | 48 | 48 (vertex dist 0.091 h) |
| retopo_b | 34 | 34 | 26 | 26 | 26 |
| retopo_c | 37 | 39 | 29 | 39 | 29 |
| sculpt_cylinder | 24 | 24 | 22 | 24 | 23 |
| basemesh_female | 140 | 146 | 119 | 128 | 128 |
| noisy_blob | 14 | 14 | 10 | 10 | 10 |
| sculpt_sphere_mirror | 92 | 96 | 74 | 80 (CV 0.178) | 84 (CV 0.167) |

M2 is at least as good as WP-20 alone on every mesh. It matches WP-23 alone on retopo_b, retopo_c and noisy_blob. It stays above WP-23 alone in four places:

- suzanne_sub2: +2;
- retopo_a: +6, the price of its distance gates;
- sculpt_cylinder: +1;
- basemesh_female: +9.

## What was fixed

The three WP-23 edits below were tuned on the M1 size field. On the thickness-aware size field and its refined proxy, each of them broke something. Details are in CHANGES.md.

1. **retopo_c: a boundary merge into a slit vertex.**
   - **Cause:** on the new proxy, two +1 cones merged into one slit vertex of the hem, with omega 367 degrees, which turned it into a flat 180 degree quad corner. S7 folded and S8 failed on every seed, so the retry dropped all WP-23 edits.
   - **Fix:** a boundary merge is now allowed only when the new quad corner angle k * 90 is within a factor 1.5 of the surface angle (`BOUNDARY_MERGE_RATIO`).
   - **Result:** 39 to 29.
2. **suzanne_sub2: far cancellation across the thickness size step on the ear rims** (h 0.79 against 1.05).
   - **Cause:** cancelling those pairs removed the dislocations that absorb the scale change, and edge CV went to 0.217.
   - **Fix:** far pairs are now cancelled only when their h ratio is at most 1.15 (`FAR_SIZE_RATIO`).
   - **Result:** CV 0.172, irregular 36.
3. **sculpt_cylinder: a -2 merge next to an S7 fold.**
   - **Cause:** S8 failed and all edits were dropped.
   - **Fix:** a failed S5 to S8 attempt now hands back its fold points. S4 runs again with those points as `keep_out`, so no edit may use a cone within 1.5 h of them. The rest of the edits are kept, and the full drop remains as the last resort.
   - **Result:** 24 to 23.
4. **The pole cap re-run cast the thickness rays and solved a size field it then threw away.** It now measures curvature only. Every S3 call uses the same thickness-aware sizing parameters (checked).
5. **`solve_scale`** now uses regula falsi instead of bisection: basemesh_female S3 1.6 s to 1.4 s, same output.
6. **aq_field `time_budget_100k_faces`: machine load, not a slowdown.**
   - It failed once in a full parallel test run.
   - Alone it passes at 2.02 s (limit 3 s).
   - The M1 workspace gives 1.98 s back to back on the same machine.

## Tests

- `cargo test --release` passes on aq_curv, aq_field, aq_quantize, aq_engine and aq_cli. `aq_field/tests/m2_poles.rs` now has 12 tests, 3 of them new.
- On aq_ffi, `abi`, `fuzz` and `perf` pass.
- One aq_ffi test is time-only and failed only under load: `all_testdata_meshes_give_valid_all_quad_meshes`.
  - In the full parallel run, basemesh_female took 30.56 s against its 30 s limit, with the perf test running at the same time.
  - Run alone, it passes: basemesh_female 26.9 s, all 12 meshes valid.
- Clippy (all targets) is clean on aq_curv, aq_field, aq_quantize, aq_engine, aq_cli and aq_ffi.

## Renders

The renders are in `renders/M2/<mesh>.png`, from seed 0, made with `tools/blender.sh --python tools/render_wire.py`. The rows are ours (M2), M1, then QuadriFlow. Poles are marked: valence 3 red, 5 blue, 6+ magenta.

- **Meshes rendered:** suzanne_sub2, retopo_a, retopo_b, retopo_c, sculpt_cylinder, basemesh_female, noisy_blob, sculpt_sphere_mirror.
- **sculpt_sphere_mirror** has no baseline, so its bottom row is the input. It is rendered from copies scaled x20 (`eval/scratch/scale_obj.py`).

What they show:

- **suzanne_sub2:** cleaner ear rims than M1 and fewer poles around the eyes. QuadriFlow has pole clusters on the brows and the chin.
- **retopo_c:** the same layout as M1, with fewer poles at the strand roots.
- **retopo_b:** fewer poles than M1. QuadriFlow breaks up at the ear.
- **sculpt_cylinder:**
  - One valence 6 pole sits at the right strand root: the one -2 merge kept by the retry.
  - The zigzag on the front panel follows the input and is the same in M1.
- **retopo_a:**
  - The spectacle frame is complete, while QuadriFlow keeps only a fragment of it.
  - The jagged shirt hem matches M1.
- **basemesh_female, noisy_blob and sculpt_sphere_mirror:** close to M1.

## What remains

- **Gates missed:**
  - **Irregular:** retopo_b 26 (gate 24), retopo_c 29 (24), sculpt_cylinder 23 (16), basemesh_female 128 (90), retopo_a 48 (24).
  - **Edge CV:**
    - retopo_c 0.214 against QuadriFlow's 0.190.
    - basemesh_female 0.394 (gate 0.20).
    - retopo_a 0.380 (gate 0.20).
    - torus 0.135 against QuadriFlow's 0.107, as in M1.
  - **Time (5 s):** retopo_c 5.5 s, sculpt_cylinder 6.8 s, basemesh_female 24.9 s.
  - **Loops:** sculpt_cylinder 0.68 closed.
  - **retopo_a no_sf:** 3 debris components of 2 to 3 triangles still go to SF. WP-20 suggested a gate ruling for this: exclude debris below 1 h^2.
- **basemesh_female 128 against WP-23's 119.** The thickness size field changes the proxy and the field: 269 initial cones and 66 cancellations, against 277 and 73. That leaves extra close +-1 pairs on the toes and the head. I tried running S4 on a size field without thickness: it gave 122 cones, but S8 then failed. I rejected it.
- **basemesh_female time: 24.9 s against M1's 17.7 s.** Measured back to back: M1 19.0 s, WP-23 alone 21.8 s, WP-20 alone 22.4 s, merged 22.7 s. Where the extra time goes:
  - S4: +1.1 s for the concurrent smooth reference of the pole cap. It never fired on the suite.
  - S7 (aq_param): +2 s, which each package shows alone. aq_param was not touched.
  - S3: +0.7 s for rays on both proxies.
  - The aq_ffi testdata test has about 3 s of margin to its 30 s limit when run alone, and none under parallel load.
- **sculpt_sphere_mirror 84 against WP-23's 74.** Edge CV 0.167 is better than both the premerge 0.178 and WP-23's 0.182.
- **retopo_a seed 1:** the targeted retry is not enough there, and the full drop gives 55.

# M1 report (WP-16, round 3)

Engine path M1: the full S2 to S11 pipeline in `aq_engine/src/pipeline.rs`, with the R0 / R1 / SF / QuadriFlow ladder. Code changes are in CHANGES.md, sections "WP-16" (round 1), "WP-16 round 2" and "WP-16 round 3".

**Verdict: M1 not met.** 7 of the 13 suite entries pass the M1 gates, on seed 0 and on 5 of 5 seeds. That is the same 7 as round 2 (round 1 passed 4).

The 7 that pass are double_torus, cube_sharp, cylinder_open, suzanne_sub2, noisy_blob, sculpt_sphere_mirror and the 328k perf input.

This round fixed two blockers:

- **basemesh_female now runs the full path.** Before, it went R0 fail, R1 fail, then SF. Hausdorff went from 5.3 h to 1.04 h and the time from 26 s to 17 s.
- **The retopo_a count error is now -0.1%,** down from +10.5%.

Neither mesh passes yet: their other gates still fail (see below).

## How it was run

```
export CARGO_TARGET_DIR=target/wp-16
cargo build --release -p aq_cli
target/wp-16/release/autoquad.exe batch --suite eval/suite.toml --out eval/results/M1 \
    --seeds 0..4 --gate M1 --compare baselines/ --markdown eval/results/M1/summary.md
```

- **Baselines:** `baselines/*_qf*.obj` (QuadriFlow), measured with the `tools/quad_stats.py` rules.
- **Results:** `eval/results/M1/summary.md` and `summary.json`, plus `<mesh>_s<seed>.obj` and `.json` for every run. The round 2 results were moved to `eval/scratch/M1_round2`.
- **Renders:** `renders/M1/<mesh>.png`, made with `tools/blender.sh --python tools/render_wire.py` from seed 0. The top row is ours and the bottom row is the QuadriFlow baseline.
  - sculpt_sphere_mirror has no baseline, so its bottom row is the input. It is rendered from copies scaled x20 (`eval/scratch/scale_obj.py`).
- **Tests:** `cargo test --release` passes on aq_param, aq_engine, aq_cli and aq_ffi. Clippy is clean on those crates (all targets).

## Gate table (seed 0; seeds passed out of 5)

| mesh | target | faces | quad ratio | irregular QF / ours (gate) | edge len CV ours (QF) | time s | fallback used | gate (seeds passed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 2000 | 2001 | 1.00 | 14 / 0 (0) | 0.135 (0.107) | 2.2 | none (all R0) | FAIL: edge_cv (0/5) |
| double_torus | 2000 | 2041 | 1.00 | 36 / 8 (<= 12) | 0.123 (0.148) | 2.9 | none (all R0) | PASS (5/5) |
| cube_sharp | 2000 | 1944 | 1.00 | 8 / 8 (8 at corners) | 0.000 (0.005) | 2.0 | none (all R0) | PASS (5/5) |
| cylinder_open | 2000 | 2015 | 1.00 | 0 / 0 (0) | 0.001 (0.014) | 2.6 | none (all R0) | PASS (5/5) |
| suzanne_sub2 | 2000 | 1986 | 1.00 | 86 / 44 (<= 50) | 0.149 (0.307) | 3.0 | none (all R0) | PASS (5/5) |
| retopo_a | 2000 | 1997 | 1.00 | 36 / 48 (<= 24) | 0.598 (0.215) | 5.8 (4.3 alone) | 5 full, 4 debris_sf | FAIL: no_sf, edge_cv, vertex_dist, hausdorff, irregular, time under load (0/5) |
| retopo_b | 2000 | 2058 | 1.00 | 38 / 34 (<= 24) | 0.160 (0.253) | 2.9 | none (all R0) | FAIL: irregular (0/5) |
| retopo_c | 2000 | 2052 | 1.00 | 50 / 37 (<= 24) | 0.201 (0.190) | 8.8 (4.3 alone) | none (all R0) | FAIL: edge_cv, irregular, time under load (0/5) |
| sculpt_cylinder | 2000 | 2047 | 1.00 | 32 / 24 (<= 16) | 0.173 (0.197) | 6.0 (3.0 alone) | none (all R0) | FAIL: irregular, loops 0.66, time under load (0/5) |
| basemesh_female | 5000 | 4977 | 1.00 | 136 / 140 (<= 90) | 0.426 (0.302) | 17.7 | none (R0, full path) | FAIL: edge_cv, hausdorff 1.04, time, irregular (0/5) |
| noisy_blob | 5000 | 4974 | 1.00 | 14 / 14 (<= 14) | 0.113 (0.221) | 4.9 | none (all R0) | PASS (5/5) |
| sculpt_sphere_mirror | 5000 | 5020 | 1.00 | - / 92 (no gate) | 0.156 (-) | 3.7 | none (all R0) | PASS (5/5) |
| perf_noisy_blob_328k | 5000 | 5145 | 1.00 | 14 / 14 (no gate) | 0.121 (0.221) | 7.3 | none (all R0) | PASS (5/5) |

The other global gates hold on every mesh unless the last column lists them:

- Quad ratio is 100%.
- Count error is within 10% on every mesh and seed. The largest are retopo_a at +3.8% (seeds 1 to 4) and perf at +2.9%.
- Angle deviation is at or below QuadriFlow on every mesh.
- Vertex distance is at or below 0.3 h, except retopo_a at 0.41 h.
- Hausdorff is at or below 1 h, except retopo_a (1.06 h) and basemesh_female (1.04 h).
- There are 0 crashes and every status is 0. Every component runs the full path, except the 4 retopo_a debris pieces.

Irregular counts do not depend on the seed, except retopo_c (36 on seed 2).

**Timing.** The batch ran while other programs loaded the machine, so retopo_b (seeds 1 to 4), retopo_c, retopo_a and sculpt_cylinder took up to 2x their standalone times. The stages this round did not touch (proxy, field) slowed down by the same factor.
- Rerun alone, every small input except basemesh_female is under 5 s: retopo_c 4.3 s, retopo_a 4.3 s, sculpt_cylinder 3.0 s.
- The three meshes that show a time failure also fail other gates, so the load does not change any verdict.

## What changed this round

Details are in CHANGES.md, "WP-16 round 3":

1. **Foldover-free untangler in the S7 repair (aq_param).**
   - The cause of the basemesh_female failure: at the crotch saddle, -1 cones were left wound 90 degrees instead of 450, and stars were mirrored. S8 then failed with "cone has 1 axis rays instead of 5".
   - Fold regions that Floater and the kernel sweeps cannot fix now go to a Garanzha et al. 2021 untangler (L-BFGS on the regularized energy). Any wrong-wound cone link is reset between attempts.
   - Every accepted region must also pass a winding check: the uv angle sum at each interior vertex must equal 2 pi - index4 pi/2.
   - This is more of the M2 WP-24 item, pulled forward.
2. **Count correction on full-path components (DESIGN 7.1 step 8, S9 to S11).** A successful rung keeps its proxy and layout. When the total count is off by more than 8%, the components that hit their own target are quantized again with a scaled target. On retopo_a this took the count from +10.5% to -0.1%.
3. **The two loop-guard solves in S7 run in parallel.** They are independent and give the same result. basemesh_female S7 dropped from 7 to 13 s to about 5 s, which also brings the aq_ffi testdata test back under its 30 s budget.
4. **Diagnostics:** `param.bad_cones`, and partial timings in failed-rung traces.
5. **Tried and removed:**
   - Cancelling the bad cones after S7 and re-running S5 to S7.
   - A larger repair growth (6).

   Neither helped basemesh_female.

## What blocks each failing mesh

- **torus: edge CV 0.135 against 0.107 (QuadriFlow).** This is a **gate conflict**, unchanged.
  - Ours is a pole-free grid: 0 irregular vertices and 110 of 110 loops closed.
  - On a pole-free grid of this torus (R/r = 4), the varying radius R + r cos(theta) alone sets a CV floor of about 0.124 (best grid 22 x 91). A conformal grid gives 0.18.
  - QuadriFlow gets 0.107 by spending 14 poles. "irregular = 0" and "edge CV <= QF" cannot both hold.
- **retopo_a.** Its main body is clean (see the render), but these gates still fail:
  - Four debris pieces (2 to 153 triangles, below the 8-quad floor) go to SF by DESIGN 7.1 step 3, and no_sf counts them. This is a **gate conflict**. The QuadriFlow baseline drops the glasses component altogether.
  - The glasses frames are thinner than h. They mesh on the full path but badly, and they drive edge CV (0.60), vertex distance (0.41 h), Hausdorff (1.06 h) and part of the irregular count.
  - This needs the M2 thickness cap, or a DESIGN rule for parts thinner than h.
- **retopo_b: 34 irregular against 24** (QuadriFlow 38).
  - 13 of the 34 are boundary vertices (valence 5 x10, valence 4 x3) at notches of a jagged hem, where the boundary turns by 4 to 37 degrees. Quad meshing forces them there.
  - The 21 interior ones are strand ends and spike tips.
  - Reaching 24 needs the M2 pole cap, or a gate that does not count geometry-forced boundary vertices.
- **retopo_c: 37 irregular against 24** (QuadriFlow 50), and **edge CV 0.2005 against 0.190**.
  - The short edges are on the thin strands: the 5% edge length quantile is 0.63 against QuadriFlow's 0.74.
  - It needs the M2 thickness cap and pole cap.
- **sculpt_cylinder: 24 irregular against 16, and loops 0.66 against 0.9.** The cones cluster where the strands join the cap. A band of stretched, zigzag quads runs down the middle strand (see the render), and it breaks the circumferential rings.
- **basemesh_female.** It is now on the full path with no fallback, but four gates still fail:
  - Irregular is 140 against a gate of 90 (QuadriFlow 136). The field keeps 140 cones (74 of index +1, 66 of index -1), and every one becomes an irregular vertex. Meeting 90 needs the cone count cut by a third in S4.
  - Edge CV is 0.43. It probably comes mostly from the tube cap: h shrinks on the fingers and toes, which get much smaller quads (see the render).
  - Hausdorff is 1.04 h, from one cluster of 14 input vertices on the front of the face (y about 9.8 of 11.6; `examples/hd_locate`). All other clusters are below 0.82 h.
  - The time is 17 s. The refined proxy (122k triangles) alone takes 7 to 9 s. The suite applies the 5 s small-input gate to every input under 50k triangles, but this is a 5000-quad target, not a 2000-quad "small input" in the DESIGN 7.4 sense. A DESIGN ruling is needed.

## Open items for the next round

1. **Gate conflicts that need a DESIGN ruling:**
   - the torus edge CV against 0 poles;
   - the retopo_a debris against no_sf;
   - the geometry-forced boundary irregular vertices on retopo_b and retopo_c;
   - the 5 s gate on the 5000-quad basemesh_female.
2. **Pole count.** retopo_b, retopo_c, sculpt_cylinder and basemesh_female need fewer cones out of S4: the M2 pole cap, or the M1 cancel_dist default.
3. **Thin parts below h.** retopo_a's glasses and the retopo_c strands need the M2 thickness cap.
4. **basemesh_female time.** The refined proxy build (7 to 9 s) is the largest stage now.

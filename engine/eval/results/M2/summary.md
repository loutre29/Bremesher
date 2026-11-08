# AutoQuad batch summary (gates M1)

Suite `eval/suite.toml`, engine path M1, seeds [0, 1, 2, 3, 4]. A mesh passes when its gates hold for seed 0 and for at least 4 of the seeds run.

| mesh | status | faces | count err | irregular (QF) | edge CV (QF) | angle dev (QF) | vertex dist / h | Hausdorff / h | loops closed | time s | components | seeds passed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 0 | 2001 | 0.1% | 0 (14) | 0.135 (0.107) | 1.34 (6.92) | 0.018 | 0.068 | 110 / 110 (1.00) | 2.93 | 1 full | 0 of 5 | FAIL |
| double_torus | 0 | 2041 | 2.1% | 8 (36) | 0.123 (0.148) | 3.28 (9.19) | 0.049 | 0.283 | - | 3.42 | 1 full | 5 of 5 | PASS |
| cube_sharp | 0 | 1944 | -2.8% | 8 (8) | 0.000 (0.005) | 0.00 (0.21) | 0.000 | 0.000 | - | 2.19 | 1 full | 5 of 5 | PASS |
| cylinder_open | 0 | 2015 | 0.8% | 0 (0) | 0.001 (0.014) | 0.02 (0.04) | 0.008 | 0.024 | 32 / 32 (1.00) | 2.77 | 1 full | 5 of 5 | PASS |
| suzanne_sub2 | 0 | 1982 | -0.9% | 36 (86) | 0.172 (0.307) | 5.62 (11.49) | 0.108 | 0.637 | - | 3.36 | 3 full | 5 of 5 | PASS |
| retopo_a | 0 | 1983 | -0.9% | 48 (36) | 0.380 (0.215) | 10.04 (11.82) | 0.091 | 0.769 | - | 4.75 | 6 full, 3 debris_sf | 0 of 5 | FAIL |
| retopo_b | 0 | 1944 | -2.8% | 26 (38) | 0.172 (0.253) | 6.15 (9.71) | 0.101 | 0.611 | - | 3.61 | 1 full | 0 of 5 | FAIL |
| retopo_c | 0 | 1990 | -0.5% | 29 (50) | 0.214 (0.190) | 8.03 (8.64) | 0.095 | 0.596 | - | 5.53 | 1 full | 0 of 5 | FAIL |
| sculpt_cylinder | 0 | 2013 | 0.7% | 23 (32) | 0.188 (0.197) | 7.08 (8.38) | 0.061 | 0.790 | 28 / 41 (0.68) | 6.78 | 1 full | 0 of 5 | FAIL |
| basemesh_female | 0 | 5132 | 2.6% | 128 (136) | 0.394 (0.302) | 7.33 (16.12) | 0.240 | 0.770 | - | 24.86 | 1 full | 0 of 5 | FAIL |
| noisy_blob | 0 | 5033 | 0.7% | 10 (14) | 0.117 (0.221) | 7.62 (9.50) | 0.032 | 0.484 | - | 6.71 | 1 full | 5 of 5 | PASS |
| sculpt_sphere_mirror | 0 | 4824 | -3.5% | 84 (-) | 0.167 (-) | 7.25 (-) | 0.058 | 0.685 | - | 4.88 | 1 full | 5 of 5 | PASS |
| perf_noisy_blob_328k | 0 | 4958 | -0.8% | 10 (14) | 0.121 (0.221) | 7.78 (9.50) | 0.060 | 0.572 | - | 8.35 | 1 full | 5 of 5 | PASS |

## Gates (first seed)

- **torus**: fails edge_cv (0.1346 <= 0.1068). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **double_torus**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **cube_sharp**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular_exact, irregular_at_corners, feature_cover.
- **cylinder_open**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **suzanne_sub2**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **retopo_a**: fails no_sf (3 components on SF or failed); edge_cv (0.3797 <= 0.2000); irregular (48 <= 24). Holds: success, quad_ratio, count_err, angle_dev, vertex_dist, hausdorff, time.
- **retopo_b**: fails irregular (26 <= 24). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.
- **retopo_c**: fails edge_cv (0.2141 <= 0.1900); time (5.53 s < 5 s); irregular (29 <= 24). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff.
- **sculpt_cylinder**: fails time (6.78 s < 5 s); irregular (23 <= 16); loops (closed 0.683 >= 0.9). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff.
- **basemesh_female**: fails edge_cv (0.3944 <= 0.2000); time (24.86 s < 5 s); irregular (128 <= 90). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff.
- **noisy_blob**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, irregular.
- **sculpt_sphere_mirror**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, vertex_dist, hausdorff, time, open_edges, seam_poles.
- **perf_noisy_blob_328k**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.

**7 of 13 meshes pass the M1 gates.**

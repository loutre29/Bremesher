# AutoQuad batch summary (gates M1)

Suite `eval/suite.toml`, engine path M1, seeds [0, 1, 2, 3, 4]. A mesh passes when its gates hold for seed 0 and for at least 4 of the seeds run.

| mesh | status | faces | count err | irregular (QF) | edge CV (QF) | angle dev (QF) | vertex dist / h | Hausdorff / h | loops closed | time s | components | seeds passed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 0 | 2001 | 0.1% | 0 (14) | 0.135 (0.107) | 1.34 (6.92) | 0.018 | 0.068 | 110 / 110 (1.00) | 2.20 | 1 full | 0 of 5 | FAIL |
| double_torus | 0 | 2041 | 2.1% | 8 (36) | 0.123 (0.148) | 3.28 (9.19) | 0.049 | 0.283 | - | 2.91 | 1 full | 5 of 5 | PASS |
| cube_sharp | 0 | 1944 | -2.8% | 8 (8) | 0.000 (0.005) | 0.00 (0.21) | 0.000 | 0.000 | - | 1.97 | 1 full | 5 of 5 | PASS |
| cylinder_open | 0 | 2015 | 0.8% | 0 (0) | 0.001 (0.014) | 0.02 (0.04) | 0.008 | 0.024 | 32 / 32 (1.00) | 2.63 | 1 full | 5 of 5 | PASS |
| suzanne_sub2 | 0 | 1986 | -0.7% | 44 (86) | 0.149 (0.307) | 5.51 (11.49) | 0.109 | 0.597 | - | 3.03 | 3 full | 5 of 5 | PASS |
| retopo_a | 0 | 1997 | -0.1% | 48 (36) | 0.598 (0.215) | 11.47 (11.82) | 0.409 | 1.062 | - | 5.82 | 5 full, 4 debris_sf | 0 of 5 | FAIL |
| retopo_b | 0 | 2058 | 2.9% | 34 (38) | 0.160 (0.253) | 7.29 (9.71) | 0.045 | 0.603 | - | 2.88 | 1 full | 0 of 5 | FAIL |
| retopo_c | 0 | 2052 | 2.6% | 37 (50) | 0.201 (0.190) | 6.99 (8.64) | 0.117 | 0.539 | - | 8.82 | 1 full | 0 of 5 | FAIL |
| sculpt_cylinder | 0 | 2047 | 2.4% | 24 (32) | 0.173 (0.197) | 6.58 (8.38) | 0.045 | 0.871 | 21 / 32 (0.66) | 6.02 | 1 full | 0 of 5 | FAIL |
| basemesh_female | 0 | 4977 | -0.5% | 140 (136) | 0.426 (0.302) | 7.67 (16.12) | 0.256 | 1.037 | - | 17.69 | 1 full | 0 of 5 | FAIL |
| noisy_blob | 0 | 4974 | -0.5% | 14 (14) | 0.113 (0.221) | 7.86 (9.50) | 0.031 | 0.493 | - | 4.92 | 1 full | 5 of 5 | PASS |
| sculpt_sphere_mirror | 0 | 5020 | 0.4% | 92 (-) | 0.156 (-) | 6.77 (-) | 0.047 | 0.746 | - | 3.74 | 1 full | 5 of 5 | PASS |
| perf_noisy_blob_328k | 0 | 5145 | 2.9% | 14 (14) | 0.121 (0.221) | 7.55 (9.50) | 0.048 | 0.496 | - | 7.31 | 1 full | 5 of 5 | PASS |

## Gates (first seed)

- **torus**: fails edge_cv (0.1346 <= 0.1068). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **double_torus**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **cube_sharp**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular_exact, irregular_at_corners, feature_cover.
- **cylinder_open**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **suzanne_sub2**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **retopo_a**: fails no_sf (4 components on SF or failed); edge_cv (0.5983 <= 0.2000); vertex_dist (0.409 h <= 0.3 h); hausdorff (1.062 h <= 1 h); time (5.82 s < 5 s); irregular (48 <= 24). Holds: success, quad_ratio, count_err, angle_dev.
- **retopo_b**: fails irregular (34 <= 24). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.
- **retopo_c**: fails edge_cv (0.2005 <= 0.1900); time (8.82 s < 5 s); irregular (37 <= 24). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff.
- **sculpt_cylinder**: fails time (6.02 s < 5 s); irregular (24 <= 16); loops (closed 0.656 >= 0.9). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff.
- **basemesh_female**: fails edge_cv (0.4257 <= 0.2000); hausdorff (1.037 h <= 1 h); time (17.69 s < 5 s); irregular (140 <= 90). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist.
- **noisy_blob**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, irregular.
- **sculpt_sphere_mirror**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, vertex_dist, hausdorff, time, open_edges, seam_poles.
- **perf_noisy_blob_328k**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.

**7 of 13 meshes pass the M1 gates.**

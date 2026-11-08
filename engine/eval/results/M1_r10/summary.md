# AutoQuad batch summary (gates M1)

Suite `eval/suite.toml`, engine path M1, seeds [0]. A mesh passes when its gates hold for seed 0 and for at least 1 of the seeds run.

| mesh | status | faces | count err | irregular (QF) | edge CV (QF) | angle dev (QF) | vertex dist / h | Hausdorff / h | loops closed | time s | components | seeds passed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 0 | 2001 | 0.1% | 0 (14) | 0.135 (0.107) | 1.34 (6.92) | 0.018 | 0.068 | 110 / 110 (1.00) | 2.10 | 1 full | 0 of 1 | FAIL |
| double_torus | 0 | 2061 | 3.0% | 8 (36) | 0.122 (0.148) | 3.19 (9.19) | 0.030 | 0.284 | - | 2.42 | 1 full | 1 of 1 | PASS |
| cube_sharp | 0 | 1944 | -2.8% | 8 (8) | 0.000 (0.005) | 0.00 (0.21) | 0.000 | 0.000 | - | 1.61 | 1 full | 1 of 1 | PASS |
| cylinder_open | 0 | 2015 | 0.8% | 0 (0) | 0.001 (0.014) | 0.02 (0.04) | 0.008 | 0.024 | 32 / 32 (1.00) | 2.08 | 1 full | 1 of 1 | PASS |
| suzanne_sub2 | 0 | 1937 | -3.1% | 44 (86) | 0.142 (0.307) | 5.10 (11.49) | 0.120 | 1.279 | - | 2.68 | 3 full | 0 of 1 | FAIL |
| retopo_a | 0 | 2154 | 7.7% | 52 (36) | 0.550 (0.215) | 9.90 (11.82) | 0.717 | 1.393 | - | 2.68 | 5 full, 4 debris_sf | 0 of 1 | FAIL |
| retopo_b | 0 | 2029 | 1.5% | 34 (38) | 0.165 (0.253) | 7.61 (9.71) | 0.044 | 0.749 | - | 2.21 | 1 full | 0 of 1 | FAIL |
| retopo_c | 0 | 2004 | 0.2% | 38 (50) | 0.228 (0.190) | 9.90 (8.64) | 0.136 | 1.152 | - | 2.90 | 1 full | 0 of 1 | FAIL |
| sculpt_cylinder | 0 | 2035 | 1.8% | 24 (32) | 0.236 (0.197) | 6.93 (8.38) | 0.043 | 1.681 | 39 / 51 (0.76) | 2.87 | 1 full | 0 of 1 | FAIL |
| basemesh_female | 1 | 4914 | -1.7% | 836 (136) | 0.181 (0.302) | 14.80 (16.12) | 0.000 | 5.267 | - | 18.64 | 1 sf | 0 of 1 | FAIL |
| noisy_blob | 0 | 4898 | -2.0% | 24 (14) | 0.113 (0.221) | 7.70 (9.50) | 0.031 | 0.540 | - | 4.71 | 1 full | 0 of 1 | FAIL |
| sculpt_sphere_mirror | 0 | 4852 | -3.0% | 92 (-) | 0.194 (-) | 7.84 (-) | 0.065 | 1.232 | - | 3.82 | 1 full | 0 of 1 | FAIL |
| perf_noisy_blob_328k | 0 | 4986 | -0.3% | 24 (14) | 0.114 (0.221) | 7.82 (9.50) | 0.054 | 0.545 | - | 6.97 | 1 full | 1 of 1 | PASS |

## Gates (first seed)

- **torus**: fails edge_cv (0.1346 <= 0.1068). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **double_torus**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **cube_sharp**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular_exact, irregular_at_corners, feature_cover.
- **cylinder_open**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **suzanne_sub2**: fails hausdorff (1.279 h <= 1 h). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, time, irregular.
- **retopo_a**: fails no_sf (4 components on SF or failed); edge_cv (0.5504 <= 0.2000); vertex_dist (0.717 h <= 0.3 h); hausdorff (1.393 h <= 1 h); irregular (52 <= 24). Holds: success, quad_ratio, count_err, angle_dev, time.
- **retopo_b**: fails irregular (34 <= 24). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.
- **retopo_c**: fails edge_cv (0.2280 <= 0.1900); angle_dev (9.90 <= 8.64); hausdorff (1.152 h <= 1 h); irregular (38 <= 24). Holds: success, quad_ratio, no_sf, count_err, vertex_dist, time.
- **sculpt_cylinder**: fails edge_cv (0.2364 <= 0.1972); hausdorff (1.681 h <= 1 h); irregular (24 <= 16); loops (closed 0.765 >= 0.9). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, time.
- **basemesh_female**: fails no_sf (1 components on SF or failed); hausdorff (5.267 h <= 1 h); time (18.64 s < 5 s); irregular (836 <= 90). Holds: success, quad_ratio, count_err, edge_cv, angle_dev, vertex_dist.
- **noisy_blob**: fails irregular (24 <= 14). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff.
- **sculpt_sphere_mirror**: fails hausdorff (1.232 h <= 1 h). Holds: success, quad_ratio, no_sf, count_err, edge_cv, vertex_dist, time, open_edges, seam_poles.
- **perf_noisy_blob_328k**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.

**4 of 13 meshes pass the M1 gates.**

# AutoQuad batch summary (gates M1)

Suite `eval/suite.toml`, engine path M1, seeds [0]. A mesh passes when its gates hold for seed 0 and for at least 1 of the seeds run.

| mesh | status | faces | count err | irregular (QF) | edge CV (QF) | angle dev (QF) | vertex dist / h | Hausdorff / h | loops closed | time s | components | seeds passed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 0 | 2001 | 0.1% | 0 (14) | 0.135 (0.107) | 1.34 (6.92) | 0.018 | 0.068 | 110 / 110 (1.00) | 2.06 | 1 full | 0 of 1 | FAIL |
| double_torus | 0 | 2061 | 3.0% | 8 (36) | 0.122 (0.148) | 3.19 (9.19) | 0.030 | 0.284 | - | 2.41 | 1 full | 1 of 1 | PASS |
| cube_sharp | 0 | 1944 | -2.8% | 8 (8) | 0.000 (0.005) | 0.00 (0.21) | 0.000 | 0.000 | - | 1.60 | 1 full | 1 of 1 | PASS |
| cylinder_open | 0 | 2015 | 0.8% | 0 (0) | 0.001 (0.014) | 0.02 (0.04) | 0.008 | 0.024 | 32 / 32 (1.00) | 2.08 | 1 full | 1 of 1 | PASS |
| suzanne_sub2 | 0 | 1937 | -3.1% | 44 (86) | 0.142 (0.307) | 5.10 (11.49) | 0.120 | 1.279 | - | 2.60 | 3 full | 0 of 1 | FAIL |
| retopo_a | 1 | 2026 | 1.3% | 116 (36) | 0.280 (0.215) | 10.84 (11.82) | 0.332 | 5.401 | - | 2.38 | 3 full, 1 retried, 1 sf, 4 debris_sf | 0 of 1 | FAIL |
| retopo_b | 0 | 2029 | 1.5% | 34 (38) | 0.165 (0.253) | 7.61 (9.71) | 0.044 | 0.749 | - | 2.24 | 1 full | 0 of 1 | FAIL |
| retopo_c | 0 | 2004 | 0.2% | 38 (50) | 0.228 (0.190) | 9.90 (8.64) | 0.136 | 1.152 | - | 2.92 | 1 full | 0 of 1 | FAIL |
| sculpt_cylinder | 0 | 2035 | 1.8% | 24 (32) | 0.226 (0.197) | 6.94 (8.38) | 0.043 | 1.683 | 39 / 51 (0.76) | 2.87 | 1 full | 0 of 1 | FAIL |
| basemesh_female | 1 | 4914 | -1.7% | 836 (136) | 0.181 (0.302) | 14.80 (16.12) | 0.000 | 5.267 | - | 18.23 | 1 sf | 0 of 1 | FAIL |
| noisy_blob | 0 | 4898 | -2.0% | 24 (14) | 0.113 (0.221) | 7.70 (9.50) | 0.031 | 0.540 | - | 4.52 | 1 full | 0 of 1 | FAIL |
| sculpt_sphere_mirror | 1 | 5088 | 1.8% | 92 (-) | 0.158 (-) | 6.54 (-) | 0.048 | 0.780 | - | 8.29 | 1 retried | 0 of 1 | FAIL |
| perf_noisy_blob_328k | 0 | 4986 | -0.3% | 24 (14) | 0.114 (0.221) | 7.82 (9.50) | 0.054 | 0.545 | - | 7.04 | 1 full | 1 of 1 | PASS |

## Gates (first seed)

- **torus**: fails edge_cv (0.1346 <= 0.1068). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **double_torus**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **cube_sharp**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular_exact, irregular_at_corners, feature_cover.
- **cylinder_open**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **suzanne_sub2**: fails hausdorff (1.279 h <= 1 h). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, time, irregular.
- **retopo_a**: fails no_sf (5 components on SF or failed); edge_cv (0.2798 <= 0.2000); vertex_dist (0.332 h <= 0.3 h); hausdorff (5.401 h <= 1 h); irregular (116 <= 24). Holds: success, quad_ratio, count_err, angle_dev, time.
- **retopo_b**: fails irregular (34 <= 24). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.
- **retopo_c**: fails edge_cv (0.2280 <= 0.1900); angle_dev (9.90 <= 8.64); hausdorff (1.152 h <= 1 h); irregular (38 <= 24). Holds: success, quad_ratio, no_sf, count_err, vertex_dist, time.
- **sculpt_cylinder**: fails edge_cv (0.2255 <= 0.1972); hausdorff (1.683 h <= 1 h); irregular (24 <= 16); loops (closed 0.765 >= 0.9). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, time.
- **basemesh_female**: fails no_sf (1 components on SF or failed); hausdorff (5.267 h <= 1 h); time (18.23 s < 5 s); irregular (836 <= 90). Holds: success, quad_ratio, count_err, edge_cv, angle_dev, vertex_dist.
- **noisy_blob**: fails irregular (24 <= 14). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff.
- **sculpt_sphere_mirror**: fails time (8.29 s < 5 s). Holds: success, quad_ratio, no_sf, count_err, edge_cv, vertex_dist, hausdorff, open_edges, seam_poles.
- **perf_noisy_blob_328k**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time.

**4 of 13 meshes pass the M1 gates.**

# AutoQuad batch summary (gates M1)

Suite `eval/suite.toml`, engine path M1, seeds [0]. A mesh passes when its gates hold for seed 0 and for at least 1 of the seeds run.

| mesh | status | faces | count err | irregular (QF) | edge CV (QF) | angle dev (QF) | vertex dist / h | Hausdorff / h | loops closed | time s | components | seeds passed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torus | 0 | 2001 | 0.1% | 0 (14) | 0.135 (0.107) | 1.34 (6.92) | 0.018 | 0.068 | 110 / 110 (1.00) | 2.05 | 1 full | 0 of 1 | FAIL |
| double_torus | 0 | 2061 | 3.0% | 8 (36) | 0.122 (0.148) | 3.19 (9.19) | 0.030 | 0.284 | - | 2.37 | 1 full | 1 of 1 | PASS |
| cube_sharp | 0 | 1944 | -2.8% | 8 (8) | 0.000 (0.005) | 0.00 (0.21) | 0.000 | 0.000 | - | 1.64 | 1 full | 1 of 1 | PASS |
| cylinder_open | 0 | 2015 | 0.8% | 0 (0) | 0.001 (0.014) | 0.02 (0.04) | 0.008 | 0.024 | 32 / 32 (1.00) | 2.02 | 1 full | 1 of 1 | PASS |
| suzanne_sub2 | 1 | 1967 | -1.7% | 317 (86) | 0.170 (0.307) | 14.77 (11.49) | 0.008 | 2.595 | - | 5.24 | 2 full, 1 sf | 0 of 1 | FAIL |
| retopo_a | 1 | 1998 | -0.1% | 351 (36) | 0.271 (0.215) | 18.95 (11.82) | 0.000 | 5.401 | - | 4.59 | 5 sf, 4 debris_sf | 0 of 1 | FAIL |
| retopo_b | 1 | 2053 | 2.6% | 350 (38) | 0.172 (0.253) | 15.00 (9.71) | 0.000 | 0.735 | - | 4.67 | 1 sf | 0 of 1 | FAIL |
| retopo_c | 1 | 2019 | 0.9% | 313 (50) | 0.167 (0.190) | 15.25 (8.64) | 0.000 | 3.842 | - | 5.77 | 1 sf | 0 of 1 | FAIL |
| sculpt_cylinder | 1 | 1988 | -0.6% | 352 (32) | 0.186 (0.197) | 15.35 (8.38) | 0.000 | 2.604 | 0 / 123 (0.00) | 5.82 | 1 sf | 0 of 1 | FAIL |
| basemesh_female | 1 | 4914 | -1.7% | 836 (136) | 0.181 (0.302) | 14.80 (16.12) | 0.000 | 5.267 | - | 18.42 | 1 sf | 0 of 1 | FAIL |
| noisy_blob | 0 | 4898 | -2.0% | 24 (14) | 0.113 (0.221) | 7.70 (9.50) | 0.031 | 0.540 | - | 4.61 | 1 full | 0 of 1 | FAIL |
| sculpt_sphere_mirror | 1 | 5128 | 2.6% | 890 (-) | 0.154 (-) | 14.53 (-) | 0.000 | 2.925 | - | 8.44 | 1 sf | 0 of 1 | FAIL |
| perf_noisy_blob_328k | 1 | 5106 | 2.1% | 890 (14) | 0.148 (0.221) | 12.89 (9.50) | 0.000 | 0.561 | - | 14.61 | 1 sf | 0 of 1 | FAIL |

## Gates (first seed)

- **torus**: fails edge_cv (0.1346 <= 0.1068). Holds: success, quad_ratio, no_sf, count_err, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **double_torus**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular.
- **cube_sharp**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular_exact, irregular_at_corners, feature_cover.
- **cylinder_open**: all gates hold. Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff, time, irregular, loops.
- **suzanne_sub2**: fails no_sf (1 components on SF or failed); angle_dev (14.77 <= 11.49); hausdorff (2.595 h <= 1 h); time (5.24 s < 5 s); irregular (317 <= 50). Holds: success, quad_ratio, count_err, edge_cv, vertex_dist.
- **retopo_a**: fails no_sf (9 components on SF or failed); edge_cv (0.2705 <= 0.2000); angle_dev (18.95 <= 11.82); hausdorff (5.401 h <= 1 h); irregular (351 <= 24). Holds: success, quad_ratio, count_err, vertex_dist, time.
- **retopo_b**: fails no_sf (1 components on SF or failed); angle_dev (15.00 <= 9.71); irregular (350 <= 24). Holds: success, quad_ratio, count_err, edge_cv, vertex_dist, hausdorff, time.
- **retopo_c**: fails no_sf (1 components on SF or failed); angle_dev (15.25 <= 8.64); hausdorff (3.842 h <= 1 h); time (5.77 s < 5 s); irregular (313 <= 24). Holds: success, quad_ratio, count_err, edge_cv, vertex_dist.
- **sculpt_cylinder**: fails no_sf (1 components on SF or failed); angle_dev (15.35 <= 8.38); hausdorff (2.604 h <= 1 h); time (5.82 s < 5 s); irregular (352 <= 16); loops (closed 0.000 >= 0.9). Holds: success, quad_ratio, count_err, edge_cv, vertex_dist.
- **basemesh_female**: fails no_sf (1 components on SF or failed); hausdorff (5.267 h <= 1 h); time (18.42 s < 5 s); irregular (836 <= 90). Holds: success, quad_ratio, count_err, edge_cv, angle_dev, vertex_dist.
- **noisy_blob**: fails irregular (24 <= 14). Holds: success, quad_ratio, no_sf, count_err, edge_cv, angle_dev, vertex_dist, hausdorff.
- **sculpt_sphere_mirror**: fails no_sf (1 components on SF or failed); hausdorff (2.925 h <= 1 h); time (8.44 s < 5 s). Holds: success, quad_ratio, count_err, edge_cv, vertex_dist, open_edges, seam_poles.
- **perf_noisy_blob_328k**: fails no_sf (1 components on SF or failed); angle_dev (12.89 <= 9.50). Holds: success, quad_ratio, count_err, edge_cv, vertex_dist, hausdorff, time.

**3 of 13 meshes pass the M1 gates.**

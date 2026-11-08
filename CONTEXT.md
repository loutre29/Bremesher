# BRemesher (ex AutoQuad Remesher): where we are (2026-11-08)

## Name and add-on
- 2026-10-30: the user liked the custom engine ("globally really good") and asked to **rename the add-on** (Bremesher, then BRemesher on 2026-11-08) and
  **drop QuadriFlow**. Done:
  - add-on source is now `bremesher/` (display name **BRemesher**, id `bremesher`, v2.1.1, operator `bremesher.remesh` labelled "BRemesh",
    settings `scene.bremesher`, result object `<name>_BRemesher`, attributes `brm_density` / `brm_guide`);
  - the DLL ships as `bin/bremesher_engine.dll` (aq_ffi still builds `autoquad_engine.dll`; copy and rename it into `bremesher/bin/`);
  - QuadriFlow is gone from the add-on (no engine choice, no whole-mesh or per-component fallback, no retries).
    The engine's own SF quadder is the only fallback; an engine failure cancels cleanly with an error.
  - Smoke suite 148/148 and sandbox install test pass. Zips: `bremesher-2.1.1.zip`, `bremesher-2.2.0.zip` (M2 engine, installed).
  - INSTALLED in real 5.2 (`bl_ext.user_default.bremesher`); `autoquad_remesher` was removed from the profile.
- Old zips `autoquad_remesher-1.0.0.zip` (QuadriFlow wrapper) and `-2.0.0.zip` are kept as archives only.
- The project folder and the engine crates (`aq_*`, `autoquad.exe`) keep their old names on purpose.

## GitHub
- Private repo https://github.com/loutre29/Bremesher (branch main, created 2026-11-08). Release v2.2.0 carries the zip.
- `.gitignore` leaves out engine/target, engine/.bl_sandbox, _wip, eval/scratch, eval result OBJs and *.zip
  (zips go up as releases: `gh release create v<ver> bremesher-<ver>.zip`). Commit only when the user asks.

## Engine
Custom Rust engine in `engine/` (spec `engine/DESIGN.md`): proxy remesh, quad-scale curvature, 4-RoSy field with pole
cancellation, closed loops, seamless param, exact motorcycle layout, Bi-MDF quantization, per-patch grids, relax,
fallback ladder. cdylib (ctypes from the add-on) plus CLI `autoquad.exe` (batch eval: see `engine/eval/M1_REPORT.md`).

### M1 status (build workflow `wf_09a1dc90-f34`, finished)
- 7 of 13 suite entries pass every gate on 5/5 seeds. Details `engine/eval/M1_REPORT.md`, renders `engine/renders/M1/`
  (top row = ours, bottom = QuadriFlow).
- Irregular verts ours vs QF: torus 0/14, double_torus 8/36, suzanne 44/86, retopo_a/b/c 48/34/37 vs 36/38/50,
  sculpt_cylinder 24/32, basemesh_female 140/136, noisy_blob 14/14. 100% quads everywhere, count within 10%.

### M2 integrated (2026-11-08)
- Thin parts (WP-20) and poles (WP-23) built in `_wip/thin` and `_wip/poles`, merged into `engine/` (pre-merge backup
  `_wip/m1_backup/`), then an integration pass fixed their interactions (boundary merge angle check, far-pair size
  ratio, targeted keep-out retry). Report `engine/eval/M2_REPORT.md`, renders `engine/renders/M2/` (ours / M1 / QF).
- Same 7 of 13 pass as M1, nothing worse. Irregular M1 to M2: suzanne 44 to 36, retopo_b 34 to 26, retopo_c 37 to 29,
  female 140 to 128, noisy_blob 14 to 10, sphere 92 to 84, sculpt_cylinder 24 to 23. retopo_a glasses frame now kept
  (Hausdorff/vdist pass). Open: female 24.9 s (M1 17.7), retopo_a seed 1 gives 55, time gates on retopo_c/sculpt_cylinder.
- BRemesher 2.2.0 built with this DLL (`bremesher-2.2.0.zip`): smoke 148/148, sandbox install 12/12.
  INSTALLED in real 5.2 on 2026-11-08 (replaces 2.1.1).
- The 2.1.1 DLL is byte-identical to AutoQuad 2.0.0. The user said 2.1.1 "looks worse"; likely the settings reset by the
  rename and no more silent QuadriFlow takeover. Still waiting for the user's model and report line.

## Tooling (engine/tools)
- `blender.sh`: headless Blender 5.2 with a sandboxed profile (`engine/.bl_sandbox`). Always use it for scripted runs.
- `render_wire.py`: wireframe PNG, 2 views per mesh, poles marked (red = valence 3, blue = 5, magenta = 6+).
- `quad_stats.py`, `quadriflow_baseline.py` (baselines in `engine/baselines/`), `export_testdata.py`.
- Add-on tests: `engine/tools/blender.sh --python-exit-code 1 --python bremesher/tests/smoke_blender.py` and
  `.../install_sandbox.py -- <zip>`. Build: `engine/tools/blender.sh --command extension build --source-dir bremesher --output-dir .`

## QuadriFlow baseline (the bar to beat)
| mesh | target | faces | irregular verts | edge len CV |
| --- | --- | --- | --- | --- |
| torus | 2000 | 2153 | 14 (ideal 0) | 0.107 |
| double_torus | 2000 | 2168 | 36 | 0.148 |
| suzanne_sub2 | 2000 | 1743 | 86 | 0.307 |
| retopo_a / b / c | 2000 | 1717 / 2065 / 1911 | 36 / 38 / 50 | 0.22 / 0.25 / 0.19 |
| sculpt_cylinder | 2000 | 1868 | 32 | 0.197 |
| basemesh_female | 5000 | 3839 | 136 | 0.302 |
| noisy_blob | 5000 | 4188 | 14 | 0.221 |
| sculpt_sphere_mirror | 5000 | QuadriFlow failed | | |

## Rules
- Test on Blender 5.2 only. When the user says install, install right away
  (`blender.exe --command extension install-file -r user_default --enable bremesher-<ver>.zip`, real profile, no --factory-startup).
- Never call `read_factory_settings` in scripts. On 2026-10-29 that uninstalled the extension wheels in
  `%APPDATA%\Blender Foundation\Blender\5.2\extensions\.local`. A normal Blender start repaired it.
- No em dashes in any file.

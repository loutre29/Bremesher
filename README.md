# BRemesher

One click, ZRemesher style quad retopology for Blender 4.2+ (tested on 5.2, Windows x64).
It runs its own engine (`bin/bremesher_engine.dll`, built from `engine/`): edge loops that follow the form,
closed rings around limbs, few well placed poles.

## Install

Download `bremesher-2.2.0.zip` from the Releases page and drag it onto the Blender window, or
Edit > Preferences > Get Extensions > (v menu, top right) > Install from Disk.

The panel is in the 3D Viewport sidebar (N), tab **Remesh**.

## Use

1. Select a mesh (Object, Sculpt or Edit mode).
2. Set **Target Quads** (in thousands, like ZBrush). Half / Same / Double set it from the current mesh.
3. Press **BRemesh**. Esc cancels while it runs.

A new object `<name>_BRemesher` is created next to the source. The source is never modified (it is only hidden).
If you started in Sculpt mode you land back in Sculpt mode on the new mesh.

| Option | What it does |
| --- | --- |
| Adaptive Size / Curve Strength / Flow Smoothness | Quad size variation, how strongly loops follow curvature, and the scale curvature is measured at. |
| Symmetry X / Y / Z | Exact mirror symmetry in local space (the half is remeshed, mirrored and welded). Keep Side picks which half is used. |
| Keep Sharp Edges | Hard surface: aligns loops to hard edges and keeps them crisp. |
| Guides | Edges marked Seam (or the edge attribute `brm_guide`) guide the flow. |
| Use Density | Vertex attribute `brm_density` (multiplier) or the active color attribute sets local density. |
| Watertight Pre-pass | Voxel remesh of a temporary copy so the engine gets a clean closed surface. Auto does it only when needed (closed meshes only). Always is best for scans, Dyntopo and booleans. |
| Smooth Flow / Strength | Relax passes that even out the quads. |
| Project to Source | Snaps every new vertex back onto the original surface. |
| Advanced > Seed | Change it for a different layout. |
| Advanced > Time Limit | Engine time budget; parts still unfinished get the engine's simple quadder. |
| Advanced > Max Solver Input | Dense sculpts are reduced to this many faces on the temporary copy, for speed. |

## Limits

- No Keep Groups (polygroup borders).
- The quad count lands near the target, not exactly on it (same as ZRemesher).
- UVs, vertex colors and other attributes are not carried over (it is a new topology).
- Windows x64 only (the engine DLL).

# SPDX-License-Identifier: GPL-3.0-or-later
#
# BRemesher: one click, ZRemesher style quad retopology for Blender.
#
# Pipeline (all on a copy, the source object is never modified):
#   1. take the evaluated source mesh (modifiers, shape keys applied)
#   2. clean it: merge doubles, drop loose geometry, unify normals
#   3. optional voxel pre pass to make it watertight (auto when needed)
#   4. optional decimation of very dense sculpts so the solver stays fast
#   5. bisect for exact symmetry, keep one half
#   6. quad solver: the BRemesher engine (bremesher_engine.dll, field aligned seamless
#      parametrization with closed loops and few poles), run in a worker thread
#   7. relax the edge flow and project every vertex back onto the source
#   8. mirror back, weld the seam, shade, hand over the result

bl_info = {
    "name": "BRemesher",
    "author": "Vincent",
    "version": (2, 2, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Remesh",
    "description": "One click ZRemesher style quad retopology",
    "category": "Mesh",
}

import math
import time
from types import SimpleNamespace

import bmesh
import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import engine

# The temporary copy is scaled so its bounding box diagonal is this long, so every tolerance
# is relative. The engine expects the same normalization (DESIGN 8.1: bbox diagonal 10).
WORK_DIAG = 10.0
MERGE_DIST = 2e-4
AXES = ("X", "Y", "Z")
SHARP_DEG = 40.0
# Offset of the voxel grid from the symmetry planes, in voxels (an irrational-looking
# fraction, so no grid plane falls on a symmetry plane).
VOXEL_SHIFT = 0.3719

_engine_tried = False


def _engine_cache_dir():
    """Folder of the shadow copy of the DLL (see engine._shadow_copy). Only for an installed
    extension; a source checkout loads bin/bremesher_engine.dll directly. It is in the user
    datafiles, outside the extension's own user folder: Blender deletes that folder on
    uninstall and cannot while it holds a loaded DLL (the next load removes stale copies)."""
    pkg = __package__ or ""
    if not pkg.startswith("bl_ext."):
        return None
    try:
        return bpy.utils.user_resource("DATAFILES", path="bremesher_engine_cache", create=True)
    except Exception:
        return None


def ensure_engine():
    """Load the DLL on first use (not at registration, so installing never locks it). A failed
    load is retried on every run, so a DLL that was briefly locked or missing recovers without
    re-enabling the add-on."""
    global _engine_tried
    if engine.available():
        return True
    _engine_tried = True
    engine.load(cache_dir=_engine_cache_dir())
    return engine.available()


class UserError(Exception):
    """A failure with a message meant for the user (no traceback)."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class BRM_Settings(bpy.types.PropertyGroup):
    target_k: FloatProperty(
        name="Target Quads",
        description="Wanted number of quads, in thousands (like ZRemesher's Target Polygons Count)",
        default=5.0, min=0.05, soft_max=100.0, max=1000.0, precision=2, step=10,
    )
    sym_x: BoolProperty(name="X", description="Exact mirror symmetry across local X", default=False)
    sym_y: BoolProperty(name="Y", description="Exact mirror symmetry across local Y", default=False)
    sym_z: BoolProperty(name="Z", description="Exact mirror symmetry across local Z", default=False)
    sym_side: EnumProperty(
        name="Keep Side",
        description="Which half is remeshed and mirrored onto the other",
        items=[("POS", "+ Side", "Keep the positive half"), ("NEG", "- Side", "Keep the negative half")],
        default="POS",
    )
    keep_sharp: BoolProperty(
        name="Keep Sharp Edges",
        description="Align edge loops to hard edges (dihedral angle above 40 degrees) and keep them crisp",
        default=False,
    )
    adaptive_size: FloatProperty(
        name="Adaptive Size",
        description="Smaller quads where the surface is curved, larger where it is flat "
                    "(accepted by the engine, used from its M2 milestone on)",
        default=0.5, min=0.0, max=1.0, subtype="FACTOR",
    )
    curve_strength: FloatProperty(
        name="Curve Strength",
        description="How strongly the edge loops follow the curvature of the surface",
        default=1.0, min=0.0, max=4.0,
    )
    flow_smoothness: FloatProperty(
        name="Flow Smoothness",
        description="Scale at which curvature is measured: larger values ignore small details "
                    "and give smoother flow",
        default=1.0, min=0.5, max=3.0,
    )
    use_density: BoolProperty(
        name="Use Density",
        description="Local quad density from the vertex attribute 'brm_density' (multiplier, 2 = twice "
                    "the quads) or else the active color attribute (white = 4x denser, black = 4x "
                    "sparser, mid grey = unchanged). Used from the engine's M2 milestone on",
        default=False,
    )
    density_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=2.0)
    use_guides: BoolProperty(
        name="Guides",
        description="Edges marked as Seam, or with the edge attribute 'brm_guide', guide the edge flow",
        default=False,
    )
    guide_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=4.0)
    cleanup: EnumProperty(
        name="Watertight Pre-pass",
        description="Voxel remesh a temporary copy first so the quad solver gets a clean, closed surface",
        items=[
            ("AUTO", "Auto", "Closed meshes: only when non-manifold. "
                             "Open meshes: never, broken edges are split instead"),
            ("ALWAYS", "Always", "Best for scans, Dyntopo and booleans. Closes small holes, "
                                 "so keep it off for open surfaces like planes or cloth"),
            ("NEVER", "Never", "Use the mesh as is"),
        ],
        default="AUTO",
    )
    relax_iterations: IntProperty(
        name="Smooth Flow",
        description="Relax passes that even out the quads (each pass is reprojected onto the source)",
        default=4, min=0, max=30,
    )
    relax_strength: FloatProperty(
        name="Strength", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
    )
    reproject: BoolProperty(
        name="Project to Source",
        description="Snap the new vertices onto the original surface so the shape is kept",
        default=True,
    )
    max_work_k: IntProperty(
        name="Max Solver Input (k)",
        description="Very dense inputs are decimated to this many faces (thousands) on the temporary copy, "
                    "for speed. The result is still projected onto the full resolution source",
        default=300, min=20, max=5000,
    )
    seed: IntProperty(name="Seed", description="Change it to get a different layout", default=0, min=0)
    time_limit: FloatProperty(
        name="Time Limit (s)",
        description="Engine time budget. Parts still unfinished then get the engine's "
                    "simple quadder",
        default=60.0, min=5.0, max=3600.0,
    )
    hide_source: BoolProperty(name="Hide Source", default=True)
    shade_smooth: BoolProperty(name="Shade Smooth", default=True)
    debug_flags: IntProperty(
        name="Engine Debug Flags", description="AqParams.debug_flags (tests only)",
        default=0, min=0, options={"HIDDEN"},
    )


def snapshot(s):
    """Plain copy of the settings, so a running job does not see later UI edits."""
    return SimpleNamespace(**{k: getattr(s, k) for k in s.bl_rna.properties.keys() if k != "rna_type"})


# --------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------

def mesh_area(me):
    a = np.empty(len(me.polygons), np.float64)
    me.polygons.foreach_get("area", a)
    return float(a.sum())


def bvh_from_mesh(me):
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loops)
    starts = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", starts)
    sizes = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", sizes)
    polys = [loops[s:s + n].tolist() for s, n in zip(starts.tolist(), sizes.tolist())]
    return BVHTree.FromPolygons(co.reshape(-1, 3).tolist(), polys)


def clean_mesh(me, weld=True):
    """Lossless-ish cleanup of the temporary copy so the engine gets a clean input."""
    bm = bmesh.new()
    bm.from_mesh(me)
    # Only exact duplicates are welded (split UV seams of imported meshes). Micro edges are
    # first relaxed apart, which keeps the topology (merging them can pinch touching
    # surfaces into non-manifold edges); collapsing them is the last resort.
    if weld:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=MERGE_DIST * 0.01)
    for _ in range(30):
        short = [e for e in bm.edges if e.calc_length() < MERGE_DIST]
        if not short:
            break
        verts = {v for e in short for v in e.verts}
        verts |= {n for v in list(verts) for le in v.link_edges for n in le.verts}
        bmesh.ops.smooth_vert(bm, verts=list(verts), factor=0.5,
                              use_axis_x=True, use_axis_y=True, use_axis_z=True)
    else:
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=MERGE_DIST)
    loose_e = [e for e in bm.edges if not e.link_faces]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
    loose_v = [v for v in bm.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()


def has_boundary(me):
    le = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", le)
    return bool((np.bincount(le, minlength=len(me.edges)) == 1).any())


def is_solver_friendly(me):
    """Manifold consistency check: no edge used by 3+ faces, no flipped neighbours, no wire
    edges, no edge shorter than 1e-4 on every axis."""
    cv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", cv)
    ce = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("edge_index", ce)
    cnt = np.bincount(ce, minlength=len(me.edges))
    if (cnt > 2).any() or (cnt == 0).any():
        return False
    order = np.argsort(ce, kind="stable")
    se, sv = ce[order], cv[order]
    pair = np.nonzero(se[1:] == se[:-1])[0]
    if (sv[pair] == sv[pair + 1]).any():
        return False
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    return not (np.abs(co[ed[:, 0]] - co[ed[:, 1]]).max(1) <= 1e-4).any()


def set_active(context, obj):
    for o in context.view_layer.objects:
        if o is not None and o.select_get():  # a just-removed object can linger as None
            o.select_set(False)
    context.view_layer.objects.active = obj
    obj.select_set(True)


def apply_modifier(obj, mod):
    bpy.ops.object.modifier_apply(modifier=mod.name)


def bisect_half(me, axes, keep_negative):
    bm = bmesh.new()
    bm.from_mesh(me)
    for a in axes:
        n = Vector((0.0, 0.0, 0.0))
        n[a] = 1.0
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(
            bm, geom=geom, dist=1e-7, plane_co=(0.0, 0.0, 0.0), plane_no=n,
            clear_outer=keep_negative, clear_inner=not keep_negative,
        )
    bm.to_mesh(me)
    bm.free()
    me.update()


def quad_ratio(me):
    if not me.polygons:
        return 0.0
    sizes = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", sizes)
    return float((sizes == 4).mean())


def snap_seam(me, axes, tol, force=None, cand=None):
    """Put seam verts exactly on their symmetry plane.

    - `force` (engine output, one int per engine vertex, 0 for most): the Symmetry chain verts
      (vflag bit 26 and kind 5, DESIGN 8.3), as a bit mask of their planes (bit a for axis a,
      taken from the input seam edge they lie along, see `seam_planes`). They go onto those
      planes whatever their distance (the engine can return them up to about h/2 off); the
      other selected planes then use the `tol` rule, for the verts where two planes cross. A
      mask with no selected plane in it falls back to the nearest selected plane.
    - `cand` (same length): further engine verts that use the `tol` rule.
    - Verts past the end of the masks, or every vert without masks: the
      border verts, with the v1 `tol` rule (|co[a]| < tol on a plane a)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    n_mask = 0 if force is None else len(force)
    for v in bm.verts:
        i = v.index
        co = v.co
        if i < n_mask:
            if force[i]:
                mine = [a for a in axes if int(force[i]) >> a & 1]
                if not mine:
                    mine = [min(axes, key=lambda a: abs(co[a]))]
                for a in axes:
                    if a in mine or abs(co[a]) < tol:
                        co[a] = 0.0
                continue
            near_ok = cand is not None and bool(cand[i])
        else:
            near_ok = v.is_boundary
        if near_ok:
            for a in axes:
                if abs(co[a]) < tol:
                    co[a] = 0.0
    bm.to_mesh(me)
    bm.free()


def seam_samples(pos, feat, kind, axes, step):
    """Points along the input Symmetry edges (kind 5) with the bit mask of the planes each edge
    lies on, every `step` at most. `snap_seam` uses them to tell which plane an engine seam
    vertex belongs to (vflag gives the chain kind, not the plane)."""
    sel = np.asarray(kind) == engine.KIND_SYMMETRY
    if not axes or not sel.any():
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    e = np.asarray(feat, np.int64)[sel]
    a, b = pos[e[:, 0]], pos[e[:, 1]]
    mask = np.zeros(len(e), np.int64)
    for ax in axes:
        on = (np.abs(a[:, ax]) < MERGE_DIST) & (np.abs(b[:, ax]) < MERGE_DIST)
        mask |= on.astype(np.int64) << ax
    n = np.clip(np.ceil(np.linalg.norm(b - a, axis=1) / max(step, 1e-9)).astype(np.int64), 1, 64)
    rep = np.repeat(np.arange(len(e)), n + 1)
    t = np.concatenate([np.linspace(0.0, 1.0, k + 1) for k in n.tolist()])
    pts = a[rep] + (b[rep] - a[rep]) * t[:, None]
    return pts, mask[rep]


def seam_plane_masks(co, idx, pts, masks):
    """Plane bit mask for the engine verts `idx`: that of the nearest seam sample."""
    out = np.zeros(len(co), np.int64)
    if len(idx) == 0 or len(pts) == 0:
        out[idx] = 1 << 30  # no sample: snap_seam falls back to the nearest plane
        return out
    kd = KDTree(len(pts))
    for i, p in enumerate(pts.tolist()):
        kd.insert(p, i)
    kd.balance()
    for i in np.asarray(idx).tolist():
        out[i] = int(masks[kd.find(co[i].tolist())[1]]) or (1 << 30)
    return out


def drop_plane_faces(me, axes):
    """Remove faces lying in a symmetry plane (every vertex within MERGE_DIST of it) and the
    geometry they leave loose: the mirror would put a second copy on top of them."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bad = [f for f in bm.faces
           if any(all(abs(v.co[a]) < MERGE_DIST for v in f.verts) for a in axes)]
    if bad:
        bmesh.ops.delete(bm, geom=bad, context="FACES_ONLY")
        loose_e = [e for e in bm.edges if not e.link_faces]
        if loose_e:
            bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
        loose_v = [v for v in bm.verts if not v.link_faces]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
        bm.to_mesh(me)
        me.update()
    bm.free()
    return len(bad)


def nonmanifold_edges(me):
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("edge_index", le)
    return int((np.bincount(le, minlength=len(me.edges)) > 2).sum())


def split_nonmanifold(me):
    """Open surfaces cannot go through the voxel pass (it would wrap them in a thin shell),
    so edges shared by 3+ faces are split into separate sheets instead."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bad = [e for e in bm.edges if len(e.link_faces) > 2]
    if bad:
        bmesh.ops.split_edges(bm, edges=bad)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()


def fill_holes(me, planes=None):
    """A small hole can remain in a closed surface, and multi axis symmetry can leave one
    where the mirror planes cross. Close them with quads where possible. With
    `planes`, only holes whose border lies entirely on those symmetry planes are filled."""
    bm = bmesh.new()
    bm.from_mesh(me)
    border = [e for e in bm.edges if e.is_boundary]
    if planes is not None:
        def on_plane(v):
            return any(abs(v.co[a]) < 1e-5 for a in planes)
        border = [e for e in border if on_plane(e.verts[0]) and on_plane(e.verts[1])]
    if border:
        new = bmesh.ops.holes_fill(bm, edges=border, sides=0)["faces"]
        if new:
            tris = bmesh.ops.triangulate(bm, faces=new)["faces"]
            bmesh.ops.join_triangles(bm, faces=tris, angle_face_threshold=math.pi, angle_shape_threshold=math.pi)
        bm.to_mesh(me)
        me.update()
    bm.free()


def locked_mask(me, lock_sharp):
    bm = bmesh.new()
    bm.from_mesh(me)
    mask = np.zeros(len(bm.verts), bool)
    sharp = math.radians(SHARP_DEG)
    for e in bm.edges:
        lock = e.is_boundary or not e.is_manifold
        if not lock and lock_sharp:
            ang = e.calc_face_angle(0.0)
            lock = ang > sharp
        if lock:
            mask[e.verts[0].index] = True
            mask[e.verts[1].index] = True
    bm.free()
    return mask


def relax_and_project(me, bvh, iterations, strength, lock_sharp, project):
    n = len(me.vertices)
    if n == 0:
        return
    co = np.empty(n * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    deg = np.bincount(ed.ravel(), minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0
    free = ~locked_mask(me, lock_sharp)
    nrm = np.empty(n * 3, np.float32)

    def project_all():
        if bvh is None:
            return
        for i in range(n):
            hit = bvh.find_nearest(co[i])
            if hit[0] is not None:
                co[i] = hit[0]

    for _ in range(iterations):
        nb = np.empty_like(co)
        for k in range(3):
            nb[:, k] = (np.bincount(ed[:, 0], weights=co[ed[:, 1], k], minlength=n)
                        + np.bincount(ed[:, 1], weights=co[ed[:, 0], k], minlength=n))
        d = nb / deg[:, None] - co
        me.vertex_normals.foreach_get("vector", nrm)
        nr = nrm.reshape(-1, 3).astype(np.float64)
        d -= (d * nr).sum(1)[:, None] * nr  # tangential only, keeps volume
        co[free] += strength * d[free]
        if project:
            project_all()
        me.vertices.foreach_set("co", co.ravel())
        me.update()

    if iterations == 0 and project:
        project_all()
        me.vertices.foreach_set("co", co.ravel())
        me.update()


# --------------------------------------------------------------------------
# Array helpers (engine input and output)
# --------------------------------------------------------------------------

def mesh_coords(me):
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def mesh_polys(me):
    """(offsets, verts) CSR arrays of the polygons."""
    n = len(me.polygons)
    starts = np.empty(n, np.int64)
    me.polygons.foreach_get("loop_start", starts)
    sizes = np.empty(n, np.int64)
    me.polygons.foreach_get("loop_total", sizes)
    lv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(sizes, out=offsets[1:])
    if n and not np.array_equal(starts, offsets[:-1]):
        idx = np.concatenate([np.arange(s, s + k) for s, k in zip(starts.tolist(), sizes.tolist())])
        lv = lv[idx]
    return offsets, lv


def mesh_triangles(me):
    """(m, 3) triangles of the loop triangulation, same winding as the polygons."""
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", t)
    return t.reshape(-1, 3)


def mesh_from_polys(name, co, offsets, verts):
    me = bpy.data.meshes.new(name)
    nv, nf = len(co), len(offsets) - 1
    me.vertices.add(nv)
    me.vertices.foreach_set("co", np.asarray(co, np.float32).ravel())
    me.loops.add(len(verts))
    me.loops.foreach_set("vertex_index", np.asarray(verts, np.int32))
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.asarray(offsets[:-1], np.int32))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    return me


def edge_faces(me):
    """Per edge: face count and the first two face indices (-1 when missing)."""
    ne = len(me.edges)
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("edge_index", le)
    offsets, _ = mesh_polys(me)
    starts = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", starts)
    sizes = np.diff(offsets)
    lp = np.empty(len(me.loops), np.int64)
    for_loops = np.repeat(np.arange(len(me.polygons)), sizes)
    loop_idx = np.repeat(starts, sizes) + (np.arange(len(lp)) - np.repeat(offsets[:-1], sizes))
    lp[loop_idx] = for_loops
    cnt = np.bincount(le, minlength=ne)
    order = np.argsort(le, kind="stable")
    sp = lp[order]
    first = np.searchsorted(le[order], np.arange(ne))
    f0 = np.where(cnt >= 1, sp[np.minimum(first, len(sp) - 1)], -1) if len(sp) else np.full(ne, -1)
    f1 = np.where(cnt >= 2, sp[np.minimum(first + 1, len(sp) - 1)], -1) if len(sp) else np.full(ne, -1)
    return cnt, f0, f1


def guide_mask(me):
    """Edges marked Seam, or with a non-zero edge attribute 'brm_guide'."""
    ne = len(me.edges)
    mask = np.zeros(ne, bool)
    try:
        seam = np.zeros(ne, bool)
        me.edges.foreach_get("use_seam", seam)
        mask |= seam
    except (AttributeError, TypeError, RuntimeError):
        pass
    attr = me.attributes.get("brm_guide")
    if attr is not None and attr.domain == "EDGE" and attr.data_type in {"BOOLEAN", "FLOAT", "INT", "INT8"}:
        vals = np.zeros(ne, np.float64 if attr.data_type == "FLOAT" else
                        (bool if attr.data_type == "BOOLEAN" else np.int64))
        attr.data.foreach_get("value", vals)
        mask |= vals != 0
    return mask


def feature_edges(me, axes, keep_sharp, use_guides):
    """Engine feature pairs of the work mesh: hard edges (the same 40 degree test as the
    relax lock, kind 1), guides (kind 2, or 3 when also hard) and the symmetry cut (kind 5:
    border edges with both ends on a plane within MERGE_DIST)."""
    ne = len(me.edges)
    if ne == 0:
        return np.zeros((0, 2), np.uint32), np.zeros(0, np.uint8)
    ev = np.empty(ne * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    cnt, f0, f1 = edge_faces(me)
    kinds = np.zeros(ne, np.uint8)
    if keep_sharp:
        pn = np.empty(len(me.polygons) * 3, np.float64)
        me.polygons.foreach_get("normal", pn)
        pn = pn.reshape(-1, 3)
        two = cnt == 2
        cosang = np.ones(ne)
        cosang[two] = (pn[f0[two]] * pn[f1[two]]).sum(1)
        kinds[two & (cosang < math.cos(math.radians(SHARP_DEG)))] = engine.KIND_HARD
    if use_guides:
        g = guide_mask(me)
        kinds[g & (kinds == engine.KIND_HARD)] = engine.KIND_GUIDE_HARD
        kinds[g & (kinds == 0)] = engine.KIND_GUIDE
    if axes:
        co = mesh_coords(me)
        on = np.zeros(ne, bool)
        for a in axes:
            on |= (np.abs(co[ev[:, 0], a]) < MERGE_DIST) & (np.abs(co[ev[:, 1], a]) < MERGE_DIST)
        kinds[(cnt == 1) & on] = engine.KIND_SYMMETRY
    sel = kinds > 0
    return ev[sel].astype(np.uint32), kinds[sel]


def density_of(me):
    """Per vertex density multipliers of `me` or None: the POINT float attribute 'brm_density',
    else the active color attribute (sRGB luminance L gives 4 ** (2L - 1))."""
    n = len(me.vertices)
    attr = me.attributes.get("brm_density")
    if attr is not None and attr.domain == "POINT" and attr.data_type == "FLOAT":
        d = np.empty(n, np.float64)
        attr.data.foreach_get("value", d)
    else:
        col = getattr(me.color_attributes, "active_color", None) if hasattr(me, "color_attributes") else None
        if col is None or col.domain not in {"POINT", "CORNER"}:
            return None
        rgba = np.empty(len(col.data) * 4, np.float64)
        try:
            col.data.foreach_get("color_srgb", rgba)
        except (AttributeError, TypeError, RuntimeError):
            col.data.foreach_get("color", rgba)
        rgba = rgba.reshape(-1, 4)
        lum = rgba[:, :3] @ np.array([0.2126, 0.7152, 0.0722])
        if col.domain == "CORNER":
            lv = np.empty(len(me.loops), np.int64)
            me.loops.foreach_get("vertex_index", lv)
            s = np.bincount(lv, weights=lum, minlength=n)
            c = np.bincount(lv, minlength=n)
            lum = np.where(c > 0, s / np.maximum(c, 1), 0.5)
        d = 4.0 ** (2.0 * np.clip(lum, 0.0, 1.0) - 1.0)
    d = np.where(np.isfinite(d) & (d > 0), d, 1.0)
    return np.clip(d, 1.0 / 16.0, 16.0)


def transfer_vertex_values(src_co, values, dst_co):
    """Nearest source vertex value for each destination vertex."""
    kd = KDTree(len(src_co))
    for i, p in enumerate(src_co.tolist()):
        kd.insert(p, i)
    kd.balance()
    out = np.empty(len(dst_co), np.float64)
    for i, p in enumerate(dst_co.tolist()):
        out[i] = values[kd.find(p)[1]]
    return out


def tri_area(pos, tris):
    if len(tris) == 0:
        return 0.0
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

def source_quad_equivalent(context, obj):
    dg = context.evaluated_depsgraph_get()
    me = obj.evaluated_get(dg).data
    return len(me.loops) / 4.0


class BRM_OT_set_target(bpy.types.Operator):
    bl_idname = "bremesher.set_target"
    bl_label = "Set Target"
    bl_description = "Set the target from the current face count of the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    factor: FloatProperty(default=1.0)

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH"

    def execute(self, context):
        s = context.scene.bremesher
        s.target_k = max(0.05, round(source_quad_equivalent(context, context.active_object) * self.factor / 1000.0, 2))
        return {"FINISHED"}


class Job:
    """State of one remesh, from the source mesh to the solver input."""

    def __init__(self, context, s, src):
        self.s = s
        self.src_name = src.name
        self.prev_mode = src.mode
        self.t0 = time.perf_counter()
        self.base = None
        self.worker = None
        self.result = None
        self.warnings = []
        self.hid_source = False


def side_error(co, axes, keep_negative):
    """Message when no vertex lies strictly on the kept side of a symmetry plane (the bisect
    would leave nothing), else None."""
    for a in axes:
        side = co[:, a] < -1e-6 if keep_negative else co[:, a] > 1e-6
        if not side.any():
            return (f"Nothing to remesh on the {'-' if keep_negative else '+'} side of the "
                    f"{AXES[a]} symmetry plane: the mesh lies on the other side. Switch Keep Side, "
                    "or move the object's origin onto its plane of symmetry")
    return None


class BRM_OT_remesh(bpy.types.Operator):
    bl_idname = "bremesher.remesh"
    bl_label = "BRemesh"
    bl_description = "Build a clean all-quad copy of the active mesh with flowing edge loops"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode in {"OBJECT", "SCULPT", "EDIT_MESH"}

    # ---- entry points ------------------------------------------------------

    def execute(self, context):
        """Blocking run (scripts, headless, redo)."""
        job = self._begin(context)
        if job is None:
            return {"CANCELLED"}

        def run():
            try:
                job.result = engine.remesh(**job.engine_args)
            except Exception as ex:  # noqa: BLE001 (handled like an engine panic)
                return self._engine_exception(context, job, ex)
            return self._finish_engine(context, job)
        try:
            return self._guard(context, job, run)
        finally:
            self._cleanup(context, job)

    def invoke(self, context, event):
        """Interactive run: the engine works in a thread, Esc cancels, the view stays usable."""
        if context.window is None:
            return self.execute(context)
        job = self._begin(context)
        if job is None:
            return {"CANCELLED"}
        job.worker = engine.Worker(**job.engine_args)
        job.worker.start()
        wm = context.window_manager
        self._job = job
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        self._status(context, "BRemesher: starting  (Esc to cancel)")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        job = self._job
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            job.worker.control.cancel()
            self._status(context, "BRemesher: cancelling...")
            return {"RUNNING_MODAL"}
        if event.type == "TIMER":
            ctl = job.worker.control
            if not job.worker.done:
                pct = int(100 * ctl.progress)
                context.window_manager.progress_update(10 + int(0.6 * pct))
                if not ctl.cancelled:
                    self._status(context, f"BRemesher: {ctl.stage} {pct}%  (Esc to cancel)")
                return {"RUNNING_MODAL"}
            self._stop_timer(context)
            try:
                if job.worker.exc is not None:
                    return self._guard(context, job, self._engine_exception, context, job, job.worker.exc)
                job.result = job.worker.result
                return self._guard(context, job, self._finish_engine, context, job)
            finally:
                self._cleanup(context, job)
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN",
                          "TRACKPADZOOM", "MOUSEROTATE", "MOUSEMOVE", "INBETWEEN_MOUSEMOVE",
                          "NDOF_MOTION"} or event.type.startswith("NDOF"):
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}  # block edits and undo while the job runs

    def cancel(self, context):
        """Blender is closing the operator (file load, quit): stop the engine and clean up."""
        job = getattr(self, "_job", None)
        if job is None:
            return
        if job.worker is not None:
            job.worker.control.cancel()
            job.worker.join(timeout=30.0)
        self._stop_timer(context)
        self._cleanup(context, job)

    # ---- helpers -------------------------------------------------------------

    def _status(self, context, text):
        try:
            context.workspace.status_text_set(text)
        except (AttributeError, RuntimeError):
            pass

    def _stop_timer(self, context):
        timer = getattr(self, "_timer", None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self._timer = None
        self._status(context, None)

    def _cleanup(self, context, job):
        if job.base is not None:
            try:
                bpy.data.meshes.remove(job.base)
            except ReferenceError:
                pass
            job.base = None
        context.window_manager.progress_end()

    def _restore_source(self, context, job):
        src = bpy.data.objects.get(job.src_name)
        if src is None:
            return
        if job.hid_source:
            src.hide_set(False)
            job.hid_source = False
        set_active(context, src)
        if job.prev_mode == "SCULPT":
            bpy.ops.object.mode_set(mode="SCULPT")
        elif job.prev_mode == "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")

    def _guard(self, context, job, fn, *args):
        """Run one step of the job. On any exception (a UserError message, or a bug) the step
        leaves nothing behind: every object and mesh it created is removed (the job's base mesh
        is left to `_cleanup`), the source is shown again, made active and put back in its mode,
        and the operator ends CANCELLED with an error report."""
        objs, meshes = set(bpy.data.objects), set(bpy.data.meshes)
        try:
            return fn(*args)
        except Exception as ex:  # noqa: BLE001 (the run must always end in a clean state)
            if isinstance(ex, UserError):
                msg = str(ex)
            else:
                import traceback
                traceback.print_exc()
                msg = f"BRemesher failed: {type(ex).__name__}: {str(ex).strip()}"
            try:
                if context.object is not None and context.object.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            for o in [o for o in bpy.data.objects if o not in objs]:
                bpy.data.objects.remove(o)
            for m in [m for m in bpy.data.meshes if m not in meshes]:
                if m != job.base and m.users == 0:
                    bpy.data.meshes.remove(m)
            try:
                self._restore_source(context, job)
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

    def _begin(self, context):
        """Shared preparation (see `_prepare`). Returns a Job or None (error reported, source
        restored)."""
        s = snapshot(context.scene.bremesher)
        src = context.active_object
        job = Job(context, s, src)
        if self._guard(context, job, self._prepare, context, job, src) != {"FINISHED"}:
            self._cleanup(context, job)
            return None
        return job

    def _prepare(self, context, job, src):
        """Snapshot, normalized clean base mesh, targets and, for the engine, its input arrays.
        Errors are raised as UserError."""
        s = job.s
        wm = context.window_manager
        wm.progress_begin(0, 100)
        if src.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        if not ensure_engine():
            raise UserError(f"BRemesher engine not available: {engine.error()}")

        dg = context.evaluated_depsgraph_get()
        base = bpy.data.meshes.new_from_object(src.evaluated_get(dg))
        job.base = base
        if not base.polygons:
            raise UserError("The mesh has no faces")
        co = mesh_coords(base)
        diag = float(np.linalg.norm(co.max(0) - co.min(0))) or 1.0
        job.norm = WORK_DIAG / diag
        base.transform(Matrix.Scale(job.norm, 4))
        clean_mesh(base)
        if not base.polygons:
            raise UserError("The mesh has no faces after cleanup")
        job.axes = [i for i, on in enumerate((s.sym_x, s.sym_y, s.sym_z)) if on]
        if job.axes:
            msg = side_error(mesh_coords(base), job.axes, s.sym_side == "NEG")
            if msg:
                raise UserError(msg)
        job.bvh = bvh_from_mesh(base) if s.reproject else None
        wm.progress_update(5)

        job.target_full = max(8, int(s.target_k * 1000))
        job.target = max(4, int(round(job.target_full / (2 ** len(job.axes)))))
        job.area = mesh_area(base)
        job.edge_len = math.sqrt(job.area / job.target_full)
        job.closed = not has_boundary(base)
        job.used_voxel = False
        self._engine_input(context, job, src)
        return {"FINISHED"}

    def _engine_input(self, context, job, src):
        """Pre-pass, bisect and arrays for the engine (DESIGN 8.3). The work copy is a
        temporary object, removed before the engine runs."""
        s, base = job.s, job.base
        if s.cleanup == "ALWAYS":
            voxel = True
        elif s.cleanup == "NEVER":
            voxel = False
        elif job.closed:
            voxel = not is_solver_friendly(base)
        else:
            if not is_solver_friendly(base):
                split_nonmanifold(base)
            voxel = False
        max_work = s.max_work_k * 1000
        colls = src.users_collection or (context.scene.collection,)
        tmp = bpy.data.objects.new("brm_work", base.copy())
        colls[0].objects.link(tmp)
        try:
            set_active(context, tmp)
            me = tmp.data
            if voxel:
                size = max(job.edge_len / 3.0, math.sqrt(job.area / max_work))
                # The voxel grid is anchored at the local origin. On a symmetric shape it puts
                # rows of output vertices exactly on a symmetry plane, with edges and faces in
                # it and coincident vertices where the two halves touch, and the bisect then
                # leaves a pinched seam (non-manifold after the mirror weld). Shifting the grid
                # off the planes makes the bisect a clean transversal cut.
                shift = Matrix.Translation(Vector((VOXEL_SHIFT * size,) * 3))
                me.transform(shift)
                me.remesh_voxel_size = size
                me.remesh_voxel_adaptivity = 0.0
                bpy.ops.object.voxel_remesh()
                me = tmp.data
                me.transform(shift.inverted())
                clean_mesh(me, weld=False)
                job.used_voxel = True
            elif len(me.polygons) > max_work:
                mod = tmp.modifiers.new("brm_decimate", "DECIMATE")
                mod.ratio = max_work / len(me.polygons)
                mod.use_collapse_triangulate = True
                apply_modifier(tmp, mod)
                me = tmp.data
                clean_mesh(me)
            if job.axes:
                bisect_half(me, job.axes, s.sym_side == "NEG")
                drop_plane_faces(me, job.axes)
                if not me.polygons:
                    raise UserError("Nothing to remesh on the kept side of the symmetry planes "
                                    "(the cut left no face). Switch Keep Side, or move the object's "
                                    "origin onto its plane of symmetry")
            pos = mesh_coords(me)
            tris = mesh_triangles(me)
            feat, kind = feature_edges(me, job.axes, s.keep_sharp, s.use_guides)
            if s.use_guides and voxel:
                job.warnings.append("guides are lost by the watertight pre-pass")
            density = None
            if s.use_density:
                d = density_of(base)
                if d is None:
                    job.warnings.append("Use Density: no 'brm_density' attribute or color attribute")
                elif len(pos) == len(base.vertices) and not job.axes and not voxel:
                    density = d
                else:
                    density = transfer_vertex_values(mesh_coords(base), d, pos)
        finally:
            set_active(context, src)
            me = tmp.data
            bpy.data.objects.remove(tmp)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        job.pos, job.tris = pos, tris
        job.seam_samples = seam_samples(pos, feat, kind, job.axes, 0.1 * job.edge_len)

        p = engine.default_params()
        p.target_quads = job.target
        p.adaptive_size = s.adaptive_size
        p.curvature_align = s.curve_strength
        p.flow_smoothness = s.flow_smoothness
        p.density_strength = s.density_strength
        p.guide_strength = s.guide_strength
        p.hard_edges = 2 if s.keep_sharp else 0
        p.seed = s.seed
        p.time_limit_s = s.time_limit
        p.threads = 0
        p.debug_flags = s.debug_flags
        job.params = p
        job.engine_args = dict(pos=pos, tris=tris, feat=feat, feat_kind=kind, density=density, params=p)
        context.window_manager.progress_update(10)

    def _new_object(self, context, job, me):
        src = bpy.data.objects[job.src_name]
        work = bpy.data.objects.new(job.src_name + "_BRemesher", me)
        colls = src.users_collection or (context.scene.collection,)
        colls[0].objects.link(work)
        work.matrix_world = src.matrix_world.copy()
        set_active(context, work)
        return work

    def _engine_exception(self, context, job, ex):
        """The engine call raised (in the worker thread or inline): handled like an engine
        panic, the source is restored and the error reported."""
        import traceback
        traceback.print_exception(ex)
        raise UserError(f"BRemesher engine call failed: {type(ex).__name__}: {str(ex).strip()}")

    def _finish_engine(self, context, job):
        res = job.result
        if res.status == engine.CANCELLED:
            self.report({"WARNING"}, "BRemesher cancelled")
            self._restore_source(context, job)
            return {"CANCELLED"}
        if res.status in (engine.INVALID_INPUT, engine.PANIC):
            raise UserError(f"BRemesher engine: {res.status_name}: {res.error}")

        co = res.pos
        offsets = res.face_offsets.astype(np.int64)
        verts = res.face_verts.astype(np.int64)
        kind = (res.vflag >> engine.VFLAG_KIND_SHIFT) & 0xF
        boundary = (res.vflag & engine.VFLAG_BOUNDARY) != 0
        # DESIGN 8.3: the boundary verts of a Symmetry chain go onto the plane whatever their
        # distance (the engine can return them about h/2 off it). Other boundary verts within
        # 1e-3 h of a plane are snapped too, as a safety net.
        seam_force = boundary & (kind == engine.KIND_SYMMETRY)
        seam_cand = np.zeros(len(co), bool)
        if job.axes and len(co):
            for a in job.axes:
                seam_cand |= np.abs(co[:, a]) < 1e-3 * job.edge_len
            seam_cand &= boundary & ~seam_force
        # Each seam vertex goes onto the plane of the input seam edge it lies along.
        seam_mask = seam_plane_masks(co, np.nonzero(seam_force)[0], *job.seam_samples)
        seam = (seam_mask, seam_cand)

        failed = res.failed_tris.astype(np.int64)
        failed = failed[failed < len(job.tris)]
        if len(failed):
            job.warnings.append(f"{len(failed):,} triangles the engine could not remesh are missing")
        if len(offsets) <= 1:
            raise UserError(f"BRemesher engine produced no faces: {res.error}")

        me = mesh_from_polys(job.src_name + "_BRemesher", co, offsets, verts)
        work = self._new_object(context, job, me)
        counts = res.comp_counts()
        parts = []
        names = {engine.COMP_FULL: "full", engine.COMP_RETRIED: "retried", engine.COMP_SF: "simple quadder",
                 engine.COMP_FAILED: "failed", engine.COMP_DEBRIS_SF: "small parts"}
        for k in sorted(counts):
            parts.append(f"{counts[k]} {names.get(k, str(k))}")
        note = f"engine {res.seconds:.1f}s, components: {', '.join(parts) or 'none'}"
        return self._finalize(context, job, work, seam, note)

    def _finalize(self, context, job, work, seam, note):
        """v1 post steps: seam snap, relax and reprojection, mirror and weld, holes, scale back,
        normals, materials, shading, source hiding and the mode round trip."""
        s = job.s
        wm = context.window_manager
        axes, edge_len = job.axes, job.edge_len
        src = bpy.data.objects[job.src_name]
        set_active(context, work)
        wm.progress_update(75)
        me = work.data
        if axes:
            snap_seam(me, axes, edge_len * 0.3, *(seam or ()))
        if s.relax_iterations or job.bvh is not None:
            relax_and_project(me, job.bvh, s.relax_iterations, s.relax_strength, s.keep_sharp,
                              job.bvh is not None)
            if axes:
                snap_seam(me, axes, edge_len * 0.3, *(seam or ()))

        wm.progress_update(90)
        if axes:
            mod = work.modifiers.new("brm_mirror", "MIRROR")
            mod.use_axis = [a in axes for a in range(3)]
            mod.use_bisect_axis = [False, False, False]
            mod.use_clip = True
            mod.use_mirror_merge = True
            mod.merge_threshold = edge_len * 0.05
            apply_modifier(work, mod)

        me = work.data
        if job.used_voxel or job.closed:
            fill_holes(me)
        elif len(axes) > 1:
            fill_holes(me, planes=axes)
        # Last check: a seam the weld closed into edges with 3+ faces (or any other non-manifold
        # edge) is never handed over silently.
        n_bad = nonmanifold_edges(me)
        if n_bad:
            job.warnings.append(f"the result has {n_bad} non-manifold edges"
                                + (" after the symmetry weld" if axes else ""))
        me.transform(Matrix.Scale(1.0 / job.norm, 4))
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()

        if not me.materials:
            for m in src.data.materials:
                me.materials.append(m)
        if s.shade_smooth:
            if s.keep_sharp and hasattr(bpy.ops.object, "shade_smooth_by_angle"):
                bpy.ops.object.shade_smooth_by_angle(angle=math.radians(SHARP_DEG))
            else:
                me.shade_smooth()
        else:
            me.shade_flat()

        if s.hide_source:
            src.hide_set(True)
            job.hid_source = True
        if job.prev_mode == "SCULPT":
            bpy.ops.object.mode_set(mode="SCULPT")
        elif job.prev_mode == "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")

        quads = int(round(quad_ratio(me) * len(me.polygons)))
        if quads < len(me.polygons):
            job.warnings.append(f"{len(me.polygons) - quads:,} of {len(me.polygons):,} faces are not quads "
                                "(hole filling)")
        msg = (f"BRemesher: {len(me.polygons):,} faces ({quads:,} quads) in "
               f"{time.perf_counter() - job.t0:.1f}s, {note}"
               f"{', watertight pre-pass used' if job.used_voxel else ''}")
        for w in job.warnings:
            self.report({"WARNING"}, "BRemesher: " + w)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

class BRM_PT_main(bpy.types.Panel):
    bl_label = "BRemesher"
    bl_idname = "BRM_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Remesh"

    def draw(self, context):
        layout = self.layout
        s = context.scene.bremesher
        obj = context.active_object

        if _engine_tried and not engine.available():
            layout.label(text="Engine unavailable (retried each run)", icon="ERROR")

        col = layout.column(align=True)
        col.prop(s, "target_k")
        row = col.row(align=True)
        row.operator("bremesher.set_target", text="Half").factor = 0.5
        row.operator("bremesher.set_target", text="Same").factor = 1.0
        row.operator("bremesher.set_target", text="Double").factor = 2.0
        if obj is not None and obj.type == "MESH":
            layout.label(text=f"Source: {len(obj.data.polygons):,} faces", icon="MESH_DATA")

        col = layout.column(align=True)
        col.prop(s, "adaptive_size")
        col.prop(s, "curve_strength")
        col.prop(s, "flow_smoothness")

        box = layout.box()
        box.label(text="Symmetry", icon="MOD_MIRROR")
        row = box.row(align=True)
        row.prop(s, "sym_x", toggle=True)
        row.prop(s, "sym_y", toggle=True)
        row.prop(s, "sym_z", toggle=True)
        sub = box.row()
        sub.enabled = s.sym_x or s.sym_y or s.sym_z
        sub.prop(s, "sym_side", expand=True)

        col = layout.column()
        col.prop(s, "keep_sharp")
        row = col.row(align=True)
        row.prop(s, "use_guides")
        sub = row.row(align=True)
        sub.active = s.use_guides
        sub.prop(s, "guide_strength")
        row = col.row(align=True)
        row.prop(s, "use_density")
        sub = row.row(align=True)
        sub.active = s.use_density
        sub.prop(s, "density_strength")
        col.prop(s, "cleanup")

        row = layout.row(align=True)
        row.prop(s, "relax_iterations")
        row.prop(s, "relax_strength")
        layout.prop(s, "reproject")

        layout.separator()
        layout.operator("bremesher.remesh", icon="MOD_REMESH")


class BRM_PT_advanced(bpy.types.Panel):
    bl_label = "Advanced"
    bl_parent_id = "BRM_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Remesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.bremesher
        col = self.layout.column()
        col.prop(s, "seed")
        col.prop(s, "time_limit")
        col.prop(s, "max_work_k")
        col.prop(s, "hide_source")
        col.prop(s, "shade_smooth")
        if engine.available():
            col.label(text="Engine: loaded", icon="CHECKMARK")
        elif _engine_tried:
            col.label(text="Engine: " + engine.error(), icon="ERROR")


classes = (BRM_Settings, BRM_OT_set_target, BRM_OT_remesh, BRM_PT_main, BRM_PT_advanced)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.bremesher = PointerProperty(type=BRM_Settings)


def unregister():
    del bpy.types.Scene.bremesher
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()

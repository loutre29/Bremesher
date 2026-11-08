# SPDX-License-Identifier: GPL-3.0-or-later
#
# Headless Blender 5.2 smoke test of the add-on (WP-06). It loads the add-on from the source
# folder (never installs it), so it only runs inside the sandbox of engine/tools/blender.sh:
#
#   engine/tools/blender.sh --python-exit-code 1 --python bremesher/tests/smoke_blender.py
#
# Checks: the ctypes binding (struct sizes, ABI, defaults, a worker thread polled through
# AqControl, cancel), the operator with the engine on several test meshes (100% quads, count,
# manifold), symmetry with the seam weld, Keep Sharp Edges, the sculpt mode round trip, Hide
# Source, Half / Same / Double, a clean error when the engine is unavailable or raises, and a
# component the engine gives up on (comp_status 3) reported missing.

import json
import os
import sys
import time
import traceback
from collections import Counter

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(os.path.dirname(HERE))
TESTDATA = os.path.join(ADDON_PARENT, "engine", "testdata")
CACHE = os.path.join(ADDON_PARENT, "engine", "target", "wp-06", "smoke_cache")

sys.path.insert(0, ADDON_PARENT)
import bremesher as aq  # noqa: E402
from bremesher import engine  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    return cond


def reset_scene():
    """Empty the scene and reset the settings (no file reload: the sandboxed profile is never
    touched)."""
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    s = bpy.context.scene.bremesher
    for k in s.bl_rna.properties.keys():
        if k not in {"rna_type", "name"}:
            s.property_unset(k)


def import_obj(name):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=os.path.join(TESTDATA, name + ".obj"))
    new = [o for o in bpy.data.objects if o not in before]
    obj = new[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def mesh_stats(me):
    sizes = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_total", sizes)
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("edge_index", le)
    cnt = np.bincount(le, minlength=len(me.edges))
    # components through shared vertices
    parent = list(range(len(me.vertices)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    ev = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    for a, b in ev.reshape(-1, 2).tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    used = set(np.unique(ev).tolist())
    comps = len({find(v) for v in used})
    return {
        "faces": len(me.polygons),
        "quad_ratio": float((sizes == 4).mean()) if len(sizes) else 0.0,
        "open_edges": int((cnt == 1).sum()),
        "nonmanifold_edges": int((cnt > 2).sum()),
        "components": comps,
    }


def run_remesh(obj, target_k, **settings):
    s = bpy.context.scene.bremesher
    s.target_k = target_k
    for k, v in settings.items():
        setattr(s, k, v)
    t = time.perf_counter()
    ret = bpy.ops.bremesher.remesh()
    dt = time.perf_counter() - t
    out = bpy.context.view_layer.objects.active
    return ret, out, dt


# --------------------------------------------------------------------------

def test_binding():
    lib = engine.load(cache_dir=CACHE)
    check("dll loads", lib is not None, engine.error())
    if lib is None:
        return False
    check("dll loaded from the shadow copy", os.path.dirname(engine.loaded_from()) == CACHE,
          engine.loaded_from())
    import ctypes as C
    check("struct sizes 72/64/12/104",
          [C.sizeof(engine.AqParams), C.sizeof(engine.AqMeshIn), C.sizeof(engine.AqControl),
           C.sizeof(engine.AqResult)] == [72, 64, 12, 104])
    check("abi version 1", lib.aq_abi_version() == 1)
    p = engine.default_params()
    check("defaults (DESIGN 8.1)",
          (p.struct_size, p.target_quads, round(p.adaptive_size, 3), p.curvature_align, p.hard_edges,
           p.close_loops, p.allow_collapse, p.ring_min, p.max_proxy_tris, p.time_limit_s, p.seed)
          == (72, 5000, 0.5, 1.0, 2, 1, 1, 8, 150000, 60.0, 0),
          str((p.target_quads, p.adaptive_size, p.hard_edges, p.time_limit_s)))

    # Torus through a worker thread, polled like the modal operator does.
    obj = import_obj("torus")
    me = obj.data
    pos = aq.mesh_coords(me)
    pos *= 10.0 / np.linalg.norm(pos.max(0) - pos.min(0))
    tris = aq.mesh_triangles(me)
    p.target_quads = 2000
    w = engine.Worker(pos=pos, tris=tris, params=p)
    w.start()
    seen = []
    while not w.done:
        seen.append(w.control.progress)
        time.sleep(0.02)
    r = w.result
    check("worker: no exception", w.exc is None, repr(w.exc))
    check("worker: status ok", r is not None and r.status == engine.OK, r and (r.status_name + " " + r.error))
    if r is not None:
        sizes = np.diff(r.face_offsets.astype(np.int64))
        check("worker: all quads", len(sizes) > 0 and bool((sizes == 4).all()), f"{len(sizes)} faces")
        check("worker: count within 10%", abs(len(sizes) - 2000) <= 200, str(len(sizes)))
        check("worker: report json", r.report.get("abi") == 1 and "components" in r.report)
        check("worker: comp status full", r.comp_status.tolist() == [engine.COMP_FULL], str(r.comp_status))
        check("worker: progress polled and reached 1", w.control.progress >= 0.999 and len(seen) > 0,
              f"{len(seen)} polls, last {w.control.progress}")

    # Cancel from the polling side.
    c = engine.Control()
    c.cancel()
    r = engine.remesh(pos, tris, params=p, control=c)
    check("cancel before the call gives status 3", r.status == engine.CANCELLED, r.status_name)
    # Invalid input (index out of range) gives 4, never a crash.
    bad = tris.copy()
    bad[0, 0] = len(pos) + 5
    r = engine.remesh(pos, bad, params=p)
    check("invalid input gives status 4", r.status == engine.INVALID_INPUT, r.status_name)
    bpy.data.objects.remove(obj)
    return True


def test_operator_meshes():
    cases = [
        ("torus", 2.0, {}),
        ("cube_sharp", 2.0, {"keep_sharp": True}),
        ("retopo_b", 2.0, {}),
        ("noisy_blob", 5.0, {}),
    ]
    for name, k, extra in cases:
        reset_scene()
        src = import_obj(name)
        ret, out, dt = run_remesh(src, k, **extra)
        ok = check(f"{name}: operator finished", ret == {"FINISHED"}, str(ret))
        if not ok:
            continue
        st = mesh_stats(out.data)
        check(f"{name}: new object", out is not src and out.name.endswith("_BRemesher"), out.name)
        check(f"{name}: 100% quads", st["quad_ratio"] == 1.0, json.dumps(st))
        check(f"{name}: count within 15%", abs(st["faces"] - k * 1000) <= 0.15 * k * 1000, str(st["faces"]))
        check(f"{name}: manifold", st["nonmanifold_edges"] == 0)
        check(f"{name}: source hidden", src.hide_get())
        check(f"{name}: time", dt < 30.0, f"{dt:.1f}s")
        if name == "torus":
            check("torus: closed", st["open_edges"] == 0)


def test_symmetry():
    reset_scene()
    src = import_obj("sculpt_sphere_mirror")
    ret, out, dt = run_remesh(src, 5.0, sym_x=True)
    if check("mirror: operator finished", ret == {"FINISHED"}, str(ret)):
        st = mesh_stats(out.data)
        check("mirror: 100% quads", st["quad_ratio"] == 1.0, json.dumps(st))
        check("mirror: 0 open edges after the weld", st["open_edges"] == 0, str(st["open_edges"]))
        check("mirror: count within 15%", abs(st["faces"] - 5000) <= 750, str(st["faces"]))
        co = aq.mesh_coords(out.data)
        # exact mirror: every vertex has a partner at (-x, y, z)
        key = {tuple(np.round(v, 5)) for v in co.tolist()}
        miss = sum(1 for v in co.tolist() if (round(-v[0], 5) + 0.0, round(v[1], 5), round(v[2], 5)) not in key)
        check("mirror: exact symmetry", miss <= len(co) * 0.001, f"{miss} unmatched of {len(co)}")


def test_sculpt_round_trip_and_targets():
    reset_scene()
    src = import_obj("suzanne_sub2")
    bpy.ops.bremesher.set_target(factor=0.5)
    half = bpy.context.scene.bremesher.target_k
    bpy.ops.bremesher.set_target(factor=1.0)
    same = bpy.context.scene.bremesher.target_k
    bpy.ops.bremesher.set_target(factor=2.0)
    double = bpy.context.scene.bremesher.target_k
    check("Half / Same / Double", abs(half * 2 - same) < 0.02 and abs(double - 2 * same) < 0.02,
          f"{half} {same} {double}")
    bpy.ops.object.mode_set(mode="SCULPT")
    ret, out, dt = run_remesh(src, 2.0)
    if check("sculpt: operator finished", ret == {"FINISHED"}, str(ret)):
        check("sculpt: back in sculpt mode on the new mesh", out is not src and out.mode == "SCULPT",
              f"{out.name} {out.mode}")
        st = mesh_stats(out.data)
        check("sculpt: 100% quads", st["quad_ratio"] == 1.0, json.dumps(st))
        bpy.ops.object.mode_set(mode="OBJECT")
    # Edit mode in, edit mode out, source shown when Hide Source is off.
    reset_scene()
    src = import_obj("torus")
    bpy.ops.object.mode_set(mode="EDIT")
    ret, out, dt = run_remesh(src, 1.0, hide_source=False)
    if check("edit: operator finished", ret == {"FINISHED"}, str(ret)):
        check("edit: back in edit mode", out.mode == "EDIT", out.mode)
        check("edit: source not hidden", not src.hide_get())
        bpy.ops.object.mode_set(mode="OBJECT")


def test_engine_unavailable():
    """No engine: the run is cancelled with an error and the source is left as it was."""
    reset_scene()
    src = import_obj("torus")
    n_meshes = len(bpy.data.meshes)
    saved = aq.ensure_engine
    aq.ensure_engine = lambda: False
    try:
        ret, msg = _run_expect_error(1.0)
    finally:
        aq.ensure_engine = saved
    check("engine unavailable: cancelled with an error",
          ret == {"CANCELLED"} and "engine not available" in msg, f"{ret} {msg.strip()[:120]}")
    objs, dm = _leftovers(src, n_meshes)
    check("engine unavailable: source untouched, nothing left behind",
          bpy.context.view_layer.objects.active == src and not src.hide_get() and not objs and dm == 0,
          f"{objs} {dm:+d}")


def test_failed_component():
    """Two components; debug flags send component 0 to R3 (comp_status 3). Its triangles come
    back in failed_tris; the add-on keeps the other part and reports the missing one."""
    reset_scene()
    a = import_obj("torus")
    b = import_obj("noisy_blob")
    b.location.x += 3.0 * max(a.dimensions)
    b.scale = (0.5, 0.5, 0.5)
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active

    # What the engine itself returns for this input.
    s = bpy.context.scene.bremesher
    s.debug_flags = engine.DEBUG_FORCE_SF | engine.DEBUG_PANIC_COMPONENT
    me = src.data
    pos = aq.mesh_coords(me)
    pos *= 10.0 / np.linalg.norm(pos.max(0) - pos.min(0))
    p = engine.default_params()
    p.target_quads = 3000
    p.debug_flags = s.debug_flags
    r = engine.remesh(pos, aq.mesh_triangles(me), params=p)
    check("failed part: engine status degraded", r.status == engine.DEGRADED, r.status_name)
    check("failed part: one component failed (3)", Counter(r.comp_status.tolist()).get(engine.COMP_FAILED) == 1,
          str(r.comp_status.tolist()))
    check("failed part: failed_tris given", len(r.failed_tris) > 0, str(len(r.failed_tris)))

    op = _FakeOp()
    s.target_k = 3.0
    ret = op.execute(bpy.context)
    out = bpy.context.view_layer.objects.active
    if check("failed part: operator finished", ret == {"FINISHED"}, f"{ret} {op.reports}"):
        st = mesh_stats(out.data)
        check("failed part: the other part only, all quads", st["components"] == 1 and st["quad_ratio"] == 1.0,
              json.dumps(st))
        check("failed part: reported missing",
              any("could not remesh" in m for k, m in op.reports if "WARNING" in k), str(op.reports))
    s.debug_flags = 0


def _run_spied(src, target_k, **settings):
    """Run the operator, capturing the engine result and the faces fill_holes added."""
    cap = {"filled": 0, "fill_calls": 0}
    real_remesh, real_fill = engine.remesh, aq.fill_holes

    def spy_remesh(**kw):
        cap["pos"], cap["feat"], cap["feat_kind"] = kw["pos"], kw["feat"], kw["feat_kind"]
        cap["res"] = real_remesh(**kw)
        return cap["res"]

    def spy_fill(me, planes=None):
        n = len(me.polygons)
        real_fill(me, planes)
        cap["fill_calls"] += 1
        cap["filled"] += len(me.polygons) - n
    engine.remesh, aq.fill_holes = spy_remesh, spy_fill
    try:
        ret, out, dt = run_remesh(src, target_k, **settings)
    finally:
        engine.remesh, aq.fill_holes = real_remesh, real_fill
    return ret, out, cap


def test_symmetry_seam():
    """DESIGN 8.3: the engine's Symmetry chain verts (vflag bit 26 and kind 5) end exactly on
    their plane, whatever their distance to it, so the weld closes the seam without cap faces.
    With the voxel pre-pass (suzanne_sub2 / sculpt_cylinder, Always) the seam must also come out
    manifold and closed: the voxel grid used to put a pinched seam on the plane."""
    cases = (
        ("torus", 2.0, {"sym_x": True}),
        ("suzanne_sub2", 2.0, {"sym_x": True}),
        ("suzanne_sub2", 2.0, {"sym_x": True, "cleanup": "ALWAYS"}),
        ("sculpt_cylinder", 2.0, {"sym_x": True, "cleanup": "ALWAYS"}),
        ("torus", 2.0, {"sym_x": True, "sym_y": True}),
    )
    for name, k, settings in cases:
        reset_scene()
        src = import_obj(name)
        axes = [a for a, key in enumerate(("sym_x", "sym_y", "sym_z")) if settings.get(key)]
        if len(axes) > 1 or settings.get("cleanup"):
            name = name + " " + "".join("XYZ"[a] for a in axes) + (" voxel" if settings.get("cleanup") else "")
        ret, out, cap = _run_spied(src, k, **settings)
        if not check(f"seam {name}: operator finished", ret == {"FINISHED"}, str(ret)):
            continue
        res = cap["res"]
        kind = (res.vflag >> engine.VFLAG_KIND_SHIFT) & 0xF
        force = np.nonzero(((res.vflag & engine.VFLAG_BOUNDARY) != 0) & (kind == engine.KIND_SYMMETRY))[0]
        ev = res.face_verts.astype(np.int64)
        off = res.face_offsets.astype(np.int64)
        nxt = np.arange(len(ev)) + 1  # next corner in the same face
        nxt[off[1:] - 1] = off[:-1]
        el = np.linalg.norm(res.pos[ev] - res.pos[ev[nxt]], axis=1)
        h = float(el.mean()) if len(el) else 1.0
        off_plane = float(np.abs(res.pos[force][:, axes]).min(1).max() / h) if len(force) else 0.0
        check(f"seam {name}: engine gave kind 5 boundary verts", len(force) > 0,
              f"{len(force)}, up to {off_plane:.2f} h off the plane before the snap")
        co = aq.mesh_coords(out.data)
        n_eng = len(res.pos)
        check(f"seam {name}: output keeps the engine vertex order", len(co) >= n_eng)
        bad = int((co[force][:, axes] != 0.0).all(1).sum()) if len(co) >= n_eng else -1
        check(f"seam {name}: every kind 5 vert exactly on a plane", bad == 0,
              f"{bad} of {len(force)} off")
        if len(axes) > 1 and len(co) >= n_eng:
            # Each one on the plane of the input seam it lies along, not merely the nearest.
            pin = np.asarray(cap["pos"])
            pts, masks = aq.seam_samples(pin, cap["feat"], cap["feat_kind"], axes, 0.05)
            from mathutils.kdtree import KDTree
            kd = KDTree(len(pts))
            for i, p in enumerate(pts.tolist()):
                kd.insert(p, i)
            kd.balance()
            wrong = 0
            for v in force.tolist():
                m = int(masks[kd.find(res.pos[v].tolist())[1]])
                wrong += any(m >> a & 1 and co[v, a] != 0.0 for a in axes)
            check(f"seam {name}: kind 5 verts on their own plane", wrong == 0, f"{wrong} on another plane")
        check(f"seam {name}: no cap faces added", cap["filled"] == 0,
              f"{cap['filled']} faces from {cap['fill_calls']} fill_holes call(s)")
        me = out.data
        le = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("edge_index", le)
        cnt = np.bincount(le, minlength=len(me.edges))
        eo = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", eo)
        eo = eo.reshape(-1, 2)
        h_out = float(np.linalg.norm(co[eo[:, 0]] - co[eo[:, 1]], axis=1).mean())
        near = np.zeros(len(eo), bool)
        for a in axes:
            near |= (np.abs(co[eo, a]) < 0.6 * h_out).any(1)
        gaps = int(((cnt == 1) & near).sum())
        check(f"seam {name}: no open edge at the seam", gaps == 0, f"{gaps} open edges near the planes")
        st = mesh_stats(me)
        check(f"seam {name}: 100% quads, manifold", st["quad_ratio"] == 1.0 and st["nonmanifold_edges"] == 0,
              json.dumps(st))
        if name.startswith("torus") or settings.get("cleanup") == "ALWAYS":
            check(f"seam {name}: closed", st["open_edges"] == 0, str(st["open_edges"]))


def test_snap_seam_planes():
    """snap_seam puts a seam vertex on the plane its mask names, even when another selected
    plane is nearer; seam_samples / seam_plane_masks give that mask from the input seam."""
    import bmesh
    me = bpy.data.meshes.new("snap")
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in ((0.3, 0.1, 0.2), (0.12, 0.2, 0.2), (0.6, 0.6, 0.2), (0.3, 0.4, 0.9))]
    bm.faces.new(vs)
    bm.to_mesh(me)
    bm.free()
    # input seam: an edge on the Y plane near vertex 0, and one on the X plane near vertex 1
    pin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    feat = np.array([[0, 1], [2, 3]])
    kinds = np.array([engine.KIND_SYMMETRY] * 2)
    pts, masks = aq.seam_samples(pin, feat, kinds, [0, 1], 0.05)
    eco = aq.mesh_coords(me)
    force = aq.seam_plane_masks(eco, np.array([0, 1]), pts, masks)
    check("snap: plane masks from the input seam", force[:2].tolist() == [2, 1] and force[2:].tolist() == [0, 0],
          str(force.tolist()))
    aq.snap_seam(me, [0, 1], 0.01, force, np.zeros(4, bool))
    co = aq.mesh_coords(me)
    check("snap: Y seam vertex goes onto Y although X is nearer", np.allclose(co[0], (0.3, 0.0, 0.2)), str(co[0]))
    check("snap: X seam vertex goes onto X", np.allclose(co[1], (0.0, 0.2, 0.2)), str(co[1]))
    check("snap: other verts untouched", np.allclose(co[2:], [(0.6, 0.6, 0.2), (0.3, 0.4, 0.9)]))
    bpy.data.meshes.remove(me)


def test_engine_failures():
    """An exception from the engine call (inline or in the worker thread) cancels cleanly with
    the error; a non-manifold result after the symmetry weld is kept with a warning."""
    real_remesh, real_nm = engine.remesh, aq.nonmanifold_edges

    def boom(**kw):
        raise MemoryError("injected engine failure")

    def warnings_of(op):
        return [m for k, m in op.reports if "WARNING" in k]

    def errors_of(op):
        return [m for k, m in op.reports if "ERROR" in k]

    reset_scene()
    src = import_obj("torus")
    bpy.context.scene.bremesher.target_k = 1.0
    n_meshes = len(bpy.data.meshes)
    op = _FakeOp()
    engine.remesh = boom
    try:
        ret = op.execute(bpy.context)
    finally:
        engine.remesh = real_remesh
    check("engine exception: cancelled with the error",
          ret == {"CANCELLED"} and any("injected engine failure" in m for m in errors_of(op)),
          f"{ret} {op.reports}")
    objs, dm = _leftovers(src, n_meshes)
    check("engine exception: nothing left behind, source shown",
          not objs and dm == 0 and not src.hide_get() and bpy.context.view_layer.objects.active == src,
          f"{objs} {dm:+d}")

    # The worker thread raising (modal path).
    reset_scene()
    src = import_obj("torus")
    bpy.context.scene.bremesher.target_k = 1.0
    n_meshes = len(bpy.data.meshes)
    op = _FakeOp()
    engine.remesh = boom
    try:
        ret = op.invoke(_FakeCtx(), _Ev("NONE"))
        t = time.perf_counter()
        while ret == {"RUNNING_MODAL"} and time.perf_counter() - t < 120:
            time.sleep(0.02)
            ret = op.modal(_FakeCtx(), _Ev("TIMER"))
    finally:
        engine.remesh = real_remesh
    check("worker exception: cancelled with the error",
          ret == {"CANCELLED"} and any("injected engine failure" in m for m in errors_of(op)), f"{ret} {op.reports}")
    objs, dm = _leftovers(src, n_meshes)
    check("worker exception: nothing left behind", not objs and dm == 0, f"{objs} {dm:+d}")

    # A non-manifold result after the weld (simulated on the first check).
    reset_scene()
    src = import_obj("torus")
    s = bpy.context.scene.bremesher
    s.target_k, s.sym_x = 2.0, True
    calls = []

    def fake_nm(me):
        calls.append(1)
        return 6 if len(calls) == 1 else real_nm(me)
    aq.nonmanifold_edges = fake_nm
    op = _FakeOp()
    try:
        ret = op.execute(bpy.context)
    finally:
        aq.nonmanifold_edges = real_nm
    out = bpy.context.view_layer.objects.active
    others = [o.name for o in bpy.data.objects if o not in (src, out)]
    check("non-manifold result: kept with a warning",
          ret == {"FINISHED"} and any("6 non-manifold edges" in m for m in warnings_of(op)), f"{ret} {op.reports}")
    check("non-manifold result: one result object, nothing else left",
          ret == {"FINISHED"} and out is not src and not others, f"{others}")


def _leftovers(src, n_meshes):
    return [o.name for o in bpy.data.objects if o != src], len(bpy.data.meshes) - n_meshes


def _run_expect_error(target_k, **settings):
    """The operator reports an error: bpy.ops raises RuntimeError with the message."""
    try:
        ret = run_remesh(None, target_k, **settings)[0]
        return ret, ""
    except RuntimeError as ex:
        return {"CANCELLED"}, str(ex)


def test_failure_paths():
    # 1. Whole mesh on the discarded side of the symmetry plane, from sculpt mode: a clear
    #    error, nothing left behind, the source back in sculpt mode and visible.
    reset_scene()
    src = import_obj("torus")
    from mathutils import Matrix
    src.data.transform(Matrix.Translation((-50.0, 0.0, 0.0)))
    bpy.ops.object.mode_set(mode="SCULPT")
    n_meshes = len(bpy.data.meshes)
    ret, msg = _run_expect_error(1.0, sym_x=True)
    check("wrong side: cancelled with a clear error",
          ret == {"CANCELLED"} and "Nothing to remesh" in msg, f"{ret} {msg.strip()[:120]}")
    objs, dm = _leftovers(src, n_meshes)
    check("wrong side: no stray object or mesh", not objs and dm == 0, f"{objs} {dm:+d} meshes")
    check("wrong side: source active, visible, back in sculpt mode",
          bpy.context.view_layer.objects.active == src and not src.hide_get() and src.mode == "SCULPT",
          f"{bpy.context.view_layer.objects.active} {src.mode}")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 2. The engine gives up on a part (a lone small triangle, component 0): the torus is kept,
    #    the part is reported missing.
    reset_scene()
    torus = import_obj("torus")
    tm = bpy.data.meshes.new("tri")
    d = max(torus.dimensions)
    x0 = 1.5 * d
    tm.from_pydata([(x0, 0, 0), (x0 + 0.03 * d, 0, 0), (x0, 0.03 * d, 0)], [], [(0, 1, 2)])
    tri = bpy.data.objects.new("tri", tm)
    bpy.context.scene.collection.objects.link(tri)
    bpy.ops.object.select_all(action="DESELECT")
    tri.select_set(True)
    torus.select_set(True)
    bpy.context.view_layer.objects.active = tri
    bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
    s = bpy.context.scene.bremesher
    s.debug_flags = engine.DEBUG_FORCE_SF | engine.DEBUG_PANIC_COMPONENT
    n_meshes = len(bpy.data.meshes)
    try:
        ret, out, cap = _run_spied(src, 2.0)
    except RuntimeError as ex:
        ret, out, cap = {"CANCELLED"}, None, {}
        print("raised:", ex)
    s.debug_flags = 0
    res = cap.get("res")
    check("lost part: engine gave up on component 0",
          res is not None and res.comp_status.tolist()[:1] == [engine.COMP_FAILED] and len(res.failed_tris) == 1,
          "" if res is None else f"{res.comp_status.tolist()} {len(res.failed_tris)} failed tris")
    if check("lost part: operator finished with the engine's result", ret == {"FINISHED"}, str(ret)):
        st = mesh_stats(out.data)
        check("lost part: the torus only, all quads", st["components"] == 1 and st["quad_ratio"] == 1.0,
              json.dumps(st))
        objs = [o.name for o in bpy.data.objects if o not in (src, out)]
        check("lost part: no stray object or mesh",
              not objs and len(bpy.data.meshes) == n_meshes + 1, f"{objs} {len(bpy.data.meshes) - n_meshes - 1:+d}")

    # 3. Any exception after the result object exists (here a bug injected into the relax
    #    step): the result is removed, the source is shown again and back in sculpt mode.
    reset_scene()
    src = import_obj("torus")
    bpy.ops.object.mode_set(mode="SCULPT")
    n_meshes = len(bpy.data.meshes)
    real = aq.relax_and_project

    def boom(*a, **k):
        raise ValueError("injected")
    aq.relax_and_project = boom
    try:
        ret, msg = _run_expect_error(1.0, sym_x=True)
    finally:
        aq.relax_and_project = real
    check("exception: cancelled with the error reported", ret == {"CANCELLED"} and "injected" in msg,
          f"{ret} {msg.strip()[:120]}")
    objs, dm = _leftovers(src, n_meshes)
    check("exception: no stray object or mesh", not objs and dm == 0, f"{objs} {dm:+d} meshes")
    check("exception: source active, visible, back in sculpt mode",
          bpy.context.view_layer.objects.active == src and not src.hide_get() and src.mode == "SCULPT",
          f"{src.mode} hidden={src.hide_get()}")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 4. A failed DLL load is retried on the next run (no need to re-enable the add-on).
    saved = engine._lib
    engine._lib, engine._error = None, "simulated failed load"
    aq._engine_tried = True
    try:
        ok = aq.ensure_engine()
    finally:
        if engine._lib is None:
            engine._lib, engine._error = saved, ""
    check("engine load retried after a failure", ok and engine.available(), engine.error())


def test_ui_mapping():
    """Every UI control reaches its AqParams field (DESIGN 8.3 table), feature kinds 1, 2, 3
    and 5 are sent, and the density map has one value per engine input vertex."""
    reset_scene()
    src = import_obj("cube_sharp")
    me = src.data
    co = aq.mesh_coords(me)
    ev = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    # mark as seams: a few edges on the cube's top face border and a few inside a face
    zmax = co[:, 2].max()
    seam = np.zeros(len(me.edges), bool)
    top = np.abs(co[ev, 2] - zmax).max(1) < 1e-6
    seam[np.nonzero(top)[0][:40]] = True
    seam[np.nonzero(~top)[0][::50]] = True
    me.edges.foreach_set("use_seam", seam)
    attr = me.attributes.new("brm_density", "FLOAT", "POINT")
    attr.data.foreach_set("value", np.full(len(me.vertices), 2.0, np.float32))

    captured = {}
    real = engine.remesh

    def spy(**kw):
        captured.update(kw)
        return real(**kw)
    engine.remesh = spy
    try:
        ret = run_remesh(src, 1.5, adaptive_size=0.3, curve_strength=2.0, flow_smoothness=1.5,
                         keep_sharp=True, seed=3, time_limit=50.0, use_guides=True, guide_strength=2.5,
                         use_density=True, density_strength=0.75, sym_x=True)[0]
    finally:
        engine.remesh = real
    check("mapping: operator finished", ret == {"FINISHED"}, str(ret))
    p = captured.get("params")
    if not check("mapping: engine called", p is not None):
        return
    check("mapping: target halved by symmetry", p.target_quads == 750, str(p.target_quads))
    check("mapping: sliders",
          (round(p.adaptive_size, 4), p.curvature_align, p.flow_smoothness, p.density_strength,
           p.guide_strength) == (0.3, 2.0, 1.5, 0.75, 2.5),
          str((p.adaptive_size, p.curvature_align, p.flow_smoothness, p.density_strength, p.guide_strength)))
    check("mapping: hard_edges 2, seed, time limit", (p.hard_edges, p.seed, p.time_limit_s) == (2, 3, 50.0))
    kinds = Counter(np.asarray(captured["feat_kind"]).tolist())
    check("mapping: feature kinds 1, 2, 3, 5 sent", all(kinds.get(k, 0) > 0 for k in (1, 2, 3, 5)), str(kinds))
    pos = np.asarray(captured["pos"])
    d = captured.get("density")
    check("mapping: density per engine vertex", d is not None and len(d) == len(pos) and
          np.allclose(d, 2.0), "none" if d is None else f"{len(d)} / {len(pos)}")
    check("mapping: symmetry pairs on the plane",
          np.abs(pos[np.asarray(captured["feat"])[np.asarray(captured["feat_kind"]) == 5]][..., 0]).max() < 2e-4)
    check("mapping: all input on the kept half", pos[:, 0].min() > -1e-6, str(pos[:, 0].min()))

    # hard_edges 0 without Keep Sharp Edges, no features at all.
    reset_scene()
    src = import_obj("cube_sharp")
    engine.remesh = spy
    captured.clear()
    try:
        ret = bpy.ops.bremesher.remesh("INVOKE_DEFAULT")
    finally:
        engine.remesh = real
    check("invoke (headless runs execute): finished", ret == {"FINISHED"}, str(ret))
    p = captured.get("params")
    check("mapping: keep sharp off gives hard_edges 0 and no features",
          p is not None and p.hard_edges == 0 and captured.get("feat") is not None and len(captured["feat"]) == 0)


class _FakeWM:
    """Window manager stand-in: timers and modal handlers are no-ops headless."""

    def __init__(self, wm):
        self._wm = wm

    def event_timer_add(self, step, window=None):
        return object()

    def event_timer_remove(self, timer):
        pass

    def modal_handler_add(self, op):
        pass

    def __getattr__(self, k):
        return getattr(self._wm, k)


class _FakeCtx:
    window = object()

    def __init__(self):
        self.window_manager = _FakeWM(bpy.context.window_manager)

    def __getattr__(self, k):
        return getattr(bpy.context, k)


class _Ev:
    def __init__(self, type, value="NOTHING"):
        self.type, self.value = type, value


class _FakeOp:
    """Runs the operator's own invoke / modal / cancel code with synthetic events."""

    def __init__(self):
        self.reports = []

    def report(self, kind, msg):
        self.reports.append((set(kind), msg))


for _name, _fn in aq.BRM_OT_remesh.__dict__.items():
    if callable(_fn) and not _name.startswith("__") and _name not in {"poll"}:
        setattr(_FakeOp, _name, _fn)


def test_modal_path():
    ctx = _FakeCtx()
    # 1. run to completion through the timer polling
    reset_scene()
    src = import_obj("noisy_blob")
    bpy.context.scene.bremesher.target_k = 3.0
    op = _FakeOp()
    ret = op.invoke(ctx, _Ev("NONE"))
    check("modal: invoke starts the worker", ret == {"RUNNING_MODAL"} and op._job.worker is not None, str(ret))
    check("modal: mouse move passes through", op.modal(ctx, _Ev("MOUSEMOVE")) == {"PASS_THROUGH"})
    check("modal: keys are blocked while running", op.modal(ctx, _Ev("Z", "PRESS")) == {"RUNNING_MODAL"})
    t = time.perf_counter()
    polls = 0
    while ret == {"RUNNING_MODAL"} and time.perf_counter() - t < 120:
        time.sleep(0.05)
        ret = op.modal(ctx, _Ev("TIMER"))
        polls += 1
    check("modal: finished after polling", ret == {"FINISHED"}, f"{ret} after {polls} polls")
    out = bpy.context.view_layer.objects.active
    if ret == {"FINISHED"}:
        st = mesh_stats(out.data)
        check("modal: result 100% quads", out is not src and st["quad_ratio"] == 1.0, json.dumps(st))
    check("modal: base mesh freed", op._job.base is None)

    # 2. Esc cancels: the engine stops, the source is restored, nothing is left behind
    reset_scene()
    src = import_obj("noisy_blob")
    bpy.ops.object.mode_set(mode="SCULPT")
    n_meshes = len(bpy.data.meshes)
    op = _FakeOp()
    ret = op.invoke(ctx, _Ev("NONE"))
    ret = op.modal(ctx, _Ev("ESC", "PRESS"))
    check("modal: Esc keeps running until the engine stops", ret == {"RUNNING_MODAL"})
    t = time.perf_counter()
    while ret == {"RUNNING_MODAL"} and time.perf_counter() - t < 60:
        time.sleep(0.02)
        ret = op.modal(ctx, _Ev("TIMER"))
    check("modal: cancelled", ret == {"CANCELLED"}, str(ret))
    check("modal: cancel took under 3 s", time.perf_counter() - t < 3.0, f"{time.perf_counter() - t:.2f}s")
    check("modal: source active, visible, back in sculpt mode",
          bpy.context.view_layer.objects.active == src and not src.hide_get() and src.mode == "SCULPT",
          f"{bpy.context.view_layer.objects.active.name} {src.mode}")
    check("modal: no leftover objects or meshes",
          len(bpy.data.objects) == 1 and len(bpy.data.meshes) == n_meshes,
          f"{len(bpy.data.objects)} objects, {len(bpy.data.meshes)} meshes (was {n_meshes})")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 3. Blender closing the operator (cancel callback) joins the worker and cleans up
    reset_scene()
    src = import_obj("noisy_blob")
    n_meshes = len(bpy.data.meshes)
    op = _FakeOp()
    op.invoke(ctx, _Ev("NONE"))
    op.cancel(ctx)
    check("modal: cancel() stops the worker", op._job.worker.done)
    check("modal: cancel() leaves nothing behind",
          len(bpy.data.objects) == 1 and len(bpy.data.meshes) == n_meshes)


def main():
    aq.register()
    try:
        for t in (test_binding, test_operator_meshes, test_symmetry, test_sculpt_round_trip_and_targets,
                  test_ui_mapping, test_modal_path, test_engine_unavailable, test_failed_component,
                  test_symmetry_seam, test_snap_seam_planes, test_engine_failures, test_failure_paths):
            try:
                t()
            except Exception:
                traceback.print_exc()
                check(t.__name__ + " raised", False)
    finally:
        try:
            aq.unregister()
        except Exception:
            traceback.print_exc()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\nSMOKE: {len(results) - n_fail} passed, {n_fail} failed", flush=True)
    if n_fail:
        sys.exit(1)


main()

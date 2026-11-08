"""Render wireframe views of OBJ meshes headless (Workbench). Poles of quad meshes are marked:
interior valence 3 = red, valence 5 = blue, 6+ = magenta.

Usage: tools/blender.sh --python tools/render_wire.py -- <out.png> <mesh.obj> [<mesh2.obj> ...] [--size 800]
Each mesh becomes one row of 2 views (front, 3/4); rows are stacked top to bottom in argument order.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
size = 800
if "--size" in argv:
    i = argv.index("--size")
    size = int(argv[i + 1])
    del argv[i:i + 2]
out_png, meshes = os.path.abspath(argv[0]), argv[1:]


def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return verts, faces


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)


def setup_scene():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "OBJECT"
    sc.display.shading.show_cavity = False
    sc.render.resolution_x = size
    sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"


def add_obj(name, verts, faces, color):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.color = color
    return ob


def interior_poles(faces):
    edges = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            e = (min(a, b), max(a, b))
            edges[e] = edges.get(e, 0) + 1
    val, bnd = {}, set()
    for (a, b), c in edges.items():
        val[a] = val.get(a, 0) + 1
        val[b] = val.get(b, 0) + 1
        if c == 1:
            bnd.update((a, b))
    return {v: k for v, k in val.items() if v not in bnd and k != 4}


def render_row(path):
    verts, faces = read_obj(path)
    clear()
    add_obj("m", verts, faces, (0.82, 0.82, 0.80, 1))
    co = np.array(verts)
    lo, hi = co.min(0), co.max(0)
    center = Vector(((lo + hi) / 2).tolist())
    radius = float(np.linalg.norm(hi - lo)) / 2
    sample = faces[: min(len(faces), 4000)]
    el = float(np.mean([np.linalg.norm(co[f[0]] - co[f[1]]) for f in sample]))
    wire = add_obj("w", verts, faces, (0.05, 0.05, 0.06, 1))
    mod = wire.modifiers.new("wf", "WIREFRAME")
    mod.thickness = el * 0.07
    if sum(1 for f in faces if len(f) == 4) > 0.5 * len(faces):
        pv = interior_poles(faces)
        for key, colr in ((3, (0.9, 0.1, 0.1, 1)), (5, (0.1, 0.3, 0.95, 1)), (6, (0.8, 0.1, 0.8, 1))):
            pts = [verts[v] for v, d in pv.items() if (d == key if key < 6 else d >= 6)]
            if not pts:
                continue
            bm = bmesh.new()
            for p in pts:
                res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=el * 0.28)
                bmesh.ops.translate(bm, verts=res["verts"], vec=p)
            m = bpy.data.meshes.new("p%d" % key)
            bm.to_mesh(m)
            bm.free()
            o = bpy.data.objects.new("p%d" % key, m)
            bpy.context.scene.collection.objects.link(o)
            o.color = colr
    cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    dist = radius / math.sin(math.radians(39.6) / 2) * 1.05
    tiles = []
    tmp = out_png + ".tmp.png"
    for d in (Vector((0, 1, 0)), Vector((1, 1, -0.7)).normalized()):
        cam.location = center - d * dist
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(tmp)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(size, size, 4)
        bpy.data.images.remove(img)
        tiles.append(px)
    os.remove(tmp)
    return np.concatenate(tiles, axis=1)


setup_scene()
rows = [render_row(p) for p in meshes]
full = np.concatenate(rows[::-1], axis=0)  # Blender image rows start at the bottom
h, w = full.shape[:2]
img = bpy.data.images.new("out", w, h, alpha=False)
img.pixels[:] = full.ravel()
img.filepath_raw = out_png
img.file_format = "PNG"
img.save()
print("RENDERED", out_png, w, "x", h)

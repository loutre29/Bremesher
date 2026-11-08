"""QuadriFlow baselines: the quality bar the custom engine must beat.

Usage: tools/blender.sh --python tools/quadriflow_baseline.py -- <out_dir> <target_quads> <mesh.obj>...
Writes <out_dir>/<stem>_qf<target>.obj (voxel pre-pass retry when QuadriFlow rejects the input).
"""
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out_dir, target, paths = argv[0], int(argv[1]), argv[2:]
os.makedirs(out_dir, exist_ok=True)


def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return verts, faces


for p in paths:
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    v, f = read_obj(p)
    me = bpy.data.meshes.new("m")
    me.from_pydata(v, [], f)
    ob = bpy.data.objects.new("m", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    t = time.time()
    kw = dict(mode="FACES", target_faces=target, use_preserve_sharp=True, use_preserve_boundary=True)
    r = bpy.ops.object.quadriflow_remesh(**kw)
    if "FINISHED" not in r:
        ob.data.remesh_voxel_size = max(ob.dimensions) / 300
        bpy.ops.object.voxel_remesh()
        r = bpy.ops.object.quadriflow_remesh(**kw)
    stem = os.path.splitext(os.path.basename(p))[0]
    outp = os.path.join(out_dir, "%s_qf%d.obj" % (stem, target))
    m = ob.data
    with open(outp, "w") as fh:
        for vv in m.vertices:
            fh.write("v %.9g %.9g %.9g\n" % tuple(vv.co))
        for pp in m.polygons:
            fh.write("f " + " ".join(str(i + 1) for i in pp.vertices) + "\n")
    print("BASELINE", stem, r, len(m.polygons), "faces", round(time.time() - t, 2), "s")

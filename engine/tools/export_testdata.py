# Builds engine/testdata/*.obj (triangulated, object local space).
# Usage: blender -b --factory-startup --python export_testdata.py -- <out_dir> [<blend> <object> <name>]...
import sys, os, bpy, bmesh
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
pairs = [argv[i:i + 3] for i in range(1, len(argv), 3)]


def write_obj(me, path):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.verts.index_update()
    with open(path, "w") as f:
        for v in bm.verts:
            f.write("v %.9g %.9g %.9g\n" % tuple(v.co))
        for fc in bm.faces:
            f.write("f %d %d %d\n" % tuple(v.index + 1 for v in fc.verts))
    print("WROTE", os.path.basename(path), len(bm.verts), "verts", len(bm.faces), "tris")
    bm.free()


def export_object(obj, name):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    write_obj(me, os.path.join(out, name + ".obj"))
    bpy.data.meshes.remove(me)


for blend, objname, name in pairs:
    bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
    export_object(bpy.data.objects[objname], name)

# Synthetic cases
bpy.ops.wm.read_homefile(use_empty=True)  # never read_factory_settings: it re-syncs extension wheels


def new(op, name, **kw):
    op(**kw)
    o = bpy.context.active_object
    o.name = name
    return o


def apply_all(o):
    bpy.context.view_layer.objects.active = o
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


tex = bpy.data.textures.new("clouds", "CLOUDS")
tex.noise_scale = 0.45
fine = bpy.data.textures.new("fine", "CLOUDS")
fine.noise_scale = 0.04

o = new(bpy.ops.mesh.primitive_monkey_add, "suzanne")
o.modifiers.new("s", "SUBSURF").levels = 2
apply_all(o); export_object(o, "suzanne_sub2"); bpy.data.objects.remove(o)

o = new(bpy.ops.mesh.primitive_torus_add, "torus", major_segments=96, minor_segments=32)
export_object(o, "torus"); bpy.data.objects.remove(o)

a = new(bpy.ops.mesh.primitive_torus_add, "t1", major_segments=96, minor_segments=32, location=(-0.9, 0, 0))
b = new(bpy.ops.mesh.primitive_torus_add, "t2", major_segments=96, minor_segments=32, location=(0.9, 0, 0))
bpy.ops.object.select_all(action="DESELECT"); a.select_set(True); b.select_set(True)
bpy.context.view_layer.objects.active = a
bpy.ops.object.join()
a.data.remesh_voxel_size = 0.02
bpy.ops.object.voxel_remesh()
dec = a.modifiers.new("d", "DECIMATE"); dec.ratio = 0.25
apply_all(a); export_object(a, "double_torus"); bpy.data.objects.remove(a)

o = new(bpy.ops.mesh.primitive_cube_add, "cube")
s = o.modifiers.new("s", "SUBSURF"); s.levels = 4; s.subdivision_type = "SIMPLE"
apply_all(o); export_object(o, "cube_sharp"); bpy.data.objects.remove(o)

o = new(bpy.ops.mesh.primitive_cylinder_add, "cyl", vertices=64, depth=3.0, end_fill_type="NOTHING")
s = o.modifiers.new("s", "SUBSURF"); s.levels = 3; s.subdivision_type = "SIMPLE"
apply_all(o); export_object(o, "cylinder_open"); bpy.data.objects.remove(o)

o = new(bpy.ops.mesh.primitive_ico_sphere_add, "blob", subdivisions=7)
d = o.modifiers.new("d", "DISPLACE"); d.texture = tex; d.strength = 0.45
d2 = o.modifiers.new("d2", "DISPLACE"); d2.texture = fine; d2.strength = 0.012
apply_all(o); export_object(o, "noisy_blob"); bpy.data.objects.remove(o)

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Installs the built zip into the SANDBOXED profile of engine/tools/blender.sh (never the real
# one), remeshes a test mesh through the installed extension, then uninstalls it while the DLL
# is loaded. This checks the packaged add-on and the DLL shadow copy (an installed extension
# loads a copy, so bin/bremesher_engine.dll stays unlocked and the add-on can be removed).
#
#   engine/tools/blender.sh --python-exit-code 1 --python bremesher/tests/install_sandbox.py -- <zip>

import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SANDBOX = os.path.normcase(os.path.realpath(os.path.join(ROOT, "engine", ".bl_sandbox")))

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        fails.append(name)


def main():
    res = os.environ.get("BLENDER_USER_RESOURCES", "")
    if not res or os.path.normcase(os.path.realpath(res)) != SANDBOX:
        print("refusing to run outside the engine/.bl_sandbox profile:", res)
        sys.exit(2)
    zip_path = sys.argv[sys.argv.index("--") + 1]
    ret = bpy.ops.extensions.package_install_files(filepath=zip_path, repo="user_default",
                                                    enable_on_install=True)
    check("install from the zip", ret == {"FINISHED"}, str(ret))
    mod = sys.modules.get("bl_ext.user_default.bremesher")
    check("extension module loaded", mod is not None)
    if mod is None:
        sys.exit(1)
    pkg_dir = os.path.dirname(mod.__file__)
    check("installed under the sandbox", os.path.normcase(os.path.realpath(pkg_dir)).startswith(SANDBOX), pkg_dir)

    bpy.ops.wm.obj_import(filepath=os.path.join(ROOT, "engine", "testdata", "torus.obj"))
    src = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = src
    bpy.context.scene.bremesher.target_k = 2.0
    ret = bpy.ops.bremesher.remesh()
    out = bpy.context.view_layer.objects.active
    check("remesh through the installed extension", ret == {"FINISHED"} and out is not src, str(ret))
    sizes = np.empty(len(out.data.polygons), np.int64)
    out.data.polygons.foreach_get("loop_total", sizes)
    check("100% quads", len(sizes) > 0 and bool((sizes == 4).all()), f"{len(sizes)} faces")
    eng = mod.engine
    check("engine loaded", eng.available(), eng.error())
    loaded = os.path.normcase(os.path.realpath(eng.loaded_from()))
    check("DLL loaded from the shadow copy, not bin/", loaded.startswith(SANDBOX)
          and os.path.normcase(os.path.join(pkg_dir, "bin")) not in loaded, eng.loaded_from())
    # The shadow copy lives outside the extension's user folder, which Blender deletes on
    # uninstall (and could not while it held the loaded DLL).
    user_dir = bpy.utils.extension_path_user(mod.__name__, create=True)
    user_parent = os.path.dirname(os.path.normcase(os.path.realpath(user_dir)))
    check("shadow copy outside the extension user folder",
          not loaded.startswith(os.path.normcase(os.path.realpath(user_dir))), f"{eng.loaded_from()} / {user_dir}")

    # Uninstall the way the Preferences do, by repo and package id.
    repos = bpy.context.preferences.extensions.repos
    idx = next((i for i, r in enumerate(repos) if r.module == "user_default"), -1)
    ret = bpy.ops.extensions.package_uninstall(repo_index=idx, pkg_id="bremesher")
    check("uninstall while the DLL is loaded", ret == {"FINISHED"}, str(ret))
    check("package folder removed (bin/ was not locked)", not os.path.exists(pkg_dir), pkg_dir)
    check("extension user folder removed", not os.path.exists(user_dir), user_dir)
    left = [n for n in (os.listdir(user_parent) if os.path.isdir(user_parent) else [])
            if "bremesher" in n]
    check("no leftover in the user folders (no .~temp~ copy)", not left, str(left))
    if fails:
        sys.exit(1)
    print("INSTALL TEST: all passed", flush=True)


main()

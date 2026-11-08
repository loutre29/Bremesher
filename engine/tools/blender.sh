#!/usr/bin/env bash
# Headless Blender 5.2 with a sandboxed user profile, so scripted runs can never touch
# the real preferences or extension wheels.
# Usage: tools/blender.sh --python tools/script.py -- args...
HERE="$(cd "$(dirname "$0")" && pwd)"
export BLENDER_USER_RESOURCES="$HERE/../.bl_sandbox"
mkdir -p "$BLENDER_USER_RESOURCES"
exec "/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup "$@"

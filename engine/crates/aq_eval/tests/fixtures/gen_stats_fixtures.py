"""Generates the quad_stats parity fixtures of aq_eval (DESIGN 6.15, WP-01).

Usage (from anywhere): python gen_stats_fixtures.py
For every baselines/*.obj it writes stats_<name>.json next to this script with:
  "py":  the dict printed by tools/quad_stats.py (rounded, as the script returns it)
  "raw": the same dict computed with round() disabled (full double precision)
The Rust test compares aq_eval::quad_stats against "raw" within 1e-6 and its rounded form
against "py" exactly.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ENGINE, "tools"))
import quad_stats  # noqa: E402

BASE = os.path.join(ENGINE, "baselines")


def run(path):
    verts, faces = quad_stats.read_obj(path)
    py = quad_stats.stats(verts, faces)
    quad_stats.round = lambda x, n=None: x
    try:
        raw = quad_stats.stats(verts, faces)
    finally:
        del quad_stats.round
    return py, raw


def main():
    for name in sorted(os.listdir(BASE)):
        if not name.endswith(".obj"):
            continue
        py, raw = run(os.path.join(BASE, name))
        out = os.path.join(HERE, "stats_" + name[:-4] + ".json")
        with open(out, "w", newline="\n") as f:
            json.dump({"mesh": "baselines/" + name, "py": py, "raw": raw}, f, indent=1, sort_keys=True)
            f.write("\n")
        print(out, py)


if __name__ == "__main__":
    main()

"""Quad mesh quality statistics (pure Python, runs in any Python 3 including Blender's).

Usage: python tools/quad_stats.py mesh.obj [mesh2.obj ...]
Prints per mesh: faces, quad ratio, irregular vertex count (interior valence != 4, boundary
valence not in {2, 3}), valence histogram (bN = boundary vertex of valence N), edge length
coefficient of variation, mean |corner angle - 90 deg| over quads, non-manifold and boundary edges.
"""
import math
import os
import sys
from collections import Counter, defaultdict


def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return verts, faces


def stats(verts, faces):
    sizes = Counter(len(f) for f in faces)
    edge_faces = defaultdict(int)
    vert_edges = defaultdict(set)
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            e = (min(a, b), max(a, b))
            edge_faces[e] += 1
            vert_edges[a].add(e)
            vert_edges[b].add(e)
    boundary_verts = set()
    for (a, b), c in edge_faces.items():
        if c == 1:
            boundary_verts.update((a, b))
    val = Counter()
    irregular = 0
    for v, es in vert_edges.items():
        k = len(es)
        if v in boundary_verts:
            val["b%d" % k] += 1
            if k not in (2, 3):
                irregular += 1
        else:
            val[str(k)] += 1
            if k != 4:
                irregular += 1
    lens = [math.dist(verts[a], verts[b]) for a, b in edge_faces]
    mean = sum(lens) / len(lens)
    cv = math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens)) / mean
    dev = []
    for f in faces:
        if len(f) != 4:
            continue
        for i in range(4):
            p, q, r = verts[f[i - 1]], verts[f[i]], verts[f[(i + 1) % 4]]
            u = [p[k] - q[k] for k in range(3)]
            w = [r[k] - q[k] for k in range(3)]
            nu, nw = math.hypot(*u), math.hypot(*w)
            if nu > 0 and nw > 0:
                c = max(-1.0, min(1.0, sum(u[k] * w[k] for k in range(3)) / (nu * nw)))
                dev.append(abs(math.degrees(math.acos(c)) - 90.0))
    return {
        "faces": len(faces),
        "quad_ratio": round(sizes.get(4, 0) / max(1, len(faces)), 4),
        "face_sizes": dict(sorted(sizes.items())),
        "irregular_verts": irregular,
        "valence": dict(sorted(val.items())),
        "edge_len_cv": round(cv, 4),
        "mean_angle_dev_deg": round(sum(dev) / max(1, len(dev)), 2),
        "nonmanifold_edges": sum(1 for c in edge_faces.values() if c > 2),
        "boundary_edges": sum(1 for c in edge_faces.values() if c == 1),
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        s = stats(*read_obj(path))
        print(os.path.basename(path), " ".join("%s=%s" % kv for kv in s.items()))

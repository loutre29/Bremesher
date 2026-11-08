# SPDX-License-Identifier: GPL-3.0-or-later
#
# ctypes binding of bremesher_engine.dll (engine/DESIGN.md section 8; aq_ffi builds it as
# autoquad_engine.dll, the add-on ships it renamed).
#
# This module does not import bpy, so it can be tested from plain Python. The structs mirror
# engine/crates/aq_ffi/include/autoquad_engine.h field by field; their sizes (72 / 64 / 12 / 104
# bytes on x86_64) are checked when the library is loaded.
#
# Rules (DESIGN 8.2): every output buffer belongs to the DLL and is released with aq_free, which
# `remesh` always does after copying the result into numpy arrays. The call blocks; ctypes drops
# the GIL during it, so the add-on runs it in a worker thread and polls the AqControl block.

import ctypes as C
import json
import os
import struct
import threading
import zlib

import numpy as np

ABI_VERSION = 1
DLL_NAME = "bremesher_engine.dll"

# AqResult.status
OK, DEGRADED, FAILED, CANCELLED, INVALID_INPUT, PANIC = range(6)
STATUS_NAMES = ("ok", "degraded", "failed", "cancelled", "invalid input", "engine panic")

# AqResult.comp_status
COMP_FULL, COMP_RETRIED, COMP_SF, COMP_FAILED, COMP_DEBRIS_SF = range(5)

# AqParams.debug_flags
DEBUG_DUMP = 1 << 0
DEBUG_FORCE_R1 = 1 << 1
DEBUG_FORCE_SF = 1 << 2
DEBUG_PANIC_COMPONENT = 1 << 30  # test only: component 0 panics (with FORCE_SF it reaches R3)

# Feature kinds (AqMeshIn.feat_kind)
KIND_HARD, KIND_GUIDE, KIND_GUIDE_HARD, KIND_SYMMETRY = 1, 2, 3, 5

# vflag bits (DESIGN 8.4)
VFLAG_CHAIN_MASK = 0x000FFFFF
VFLAG_KIND_SHIFT = 20
VFLAG_CORNER = 1 << 24
VFLAG_CONE = 1 << 25
VFLAG_BOUNDARY = 1 << 26
VFLAG_SF = 1 << 27

STAGE_NAMES = ("Ingest", "Proxy", "Curvature", "Field", "Loops", "Cut", "Param", "Layout",
               "Quantize", "Patch", "Post", "Fallback", "Engine", "FFI")


class AqParams(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("target_quads", C.c_uint32),
        ("adaptive_size", C.c_float),
        ("curvature_align", C.c_float),
        ("flow_smoothness", C.c_float),
        ("density_strength", C.c_float),
        ("guide_strength", C.c_float),
        ("hard_edges", C.c_uint8),
        ("close_loops", C.c_uint8),
        ("allow_collapse", C.c_uint8),
        ("reserved0", C.c_uint8),
        ("ring_min", C.c_uint32),
        ("max_proxy_tris", C.c_uint32),
        ("threads", C.c_uint32),
        ("time_limit_s", C.c_float),
        ("debug_flags", C.c_uint32),
        ("seed", C.c_uint64),
        ("dump_dir", C.c_char_p),
    ]


class AqMeshIn(C.Structure):
    _fields_ = [
        ("n_verts", C.c_uint32), ("pos", C.POINTER(C.c_double)),
        ("n_tris", C.c_uint32), ("tris", C.POINTER(C.c_uint32)),
        ("n_feat", C.c_uint32), ("feat", C.POINTER(C.c_uint32)),
        ("feat_kind", C.POINTER(C.c_uint8)),
        ("density", C.POINTER(C.c_float)),
    ]


class AqControl(C.Structure):
    _fields_ = [
        ("cancel", C.c_int32),
        ("progress_bits", C.c_uint32),
        ("stage", C.c_uint32),
    ]


class AqResult(C.Structure):
    _fields_ = [
        ("status", C.c_int32),
        ("n_verts", C.c_uint32), ("pos", C.POINTER(C.c_double)),
        ("n_faces", C.c_uint32), ("face_offsets", C.POINTER(C.c_uint32)),
        ("face_verts", C.POINTER(C.c_uint32)),
        ("vflag", C.POINTER(C.c_uint32)),
        ("face_comp", C.POINTER(C.c_uint32)),
        ("n_comp", C.c_uint32), ("comp_status", C.POINTER(C.c_uint8)),
        ("n_failed_tris", C.c_uint32), ("failed_tris", C.POINTER(C.c_uint32)),
        ("report_json", C.c_void_p),
        ("error", C.c_void_p),
    ]


EXPECTED_SIZES = {AqParams: 72, AqMeshIn: 64, AqControl: 12, AqResult: 104}

_lock = threading.Lock()
_lib = None
_error = "not loaded yet"
_loaded_from = ""


def default_dll_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", DLL_NAME)


def _shadow_copy(src, cache_dir):
    """Windows cannot unload or overwrite a DLL that is in use. Loading a copy named after the
    file's checksum keeps bin/bremesher_engine.dll unlocked, so the add-on can be updated or
    removed while Blender runs (the new DLL is used after the add-on is re-enabled). Stale copies
    that are no longer locked are deleted."""
    with open(src, "rb") as f:
        data = f.read()
    name = "bremesher_engine-%08x-%d.dll" % (zlib.crc32(data) & 0xFFFFFFFF, len(data))
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, name)
    if not os.path.isfile(dst) or os.path.getsize(dst) != len(data):
        tmp = dst + ".part%d" % os.getpid()
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    for other in os.listdir(cache_dir):
        if other.startswith("bremesher_engine-") and other != name:
            try:
                os.remove(os.path.join(cache_dir, other))
            except OSError:
                pass  # still loaded by another Blender session
    return dst


def load(path=None, cache_dir=None):
    """Load the DLL once. Returns the library, or None (then `error()` says why and the add-on
    reports it). `cache_dir`, when given, holds the shadow copy that is actually loaded."""
    global _lib, _error, _loaded_from
    with _lock:
        if _lib is not None:
            return _lib
        try:
            for cls, size in EXPECTED_SIZES.items():
                if C.sizeof(cls) != size:
                    raise OSError("%s is %d bytes, expected %d (64-bit Python required)"
                                  % (cls.__name__, C.sizeof(cls), size))
            if os.name != "nt":
                raise OSError("the BRemesher engine is only built for Windows x64")
            path = path or default_dll_path()
            if not os.path.isfile(path):
                raise OSError("missing " + path)
            target = path
            if cache_dir:
                try:
                    target = _shadow_copy(path, cache_dir)
                except OSError:
                    target = path
            lib = C.CDLL(target)
            lib.aq_abi_version.argtypes = []
            lib.aq_abi_version.restype = C.c_uint32
            lib.aq_params_default.argtypes = [C.POINTER(AqParams)]
            lib.aq_params_default.restype = None
            lib.aq_remesh.argtypes = [C.POINTER(AqMeshIn), C.POINTER(AqParams),
                                      C.POINTER(AqControl), C.POINTER(C.POINTER(AqResult))]
            lib.aq_remesh.restype = C.c_int32
            lib.aq_free.argtypes = [C.POINTER(AqResult)]
            lib.aq_free.restype = None
            abi = lib.aq_abi_version()
            if abi != ABI_VERSION:
                raise OSError("engine ABI %d, the add-on expects %d" % (abi, ABI_VERSION))
            _lib = lib
            _error = ""
            _loaded_from = target
        except (OSError, AttributeError) as ex:
            _error = str(ex)
            _lib = None
        return _lib


def available():
    return _lib is not None


def error():
    return _error


def loaded_from():
    return _loaded_from


def default_params():
    """AqParams filled by the engine's own defaults (DESIGN 8.1)."""
    lib = _require()
    p = AqParams()
    lib.aq_params_default(C.byref(p))
    return p


class Control:
    """The caller-owned AqControl block: the worker thread runs `remesh`, the UI polls this."""

    def __init__(self):
        self.raw = AqControl()

    def cancel(self):
        self.raw.cancel = 1

    @property
    def cancelled(self):
        return self.raw.cancel != 0

    @property
    def progress(self):
        v = struct.unpack("<f", struct.pack("<I", self.raw.progress_bits & 0xFFFFFFFF))[0]
        return min(1.0, max(0.0, v)) if v == v else 0.0

    @property
    def stage(self):
        s = self.raw.stage
        return STAGE_NAMES[s] if s < len(STAGE_NAMES) else str(s)


class Result:
    """A copy of AqResult in numpy arrays; nothing in it points into DLL memory."""

    def __init__(self):
        self.status = PANIC
        self.pos = np.zeros((0, 3), np.float64)
        self.face_offsets = np.zeros(1, np.uint32)
        self.face_verts = np.zeros(0, np.uint32)
        self.vflag = np.zeros(0, np.uint32)
        self.face_comp = np.zeros(0, np.uint32)
        self.comp_status = np.zeros(0, np.uint8)
        self.failed_tris = np.zeros(0, np.uint32)
        self.report = {}
        self.error = ""
        self.seconds = 0.0

    @property
    def n_faces(self):
        return len(self.face_offsets) - 1

    @property
    def status_name(self):
        return STATUS_NAMES[self.status] if 0 <= self.status < len(STATUS_NAMES) else str(self.status)

    def comp_counts(self):
        """Number of components per comp_status value."""
        return {int(k): int(n) for k, n in zip(*np.unique(self.comp_status, return_counts=True))}


def _copy(ptr, n, dtype):
    if n == 0 or not ptr:
        return np.zeros(n if ptr else 0, dtype)
    return np.ctypeslib.as_array(ptr, shape=(n,)).astype(dtype, copy=True)


def _cstr(addr):
    return C.string_at(addr).decode("utf-8", "replace") if addr else ""


def _require():
    if _lib is None:
        raise RuntimeError("BRemesher engine not available: " + (_error or "not loaded"))
    return _lib


def _ptr(arr, ctype):
    return arr.ctypes.data_as(C.POINTER(ctype)) if arr is not None and arr.size else None


def remesh(pos, tris, feat=None, feat_kind=None, density=None, params=None, control=None):
    """Run aq_remesh. `pos` (n, 3) floats (bbox diagonal 10), `tris` (m, 3) CCW indices, `feat`
    (k, 2) vertex pairs with `feat_kind` (k,) kinds, `density` (n,) multipliers or None. Blocks
    until the engine returns; safe to call from a worker thread. Returns a `Result`."""
    import time
    lib = _require()
    pos = np.ascontiguousarray(np.asarray(pos, np.float64).reshape(-1, 3))
    tris = np.ascontiguousarray(np.asarray(tris, np.uint32).reshape(-1, 3))
    if feat is not None and len(feat):
        feat = np.ascontiguousarray(np.asarray(feat, np.uint32).reshape(-1, 2))
        feat_kind = np.ascontiguousarray(np.asarray(feat_kind, np.uint8).reshape(-1))
        if len(feat_kind) != len(feat):
            raise ValueError("feat and feat_kind lengths differ")
    else:
        feat = feat_kind = None
    if density is not None:
        density = np.ascontiguousarray(np.asarray(density, np.float32).reshape(-1))
    if params is None:
        params = default_params()
    params.struct_size = C.sizeof(AqParams)

    mesh_in = AqMeshIn()
    mesh_in.n_verts = len(pos)
    mesh_in.pos = _ptr(pos, C.c_double)
    mesh_in.n_tris = len(tris)
    mesh_in.tris = _ptr(tris, C.c_uint32)
    mesh_in.n_feat = 0 if feat is None else len(feat)
    mesh_in.feat = _ptr(feat, C.c_uint32)
    mesh_in.feat_kind = _ptr(feat_kind, C.c_uint8)
    mesh_in.density = _ptr(density, C.c_float)

    ctl = control.raw if control is not None else None
    out = C.POINTER(AqResult)()
    t0 = time.perf_counter()
    lib.aq_remesh(C.byref(mesh_in), C.byref(params), C.byref(ctl) if ctl is not None else None,
                  C.byref(out))
    res = Result()
    res.seconds = time.perf_counter() - t0
    if not out:
        res.status = INVALID_INPUT
        res.error = "the engine returned no result"
        return res
    try:
        r = out.contents
        res.status = int(r.status)
        nv, nf = int(r.n_verts), int(r.n_faces)
        res.pos = _copy(r.pos, 3 * nv, np.float64).reshape(-1, 3)
        res.face_offsets = _copy(r.face_offsets, nf + 1, np.uint32)
        if len(res.face_offsets) != nf + 1:
            res.face_offsets = np.zeros(1, np.uint32)
        res.face_verts = _copy(r.face_verts, int(res.face_offsets[-1]), np.uint32)
        res.vflag = _copy(r.vflag, nv, np.uint32)
        res.face_comp = _copy(r.face_comp, nf, np.uint32)
        res.comp_status = _copy(r.comp_status, int(r.n_comp), np.uint8)
        res.failed_tris = _copy(r.failed_tris, int(r.n_failed_tris), np.uint32)
        text = _cstr(r.report_json)
        try:
            res.report = json.loads(text) if text else {}
        except ValueError:
            res.report = {"raw": text}
        res.error = _cstr(r.error)
    finally:
        lib.aq_free(out)
    return res


class Worker(threading.Thread):
    """Runs `remesh` off the main thread. The UI polls `control` and `done`."""

    def __init__(self, **kwargs):
        super().__init__(name="bremesher-engine", daemon=True)
        self.kwargs = kwargs
        self.control = kwargs.setdefault("control", Control())
        self.result = None
        self.exc = None

    def run(self):
        try:
            self.result = remesh(**self.kwargs)
        except Exception as ex:  # reported on the main thread
            self.exc = ex

    @property
    def done(self):
        return not self.is_alive()

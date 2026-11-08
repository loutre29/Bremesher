/* autoquad_engine.h: C ABI of autoquad_engine.dll (DESIGN.md 8.1, normative).
 *
 * Rules (DESIGN.md 8.2):
 *   - All output memory is allocated by the engine and freed only through aq_free.
 *     Input pointers are borrowed for the duration of the call and never retained.
 *   - aq_remesh returns the same value as (*out)->status. *out is always set, except when
 *     out itself is NULL, which returns 4. No global state: concurrent calls are safe.
 *   - Every exported function catches panics; a panic gives status 5 with its message.
 *   - The engine writes progress_bits (f32 bits) and stage with relaxed atomic stores and
 *     polls cancel. Nothing calls back into the caller.
 *   - The call blocks; run it in a worker thread and poll AqControl.
 *
 * x86_64 layout (pinned by crates/aq_ffi/tests/abi.rs):
 *   sizeof(AqParams) = 72, sizeof(AqMeshIn) = 64, sizeof(AqControl) = 12,
 *   sizeof(AqResult) = 104.
 */
#ifndef AUTOQUAD_ENGINE_H
#define AUTOQUAD_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AQ_ABI_VERSION 1

/* AqResult.status */
#define AQ_OK 0
#define AQ_DEGRADED 1
#define AQ_FAILED 2
#define AQ_CANCELLED 3
#define AQ_INVALID_INPUT 4
#define AQ_PANIC 5

/* AqResult.comp_status */
#define AQ_COMP_FULL 0
#define AQ_COMP_RETRIED 1
#define AQ_COMP_SF 2
#define AQ_COMP_FAILED 3
#define AQ_COMP_DEBRIS_SF 4

/* AqParams.debug_flags (bits 30 and 31 are test-only panic injections) */
#define AQ_DEBUG_DUMP (1u << 0)
#define AQ_DEBUG_FORCE_R1 (1u << 1)
#define AQ_DEBUG_FORCE_SF (1u << 2)

/* vflag bits (DESIGN.md 8.4) */
#define AQ_VFLAG_CHAIN_MASK 0x000FFFFFu /* chain id + 1, 0 = none */
#define AQ_VFLAG_KIND_SHIFT 20          /* 4 bits: FeatureKind of that chain */
#define AQ_VFLAG_CORNER (1u << 24)
#define AQ_VFLAG_CONE (1u << 25)
#define AQ_VFLAG_BOUNDARY (1u << 26)
#define AQ_VFLAG_SF (1u << 27)

typedef struct AqParams {
    uint32_t struct_size;      /* = sizeof(AqParams); mismatch -> status 4 */
    uint32_t target_quads;     /* 5000      Target Quad Count */
    float    adaptive_size;    /* 0.5 [0,1] Adaptive Size (M1 forces 0 internally until WP-20 lands) */
    float    curvature_align;  /* 1.0 [0,4] Curve Strength: alignment to curvature */
    float    flow_smoothness;  /* 1.0 [0.5,3] scale c_s at which curvature is measured; larger = smoother flow */
    float    density_strength; /* 1.0 [0,2] exponent on the density map */
    float    guide_strength;   /* 1.0 [0,4] */
    uint8_t  hard_edges;       /* 2: 0 ignore, 1 soft (field only), 2 crisp (field + iso + arcs) */
    uint8_t  close_loops;      /* 1 */
    uint8_t  allow_collapse;   /* 1 */
    uint8_t  reserved0;        /* 0 */
    uint32_t ring_min;         /* 8 */
    uint32_t max_proxy_tris;   /* 150000 */
    uint32_t threads;          /* 0 = all logical cores */
    float    time_limit_s;     /* 60 */
    uint32_t debug_flags;      /* 0; bit0 dump stages, bit1 force R1, bit2 force SF (tests) */
    uint64_t seed;             /* 0 */
    const char* dump_dir;      /* NULL or UTF-8 path */
} AqParams;

typedef struct AqMeshIn {
    uint32_t n_verts;  const double*   pos;        /* 3*n_verts, bbox diagonal 10 */
    uint32_t n_tris;   const uint32_t* tris;       /* 3*n_tris, CCW */
    uint32_t n_feat;   const uint32_t* feat;       /* 2*n_feat vertex index pairs */
    const uint8_t* feat_kind;                      /* n_feat: 1 Hard, 2 Guide, 3 GuideHard, 5 Symmetry */
    const float*   density;                        /* n_verts multipliers (2 = twice the quads) or NULL */
} AqMeshIn;

typedef struct AqControl {            /* allocated by the caller, polled by the caller */
    volatile int32_t  cancel;         /* caller sets 1 to cancel */
    volatile uint32_t progress_bits;  /* f32 bits in [0,1], written by the engine */
    volatile uint32_t stage;          /* Stage enum, written by the engine */
} AqControl;

/* AqControl.stage values */
enum {
    AQ_STAGE_INGEST = 0, AQ_STAGE_PROXY = 1, AQ_STAGE_CURV = 2, AQ_STAGE_FIELD = 3,
    AQ_STAGE_LOOPS = 4, AQ_STAGE_CUT = 5, AQ_STAGE_PARAM = 6, AQ_STAGE_LAYOUT = 7,
    AQ_STAGE_QUANTIZE = 8, AQ_STAGE_PATCH = 9, AQ_STAGE_POST = 10, AQ_STAGE_FALLBACK = 11,
    AQ_STAGE_ENGINE = 12, AQ_STAGE_FFI = 13
};

typedef struct AqResult {
    int32_t   status;                 /* 0 ok, 1 degraded, 2 failed, 3 cancelled, 4 invalid input, 5 caught panic */
    uint32_t  n_verts;  double*   pos;
    uint32_t  n_faces;  uint32_t* face_offsets;   /* n_faces + 1 */
    uint32_t* face_verts;
    uint32_t* vflag;                  /* n_verts */
    uint32_t* face_comp;              /* n_faces */
    uint32_t  n_comp;   uint8_t*  comp_status;    /* 0 full, 1 retried (R1), 2 SF, 3 failed (QuadriFlow), 4 debris SF */
    uint32_t  n_failed_tris; uint32_t* failed_tris; /* input triangle indices of failed components */
    char*     report_json;            /* UTF-8, NUL-terminated, never NULL */
    char*     error;                  /* UTF-8 message, "" when status 0 */
} AqResult;
/* Buffers with a zero count are NULL, except face_offsets (always n_faces + 1 entries). */

uint32_t aq_abi_version(void);
void     aq_params_default(AqParams* p);
int32_t  aq_remesh(const AqMeshIn* in, const AqParams* p, AqControl* ctl /* nullable */, AqResult** out);
void     aq_free(AqResult* r);           /* NULL-safe; frees every buffer allocated by Rust */

#ifdef __cplusplus
}
#endif

#endif /* AUTOQUAD_ENGINE_H */

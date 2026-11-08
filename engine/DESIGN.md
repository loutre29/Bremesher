# AutoQuad Engine: Final Design (implementation contract)

Status: final architecture, written 2026-10-29 by the lead architect from three proposals (A integer grid, B quantized layout, C free tracing) and three judgments (quality, robustness, buildability). This file is the only specification the implementing agents get. Where it says MUST, a deviation needs the integrator's written approval in `engine/CHANGES.md`.

Typography rule of the project: no em dashes and no en dashes in any file (code, comments, docs, reports). Use colons, commas, parentheses or plain hyphens.

---

## 1. Decision summary

### 1.1 Chosen approach (base B, hardened with A and C)

1. Build an internal isotropic **proxy** mesh (edge length h/3) so the solvers never see the user's triangles.
2. Measure curvature **at the quad scale** (normal-cycle tensor, implicit heat diffusion over a distance of about h). Build a sizing field and normalize it to the target count.
3. Solve a smooth **4-RoSy cross field** (Knoppel power field with curvature alignment and hard feature constraints). Then remove close pole pairs **deterministically**, by editing the matching along a path (C), and finish with a fixed-matching re-solve.
4. Find **closed streamline loops** (limbs, torus rings, eye sockets) and cut the mesh to a disk.
5. Solve a **seamless (not integer) parametrization**. Every transition constraint, feature or boundary iso-line, loop closure (transversal translation = 0) and cone condition is **eliminated exactly** with ±1 pivots (A). The system stays SPD and is solved with sparse Cholesky. Stiffening and local injectivity repair follow. The uv is snapped to a dyadic grid so every transition is bit-exact.
6. Trace an exact **motorcycle layout** in uv. Traces are iso-lines, so every crossing test is an exact comparison or orient2d. The pure combinatorial **T-mesh** is rebuilt from the embedded mesh.
7. **Quantize** arc lengths to integers with a bi-directed min-deviation flow (B): zero arcs are allowed, so ± poles merge and T-offsets collapse at coarse targets. Validity is checked on the combinatorial quad mesh (mesh-free), with lazy cuts, and the exact integer count is targeted by a secant method.
8. Mesh each 4-sided patch as an exact W x H grid: field-aligned least squares, then a barrier symmetric Dirichlet repair, then Floater as a last resort. Every map is checked for bijectivity, and stitching is exact through shared arc samples.
9. Run a global **relaxation across patch borders**, then cleanup and a count check.
10. **Per-component fallback ladder**, owned by the orchestrator and bounded by the time budget: retry with a smoother field, then the engine's own always-succeeding quadder (SF, from C), then Python QuadriFlow for that component.

### 1.2 Why, and what was rejected

The quality and robustness judges both chose B. Its seamless map plus layout quantization keeps rounding noise out of the structure and handles coarse targets (the user's 1k to 5k quad retopo case) by collapsing arcs.

- **A's global MIQ rounding plus QEx is rejected.** Greedy rounding folds near close cones at coarse targets, which produces holes and spurious poles, the "looks like QuadriFlow" artifact. QEx is also the costliest module to make robust.
- **C's 3D tracing without a parametrization is rejected.** The field is not integrable, so opposite patch sides disagree, and the termination rules are heuristic tolerances.
- **Kept from A:** the conventions, exact elimination, feature isolation, dyadic snap, loop-closure constraints, tube cap, h0 = 1 units, and the polled FFI.
- **Kept from C:** the walking skeleton (M0), the SF quadder, deterministic matching-edit cancellation, local projection with a normal check, symmetry push-off, the explicit quality gates, and orchestrator-owned retries.

Inside B, the following are rejected or replaced:

| B element | Replacement or reason |
| --- | --- |
| KKT LDLT solve | Replaced by elimination, because LDLT is fragile with redundant constraints. |
| Cone 1-ring 1-to-3 refinement | Dropped: it changes indices mid-pipeline and creates slivers. Injectivity repair replaces it. |
| `trace_unsafe` 3D stepping | Dropped: it breaks exactness. |
| Crossing snapping to edge endpoints | Dropped: the embedding is exact, so slivers are tolerated. |
| Diffusion-based pair cancellation | Replaced by C's matching edits. |
| Confidence a²/(1+a²) | Replaced, because it vanishes on large cylinders. |
| Proxy at h/4 | Changed to h/3 for cost. |

Also rejected:

| Rejected | Reason |
| --- | --- |
| C's N-gon patterns | Every patch is a 4-corner rectangle by construction. |
| MBO sharpening | The fixed-matching re-solve already gives the angular accuracy. |
| Ray 2008 nonlinear smoothing | Same reason as MBO sharpening. |
| Progress callbacks into Python | Risk of GIL deadlocks. |
| Global `aq_last_error` or `aq_cancel` state | Not re-entrant. |

---

## 2. Pipeline overview

All stages after S2 run **per connected component**, with components in parallel (rayon). Units: after S1 every component is scaled so that the global provisional quad size `h0 = sqrt(A_total / N_target) = 1` ("quad units"), and the output is scaled back.

| Stage | Crate | Input | Output |
| --- | --- | --- | --- |
| S0 ingest | aq_prep | `TriMesh` (diag 10, from Python), feature pairs + kinds, density | `Ingested`: manifold oriented mesh, `FeatureSet`, src maps |
| S1 components and budget | aq_engine | `Ingested` | `ComponentMesh` list, scale factor, per-component N_c, time share |
| S2 proxy | aq_prep | component, source BVH | `Proxy`: isotropic mesh l = h/3, chains with corners and segments, feature isolation done |
| S3 curvature and sizing | aq_curv | proxy | `CurvField` (per vertex/face), `SizeField` (h per vertex/face, normalized to N_c) |
| S4 field | aq_field | proxy, features, curv, size | `CrossField` (theta, authoritative matching p, delta, index4) |
| S5 loops | aq_loops | proxy, field, curv, size | `Vec<LoopCand>` (closed streamlines, dual cycles) |
| S6 cut | aq_cut | proxy, field | `CutGraph` (cut halfedges, arcs, nodes, combed theta and matching) |
| S7 param | aq_param | proxy, features, field, cut, size, loops | `SeamlessParam` (dyadic per-corner uv, exact transitions, chain families) |
| S8 layout | aq_layout | proxy, features, param, loops | `Layout` (W' refined mesh, `TMesh`, arc chains, patch faces) |
| S9 quantize | aq_quantize | `TMesh`, N_c | `QuantResult` (n per arc, `QComb` combinatorial quad mesh) |
| S10 patch | aq_patch | `Layout`, `QuantResult` | `QuadMesh` of the component |
| S11 post | aq_post | `QuadMesh`, proxy | relaxed, cleaned `QuadMesh`, count check |
| S12 verify and merge | aq_engine | component results | final `QuadMesh`, statuses, report |
| SF fallback | aq_fallback | component mesh | all-quad `QuadMesh` (poor flow, always succeeds on manifold input) |

Invariants are checked between every pair of stages (section 7.3). A failed check becomes a typed error, and the error drives the fallback ladder.

---

## 3. Workspace, crates, dependencies, coding rules

### 3.1 Layout

```
engine/
  Cargo.toml                 workspace, [profile.release] panic = "unwind", lto = "thin", codegen-units = 1
  rust-toolchain.toml        channel = "1.99", target x86_64-pc-windows-msvc
  crates/
    aq_core/       contracts, conventions, topology, frames, BVH, projection, Ctl, errors, rng
    aq_testkit/    (dev-dependency only) synthetic meshes, analytic fields/params/TMeshes
    aq_linalg/     sparse matrices, Cholesky wrapper (faer), PCG, Elim, Laplacians
    aq_io/         OBJ read/write, debug dumps
    aq_eval/       metrics (quad_stats.py parity), Hausdorff, loop checker, bisect, mirror-weld
    aq_prep/       S0 ingest, S2 proxy remesh, chain classification, feature isolation
    aq_curv/       S3 curvature at scale, sizing
    aq_field/      S4 cross field, singularity edits, fixed-matching re-solve
    aq_loops/      S5 closed streamline detection
    aq_cut/        S6 cut graph, combing, cut arcs
    aq_param/      S7 seamless parametrization
    aq_layout/     S8 exact uv motorcycle tracing, embedding, T-mesh extraction
    aq_quantize/   S9 Bi-MDF quantization, validity, QComb
    aq_patch/      S10 patch maps, sampling, assembly
    aq_post/       S11 relaxation, cleanup
    aq_fallback/   SF quadder
    aq_engine/     orchestration, components, ladder, budgets, report
    aq_ffi/        cdylib, [lib] name = "autoquad_engine" -> autoquad_engine.dll
    aq_cli/        bin "autoquad" (autoquad.exe)
  eval/suite.toml            per-mesh targets and gates (section 10)
  testdata/ baselines/ renders/ tools/   (existing)
```

### 3.2 Dependency graph

Every contract type lives in `aq_core`. Every stage crate depends only on `aq_core` (plus `aq_linalg` when it solves systems), so each crate can be tested on synthetic contract instances from `aq_testkit` without the upstream crates.

```
aq_core            <- everyone
aq_linalg          <- aq_core
aq_io              <- aq_core
aq_eval            <- aq_core, aq_io
aq_testkit (dev)   <- aq_core, aq_io
aq_prep            <- aq_core, aq_linalg
aq_curv            <- aq_core, aq_linalg
aq_field           <- aq_core, aq_linalg
aq_loops           <- aq_core
aq_cut             <- aq_core
aq_param           <- aq_core, aq_linalg
aq_layout          <- aq_core
aq_quantize        <- aq_core
aq_patch           <- aq_core, aq_linalg
aq_post            <- aq_core
aq_fallback        <- aq_core, aq_prep
aq_engine          <- all stage crates, aq_eval
aq_ffi, aq_cli     <- aq_engine (aq_cli also aq_io, aq_eval)
```

### 3.3 Allowed external crates

The versions are pinned with `=` in the workspace at WP-00, using the latest release available that day, and recorded in `CHANGES.md`.

- `faer` (sparse Cholesky LLT with symbolic/numeric split, dense LU/eigen for small blocks). Only `aq_linalg` imports it.
- `rayon`
- `robust` (Shewchuk orient2d/orient3d)
- `serde` and `serde_json` (report JSON, CLI config)

Nothing else: no remeshing, geometry processing, clap or rand crates. CLI argument parsing is hand-written. The RNG is `aq_core::rng::SplitMix64`.

### 3.4 Coding rules (all crates)

- Edition 2024. Library crates carry `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`; tests may use unwrap.
- No recursion. Use explicit stacks.
- Use `try_reserve` for every buffer larger than 16 MB. A failure returns `ErrKind::Budget`.
- Determinism: same input + params + seed give a bit-identical output for any thread count.
  - Parallel reductions use a fixed chunking.
  - Iteration order is sorted where a hash map would decide it. Use `BTreeMap` or sorted `Vec`, never `HashMap` iteration order.
- Every long loop calls `ctl.check()?` at least every 10 ms of work: remesh iterations, solver iterations, trace queue pops (every 1024), quantization rounds, and patch loops.
- Public functions return `Result<T, AqError>`. A function never panics on user data. `debug_assert!` is allowed.
- Angles are in radians internally. A parameter or report field in degrees ends in `_deg`.

---

## 4. Global conventions (owned by aq_core, binding for every crate)

Each convention exists as a function in `aq_core`. **No crate re-derives these formulas.** The executable tests in 6.1 pin every sign.

**C1 Mesh.** `TriMesh { pos: Vec<[f64;3]>, tris: Vec<[u32;3]> }`. Triangles are CCW seen from outside. The normal is `n_f = normalize((p1-p0) x (p2-p0))`. `NONE = u32::MAX`.

**C2 Halfedges are implicit.**
- `he = 3f + k` runs from `tail = tris[f][k]` to `head = tris[f][(k+1)%3]`. `face(he) = he/3`, `next(he) = 3f + (k+1)%3`, `prev(he) = 3f + (k+2)%3`.
- `twin[he]` is `NONE` on a boundary.
- Corner `c = 3f + k` sits at vertex `tris[f][k]` (the tail of halfedge c). Corner and halfedge ids coincide on purpose.
- **Canonical halfedge:** `he` with `twin == NONE || he < twin`. Quantities defined per edge are computed on the canonical halfedge and derived for the twin by the rules below, never recomputed.

**C3 Fans.**
- `vert_he[v]` is one outgoing halfedge. For a boundary vertex it is the outgoing **boundary** halfedge, which is the CW-most one.
- CCW successor of an outgoing halfedge: `ccw_out(he) = twin(prev(he))`, which is `NONE` at the end of an open fan.
- `H_in(v)` = all interior halfedges whose head is v. Crossing `he` from `face(he)` to `face(twin he)` is one CCW step around `head(he)`.

**C4 Face frame.** `e1 = normalize(p1 - p0)`, `e2 = n x e1` (right-handed), angles CCW about n from e1. `edge_angle(he) = atan2(d.e2, d.e1)` with `d = pos[head] - pos[tail]`, in the frame of `face(he)`.

**C5 Transport (discrete Levi-Civita).**
- For a canonical interior halfedge `he` (f = face(he), g = face(twin)): `kappa[he] = wrap_pi(edge_angle(twin) + pi - edge_angle(he))`, and `kappa[twin] = -kappa[he]` exactly.
- A tangent direction at angle a in f has angle `a + kappa[he]` in g.
- `wrap_pi` maps to (-pi, pi]. Gauss-Bonnet check: `sum_{he in H_in(v)} kappa[he] == K(v) mod 2pi`, where `K(v) = 2pi - sum of corner angles` for an interior vertex.

**C6 Cross field.**
- Per face one real `theta[f]`, the representative of {theta + k pi/2}. It is not restricted to [0, pi/2).
- Branch k of face f is the direction `theta[f] + k pi/2`. The **u-axis** of f is branch 0, `d_f = cos(theta) e1 + sin(theta) e2`, and the v-axis is branch 1, `n x d_f`.

**C7 Matching (authoritative).**
- For a canonical interior he: `p[he] = round((theta[g] - theta[f] - kappa[he]) / (pi/2)) mod 4`, with rounding half away from zero.
- Twin: `p[twin] = (4 - p[he]) mod 4`.
- Meaning: branch k of f maps to branch `(k - p[he]) mod 4` of g.
- Once a stage (S4) has produced p, **p is the source of truth**. Later stages never re-round it from theta. Singularity edits change p directly.

**C8 Residual.** For a canonical he: `delta[he] = theta[g] - theta[f] - kappa[he] - p[he]*pi/2`, which is not wrapped. `delta[twin] = -delta[he]` by definition. When p is freshly rounded, delta lies in [-pi/4, pi/4]. After edits it may be larger.

**C9 Index.**
- Interior vertex: `index4(v) = round((2/pi) * (K(v) + sum_{he in H_in(v)} delta[he]))`, with `|residual| < 1e-6` (otherwise `ErrKind::Invariant`). The index is `I = index4/4`, and the valence of the quad vertex is `4 - index4` (+1 gives a valence 3 pole, -1 gives a valence 5 pole).
- Boundary vertex: let `omega` be the sum of its corner angles. Then `n_c(v) = round((omega - sum_{he in H_in(v)} delta[he]) / (pi/2))` with residual < 1e-6, and `index4(v) = 2 - n_c(v)`. n_c = 2 is a straight boundary, 1 a convex 90 degree corner (index4 +1), 3 a reflex corner.
- **Poincare-Hopf:** `sum_v index4(v) = 4 chi` per component, including boundary vertices. This test is mandatory for every field produced anywhere.
- The index depends only on p and the geometry, not on theta, so a fixed-matching re-solve never changes indices.

**C10 Rotations and transitions.**
- `R(k)` is the CCW rotation by `k pi/2`, `k in Z4`, with exact entries in {0, ±1}. `R(1)(u,v) = (-v, u)`.
- Across a cut halfedge he from f to g, the charts satisfy `phi_g = R(-r[he]) phi_f + t[he]`. Here `r[he]` is the **combed** matching (C12) and `t[he]` is real (seamless, not integer).
- Twin: `r[twin] = (4 - r[he]) mod 4`, `t[twin] = -R(r[he]) t[he]`.
- A non-cut interior halfedge has r = 0 and t = 0 after combing.
- `Transition::{apply, inverse, compose}` are exact in f64 for dyadic inputs.

**C11 Orientation and folds.**
- `J_f` is the 2x2 Jacobian of (u, v) with respect to the face frame (e1, e2): row 0 is grad u, row 1 is grad v.
- The face is valid when `det J_f > eps_det`, with `eps_det = 1e-10 * median(|det J|)`. A face with `det <= eps_det` is "folded".
- Output quads are CCW seen from outside, which follows from det > 0.

**C12 Combing.** S6 chooses an integer `c_f` per face. Then `theta'[f] = theta[f] + c_f pi/2` and `r[he] = (p[he] + c_g - c_f) mod 4`, which is 0 on every dual-tree edge. Downstream crates use only theta' and r. A family "branch k in raw theta" becomes `(k + c_f) mod 4` in combed terms.

**C13 Edge family.** For a halfedge on a feature, boundary or loop chain in face f: family `U` (0) if the edge is parallel to the u-axis of f (|wrap_pi(edge_angle - theta')| < pi/4 or > 3pi/4), else `V` (1). A U-family edge lies on a **v-iso line** (v is constant along it).

**C14 Units and tolerances.**
- The Python side sends meshes with a bounding box diagonal of 10. The engine computes `h0 = sqrt(A_total / N_target)`, scales all positions by `1/h0`, and works in quad units, where the target edge length is 1 and the local size is `h(x) = g(x)` (about 1).
- Absolute tolerances are written in h:

| Quantity | Tolerance |
| --- | --- |
| Geometric eps | 1e-9 |
| Zero-area face | area < 1e-14 * diag_q², where diag_q is the diagonal in quad units |
| uv dyadic quantum | 2^-24 |
| Bound on uv | |uv| < 2^27, otherwise `ErrKind::Numerical` |
| Angle tests | 1e-12 rad |

- Weights are clamped to [1e-6, 1e6] times their mean.

**C15 Determinism and seed.** `seed` drives: the dual tree root (the face of largest area, ties broken by the rng), the loop seed order, and the h0 jitter in the ladder. Nothing else is random.

---

## 5. Shared data contracts (aq_core)

Each type has a `validate(&self, ...) -> Result<(), AqError>` that checks its invariants. It is called by the engine between stages (cheap checks in release, all checks in debug) and by the tests.

```rust
// aq_core::mesh
pub type V2 = [f64; 2]; pub type V3 = [f64; 3]; pub const NONE: u32 = u32::MAX;
pub struct TriMesh { pub pos: Vec<V3>, pub tris: Vec<[u32; 3]> }
pub struct QuadMesh {
    pub pos: Vec<V3>,
    pub face_off: Vec<u32>,      // n_faces + 1 offsets into face_verts (CSR); quads have 4
    pub face_verts: Vec<u32>,
    pub vflag: Vec<u32>,         // section 8.4
    pub face_comp: Vec<u32>,
}

// aq_core::topo
pub struct Topology {
    pub twin: Vec<u32>, pub vert_he: Vec<u32>, pub face_comp: Vec<u32>, pub n_comp: u32,
    pub boundary_loops: Vec<Vec<u32>>,   // boundary halfedges in loop order
    pub is_bnd_vert: Vec<bool>,
}
impl Topology {
    pub fn build(m: &TriMesh) -> Result<Self, AqError>;           // requires manifold + consistent orientation
    pub fn face(h: u32) -> u32; pub fn next(h: u32) -> u32; pub fn prev(h: u32) -> u32;
    pub fn tail(&self, m: &TriMesh, h: u32) -> u32; pub fn head(&self, m: &TriMesh, h: u32) -> u32;
    pub fn is_canonical(&self, h: u32) -> bool;
    pub fn fan_out(&self, v: u32) -> FanIter;                     // outgoing halfedges CCW (C3)
    pub fn incoming(&self, m: &TriMesh, v: u32) -> InIter;        // H_in(v), CCW order
    pub fn euler_char(&self, m: &TriMesh, comp: u32) -> i64;
}

// aq_core::frames
pub struct Frames { pub e1: Vec<V3>, pub e2: Vec<V3>, pub n: Vec<V3>, pub area: Vec<f64>,
                    pub corner_angle: Vec<f64> /* per corner */, pub kappa: Vec<f64> /* per halfedge, 0 on boundary */ }
impl Frames {
    pub fn new(m: &TriMesh, t: &Topology) -> Result<Self, AqError>;
    pub fn edge_angle(&self, m: &TriMesh, h: u32) -> f64;
    pub fn angle_defect(&self, t: &Topology, v: u32) -> f64;      // K(v), or pi - omega on the boundary
    pub fn to_face_2d(&self, f: u32, d: V3) -> V2; pub fn from_face_2d(&self, f: u32, a: V2) -> V3;
}

// aq_core::cross  (C5..C10)
pub fn wrap_pi(x: f64) -> f64;
pub fn rot(k: u8, a: V2) -> V2;
#[derive(Clone, Copy, PartialEq)] pub struct Transition { pub r: u8, pub t: V2 }
impl Transition { pub fn apply(&self, a: V2) -> V2; pub fn inverse(&self) -> Self; pub fn compose(&self, then: &Self) -> Self; }
pub struct CrossField {
    pub theta: Vec<f64>,        // per face
    pub p: Vec<u8>,             // per halfedge, authoritative (0 on boundary)
    pub delta: Vec<f64>,        // per halfedge, C8
    pub index4: Vec<i8>,        // per vertex, C9
    pub hard: Vec<bool>,        // per face: theta fixed by a constraint
}
pub fn matching_from_theta(m: &TriMesh, t: &Topology, fr: &Frames, theta: &[f64]) -> (Vec<u8>, Vec<f64>);
pub fn delta_from_p(m: &TriMesh, t: &Topology, fr: &Frames, theta: &[f64], p: &[u8]) -> Vec<f64>;
pub fn index4_all(m: &TriMesh, t: &Topology, fr: &Frames, delta: &[f64]) -> Result<Vec<i8>, AqError>;
pub fn edge_family(fr: &Frames, m: &TriMesh, h: u32, theta_combed: f64) -> u8;   // C13

// aq_core::features
#[repr(u8)] #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeatureKind { None = 0, Hard = 1, Guide = 2, GuideHard = 3, Boundary = 4, Symmetry = 5 }
pub struct Chain {
    pub hes: Vec<u32>,          // consecutive halfedges, head(hes[i]) == tail(hes[i+1]); boundary chains use the boundary halfedges
    pub kind: FeatureKind, pub closed: bool,
    pub corners: Vec<u32>,      // indices i into hes: a segment starts at tail(hes[i]); sorted; for a closed chain with no corner: empty
}
pub struct FeatureSet {
    pub he_kind: Vec<FeatureKind>,   // per halfedge, both halves of an interior feature edge carry the kind
    pub he_chain: Vec<u32>,          // chain id or NONE
    pub chains: Vec<Chain>,
    pub vert_corner: Vec<bool>,      // feature or boundary corner vertex
}

// aq_core::fields
pub struct CurvField { pub k1: Vec<f64>, pub k2: Vec<f64>,    // per face, |k1| >= |k2|
                       pub phi: Vec<f64>,                      // per face, angle of the k1 direction in the face frame
                       pub conf: Vec<f64>,                     // per face in [0,1]
                       pub kappa_hat: Vec<f64> }               // per vertex, sqrt(k1^2 + k2^2)
pub struct SizeField { pub h_vert: Vec<f64>, pub h_face: Vec<f64>, pub n_pred: f64 }

// aq_core::cut
pub struct CutArc { pub hes: Vec<u32>, /* cut halfedges of side A in path order */ pub nodes: [u32; 2], pub r: u8 /* combed rotation of side A */ }
pub struct CutGraph {
    pub is_cut: Vec<bool>,           // per halfedge (both halves); boundary halfedges false
    pub he_arc: Vec<(u32, i8)>,      // (arc id, +1 side A / -1 side B) or (NONE, 0)
    pub arcs: Vec<CutArc>,
    pub node_verts: Vec<u32>,        // cut-graph vertices of degree != 2, cones, cut/boundary junctions
    pub theta_combed: Vec<f64>, pub r: Vec<u8>, pub comb_offset: Vec<u8>,   // C12
}

// aq_core::loops
pub struct SurfacePoint { pub face: u32, pub bary: V3 }
pub struct LoopCand {
    pub hes: Vec<u32>,               // dual cycle: halfedges crossed in order, ends back in start_face
    pub start_face: u32, pub seed: SurfacePoint,
    pub branch: u8,                  // raw-theta branch (0..3) tangent to the loop at the seed
    pub length: f64, pub ring: bool, // ring = wraps a tube (length < 1.5 * 2pi / mean|k1|)
}

// aq_core::param
pub struct SeamlessParam {
    pub uv: Vec<[V2; 3]>,            // per face corner, dyadic multiples of 2^-24, quad units
    pub trans: Vec<Transition>,      // per halfedge (identity off-cut and on the boundary)
    pub is_cut: Vec<bool>,
    pub cones: Vec<(u32, i8)>,       // vertex, index4
    pub he_family: Vec<u8>,          // per halfedge on chains: 0 U, 1 V, 255 none
    pub loops_exact: Vec<u32>,       // indices of LoopCand whose closure was eliminated exactly
    pub folded: Vec<bool>,           // per face after repair
    pub report: ParamReport,
}

// aq_core::tmesh   (mesh-free; used by aq_layout, aq_quantize, aq_patch)
#[repr(u8)] pub enum NodeKind { Regular = 0, Cone = 1, FeatureCorner = 2, BoundaryCorner = 3, ChainEnd = 4, Seed = 5 }
#[repr(u8)] pub enum ArcKind { Separatrix = 0, SeedTrace = 1, Loop = 2, Feature = 3, Boundary = 4, Symmetry = 5, Guide = 6 }
pub struct TNode { pub kind: NodeKind, pub index4: i8, pub chain: u32 }
pub struct TArc { pub ends: [u32; 2], pub len: f64, pub kind: ArcKind, pub lb: u32, pub weight: f64, pub wrap: bool, pub chain: u32 }
pub struct TPatch { pub sides: [Vec<(u32, bool)>; 4] }   // (arc, reversed); CCW; side0 runs +u in the patch chart, side1 +v, side2 -u, side3 -v
pub struct TMesh { pub nodes: Vec<TNode>, pub arcs: Vec<TArc>, pub patches: Vec<TPatch>,
                   pub rings: Vec<Vec<(u32, bool)>>,      // closed arc cycles that must keep >= 3 segments
                   pub chi: i32 }
pub struct QComb {                                       // combinatorial quantized quad mesh
    pub n_verts: u32,
    pub node_id: Vec<u32>,                               // per TNode
    pub arc_ids: Vec<Vec<u32>>,                          // per arc, n_a + 1 ids from ends[0] to ends[1]
    pub patch_dims: Vec<[u32; 2]>,                       // W, H (0 for a glued zero patch)
    pub patch_ids: Vec<Vec<u32>>,                        // (W+1)*(H+1), row-major, j*(W+1)+i
    pub quads: Vec<[u32; 4]>, pub quad_patch: Vec<u32>,
}

// aq_core::layout
pub struct Layout {
    pub wmesh: TriMesh, pub wtopo: Topology,            // W' = proxy split along all arcs
    pub wuv: Vec<[V2; 3]>, pub wtrans: Vec<Transition>, // param restricted to W' (exact)
    pub parent_face: Vec<u32>,                           // W' face -> proxy face
    pub tmesh: TMesh,
    pub node_vert: Vec<u32>,                             // TNode -> W' vertex
    pub arc_verts: Vec<Vec<u32>>,                        // W' vertex chain ends[0] -> ends[1]
    pub arc_s: Vec<Vec<f64>>,                            // cumulative parametric length along arc_verts
    pub patch_faces: Vec<Vec<u32>>, pub patch_chart_face: Vec<u32>,
    pub vert_chain: Vec<u32>,                            // W' vertex -> feature chain id or NONE
    pub health: LayoutHealth,                            // rectangle mismatch stats
}

// aq_core::ctl, aq_core::error
pub struct Ctl<'a> { /* &AtomicI32 cancel, &AtomicU32 progress bits, &AtomicU32 stage, deadline: Instant */ }
impl Ctl<'_> { pub fn check(&self) -> Result<(), AqError>; pub fn progress(&self, stage: Stage, frac: f32);
               pub fn remaining(&self) -> std::time::Duration; pub fn sub(&self, deadline: std::time::Instant) -> Ctl<'_>; }
#[derive(Clone, Copy, Debug)] pub enum Stage { Ingest, Proxy, Curv, Field, Loops, Cut, Param, Layout, Quantize, Patch, Post, Fallback, Engine, Ffi }
#[derive(Clone, Copy, Debug)] pub enum ErrKind { InvalidInput, Topology, Numerical, Invariant, Infeasible, Cancelled, Timeout, Budget, Internal }
pub struct AqError { pub stage: Stage, pub kind: ErrKind, pub msg: String }

// aq_core::bvh, aq_core::project
pub struct Bvh; impl Bvh { pub fn build(m: &TriMesh) -> Self; pub fn closest(&self, p: V3) -> (SurfacePoint, V3, f64); }
pub struct Projector<'a> { /* source mesh, topology, frames, bvh */ }
impl Projector<'_> {
    // Walks from hint face over a growing neighbourhood (<= 3 rings); accepts the closest point only if
    // n_src . n_ref > 0.5; falls back to the BVH with the same test; otherwise returns None.
    pub fn project(&self, p: V3, hint_face: u32, n_ref: V3) -> Option<(SurfacePoint, V3)>;
}
```

---

## 6. Modules

Each module gives its purpose, algorithm, API (on top of the contracts), errors, complexity and tests. Unless stated otherwise, the test thresholds apply to release builds on synthetic inputs from `aq_testkit`.

### 6.1 aq_core (WP-00)

**Purpose:** conventions C1 to C15, topology, frames, BVH, projection, the contract types and their validators, `Ctl`, errors and rng.

**Algorithms:**
- `Topology::build` sorts the directed edges (tail, head) to find twins.
  - A directed edge that appears twice means inconsistent orientation, which is `ErrKind::Topology`. aq_prep repairs this before calling `build`.
  - Boundary loops follow `next` on boundary halfedges through their fans.
- BVH: median split over centroids, 4 triangles per leaf, closest point with a stack traversal.

**Tests** (numeric thresholds):
- **Euler characteristic:** icosphere(4) χ = 2, torus(R=3, r=1) χ = 0, double torus χ = -2, open cylinder χ = 0 with 2 boundary loops.
- **Topology identities:** `twin(twin h) = h` for every interior h, and next³ = id.
- **Frames:** |e1| - 1 < 1e-12, |e1.e2| < 1e-12, |e1.n| < 1e-12.
- **Transport:**
  - `kappa[twin] == -kappa[he]` bitwise.
  - Planar jittered grid: |kappa| < 1e-12.
  - Noisy icosphere: `sum_{H_in(v)} kappa == K(v) mod 2pi` within 1e-9 for every vertex.
- **Matching:** random theta on a noisy sphere gives `p[twin] == (4-p)%4` and `delta[twin] == -delta` bitwise, with delta in [-pi/4, pi/4].
- **Index and Poincare-Hopf** (sum index4 = 4χ exactly):
  - icosphere with random theta: 8
  - torus: 0
  - double torus: -8
  - open cylinder with a boundary-aligned field: 0
  - planar square with a boundary-aligned field: the 4 corners each have index4 = +1, sum 4
- **Planted planar singularities:** `theta = atan2(y,x)/4` gives index4 = +1 at the center vertex, and `theta = -atan2(y,x)/4` gives -1.
- **Cube corner:** a cube surface with theta aligned to its edges has index4 = +1 at the 8 corners and 0 elsewhere.
- **Transitions:** `compose(T, T.inverse())` is the identity bitwise for dyadic t. The twin formula round-trips exactly.
- **BVH:** the closest point equals brute force within 1e-12 on 10k random queries.
- **Projector:**
  - Smooth sphere with correct hints: agrees with the BVH on at least 99.9% of points.
  - Two parallel planes 0.05 apart, queries offset 0.03 from plane 1 with plane 1's hint and normal: 0 results on plane 2.

### 6.2 aq_testkit (WP-01b, dev-only)

**Generators** (all parameterized and deterministic):
- Plane grids (rotation, jitter) and a planar square disk.
- uv sphere and icosphere, open and closed cylinder (radius R and length L, triangulation jitter), torus (R, r), double torus (implicit, marching tetrahedra then remeshed).
- Cube with sharp edges, hemisphere (a bisected sphere whose cut chain is flagged Symmetry).
- Noise adder (amplitude, wavelength), twisted-cylinder field, and an OBJ loader for testdata.

**Analytic contracts:** these make the downstream crates testable without the upstream ones.
- `analytic_field(mesh, kind)`: constant, circumferential on a cylinder or torus, cube-aligned, or planted ± pair.
- `analytic_cut(mesh)`.
- `plane_param(alpha, s)`.
- `cylinder_param(n_around)` and `torus_param(n, m)`, both on jittered triangulations so that iso-lines cross faces generically. Grid-aligned variants are provided to exercise SoS.
- `cube_sphere_param(n)`: a cube subdivided n x n per face and projected to the sphere. uv is the cube-face coordinates, transitions are quarter rotations plus integer translations, and the 8 cones have index4 = +1. It is an exact seamless param.
- `perturbed_cube_sphere_param(n, amp, seed)`: cone positions moved off integers while staying exactly seamless (built by solving aq_param-like constraints with a tiny private solver in testkit).
- TMesh generators:
  - `tmesh_rect(l0, l1)`
  - `tmesh_grid(rows, cols, t_offsets)` (with T-junctions)
  - `tmesh_cube()`
  - `tmesh_random_bidirected(n_arcs, seed)`: abstract constraint graphs with random signs, including (+,+) and (-,-) edges.

### 6.3 aq_linalg (WP-02)

**API:**
```rust
pub struct Triplets { /* push(i, j, v) */ } pub struct Csc;  // symmetric lower storage
pub struct Chol { /* faer symbolic + numeric */ }
impl Chol {
    pub fn analyze(a: &Csc) -> Result<Symbolic, AqError>;           // AMD ordering
    pub fn factor(sym: &Symbolic, a: &Csc) -> Result<Self, AqError>; // numeric; reuse sym across same pattern
    pub fn solve(&self, b: &mut [f64]); pub fn solve_multi(&self, b: &mut [Vec<f64>]);
}
pub fn pcg(a: &Csc, b: &[f64], x: &mut [f64], pre: &dyn Fn(&[f64], &mut [f64]), tol: f64, max_it: usize) -> usize;
pub fn cotan_laplacian(m: &TriMesh, t: &Topology, fr: &Frames, edge_scale: Option<&[f64]>) -> Csc; // clamped weights
pub fn lumped_mass(m: &TriMesh, t: &Topology, fr: &Frames) -> Vec<f64>;
pub fn hermitian_to_real(...) -> Csc;   // complex n x n -> real 2n x 2n

// Exact linear elimination of constraints with integer coefficients (used by aq_param)
pub struct Elim { /* n_vars; per var: Free(idx) | Expr(sparse Vec<(var, i64)>) */ }
pub enum RowOutcome { Eliminated { pivot: u32 }, Redundant, Inconsistent, NoUnitPivot }
impl Elim {
    pub fn new(n_vars: usize) -> Self;
    pub fn add_row(&mut self, row: &[(u32, i64)], prefer: &dyn Fn(u32) -> i32) -> RowOutcome; // homogeneous rows: sum c_i x_i = 0
    pub fn pin(&mut self, var: u32, value_zero: bool);                                       // gauge
    pub fn expr(&self, var: u32) -> &[(u32, i64)];                                           // in terms of free vars
    pub fn n_free(&self) -> usize; pub fn free_index(&self, var: u32) -> Option<u32>;
}
```

**Elim algorithm:**
1. Substitute the current expressions into the row. Merge equal variables and drop zero coefficients.
2. An empty row is `Redundant`.
3. Otherwise choose the pivot: a free variable with coefficient ±1, maximizing `prefer` (aq_param prefers translation variables over vertex variables, then the newest variable), with ties broken by the lowest id.
4. Express the pivot as `x_p = -sign * sum_{i != p} c_i x_i` and substitute it into every existing expression that references p (reverse index).
5. If no coefficient is ±1, return `NoUnitPivot`. The caller then treats the row as a penalty.

Every coefficient stays an integer. Every constraint in this design is homogeneous, so `Inconsistent` can only arise from a pinned gauge, and it is reported.

**Errors:** a non-SPD factorization gives `ErrKind::Numerical` with the column index.

**Tests:**
- Random SPD 10k: residual < 1e-10.
- Laplacian plus mass on 200k vertices: analyze and factor < 1.5 s on 8 threads; numeric refactor with the same symbolic < 60% of the full time.
- `hermitian_to_real` solve equals a dense complex reference within 1e-10 (n = 200).
- PCG on 2D Poisson converges to 1e-10 within 500 iterations with a Jacobi preconditioner.
- Elim:
  - The chain `x1 = x2, x2 = -x3, x3 + x4 - x5 = 0` matches the hand-computed expressions exactly.
  - A duplicate row gives `Redundant`.
  - A row `2 x1 + 2 x2 = 0` gives `NoUnitPivot`.
  - 10k random ±1 rows keep integer coefficients with max |coeff| < 2^20.
  - Expressions satisfy every accepted row exactly in i64 arithmetic.
- Cotan Laplacian: `L * 1 == 0` within 1e-12.

### 6.4 aq_prep (WP-03): S0 ingest, S2 proxy, chains, feature isolation

**API:**
```rust
pub struct IngestReport { pub dropped_faces: u32, pub collapsed_faces: u32, pub split_edges: u32, pub split_verts: u32,
                          pub flipped_faces: u32, pub dropped_features: u32, pub rerouted_features: u32, pub components: u32 }
pub struct Ingested { pub mesh: TriMesh, pub topo: Topology, pub feats: FeatureSet,
                      pub vert_src: Vec<u32>, pub face_src: Vec<u32>, pub report: IngestReport }
pub fn ingest(mesh: TriMesh, feat_pairs: &[[u32; 2]], feat_kind: &[u8], ctl: &Ctl) -> Result<Ingested, AqError>;
pub struct ComponentMesh { pub mesh: TriMesh, pub topo: Topology, pub feats: FeatureSet, pub face_src: Vec<u32>,
                           pub vert_src: Vec<u32>, pub area: f64, pub chi: i64, pub closed: bool }
pub fn split_components(ing: &Ingested) -> Vec<ComponentMesh>;
pub struct ProxyParams { pub iters: u32 /*5*/, pub max_tris: u32, pub corner_deg: f64 /*50*/, pub sym_corner_deg: f64 /*70*/,
                         pub hard_mode: u8 }
pub struct Proxy { pub mesh: TriMesh, pub topo: Topology, pub frames: Frames, pub feats: FeatureSet,
                   pub src_face: Vec<u32>, pub src_bary: Vec<V3>, pub density: Vec<f64> }
pub fn build_proxy(src: &ComponentMesh, proj: &Projector, len: &dyn Fn(V3) -> f64, p: &ProxyParams, ctl: &Ctl) -> Result<Proxy, AqError>;
pub fn isolate_features(px: &mut Proxy) -> Result<u32, AqError>;
pub fn classify_chains(px: &mut Proxy, h_vert: &[f64], p: &ProxyParams);
pub fn transfer_density(px: &mut Proxy, src: &ComponentMesh, density: Option<&[f32]>, strength: f64);
```

**Ingest:**
1. **Validate.** Return `InvalidInput` for:
   - a NaN or Inf position,
   - an index out of range,
   - fewer than 1 triangle,
   - n_feat pairs with an out-of-range vertex.
2. **Collapse degenerate faces.** A face with area < 1e-14 diag² or a repeated index is collapsed topologically: merge its shortest edge's endpoints when the link condition holds, otherwise delete it. Repeat. Deleting must never open a hole in a closed fan unless no collapse is possible, and the report counts it.
3. **Split non-manifold edges** (3 or more faces): keep the two faces whose normals are most consistent, and detach the others by duplicating the vertices, which turns them into boundaries.
4. **Split non-manifold vertices** so that each fan becomes its own vertex.
5. **Orient each component** by BFS from the face with the largest area: flip the faces whose shared edge appears in the same direction twice. On a closed component, flip the whole component when its signed volume is negative.
6. **Map features.** A vertex pair that is an edge marks both halfedges. A pair that is not an edge is routed along the shortest edge path between its endpoints, limited to 2 rings; otherwise it is dropped and counted. Boundary edges always get `Boundary`, unless they carry `Symmetry` (a symmetry kind given on a boundary edge wins).

**Proxy remesh** (Botsch-Kobbelt 2004), with `len(x) = h(x)/3`:
- **Budget.** The expected proxy triangle count is about `20.8 * N_c`. If it exceeds `max_tris` (the component's share of `max_proxy_tris`, default 150000 in total), every length is scaled by `sqrt(count / max_tris)`. The minimum is 2000 triangles per component.
- **Iteration** (`iters` = 5):
  1. **Split** edges longer than 4/3 len.
  2. **Collapse** edges shorter than 4/5 len toward the better endpoint. A collapse is rejected when it:
     - violates the link condition,
     - creates an edge longer than 4/3 len,
     - flips any face normal by more than 60 degrees,
     - drops a minimum angle below 10 degrees,
     - merges a vertex on one chain into a vertex not on that chain,
     - moves a corner vertex,
     - or collapses a chain edge into a non-chain edge.
  3. **Flip** toward valence 6 (4 on the boundary) unless the edge is a chain or boundary edge, or the flip creates a normal deviation of more than 30 degrees.
  4. **Smooth tangentially**: area-weighted centroid with lambda 0.3, projected onto the tangent plane. Chain vertices move only along their chain (1D smoothing). Corners are frozen.
  5. **Project** with `Projector::project` (hint = the vertex's src_face, n_ref = the current vertex normal). A failed projection keeps the previous position.
- **After each iteration** verify that the proxy is manifold, that every component's χ equals the input's, and that the chains are intact (same count, same endpoints). On a violation, revert to the previous iteration's mesh and stop. If the first iteration already fails, return `ErrKind::Topology`, and the engine goes to SF.
- **Tracking.** Every proxy vertex keeps (src_face, bary) for density transfer and later projection.

**Feature isolation:** repeat until stable, at most 4 passes.
- Split at its midpoint every non-chain edge whose two endpoints both lie on chains (bridge edge), then re-project.
- Split every face with 2 or more chain edges by splitting its non-chain edge. A face with 3 chain edges gets its centroid inserted.

Postcondition: every face has at most 1 chain edge, and no non-chain edge joins two chain vertices.

**Chain classification:**
- Chains are maximal sequences of feature edges between vertices of chain degree other than 2. A closed loop forms one chain.
- Corners:
  - every chain vertex of chain degree other than 2;
  - every point where a Symmetry chain meets another kind;
  - every vertex where the windowed turning angle (the angle between the mean tangents over 0.5h on each side) is at least `corner_deg` (50 degrees) for Hard, Guide, GuideHard and Boundary, or at least `sym_corner_deg` (70 degrees) for Symmetry.
- Segments run between corners.
- `hard_mode` = 0 drops Hard chains except Symmetry and Boundary; 1 keeps Hard chains as soft guides (kind Guide).

**Complexity:** O(n log n). Targets: 300k to 100k triangles in < 4 s on 8 threads, and ingest of 300k in < 0.5 s.

**Tests:**
- **Ingest repairs:**
  - Bowtie vertex: split count 1, manifold output.
  - An edge with 3 faces: manifold output with 1 extra boundary loop.
  - A sliver of area 1e-16 inside a closed mesh: collapsed, χ unchanged, 0 holes.
- **Invalid input:** NaN, an out-of-range index and an empty mesh each give `InvalidInput`.
- **Orientation:** randomly flipped faces on a sphere are re-oriented with positive volume.
- **Feature mapping:** a non-edge pair at distance 2 rings is rerouted. One at 5 rings is dropped and counted.
- **Proxy quality (noisy sphere, len 0.1):**
  - At least 95% of edges within [0.8, 1.33] len.
  - At least 98% of faces with a minimum angle of at least 25 degrees, and none below 5 degrees.
  - Hausdorff to the source at most 0.1 len.
  - χ preserved.
- **Cube with 12 Hard chains:** 12 chains, the 8 corners exact (1e-12), every chain vertex on its source edge within 1e-9.
- **Two parallel planes 0.05 apart, len 0.1:** 0 vertices cross sheets.
- **Isolation:** after `isolate_features` on the cube and on random chain networks, 0 bridge edges and 0 faces with 2 chain edges.
- **Chain classification:**
  - Hemisphere Symmetry loop: 0 corners.
  - A 45 degree polyline kink on a Hard chain: no corner. A 90 degree kink: 1 corner.
- **Performance:** noisy_blob x4 (328k triangles) to l for 5k quads in < 4 s.
- **Determinism:** 3 runs with thread counts 1, 4 and 16 are bitwise equal.

### 6.5 aq_curv (WP-07): S3 curvature at the quad scale, sizing

**Curvature:**
1. Per edge, the signed dihedral `beta_e = atan2((n_f x n_g) . e_hat, n_f . n_g)`, with `e_hat` along the canonical halfedge in f. It is positive for convex edges.
2. Per vertex, the normal-cycle tensor `T0(v) = (1/A_v) sum_{e at v} (1/2) beta_e |e| e_hat e_hat^T`, an ambient symmetric 3x3 tensor (6 scalars). `A_v` is the barycentric area.
3. Diffuse: `(M + L_t) T = M T0` for the 6 components with one factorization. L_t is the cotan Laplacian with each edge weight multiplied by `t_e = (c_s * hbar_e)^2`, where `c_s = flow_smoothness` (default 1.0) and hbar_e is the mean h of the endpoints (the uniform h0 = 1 in pass 1).
4. Per face, average the 3 vertex tensors, project to the face plane (`T_f = P T P`), and rotate by 90 degrees in the plane: `C_f = J T_f J^T`.
   - The eigenvectors of C_f are the principal directions, and its eigenvalues are the signed principal curvatures. This rotation is required because the dominant eigenvector of the normal-cycle tensor is the **minimum** curvature direction. The cylinder test locks it.
   - Order the eigenvalues so that |k1| >= |k2|. phi_f is the angle of the k1 eigenvector in the face frame.
5. Confidence: `conf_f = smoothstep(h_f |k1 - k2|; 0.02, 0.25) * |k1 - k2| / (|k1| + |k2| + 1e-12)`, where `smoothstep(x; a, b) = s^2 (3 - 2s)` with `s = clamp((x-a)/(b-a), 0, 1)`.
   - The relative anisotropy |k1-k2|/(|k1|+|k2|) is 1 on cylinders and saddles and 0 on umbilics.
   - The gate rejects turns of less than about 0.02 rad per quad, which is noise. For a cylinder of radius 10h the gate gives 0.28.
   - This deliberately departs from A's |k1|-|k2|, which zeroes saddles.
6. `kappa_hat(v) = sqrt(k1^2 + k2^2)` from the vertex tensor.

**Sizing** (all in quad units):
- **Base:** `g(v) = clamp((1 + kappa_hat/kappa_ref)^(-alpha) * rho(v)^(-1/2), 1/4, 4)`.
  - `alpha = 0.7 * adaptive_size`.
  - kappa_ref is the area-weighted median of kappa_hat.
  - rho is the density multiplier raised to `density_strength`, with a default of 1. M1 runs with adaptive = 0 and rho = 1 (section 11).
- **Tube cap:** where `conf > 0.5` and the relative anisotropy exceeds 0.6 (cylinder-like), `g <= 2pi / (ring_min * |k1|)`, with ring_min = 8.
- **Thickness cap (M2):** a shape-diameter estimate d(v): 5 rays in a 30 degree cone around -n, using the median of hits with a backface test. Then `g <= pi d / ring_min`.
- **Gradation** (Persson): enforce `log g(y) <= log g(x) + grad * |xy| / g(x)` with grad = 0.25. It is done with a Dijkstra-style pass in increasing log g over proxy edges.
- **Normalization:** `s = sqrt(sum_f A_f / g_f^2 / N_c)` and `h = s * g`, so `n_pred = sum A/h^2 = N_c` exactly. Apply the caps once more after normalization, then renormalize; the result is exact by construction.

Two passes: pass 1 with uniform t; pass 2 (only when adaptive > 0 or a density is present) with t from pass 1's h. M2 adds an adaptive proxy re-remesh between the passes (3 iterations with len = h/3).

**API:**
```rust
pub struct CurvParams { pub scale: f64, pub conf_lo: f64, pub conf_hi: f64 }
pub fn curvature(px: &Proxy, h_vert: &[f64], p: &CurvParams, ctl: &Ctl) -> Result<CurvField, AqError>;
pub struct SizingParams { pub n_target: f64, pub adaptive: f64, pub ring_min: u32, pub grad: f64, pub thickness: bool }
pub fn sizing(px: &Proxy, c: &CurvField, p: &SizingParams, ctl: &Ctl) -> Result<SizeField, AqError>;
```

**Complexity:** one sparse factorization of V unknowns and 6 right-hand sides, < 1 s at 100k triangles.

**Tests:**
- **Sphere of radius 2:** k1 and k2 equal 0.5 within 3%, conf < 0.05.
- **Cylinders with R/h = 2, 5 and 10** (h = 1, so R = 2, 5 and 10):
  - |k1| = 1/R within 3%, and |k2| < 0.03/R, measured more than R from the ends.
  - The k1 direction is circumferential within 3 degrees on at least 95% of those faces.
  - conf is at least 0.95, 0.85 and 0.25 respectively.
- **Noisy cylinder** (amplitude 0.02R, wavelength 0.2h): direction error < 5 degrees on at least 90% of faces.
- **Saddle z = (x^2 - y^2)/4:** conf > 0.5 near the origin.
- **Sizing:**
  - adaptive = 0 with no density: h uniform within 1e-12, and `sum A/h^2 = N` within 1e-9.
  - The gradation inequality holds within 1e-9 on every edge.
  - Thin cylinder R = 0.5: `2piR / h >= 8`.
  - Sphere with a bump, adaptive = 1 (M2): quad density (1/h^2) on the bump at least 2x that of the flat region.
- **Performance:** 100k triangles, both passes, < 1.5 s.

### 6.6 aq_field (WP-08): S4 cross field

**Step 1, constraints:**
- **Hard faces:** every face carrying a chain edge (after isolation there is exactly one) of kind Hard, GuideHard, Boundary or Symmetry (Hard and GuideHard only when hard_mode = 2).
- **Target direction psi_f:** the **segment-smoothed tangent** of that chain at the face. It is the mean of the edge directions over a window of 0.5h along the chain, measured in the face frame, and **never smoothed across a corner**.
  - On a straight segment psi_f is the edge direction. Along a gently curved segment the field rotates smoothly without changing family. At a corner the family switch is abrupt.
  - This makes the field's family structure agree with the corners of 6.4: a turn below the corner threshold never switches family.
- **Constraint value:** `z_f = exp(4i psi_f)`. These faces are eliminated from the system.
- **Guide faces** (kind Guide, faces within 0.5h of a Guide chain, M2): a soft target `q = exp(4i psi)` with confidence 1 and weight `lambda_guide = guide_strength * 20 / h^2`.

**Step 2, Knoppel power field.** Unknowns are `z_f in C` for the free faces. Minimize

`E(z) = sum_{interior canonical he} w_e |z_g - exp(4i kappa~_he) z_f|^2 + sum_f lambda_f A_f conf_f |z_f - q_f|^2`

- `w_e = |e|^2 / (A_f + A_g)`, clamped.
- `q_f = exp(4i phi_f)`.
- `lambda_f = curvature_align / h_f^2`, with a default curvature_align of 1.0. Smoothness is dimensionless and alignment scales with area times 1/h², so the field's correlation length is about one quad.
- `kappa~` is kappa in M1. M2 adds the size-consistent connection `kappa~_he = kappa_he + beta_c * w_cot(e) * (phi_s(b) - phi_s(a))`.
  - `phi_s = -log(h/h0)` at the edge endpoints a = tail and b = head of he.
  - `w_cot = (cot alpha + cot beta)/2`.
  - `beta_c = size_consistency` (default 1 in M2).
  - Its sign is pinned by the test below. The field's singularities then appear where density changes demand pole pairs. Matchings are computed with kappa~, while delta and index4 use kappa (C8, C9). The index is unchanged by construction, because the correction telescopes.
- Solve the normal equations as a real 2n system with Cholesky.
- **Degenerate path:** when there are no hard faces and the area-weighted mean conf is below 0.01, use the smallest eigenvector of the connection Laplacian by inverse power iteration: 8 iterations, shift 1e-8 * mean diagonal, a deterministic start from seed, normalized in the mass norm, with the factorization reused.
- Then `theta_f = arg(z_f)/4`. A face with `|z_f| < 1e-9 * mean|z|` takes the mean angle of its neighbours (transported). p comes from `matching_from_theta` with kappa~, delta and index4 from aq_core.

**Step 3, split high indices:**
- At an interior vertex with |index4| >= 2 that is not a feature corner, move one quarter of sign(index4) at a time to a 1-ring neighbour w with a **path edit** (Step 4).
- w is the non-chain neighbour that maximizes the geodesic distance to other cones. Repeat until every non-corner interior vertex has |index4| <= 1.
- Corner vertices keep any index.

**Step 4, path edit (telescoping).** This moves one quarter of index from v0 to vk along the vertex path v0, ..., vk.
- For each i, take the halfedge he_i with `tail = v_i` and `head = v_{i+1}`. Apply `p[he_i] = (p[he_i] + s) mod 4` and `p[twin] = (4 - p[he_i]) mod 4`, with s = +1, then recompute delta for that edge.
- **Effect:** delta[he_i] drops by s·pi/2 and delta[twin] rises by s·pi/2. index4(v_{i+1}) changes by -s and index4(v_i) by +s, so only the endpoints change: index4(v0) += s and index4(vk) -= s.
- Usage:
  - To move a quarter of positive index from a cone at a to vertex b, call `edit_path([a, ..., b], -1)`: index4(a) -= 1 and index4(b) += 1.
  - To cancel a pair (a with index4 -1, b with +1), call `edit_path([a, ..., b], +1)`: both reach 0.
  - The unit test on a planted cone pins these signs.
- Paths never touch chain vertices (Dijkstra restricted to non-chain interior vertices).

**Step 5, pair cancellation:**
1. Candidate pairs are a +1 and a -1 cone with geodesic distance (Dijkstra on edges) `d < D_c`.
   - `D_c = cancel_dist * hbar`, with cancel_dist = 1.2 by default.
   - D_c becomes 2.5 hbar when the area-weighted mean conf within a disc of radius d around the midpoint is below 0.2.
2. Match greedily by increasing d, with each cone in at most one pair.
3. For each pair, edit the path from the -1 cone to the +1 cone so that both reach 0.
4. After all edits, run one fixed-matching re-solve (Step 7).
5. On every edited path, check that each dual edge has |delta| <= pi/3. Pairs that violate it are reverted, with the opposite edit, and re-solved.
6. Repeat up to max_cancel_rounds (3), including pairs that become close in later rounds.

**Step 6, push-off (symmetry and features):**
- A cone within `sym_clearance` (0.5h) of a Symmetry chain, or within 0.35h of any other chain, moves along a path edit to the nearest non-chain vertex at distance >= 0.75h from all chains. The distance is geodesic on edges.
- M2: a cone within 0.5h of an interior feature corner merges into that corner when the merged |index4| <= 1.

**Step 7, fixed-matching re-solve** (always runs last, and after every edit batch):
- Real unknowns theta_f for the free faces. With p fixed, minimize `sum_e w_e (theta_g - theta_f - kappa~_e - p_e pi/2)^2 + sum_f lambda_f A_f conf_f (theta_f - phihat_f)^2`.
- `phihat_f = phi_f + k pi/2` is the branch nearest the current theta_f. Hard faces are fixed at `psi_f + k pi/2`, again the branch nearest their current theta.
- If there are no hard faces and no alignment, pin one face.
- Cholesky on F unknowns. Recompute delta. index4 is unchanged; assert it.

**M2 pole sanity cap:** S_ref is the cone count of a pure smooth run (alignment 0). When the count exceeds `max(1.5 S_ref, S_ref + 8)`, report `FieldReport.excess_poles` so that the orchestrator can rerun with curvature_scale x1.5 and curvature_align x0.7. Also M2: same-sign repulsion. Two same-sign cones closer than 0.7h move apart: the lower-confidence one moves to the 1-ring vertex that minimizes the local fixed-matching energy, at most 5 moves.

**API:**
```rust
pub struct FieldParams { pub align: f64, pub guide_w: f64, pub hard_mode: u8, pub cancel_dist: f64, pub cancel_dist_lowconf: f64,
                         pub max_cancel_rounds: u32, pub sym_clearance: f64, pub size_consistency: f64, pub seed: u64 }
pub struct FieldReport { pub cones_initial: u32, pub cones_final: u32, pub split: u32, pub cancelled: u32, pub reverted: u32,
                         pub pushed: u32, pub max_abs_delta: f64, pub eigen_path: bool, pub excess_poles: bool }
pub fn solve_field(px: &Proxy, c: &CurvField, s: &SizeField, p: &FieldParams, ctl: &Ctl) -> Result<(CrossField, FieldReport), AqError>;
pub fn knoppel(px: &Proxy, c: &CurvField, s: &SizeField, p: &FieldParams, ctl: &Ctl) -> Result<Vec<f64>, AqError>;
pub fn edit_path(px: &Proxy, f: &mut CrossField, path: &[u32], s: i8);
pub fn resolve_fixed_matching(px: &Proxy, c: &CurvField, s: &SizeField, f: &mut CrossField, p: &FieldParams) -> Result<(), AqError>;
```

**Errors:** a Poincare-Hopf failure or an index residual is `Invariant`. A factorization failure is `Numerical`. The engine catches either.

**Complexity:** at 100k faces, one factorization (2F) for the Knoppel solve, plus one F-size factorization for each re-solve, at most 4 of them (pattern reused). Total < 3 s.

**Tests:**
- **Poincare-Hopf** exact on every testkit generator and on the 12 testdata meshes (through aq_prep).
- **Planar square with boundary:** 0 interior cones, constrained faces exact (1e-12).
- **Open cylinder and torus:** 0 cones, field within 2 degrees of the circles on at least 95% of faces.
- **Cube with Hard chains:** exactly 8 cones of index4 +1, each at a corner vertex.
- **Icosphere** (eigen path): exactly 8 cones of +1, none negative, minimum pairwise geodesic distance at least 0.85R.
- **Hemisphere** (Symmetry loop with no corners): exactly 4 cones of +1, each at distance >= 0.5h from the chain.
- **Planted pairs:** ± pairs at 0.5h and at 1.0h (50 random placements each on a plane and on a sphere) are removed in 100% of cases. Afterwards |delta| <= pi/3 and the energy increase is < 5%.
- **Planted +2 vertex:** split into two +1 cones.
- **Index stability:** `resolve_fixed_matching` leaves p and index4 bitwise unchanged.
- **Noise robustness:** noisy blob at N = 5000 has at most 1.5x the cone count of the same blob without noise.
- **Edit sign:** `edit_path` with s = +1 from a to b increases index4(a) by 1 and decreases index4(b) by 1.
- **M2 size-consistent connection:** on a plane disc whose density doubles inside a central disc, 2 to 8 cone pairs appear on the transition ring, with +1 cones on the dense side. This pins the sign of beta_c. With beta_c = 0, 0 cones.

### 6.7 aq_loops (WP-09): S5 closed streamline loops

**Purpose:** find closed field lines (rings around limbs and tori, loops around eye sockets and the mouth). aq_param uses them to force zero transversal translation, which kills helices, and aq_layout pre-inserts them as exact rings.

**Algorithm:**
- **Seeds:** deterministic greedy farthest-point sampling over proxy vertices, keeping vertices whose incident faces have mean conf >= 0.3 and spacing >= `stride` (2h). The order is shuffled by seed.
- From each seed, trace branch 0 and branch 1 (forward only, since a closed loop is found from either direction).
- **Streamline tracing** (Ray-Sokolov style):
  - Inside face f, follow the straight line along branch k: `d = cos(theta_f + k pi/2) e1 + sin(...) e2`. The exit edge is found in 2D face coordinates (the smallest positive ray parameter, the entry edge excluded).
  - Cross to g = face(twin(he_exit)). The new branch is `k' = (k - p[he_exit]) mod 4`; it comes from the matching, never from a nearest-dot heuristic. Re-express the point in g's barycentric coordinates.
  - An exact vertex hit is nudged by 1e-9 h toward the CCW side.
  - Stop at a boundary, a hard chain, a cone face (a face incident to a cone vertex), or a length above `max_len` (40h).
- **Closure:** each time the trace enters the seed face or a face of its 1-ring, compute the distance from the seed to the trace segment. If it is below `close_tol` (0.3h) and the current branch equals the starting branch (mod 4), the loop is closed.
  - The closing dual path back to the seed face is the shortest dual path inside the seed's 1-ring.
  - The total geodesic turning must be below 60 degrees beyond the holonomy. This rejects loops that spiral around cones.
- **Ring flag:** `ring = length < 1.5 * 2pi / mean_along(|k1|)`.
- **Clustering:** a new loop is discarded if, along at least 80% of its length, it stays within `cluster` (6h) of an accepted loop of the same combed family. Rings are preferred: process rings first, then shortest first.

**API:**
```rust
pub struct LoopParams { pub stride: f64, pub max_len: f64, pub close_tol: f64, pub min_conf: f64, pub cluster: f64, pub seed: u64 }
pub fn find_loops(px: &Proxy, f: &CrossField, c: &CurvField, s: &SizeField, p: &LoopParams, ctl: &Ctl) -> Result<Vec<LoopCand>, AqError>;
pub fn trace_streamline(px: &Proxy, f: &CrossField, from: SurfacePoint, branch: u8, max_len: f64) -> Streamline;
```

**Complexity:** about N/4 seeds x 2 branches x up to 120 faces each, < 0.5 s at 100k faces.

**Tests:**
- **Torus with circumferential field:** both families found, closure error < 0.05h, at least 1 loop per family per 6h of extent, `ring` true for the meridians.
- **Open and closed cylinders:** circumferential rings found.
- **Twisted cylinder field** (pitch 2h): no loop accepted. Pitch 0.2h: accepted (closure within tol).
- **Sphere with the 8-cone field:** no loop passes through a cone face; accepted loops have turning < 60 degrees.
- **Noisy cylinder:** at least 1 ring per 6h of length.
- **Clustering:** no two accepted loops of the same family within 6h over 80% of their length.
- **Determinism** with a fixed seed.

### 6.8 aq_cut (WP-09): S6 cut graph and combing

**Algorithm:**
1. **Dual spanning tree:** Dijkstra over faces from the root (the face of largest area, ties broken by seed), with cost equal to the dual edge length. Interior edges only.
2. **Initial cut:** all interior edges not crossed by the tree.
3. **Prune:** repeatedly remove a cut edge whose endpoint has cut degree 1, provided that endpoint is not a cone (index4 != 0), not a boundary vertex, and not a feature corner.
4. **Check:** the cut mesh (faces glued along non-cut edges) must be a topological disk per component: `V_cut - E_cut + F = 1`, counting duplicated vertices. Otherwise return `Invariant`.
5. **Combing:** BFS on the dual tree from the root. `c_root = 0` and `c_g = (c_f - p[he]) mod 4` across tree edges, which gives combed `r = 0` on tree edges (C12). `theta_combed = theta + c pi/2`.
6. **Nodes and arcs:**
   - Nodes are cut-graph vertices with cut degree other than 2, every cone on the cut, and every vertex where the cut touches a boundary.
   - Arcs are maximal paths between nodes. Side A of an arc is the set of halfedges whose face lies on the left when walking from nodes[0] to nodes[1].
   - Assert that r is constant along each arc.

**API:**
```rust
pub fn cut_and_comb(px: &Proxy, f: &CrossField, seed: u64) -> Result<CutGraph, AqError>;
pub fn homology_generators(px: &Proxy, cut: &CutGraph) -> Vec<Vec<u32>>;   // diagnostics and tests
```

**Errors:** a genus-0 component without cones and without boundary is impossible by Poincare-Hopf, so it is reported as `Invariant`, which reveals a convention bug.

**Tests:**
- χ(cut) = 1 per component on every generator and every testdata mesh.
- Every cone lies on the cut.
- Combed r = 0 on all non-cut interior halfedges.
- Torus: 2 homology generators. Double torus: 4.
- r is constant along each arc.
- Determinism.
- < 0.3 s at 100k faces.

### 6.9 aq_param (WP-10): S7 seamless parametrization

**Variables:**
- **Wedges.** For each vertex v, its fan is split into wedges: maximal runs of faces connected through non-cut interior edges. One rep wedge is chosen per vertex, the wedge containing v's lowest-index face. It gets two real variables, `x_v = (x_v^u, x_v^v)`.
- **Translations.** Each cut arc a gets two real variables `t_a` (the translation of side A, C10).
- **Corner expressions.** Walk the fan from the rep wedge in the CCW direction. When crossing cut halfedge he from wedge W_i to W_{i+1}, `phi(W_{i+1}) = R(-r_he) phi(W_i) + t_he`, where `t_he = t_a` on side A and `-R(r) t_a` on side B (C10). Each corner's uv is therefore an affine expression with integer coefficients, in the form `R(k) x_v + sum_j R(k_j) (±t_{a_j})`.
- **Closure.** Interior vertices that are cut nodes need a closure row (non-node vertices close automatically). Composing all transitions around the full fan gives `(R_tot, tau_tot)`, and the condition `x = R_tot x + tau_tot` yields 2 rows in the form `(I - R_tot) x_v - tau_tot = 0`. Their coefficients are in {0, ±1}, or {±2} for an index4 = ±2 corner vertex, which is then treated as a penalty row.

**Constraints**, fed to `Elim`, in this order:
1. **Closure rows.** The pivot prefers translation variables.
2. **Chain iso rows.** For every segment of every chain, for each chain edge (a, b) taken on the side of face f = face(he):
   - if `edge_family(he) = U`, add the row `v(phi_{a,f}) - v(phi_{b,f}) = 0`;
   - if it is V, use u instead.
   - At a corner, both families meet, so the corner gets both coordinates constrained by the two adjacent segments.
3. **Loop rows** (when close_loops is on). For each LoopCand, compose the transitions of the cut halfedges it crosses, in order, giving `(R, tau)`.
   - R must be the identity. Otherwise skip the loop (holonomy).
   - The combed family k at the start face decides the row: `(k mod 2) == 0` means the loop runs along u, so the row is `tau_v = 0`. Otherwise the row is `tau_u = 0`.
   - The row involves only t variables.
4. **Gauge:** pin the rep x of the lowest-index vertex per component to (0, 0).

`Redundant` rows are dropped. `NoUnitPivot` rows become penalties with weight `penalty_w = 1e4 * mean(diag)`. `Inconsistent` rows are dropped and logged. A loop counts as **exact** if its row was `Eliminated` or `Redundant`.

**Energy:** `E = sum_f w_f A_f (|grad u - d'_f/h_f|^2 + |grad v - n x d'_f / h_f|^2)`.
- d'_f is the combed u-axis.
- `grad u = sum_k u_k grad B_k`, with the barycentric gradients in the face frame.
- The corner values are expressions over the free variables, so assembly is `H = sum_f G_f^T G_f`.
- Solve `H y = b` with Cholesky, then reconstruct every corner. `w_f` starts at 1.

**Loop energy guard:**
- E0 is the energy without loop rows; E1 is the energy with all loop rows.
- If `E1 > 1.3 E0`, keep only `ring` loops and solve again (E2). If `E2 > 1.3 E0`, use no streamline loops.
- At most 3 symbolic analyses happen here, all before stiffening.

**Count scaling:**
- `N_pred = sum_f A_f det J_f`. All rows are homogeneous apart from the zero gauge, so scaling every free variable by `nu = sqrt(N_c / N_pred)` is exact and needs no re-solve.
- If `N_pred <= 0` (heavily folded), return `Numerical`.

**Stiffening** (at most `stiff_iters` = 6):
- Per face, `dist_f = |sigma1 h_f - 1| + |sigma2 h_f - 1|`, plus 10 if the face is folded, where sigma are the singular values of J_f.
- Update `w_f += min(dist_f, 5)`, then take one Jacobi average over the face 1-ring.
- Numeric refactor with the same symbolic. Stop when there are 0 folds, or when the fold count does not decrease, keeping the best iterate.

**Injectivity repair** (before the snap):
1. Fold faces are grown by 2 rings (vertex adjacency), and the regions are merged with union-find.
2. For each region R that is a topological disk, contains no cone, and unfolds consistently into one chart (BFS applying transitions):
   1. Take the boundary polygon B in uv. If B is simple (exact orient2d segment tests), run Floater mean value coordinates with B fixed.
   2. If every det > 0, run up to 10 iterations of barrier symmetric Dirichlet toward the targets `d'/h` (the local-global scheme of 6.12) with B fixed.
   3. Otherwise grow by 2 more rings and retry, up to `repair_max_grow` (3) times.
3. Regions containing cones are left as they are in M1 and counted in `folds_after_repair`. M2 handles them by cutting the region along the shortest edge path from the cone to the region boundary and holding that path fixed on both sides.
4. Repaired vertex values are written back as free-variable values. Only rep x variables are inside R, since cut arcs inside R would have prevented a consistent unfolding.

**Dyadic snap:**
- Round every free variable to a multiple of 2^-24, after the scaling.
- Recompute every corner from its integer-coefficient expression in exact f64. Assert |uv| < 2^27.
- Transitions are now exact for both endpoints of every cut edge, iso chains are exact, and exact loops close exactly.
- Penalty rows hold only approximately; their loops are not listed in `loops_exact`.

**Output:** a `SeamlessParam` with `he_family` for every chain halfedge, which is used by aq_layout.

**API:**
```rust
pub struct ParamParams { pub stiff_iters: u32, pub repair: bool, pub repair_max_grow: u32, pub loops: bool, pub n_target: f64, pub penalty_w: f64 }
pub struct ParamReport { pub n_free: u32, pub rows: u32, pub redundant: u32, pub inconsistent: u32, pub penalty_rows: u32,
                         pub loops_exact: u32, pub loops_dropped: u32, pub e0: f64, pub e_final: f64, pub flips_initial: u32,
                         pub flips_after_stiff: u32, pub folds_after_repair: u32, pub n_pred: f64, pub scale: f64 }
pub fn seamless(px: &Proxy, f: &CrossField, cut: &CutGraph, s: &SizeField, loops: &[LoopCand], p: &ParamParams, ctl: &Ctl)
    -> Result<SeamlessParam, AqError>;
```

**Errors:**
- `Numerical`: a NaN, a non-SPD factorization after retrying with a regularization of 1e-10·mean diagonal, or N_pred <= 0.
- `Invariant`: transitions not exact after the snap.

**Complexity:** 50k vertices give about 100k free variables. Each analysis + factor takes about 0.5 s and each numeric refactor about 0.3 s. Total < 5 s including the guard and stiffening.

**Tests:**
- **Plane with a constant field:** affine map with error < 1e-9, det > 0 everywhere.
- **Exactness after the snap:** for every cut halfedge, both endpoints satisfy the transition bitwise. Chain iso coordinates are bitwise equal. Exact loops close bitwise.
- **Open cylinder:** boundary loops iso exact, `t_v = 0` exactly, and the circumference translation `t_u = 2piR/h` within 1e-3 relative.
- **Torus with loops:** transversal translations exactly 0 for both families, `||grad u| h - 1| < 5%` on 95% of faces, 0 folds.
- **Twisted torus field** (0.3 rad twist): with loops, `tau_perp` = 0 exactly; without loops (control), `|tau_perp| > 0.05`.
- **Cube-sphere with the 8-cone field** (from aq_field or testkit): at most 0.1% folds before repair, 0 after, N_pred = N within 1e-9 after scaling.
- **cube_sharp synthetic:** chains iso exact, corners on both iso lines, 0 folds.
- **Stiffening:** the kept fold count never increases.
- **Repair:** a planted fold (10 vertices perturbed) ends with 0 folds after repair; vertices outside the region are bitwise unchanged.
- **Gauge invariance:** on a genus-2 mesh with permuted vertex order, uv equal up to translation within 1e-8.
- **Performance:** 100k-face proxy < 5 s.

### 6.10 aq_layout (WP-11): S8 exact motorcycle layout

**Principle:** every curve is an iso-line in uv.
- Inside face f (chart f), a curve of family U is `v = c`, and one of family V is `u = c`, with c dyadic.
- Crossing a cut halfedge maps the curve through the exact transition: `v_f = c` becomes `u_g = ±c + t` or `v_g = ±c + t`, depending on r.
- So entry and exit edges are decided by **exact comparisons** of c with the corner coordinates. Crossing two curves of different families in a face is the point (c1, c2), whose inclusion is decided by exact orient2d (the robust crate).
- Two curves of the same family never cross unless they are equal.

This pure exactness replaces every tolerance rule of proposals B and C.

**Constraint curves** (pre-inserted, never crossed):
- every chain segment (Hard, GuideHard, Boundary, Symmetry; Guide in M2), using `he_family`;
- every exact loop: the iso-line through the loop's seed point with `c = snap(transverse coordinate of the seed)`, traced in uv in both directions until it closes on itself exactly (same face, same c after transitions). One that does not close within `max_trace_len` is dropped.

**Emitters:**
- **Cones.** At each cone vertex, for each incident face, test the 4 axis directions from the corner point p in chart f.
  - Direction a is emitted from that face if it lies strictly inside the corner wedge (orient2d(e_a, a) > 0 and orient2d(a, e_b) > 0, with e_a and e_b the outgoing and incoming edge vectors in CCW order).
  - A direction exactly along e_a belongs to this face, and one exactly along e_b belongs to the next face (the CCW rule).
  - Directions already occupied by a constraint curve leaving the vertex are not emitted.
  - The total must equal `4 - index4`, minus the occupied directions. Otherwise return `Invariant`.
- **Feature and boundary corners.** Use the same wedge rule over interior wedges not covered by chains. For example, a cube corner gives 0, a 90 degree boundary corner 0, and a 270 degree corner 1.
- **Open chain ends** (M2, guides): 1 continuation trace along the chain direction.
- **Seeds:** see validation below.

**Simultaneous motorcycles:**
- A priority queue of (entry_time, trace_id). Time is the parametric distance travelled: |Δu| or |Δv|, which is invariant under transitions.
- **Pop:** the trace crosses the current face from its entry point to its exit point. Gather the segments already placed in that face:
  - **Collision.** If its segment intersects a placed segment S at point x, compare the arrival times at x. If this trace arrives later, it stops at x (a T-node). If S's owner arrives later, S's owner is truncated at x and its pending queue entry is cancelled.
    - This needs no deeper rollback: a trace is truncated only within the face it is currently crossing, since its next pop has a later time.
    - Equal times: the higher id stops.
  - **Constraint curve:** the trace stops (T-node on the chain or ring).
  - **Cone:** the iso-line passes exactly through a cone vertex lying on the segment. The trace stops at the cone.
  - **Head-on:** an opposite-direction trace on the same iso-line, with the segments overlapping. Both stop at the midpoint in time, joined, and the meeting point becomes a Regular node.
  - **Boundary:** stop (boundaries are constraint curves).
  - **Length cap:** a trace beyond `max_trace_len = 4 sqrt(A_c)` returns `LayoutIssue::Runaway`.
- **SoS:** when c equals a corner coordinate exactly, the curve passes **through** that vertex (the embedding uses the vertex). The next face is chosen as if the curve were at `c + epsilon`, where the sign of epsilon is the trace's transverse orientation, carried through transitions by R.

**Embedding** (no snapping):
- Each face is cut by its segments. The cells are convex (an intersection of half-planes in uv), so each cell is fan-triangulated.
- New vertex uv is exact: one coordinate is c, the other comes from exact intersection. The coordinate along the edge is computed in f64 as `t = (c - v_a)/(v_b - v_a)`; it is used only for 3D positions.
- 3D positions come from linear interpolation on the proxy face.
- Slivers are tolerated (downstream weights are floored).
- The result is W', with `parent_face` and with transitions inherited on the cut halfedges.

**T-mesh extraction**, a pure function of W' and the curve labels:
- **Nodes:** cones, corners, T-nodes, crossings, chain ends and seeds. A closed chain or ring with no node gets a Regular node at its lowest-index vertex.
- **Arcs:** maximal W' edge chains between nodes.
- **Patches:** flood fill of W' faces without crossing arc edges.
- **Per patch:**
  - **Disk check:** χ = 1 with 1 boundary loop.
  - **Corners:** at each boundary node, the patch's uv angle sum (corner angles in the patch chart) divided by pi/2 must be an integer within 0.02, giving quarter-turn counts. Corners are the nodes with count 1. The patch must have exactly 4 corners and no node with count above 2.
  - **Chart:** BFS from `patch_chart_face`, unfolding through transitions; no cone lies strictly inside a patch. Sides are assigned so that side0 runs +u.
- **Arc length:** `len` is the sum of |Δ(varying coordinate)| over the arc's W' edges. ArcKind follows the source curve.
- **Wrap flag:** an arc on a curve T whose two end nodes are T-nodes created by the same trace S, which crossed T one turn apart (consecutive crossings of S with T), with `len < 1.5`.
- **rings** = the closed arc cycles of exact loops and closed chains.
- **chi** = the component χ.
- **Health:** per patch `m_P = max(|l0 - l2| / max(l0, l2, 1), |l1 - l3| / max(l1, l3, 1))`, reporting mean, max and the fraction above 0.1. With an exact param on unfolded faces, m_P < 1e-6.

**Validation loop** (pure; `build_layout` re-runs tracing and embedding from scratch, at most `max_rounds` = 3 times):
- A non-disk patch or a patch without 4 corners produces `ValidationRequest { seeds }`.
  - Patch with a boundary component: the seed is the lowest-index node, or a chain vertex promoted to a Seed node, on that component. It emits along the interior axis perpendicular to the chain.
  - Patch without boundary (for example a bare torus): a Seed node at its lowest-index vertex, emitting all 4 axes.
- Seeds accumulate across rounds. After 3 rounds the result is `Err(Invariant)`.

**API:**
```rust
pub struct LayoutParams { pub max_trace_len: f64, pub max_rounds: u32, pub guide_continuation: bool, pub simplify: bool }
pub fn build_layout(px: &Proxy, prm: &SeamlessParam, loops: &[LoopCand], p: &LayoutParams, ctl: &Ctl) -> Result<Layout, AqError>;
pub fn trace_all(px: &Proxy, prm: &SeamlessParam, curves: &[ConstraintCurve], emitters: &[Emitter], p: &LayoutParams, ctl: &Ctl)
    -> Result<Vec<Trace>, LayoutIssue>;
pub fn embed(px: &Proxy, prm: &SeamlessParam, curves: &[CurveRef]) -> Result<Embedded, AqError>;
pub fn extract(e: &Embedded, px: &Proxy) -> Result<Layout, LayoutIssue>;
pub enum LayoutIssue { Validation(Vec<Seed>), Runaway(u32), Fatal(AqError) }
```

**M2 simplification:**
- For each separatrix ending in a T-node, try removing its tail back to its previous node. Do this greedily, longest tails first, in a deterministic order.
- Keep the removal if every patch of the rebuilt layout is still a 4-corner disk and no patch exceeds 16h per side.
- It is implemented as a per-trace truncation list fed back to `trace_all`, so it stays a pure function.

**Complexity:** O(total trace length / l), and < 1 s at 100k faces.

**Tests** (on testkit analytic params only):
- **Planar square** (rotated grid param, 4 boundary chains): 1 patch, 4 corners, arc lengths equal to side/h within 1e-12.
- **Planar square with one forced seed:** 4 patches.
- **cube_sphere_param:** every cone trace reaches another cone head-on or at the cone; 6 patches (or 6 after joins), all lengths integers exactly.
- **perturbed_cube_sphere_param** (20 seeds):
  - every patch is a disk with 4 corners;
  - every interior arc lies on exactly 2 patch sides;
  - the T-mesh satisfies V - E + F = 2;
  - the patch areas sum to the mesh area within 1e-9;
  - health max < 1e-6.
- **torus_param with exact loops:** all patches disks; at least 1 patch; no Runaway.
- **cylinder_param, open:** 1 patch after 1 seed round.
- **Grid-aligned variants** (iso values hit vertices exactly): valid layouts, no duplicate nodes, determinism.
- **Hand-built 2-cone motorcycle instance:** the later-arriving trace stops; truncation of the earlier trace is tested on a crafted case.
- **W' quality:** manifold, `parent_face` valid, every arc an edge chain.
- **Determinism:** 10 runs and thread counts 1/8 give bitwise identical results.
- **Performance:** a perturbed cube-sphere with 100k faces in < 1 s.

### 6.11 aq_quantize (WP-12): S9 integer arc lengths

**Constraints:**
- Node set: Q = {(P, axis) for every patch P, axis in {0, 1}} plus a ground node G with free balance.
- An arc occurrence on side k of P contributes to node (P, k mod 2) with sign +1 if k is 0 or 1, and -1 if k is 2 or 3.
- An interior arc has 2 occurrences (a bi-directed edge). A boundary arc has 1, and its other end is G with the opposite sign.
- An arc occurring twice in the same node with opposite signs cancels (no constraint).
- For every q other than G, `sum_ends sigma * n_a = 0`. Also `n_a >= lb_a`.

**Cost:** `f_a(n) = weight_a (n - sigma l_a)^2 / max(sigma l_a, 0.5)`. For wrap arcs the target is 0 instead of sigma l_a.

| Arc kind | weight | lb |
| --- | --- | --- |
| Separatrix, SeedTrace | 1 | 1, or 0 when allow_collapse |
| Feature, Boundary, Guide, Loop | 4 | 1 when sigma l >= 0.5, else 0 |
| Symmetry | 8 | 1 |
| Wrap arcs | 8 (target 0) | 0 |

**Solver:**
1. **Double cover** (Bi-MDF):
   - Node q becomes q+ and q-. An edge with ends (u, su) and (v, sv) becomes two directed copies, `u^{su} -> v^{-sv}` and `v^{sv} -> u^{-su}`, each with convex cost `f_a(x)/2` in integer x. Then `n_a = (x1 + x2)/2`.
   - Start every copy at its unconstrained integer minimizer, clamped to `[lb, +inf)`, so every residual marginal cost is >= 0. Then run successive shortest paths with Dijkstra and potentials from excess to deficit nodes.
   - The marginal cost of +1 or -1 on a copy is `f(x±1)/2 - f(x)/2`.
   - Use capacity scaling when the total excess is above 2000.
   - The symmetrized result is an optimal half-integral relaxation.
2. **Parity repair:**
   - H = the arcs with half-integral n. Every node other than G has an even number of H-ends, and so does G.
   - Decompose H into closed walks (Hierholzer). Along a walk, set delta_1 = +1/2 and propagate `delta_{i+1} = -sigma_in * sigma_out * delta_i`. Here sigma_in and sigma_out are the end signs of the entering and leaving arcs at the shared node.
   - A walk through G, or one that closes consistently (`sigma_last delta_last + sigma_first delta_1 = 0`), is applied with the cheaper global sign (±).
   - An inconsistent walk leaves an imbalance of ±1 at its start node. Pair imbalanced nodes greedily: each node is connected to the nearest other imbalanced node, or to G, by a shortest path in the double cover from `q1^{s1}` to `q2^{s2}`, where the states s encode the needed sign. The ±1 changes are applied along the path, respecting lb.
3. **Integer repair** (fallback, Lyon-style strip moves):
   - While any constraint is violated, take a violated node and find, in the double cover, the cheapest path to a node with the compensating violation or to G. Apply ±1.
   - Cap: 4 * |arcs| moves, then `Err(Infeasible)`.
4. **Validity** (lazy cuts, at most `max_valid_rounds` = 8):
   - Build `QComb` with `build_comb`. Zero arcs merge their end nodes (union-find). A zero-height patch (all side1/side3 arcs zero) glues side0 to side2 index by index. A zero-width patch is handled the same way.
   - Violations:
     - (i) the QComb is non-manifold, has a repeated vertex in a quad, or has a doublet (two quads sharing two edges);
     - (ii) a merged node set has an interior valence <= 2 or >= 6, or a combined index4 outside {-1, 0, 1};
     - (iii) a feature corner is merged with a cone, or with another corner of a different chain;
     - (iv) two different boundary loops or chains are identified by a glued patch;
     - (v) a ring has a sum below max(3, ring_min/2);
     - (vi) the Euler characteristic of the QComb is not chi.
   - For each violation, set lb = 1 on the zero arcs whose collapse caused it (tracked through the union-find history), then re-solve.
   - After 8 rounds, set lb = 1 on all arcs and solve once more. If that still fails, `Err(Infeasible)`.
5. **Count targeting:**
   - `Q(sigma) = sum_P W_P H_P` from the solved n. The iteration starts at `sigma0 = 1` (the param is already scaled to N_c).
   - Update sigma with a secant method on `log Q - log N_c` until `|Q - N_c| / N_c < count_tol` (4%) or after `max_secant` (5) iterations.
   - Validity rounds run inside each secant step, and cuts persist across steps. At most 24 solves in total.

**API:**
```rust
pub struct QuantParams { pub allow_collapse: bool, pub count_tol: f64, pub max_secant: u32, pub max_valid_rounds: u32, pub ring_min: u32 }
pub struct QuantResult { pub n: Vec<u32>, pub sigma: f64, pub count: u64, pub solves: u32, pub cuts: u32, pub comb: QComb, pub warnings: Vec<String> }
pub fn quantize(t: &TMesh, n_target: f64, p: &QuantParams, ctl: &Ctl) -> Result<QuantResult, AqError>;
pub fn solve_bimdf(t: &TMesh, sigma: f64, lb: &[u32]) -> Result<Vec<u32>, AqError>;
pub fn build_comb(t: &TMesh, n: &[u32]) -> Result<QComb, Vec<Violation>>;
pub enum Violation { NonManifold(Vec<u32>), BadValence(u32), CornerMerge(u32), Pinch(u32), ShortRing(u32), Euler(i64) }
```

**Tests** (mesh-free, testkit TMeshes):
- **Rectangle** with lengths (3.4, 5.6, 3.4, 5.6) and no count targeting: (3, 6, 3, 6).
- **2x2 grid with T-offsets of 0.3:** collapse to 0 with allow_collapse, and to 1 without.
- **200 random bi-directed instances** of at most 8 arcs:
  - always feasible;
  - the objective equals the brute-force optimum in at least 95% of cases and is at most 1.5x everywhere;
  - lb always respected.
- **Random 500-patch grids:** count error < 3% after the secant, 0 violations.
- **Planted merge of two +1 cones through a zero arc:** rejected; the result has lb = 1 there.
- **Zero-height patch between two boundary loops:** rejected.
- **Helix layout** (a spiral with pitch 0.6, from testkit): every wrap arc gets 0, and walking the QComb rows gives closed rings.
- **QComb:** Euler = chi, manifold, `tmesh_cube` gives 8 valence-3 vertices.
- **Performance:** 5k arcs, a full quantize with 24 solves, < 300 ms.

### 6.12 aq_patch (WP-13): S10 patch meshing and assembly

**Arc samples:**
- For arc a with n_a > 0, sample j (0 <= j <= n_a) lies at parametric length `s = j * len_a / n_a` along `arc_verts`, interpolated linearly in s on the W' edge that contains it. Position is in 3D, uv is in the arc's charts.
- Each arc is computed **once**, so both adjacent patches use bitwise identical samples.
- Ids come from `QComb.arc_ids`. Node positions are the averages over their union-find classes.

**Boundary map per patch:**
- Rectangle [0, W] x [0, H]. Side k's arcs are laid out consecutively.
- A W' boundary vertex at parameter s of arc a maps to `offset_a + n_a * s / len_a`.
- A zero arc occupies the interval [k, k + m*eps], where m is the number of consecutive zero arcs and eps = 1e-3. That interval is carved from the start of the next positive arc on the side, whose interval becomes [k + m*eps, k + n]. This keeps the boundary map injective and the polygon weakly convex.
- A zero-height patch is not meshed; its sides were glued by QComb.

**Interior map, in order:**
1. **Pre-split** every W' triangle whose 3 vertices all lie on the patch boundary, and every triangle whose 3 vertices lie on one side, by splitting its interior edge at the midpoint.
2. **Field-aligned least squares**, with the boundary fixed. Minimize `sum A_f |J - T_f|^2`, where `T_f = diag(W/l0, H/l1) * J_f^seam`: the seamless Jacobian in the patch chart, rescaled by the quantization stretch. l0 and l1 are the mean side lengths. Then check det > 0 with exact orient2d on every triangle.
3. If any triangle is flipped, run the **barrier symmetric Dirichlet** repair (never cut).
   - Start from Floater mean value coordinates with the same boundary. The boundary is a convex rectangle and the weights are positive, so the start is bijective.
   - Then run up to `sd_iters` (20) local-global iterations:
     - local step: the per-face target rotation from the SVD of J (relative to T_f);
     - global step: a weighted Laplacian solve with the boundary fixed;
     - line search with the maximal step that keeps every det > 0 (from the roots of the quadratic det(t) per face, times 0.8), plus Armijo energy decrease.
   - Smith-Schaefer 2015 is the reference for the step bound.
4. If that fails, fall back to the Floater map itself, which is valid by construction.

Mean value coordinates floor `tan(alpha/2)` at alpha = 1e-3 rad and edge lengths at 1e-9.

**Sampling:**
- Interior grid points (i, j) with 0 < i < W and 0 < j < H are located in the (U, V) triangulation with a uniform bucket grid and exact orient2d.
- A point on an edge or vertex uses any adjacent face, consistently: the lowest face id.
- The point is lifted through barycentric coordinates on the W' positions. Ids come from `patch_ids`.

**Assembly:**
- Quads come from `QComb.quads`. The orientation is CCW, inherited from the orientation-preserving maps.
- vflag is set per section 8.4.
- **Final check:** every quad corner Jacobian is positive relative to the proxy normal, the mesh is manifold, twins are consistent, and χ matches.
  - A failed patch is re-meshed with plain Floater.
  - A failed component gives `Err(Invariant)`, and the engine falls to SF.

**API:**
```rust
pub struct PatchParams { pub sd_iters: u32, pub eps_zero: f64 }
pub struct PatchReport { pub field_aligned: u32, pub barrier: u32, pub floater: u32, pub remeshed: u32 }
pub fn mesh_patches(lay: &Layout, q: &QuantResult, p: &PatchParams, ctl: &Ctl) -> Result<(QuadMesh, PatchReport), AqError>;
pub fn floater_mvc(disk: &TriMesh, t: &Topology, boundary_uv: &[(u32, V2)]) -> Result<Vec<V2>, AqError>;
pub fn sd_barrier(disk: &TriMesh, t: &Topology, uv: &mut [V2], targets: &[[[f64; 2]; 2]], fixed: &[bool], iters: u32) -> Result<u32, AqError>;
```

**Tests:**
- **Hemisphere disk patch** with sides (8, 8): 64 quads, all Jacobians > 0, boundary vertices on their polylines within 1e-9.
- **Floater:** 1000 random star-shaped disks give 0 flipped triangles.
- **Barrier:** 200 random folded initial maps end with 0 flips and the energy decreasing monotonically.
- **Two patches with a T-junction:** manifold, no duplicate vertices.
- **Zero arc and zero-height patch:** the edge count equals the expected value exactly.
- **Shared arc samples:** bitwise identical.
- **Boundary-only triangles:** pre-split as specified.
- **Cylinder patch, field-aligned map:** mean corner angle deviation < 3 degrees.
- **Performance:** 5k quads from a 100k-face W' in < 1.5 s.

### 6.13 aq_post (WP-14): S11 relaxation and cleanup

**Relaxation:**
- `iters` = 8 iterations. Arc vertices are free, so loops straighten across patch borders and arc kinks disappear.
- An interior vertex moves to the area-weighted mean of its incident quad centroids, projected to its tangent plane, then onto the proxy with `Projector` (hint = its source face).
- A chain vertex slides along its chain polyline, staying on the polyline within 1e-9.
- Corner and cone vertices are fixed for the first `freeze_nodes_iters` (4) iterations, then move as interior vertices. Corners stay fixed throughout.
- A move is rejected if any incident quad corner normal turns more than 60 degrees from the surface normal.

**Cleanup:**
- Remove interior valence-2 vertices by merging their two quads.
- Collapse 3-3 diagonal pairs on quads of area < 0.3 h^2.
- Report the count change.

**API:**
```rust
pub struct PostParams { pub iters: u32, pub freeze_nodes_iters: u32 }
pub fn relax(q: &mut QuadMesh, px: &Proxy, chain_of_vert: &[u32], p: &PostParams, ctl: &Ctl) -> Result<RelaxReport, AqError>;
pub fn cleanup(q: &mut QuadMesh) -> CleanupReport;
```

**Tests:**
- A jittered planar grid's edge CV drops by at least 50%.
- A perfect grid moves < 1e-12.
- Chain vertices stay on their polylines within 1e-9.
- No inversions on the sphere output.
- A synthetic kinked arc between two patches has its kink angle reduced by at least 50%.
- A planted doublet is removed.

### 6.14 aq_fallback (WP-04): SF quadder

**Algorithm:**
1. Isotropic remesh of the component with aq_prep at `l_sf = 2.15 h`. That gives a triangle count of about N/2, which after the split yields about N quads.
2. Greedy pairing: for each interior edge whose dihedral is below 30 degrees, compute the quality of the quad formed by its two triangles (the maximum |corner angle - 90 degrees|). Sort ascending and pair each triangle at most once. Chain edges are never paired across.
3. Topological split: each quad becomes 4 quads (edge midpoints + face center) and each unpaired triangle becomes 3 quads (edge midpoints + center). Shared midpoints keep the result conforming, and it is 100% quads.
4. Relax 5 iterations, then project.

**API:**
```rust
pub struct SfParams { pub target: u32, pub relax_iters: u32 }
pub fn sf_quadrangulate(src: &ComponentMesh, proj: &Projector, p: &SfParams, ctl: &Ctl) -> Result<QuadMesh, AqError>;
pub fn sf_batch(srcs: &[&ComponentMesh], ...) -> Result<Vec<QuadMesh>, AqError>;   // tiny components, no proxy each
```

**Tests:**
- Sphere, torus, cube, noisy blob, open cylinder and a 20-triangle piece each give 100% quads, manifold output, 0 flips, and a count within 25% of the target.
- 100k triangles in < 2 s.

### 6.15 aq_io and aq_eval (WP-01)

**aq_io:**
- OBJ reading (`v`, `f` with any `/` suffixes, polygons fan-triangulated for TriMesh input) and writing (quads as polygons).
- Debug dumps:
  - `proxy.obj`
  - `field.obj` (line segments per face)
  - `cones.obj` (points with index groups)
  - `cut.obj`
  - `param.obj` (with `vt` per corner)
  - `layout.obj` (arcs as lines, patches as groups)
  - `quant.json`
  - `patches.obj`

**aq_eval:**
```rust
pub struct QuadStats { pub faces: u32, pub quad_ratio: f64, pub face_sizes: BTreeMap<u32, u32>, pub irregular: u32,
                       pub valence: BTreeMap<String, u32>, pub edge_cv: f64, pub angle_dev_deg: f64,
                       pub nonmanifold_edges: u32, pub boundary_edges: u32 }
pub fn quad_stats(q: &QuadMesh) -> QuadStats;           // identical rules to tools/quad_stats.py
pub struct Hausdorff { pub q_to_in_max: f64, pub q_to_in_mean: f64, pub in_to_q_max: f64, pub in_to_q_mean: f64 }
pub fn hausdorff(q: &QuadMesh, input: &TriMesh, samples_per_face: u32) -> Hausdorff;   // units: input units and h
pub struct LoopReport { pub loops: u32, pub closed: u32, pub closed_frac: f64, pub helix_suspects: u32 }
pub fn loop_closure(q: &QuadMesh, axis: Option<V3>) -> LoopReport;
pub fn bisect(m: &TriMesh, plane: [f64; 4], keep_positive: bool) -> (TriMesh, Vec<[u32; 2]>);   // cut edges for kind Symmetry
pub fn mirror_weld(q: &QuadMesh, plane: [f64; 4], tol: f64) -> (QuadMesh, WeldReport);           // open edges, seam poles
```

**quad_stats rules:**
- Irregular: an interior vertex of valence other than 4, or a boundary vertex of valence not in {2, 3}.
- The valence keys are "N" for interior vertices and "bN" for boundary vertices.
- Edge CV is the population standard deviation over the mean, over unique edges.
- The angle deviation is the mean |corner angle - 90| over quads.

**loop_closure:**
- Walk edge loops straight through valence-4 vertices (the opposite edge).
- A loop is closed when it returns to its start edge, and open when it hits a boundary or an irregular vertex.
- With an axis given, count only loops whose mean direction has |dot(axis)| < 0.3 (circumferential).
- A helix suspect is a loop that crosses the same axial slab twice with an axial drift above 0.5 h per turn.

**Tests:**
- Stats on `baselines/torus_qf2000.obj` and `retopo_a_qf2000.obj` equal `quad_stats.py` output within 1e-6 (compared to fixture JSON produced by the script).
- Hausdorff of a mesh with itself is 0.
- A perfect torus grid gives closed_frac 1.0.
- `bisect` then `mirror_weld` of a quad sphere gives 0 open edges.

---

## 7. Orchestration (aq_engine)

### 7.1 Flow

1. **Ingest** (S0).
2. **Split components.** For each component compute area and χ, then the provisional global scale `1/h0`.
3. **Debris.** Components with area < 6 h² (quad units) and closed, or with N_c below the floor, are batched to SF directly, with no proxy each.
4. **Budget per component:** `N_c = N_target * A_c / A_total` (pass 1). Floors: 24 quads closed, 8 open. Renormalize the rest so the total stays N_target. Each component's proxy triangle budget is proportional to N_c.
5. **Time share:** `T_c = (time_limit - elapsed) * N_c / sum N`, at least 2 s each.
6. **Per component, in parallel** inside one rayon pool, built per call with `threads` and `install()`:
   1. Run the rung pipeline under `catch_unwind`. A panic in one component sends only that component to SF.
   2. Check the stage invariants (7.3) between stages.
7. **Merge:** offset the ids, set face_comp and vflag, scale back by h0.
8. **Count correction:** if the total count error exceeds 8%, and time remains, re-run S9 to S11 for the components with the largest error, with N_c scaled by target/count (once).
9. **Report.**

### 7.2 Fallback ladder (per component, owned only by aq_engine)

Lower crates return typed errors and never retry upstream stages.

| Rung | What runs | Entered when |
| --- | --- | --- |
| R0 | Full pipeline with the user's parameters | always first |
| R1 | Re-run S3 to S11: curvature_align x0.5, adaptive 0, cancel_dist x1.5, streamline loops limited to rings, allow_collapse kept, h0 jitter x0.95 | R0 failed and elapsed time < 40% of T_c |
| R2 | SF quadder on the component | R1 failed or was skipped, or the deadline is near |
| R3 | comp_status = Failed, and the component's input triangles go to `failed_tris` (Python runs QuadriFlow on them) | SF failed (non-manifold proxy, or Budget) |

**Re-run triggers inside R0** (count as R0, at most 1 each):
- `FieldReport.excess_poles` (M2): re-run S4 with curvature_scale x1.5 and align x0.7.
- A layout health fraction above 0.1 over more than 5% of patches: flag the component Degraded, then continue.

A cancel at any point returns status 3 at once. A timeout sends the components that are still unfinished to SF if their share allows it, otherwise to R3.

### 7.3 Invariants between stages (checked by the engine)

| After | Checks |
| --- | --- |
| S0 | manifold, oriented, no NaN, all indices valid, χ per component |
| S2 | manifold; χ per component equals S0's; chains intact; at most 0.1% of faces with min angle < 1 degree |
| S3 | `sum A/h^2 = N_c` within 1e-6; h finite and positive |
| S4 | `p[twin] = (4-p)%4`; Poincare-Hopf; interior non-corner cones have |index4| <= 1; hard faces exact |
| S6 | χ(cut) = 1; cones on the cut; combed r = 0 off the cut |
| S7 | exact transitions; exact chain iso lines; |uv| bound; N_pred = N_c within 1e-6 before quantization |
| S8 | every patch a 4-corner disk; every interior arc on exactly 2 patch sides; T-mesh Euler = χ; patch areas sum to the area within 1e-9 |
| S9 | every constraint holds exactly; lb respected; QComb manifold with Euler = χ |
| S10 | all quad corner Jacobians > 0; shared arc samples bitwise equal; manifold; χ |
| S11 | same as S10, plus the count error |

### 7.4 Budgets (300k-triangle input to 5k quads on an 8-core desktop)

| Step | Time |
| --- | --- |
| Ingest | 0.5 s |
| Proxy | 4 s |
| Curvature + sizing | 1.5 s |
| Field + edits | 3 s |
| Loops | 0.5 s |
| Cut | 0.3 s |
| Param | 5 s |
| Layout | 1 s |
| Quantize | 0.3 s |
| Patch | 1.5 s |
| Post | 1 s |
| Total | about 19 s (gate: < 30 s; M2 target < 20 s) |

Small inputs (< 50k triangles, 2k quads): < 5 s.

---

## 8. C ABI (aq_ffi, autoquad_engine.dll) and Python integration

### 8.1 Header (normative; the Rust `#[repr(C)]` structs mirror it field by field)

```c
#define AQ_ABI_VERSION 1
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

uint32_t aq_abi_version(void);
void     aq_params_default(AqParams* p);
int32_t  aq_remesh(const AqMeshIn* in, const AqParams* p, AqControl* ctl /* nullable */, AqResult** out);
void     aq_free(AqResult* r);           /* NULL-safe; frees every buffer allocated by Rust */
```

### 8.2 Rules

- **Ownership:** all output memory is allocated by Rust and freed only through `aq_free`. Input pointers are borrowed for the duration of the call and never retained.
- **Errors:** `aq_remesh` returns the same value as `(*out)->status`. `*out` is always set, except when the out pointer itself is NULL, which returns 4. There is no global state: the error string lives in the result, and cancellation in AqControl, so concurrent calls are safe.
- **catch_unwind** wraps every exported function. `panic = "unwind"` is set in the release profile. Panics from rayon workers propagate to the per-component catch, or to the top-level catch, which returns status 5 with the message.
- **Progress:** the engine writes `progress_bits` with `AtomicU32::store(Relaxed)` through a pointer cast. Python reads it with `struct.unpack('f', ...)`. Nothing calls back into Python.
- **Threads:** the call blocks. Python runs it in a worker thread and polls AqControl from a modal timer. The engine builds its own rayon pool per call.

### 8.3 Python side (WP-06, add-on `autoquad_remesher`)

- New module `engine.py` with the ctypes mirrors of the structs. It loads `bin/autoquad_engine.dll` from the add-on folder and checks `aq_abi_version() == 1`, otherwise it uses QuadriFlow. Windows cannot unload a DLL in use: updating the DLL requires restarting Blender.
- UI: "Engine: AutoQuad / QuadriFlow". New controls map to AqParams:

| UI control | AqParams field |
| --- | --- |
| Target Quads | target_quads |
| Adaptive Size | adaptive_size |
| Curve Strength | curvature_align |
| Flow Smoothness | flow_smoothness |
| Keep Sharp Edges | hard_edges = 2 / 0 |
| Use Density (vertex attribute `aq_density` or color) | density + density_strength |
| Guides (edges marked Seam, or the attribute `aq_guide`) | kind Guide |
| Seed | seed |

- Feature edges come from the existing `keep_sharp` detection (kind 1). Symmetry cut edges come from `bisect_half` (kind 5): the edges whose both vertices lie on the plane within MERGE_DIST.
- Python keeps its existing steps: cleanup, voxel pre-pass, normalization to diag 10, bisect, mirror and weld, `snap_seam` on vertices with vflag bit 26 and kind 5, BVH reprojection, relax, and the UI.
- For `comp_status == 3`, run QuadriFlow on the subset `failed_tris` and merge the result.
- M2: optional "Safe mode", which runs `autoquad.exe` as a subprocess on a temporary OBJ (it survives stack overflow, aborts and OOM).

### 8.4 Vertex flags (vflag)

| Bits | Meaning |
| --- | --- |
| 0..19 | chain id + 1 (0 = none), globally renumbered |
| 20..23 | FeatureKind of that chain |
| 24 | corner node |
| 25 | cone (singular vertex of the layout) |
| 26 | boundary vertex |
| 27 | vertex of an SF (fallback) component |
| 28..31 | reserved (0) |

---

## 9. CLI (aq_cli, autoquad.exe)

```
autoquad remesh <in.obj> -o <out.obj> [--target N] [--adaptive F] [--align F] [--smooth F]
        [--features auto|none|<file>] [--hard-angle DEG(30)] [--density <file>] [--sym x|y|z] [--keep-neg]
        [--seed N] [--threads N] [--time-limit S] [--no-loops] [--no-collapse] [--upsample K]
        [--reproject] [--report <r.json>] [--dump <dir>] [--stop-after S0..S11]
autoquad batch --suite eval/suite.toml --out <dir> [--seeds 0..4] [--compare baselines/] [--markdown summary.md]
autoquad stats <quad.obj> [--ref <input.obj>] [--axis x,y,z] [--json]
```

- The CLI does exactly what Python does before and after the call, so that the test results represent the add-on:
  - normalize to diag 10 and scale back;
  - `--features auto` marks edges with dihedral > hard-angle as kind 1;
  - `--sym` bisects with `aq_eval::bisect`, marks the cut kind 5, remeshes, mirrors and welds with `aq_eval::mirror_weld`;
  - `--reproject` projects onto the input with the BVH;
  - `--upsample K` subdivides each triangle 1-to-4, K times (a 300k performance input from noisy_blob with K = 2).
- `--report` writes the engine report JSON plus the stats below.

**Stats report** (`autoquad stats` and the report `stats` block):
- faces, quad_ratio, face_sizes, irregular, the valence histogram, edge_cv, angle_dev_deg, nonmanifold_edges, boundary_edges;
- count error vs target;
- Hausdorff, both directions, max and mean, in input units, in % of the diagonal and in h;
- loop closure, when `--axis` is given or for the suite meshes that define an axis;
- mirror weld: open edges and seam poles;
- timing per stage and in total.

**Report JSON** (from the engine):
```json
{"abi":1,"status":0,"timing_ms":{"ingest":0,"proxy":0,"curv":0,"field":0,"loops":0,"cut":0,"param":0,"layout":0,"quantize":0,"patch":0,"post":0,"total":0},
 "components":[{"id":0,"status":"full","rung":0,"target":0,"quads":0,"cones":{"v3":0,"v5":0,"other":0},
   "field":{},"param":{},"layout":{"patches":0,"arcs":0,"health_mean":0,"health_max":0,"seed_rounds":0},
   "quant":{"sigma":0,"solves":0,"cuts":0,"count_err":0},"patch":{},"post":{},"warnings":[]}],
 "count":{"target":0,"result":0,"err":0}}
```

---

## 10. Evaluation protocol

### 10.1 Suite (`eval/suite.toml`)

Each entry gives: mesh, target, features, sym, axis for the loop check, and the gate values from 10.3.

| mesh | target | features | sym | loop axis |
| --- | --- | --- | --- | --- |
| torus | 2000 | none | - | z (both families) |
| double_torus | 2000 | none | - | - |
| cube_sharp | 2000 | auto 30 | - | - |
| cylinder_open | 2000 | none | - | cylinder axis |
| suzanne_sub2 | 2000 | none | - | - |
| retopo_a, retopo_b, retopo_c | 2000 | none | - | - |
| sculpt_cylinder | 2000 | none | - | PCA axis |
| basemesh_female | 5000 | none | - | - |
| noisy_blob | 5000 | none | - | - |
| sculpt_sphere_mirror | 5000 | none | x | - |
| perf: noisy_blob --upsample 2 (328k triangles) | 5000 | none | - | - |

### 10.2 Procedure

1. `autoquad batch --suite eval/suite.toml --out results/<milestone> --seeds 0..4 --compare baselines/`
2. `tools/blender.sh --python tools/render_wire.py -- renders/<milestone>/<mesh>.png results/<milestone>/<mesh>_s0.obj baselines/<mesh>_qf<N>.obj` gives two rows: ours on top, QuadriFlow below. Poles are colored.
3. Visual checklist per mesh:
   - loops follow the form;
   - closed rings on the torus, cylinders and limbs;
   - no spirals;
   - poles where geometry demands them;
   - hard edges are quad edges;
   - the seam is clean after the mirror;
   - no stretched patches.
4. **Gates must hold for seed 0 and for at least 4 of the 5 seeds.** The summary markdown lists every metric against QuadriFlow and against the gate.

### 10.3 Target numbers (irregular vertex counts use the quad_stats.py rules)

| mesh | lower bound | QuadriFlow | M1 gate | M2 gate |
| --- | --- | --- | --- | --- |
| torus | 0 | 14 | 0 | 0 |
| double_torus | 8 (χ = -2) | 36 | <= 12 | <= 10 |
| cube_sharp | 8 | (baseline) | exactly 8, at the corners | same |
| cylinder_open | 0 | (baseline) | 0 | 0 |
| suzanne_sub2 | - | 86 | <= 50 | <= 40 |
| retopo_a / b / c | - | 36 / 38 / 50 | <= 24 each | <= 20 each |
| sculpt_cylinder | - | 32 | <= 16 | <= 12 |
| basemesh_female | - | 136 | <= 90 | <= 80 |
| noisy_blob | 8 | 14 | <= 14 | <= 10 |
| sculpt_sphere_mirror | - | failed | success, 0 open edges, 0 seam poles after the weld | same |

Global gates:

| Gate | M1 | M2 |
| --- | --- | --- |
| Quad ratio | 100% on every mesh, with no component on SF in the suite | same |
| Count error | <= 10% | <= 8% |
| Edge CV (adaptive 0) | <= min(QuadriFlow value, 0.20) | same |
| Angle deviation | <= QuadriFlow value | same |
| Distance from output vertices to the input | max <= 0.3 h | same |
| Symmetric Hausdorff | <= 1.0 h | same |
| Closed loops | torus 100% (both families), cylinder_open 100% circumferential, sculpt_cylinder >= 90% | sculpt_cylinder 100%, 0 helix suspects |
| Robustness | 0 crashes, 0 panics over the suite and the fuzz suite | same |
| Time, 328k triangles to 5k quads | < 30 s | < 20 s |
| Time, small inputs | < 5 s each | same |
| cube_sharp | every input feature edge covered by quad edges within 1e-6 | same |

M2 additions:
- **Adaptive:** adaptive = 1 on a sphere with a bump, and on basemesh_female (face vs back), gives a density ratio at least 2x that of adaptive = 0, with the angle deviation no worse than +2 degrees.
- **Guide test:** a guide loop drawn on the sphere is followed by an edge loop with mean distance < 0.25h.
- **Induced twist test:** sculpt_cylinder with a synthetic field twist gives 100% closed rings.

**Fuzz suite** (WP-05): NaN, Inf, out-of-range indices, an empty mesh, a single triangle, all faces degenerate, duplicate vertices, a bowtie, 500 floating components, a mesh with 3-face edges, and random triangle soup (1000 cases). The only allowed results are statuses 0, 1, 2 or 4.

---

## 11. Milestones and work packages

### 11.1 Milestone scopes

- **M0, walking skeleton:** ingest, uniform proxy, SF on every component, the FFI, the CLI and the Python wiring. Every mesh already produces valid all-quad output through the DLL inside Blender 5.2. Later stages replace SF component by component.
- **M1, the smallest path that beats QuadriFlow:**
  - uniform sizing (adaptive and density inputs accepted but ignored, with a report warning), tube cap, per-component floors;
  - field with hard constraints, high-index split, cancellation, symmetry push-off and the fixed-matching re-solve;
  - loops (rings and torus);
  - cut; param with elimination, iso, loops, stiffening and basic repair;
  - exact layout with seeds; quantize with collapse (`allow_collapse` can be switched off by the integrator if WP-13 zero-arc meshing lags, with lb = 1 everywhere);
  - patch maps with the barrier repair; post; the ladder.
  - Guides are mapped to soft field alignment only.
- **M2, ZRemesher character:**
  - adaptive sizing, adaptive proxy, density map, size-consistent connection, thickness cap;
  - guides (iso, arcs, continuation);
  - layout simplification; same-sign repulsion; pole sanity cap; corner merge;
  - cone-aware injectivity repair; performance pass; Python M2 UI and safe mode.
- **M3, polish:**
  - pole relocation toward high |K| (local field re-solve with an attraction term);
  - quad layout optimization, choosing trace subsets QuadWild-style (greedy plus local swaps, minimizing distortion plus pole count);
  - output pole-pair collapse (Tarini 2010 quad simplification);
  - an optional anisotropic size along limbs.

### 11.2 Work packages

Each WP has one owner agent. Contract changes in aq_core after the WP-00 freeze go through the integrator (WP-16) and `CHANGES.md`.

| WP | Owner scope | Files | Depends on | Parallel | Definition of done |
| --- | --- | --- | --- | --- | --- |
| WP-00 | aq_core + workspace | `engine/Cargo.toml`, `rust-toolchain.toml`, `crates/aq_core/**` | none | first, alone (about 2 days) | Every 6.1 test passes; every contract type with `validate()`; rustdoc on each convention; API frozen (tag `core-v1`) |
| WP-01 | aq_io + aq_eval | `crates/aq_io/**`, `crates/aq_eval/**` | WP-00 | yes | 6.15 tests; stats parity fixtures generated with quad_stats.py; bisect and mirror-weld utilities |
| WP-01b | aq_testkit | `crates/aq_testkit/**` | WP-00 | yes (high priority) | All 6.2 generators; analytic params validated by `SeamlessParam::validate`; TMesh generators validated |
| WP-02 | aq_linalg | `crates/aq_linalg/**` | WP-00 | yes | 6.3 tests including Elim properties and timings |
| WP-03 | aq_prep | `crates/aq_prep/**` | WP-00, WP-02 | yes | 6.4 tests; ingest, proxy and isolation on all 12 testdata meshes without error |
| WP-04 | aq_fallback | `crates/aq_fallback/**` | WP-03 (remesh) | yes, after WP-03's remesh API exists | 6.14 tests; all 12 meshes give 100% quads |
| WP-05 | aq_engine + aq_ffi + aq_cli (M0 path) | `crates/aq_engine/**`, `crates/aq_ffi/**`, `crates/aq_cli/**`, `eval/suite.toml` | WP-01, WP-03, WP-04 | yes | M0 gate (11.3); fuzz suite; FFI tests (re-entrancy, cancel, panic gives 5, NULL-safe free); batch runner and markdown summary |
| WP-06 | Python integration | `autoquad_remesher/engine.py`, `autoquad_remesher/__init__.py`, packaging | WP-05 (ABI) | yes, can start from the header | Headless Blender 5.2 ctypes smoke test via `tools/blender.sh`; UI mapping; per-component QuadriFlow fallback; zip rebuilt; installed when the user asks |
| WP-07 | aq_curv | `crates/aq_curv/**` | WP-00, WP-02 (WP-01b for tests) | yes | 6.5 tests (M1 part) |
| WP-08 | aq_field | `crates/aq_field/**` | WP-00, WP-02, WP-01b | yes | 6.6 tests (M1 part); Poincare-Hopf on all testdata through WP-03 proxies |
| WP-09 | aq_cut + aq_loops | `crates/aq_cut/**`, `crates/aq_loops/**` | WP-00, WP-01b | yes | 6.7 and 6.8 tests |
| WP-10 | aq_param | `crates/aq_param/**` | WP-00, WP-02, WP-01b | yes (critical path; strong agent) | 6.9 tests; exactness on all analytic and synthetic inputs |
| WP-11 | aq_layout | `crates/aq_layout/**` | WP-00, WP-01b | yes (critical path; strongest agent) | 6.10 tests on analytic params |
| WP-12 | aq_quantize | `crates/aq_quantize/**` | WP-00, WP-01b | yes | 6.11 tests including brute-force comparisons |
| WP-13 | aq_patch | `crates/aq_patch/**` | WP-00, WP-02, WP-01b | yes | 6.12 tests; zero-arc and zero-patch handling |
| WP-14 | aq_post | `crates/aq_post/**` | WP-00 | yes | 6.13 tests |
| WP-16 | Integration and evaluation | `crates/aq_engine/src/pipeline.rs`, `eval/**`, `renders/M1/**`, `CHANGES.md` | everything above | last; starts wiring as crates land | M1 gates (10.3) on seed 0 and on at least 4 of 5 seeds; renders reviewed; parameter defaults tuned on retopo_a/b/c first |
| WP-20 (M2) | adaptive sizing, adaptive proxy, density, thickness cap, size-consistent connection | aq_curv, aq_prep, aq_field | M1 | yes | 6.5 and 6.6 M2 tests; adaptive gate |
| WP-21 (M2) | guides end to end (field soft and hard, iso rows, arcs, continuation traces) | aq_prep, aq_field, aq_param, aq_layout | M1 | yes | guide gate test |
| WP-22 (M2) | layout simplification, wrap-arc refinements | aq_layout, aq_quantize | M1 | yes | fewer T-nodes (at least 30% fewer arcs on retopo_a with the same gates) |
| WP-23 (M2) | same-sign repulsion, pole cap, corner merge | aq_field, aq_engine | M1 | yes | noisy_blob <= 10; no regression elsewhere |
| WP-24 (M2) | cone-aware injectivity repair | aq_param | M1 | yes | folds after repair = 0 on the whole suite |
| WP-25 (M2) | performance pass | all | M1 | after WP-20..24 | 328k input < 20 s |
| WP-26 (M2) | Python M2 controls, safe mode | add-on | WP-05 | yes | UI test in Blender 5.2 |
| WP-27 (M2) | evaluation M2 | eval | all M2 | last | M2 gates |

**Parallel lanes after the WP-00 freeze:**
- Lane A: WP-01, WP-01b, WP-02.
- Lane B: WP-03, then WP-04, then WP-05 and WP-06 (M0).
- Lane C, the M1 stage crates, in parallel against testkit contracts: WP-07, WP-08, WP-09, WP-10, WP-11, WP-12, WP-13 and WP-14.
- WP-16 integrates as the crates land.

The critical path is WP-00, then WP-01b, then WP-10 and WP-11, then WP-16.

### 11.3 Milestone gates

- **M0:**
  - the CLI and the DLL return valid all-quad meshes on all 12 testdata meshes and the perf input;
  - the fuzz suite passes;
  - the ctypes smoke test passes in Blender 5.2;
  - the report JSON is produced;
  - the add-on with Engine = AutoQuad works end to end, using SF.
- **M1:** every M1 gate in 10.3, plus every unit test of the M1 parts of 6.1 to 6.15.
- **M2:** every M2 gate in 10.3.

---

## 12. Risks (ranked) and cut order

| # | Risk | Mitigations |
| --- | --- | --- |
| 1 | Param folds near cones corrupt the layout (geometry only, combinatorics stay exact) | h/3 proxy, feature isolation, cancellation and push-off, stiffening, injectivity repair (cone-aware in M2), health metric, per-patch bijective maps, ladder |
| 2 | Layout combinatorics (exact tracing bugs, wedge counts) | exact arithmetic, SoS rule, analytic testkit params, a pure re-runnable layout, strongest agent |
| 3 | Pole count on noisy AI meshes | quad-scale curvature, confidence gate, lambda proportional to 1/h², deterministic cancellation, pole cap (M2); tuning on retopo_a/b/c first |
| 4 | Quantization parity or validity loops | brute-force-tested Bi-MDF, strip repair, all-lb = 1 last try, ladder |
| 5 | Loop constraints distorting the param | energy guard 1.3x, ring-only retry, drop |
| 6 | Performance at 300k | proxy budget, symbolic reuse, at most 6 stiffening refactors, time-shared ladder |
| 7 | Convention mismatch between agents | single aq_core implementation, Poincare-Hopf, kappa-sum and exactness validators run between stages and in every crate's CI |

**Cut order if time is short** (first cut first):
1. M3 entirely.
2. Layout simplification (WP-22).
3. Thickness cap.
4. Same-sign repulsion and corner merge.
5. Guide continuation traces (guides stay soft field plus iso).
6. Cone-aware repair (M1 repair only).
7. The size-consistent connection: adaptive then stays weak, which the report states explicitly.

**Never cut:**
- the proxy;
- exact elimination and the dyadic snap;
- exact uv tracing;
- collapse-capable quantization;
- the barrier repair in patch maps;
- deterministic cancellation;
- the SF fallback;
- catch_unwind and time-budgeted retries;
- the evaluation gates.

---

## 13. Parameter reference (internal defaults)

| Name | Default | Where |
| --- | --- | --- |
| proxy len | h/3 | 6.4 |
| proxy iters | 5 | 6.4 |
| max_proxy_tris | 150000 | 6.4 |
| corner_deg / sym_corner_deg | 50 / 70 | 6.4 |
| curvature scale c_s (flow_smoothness) | 1.0 | 6.5 |
| conf gate (lo, hi) | 0.02, 0.25 | 6.5 |
| alpha (adaptive) | 0.7 * adaptive_size | 6.5 |
| g clamp | [1/4, 4] | 6.5 |
| gradation | 0.25 | 6.5 |
| ring_min | 8 | 6.5, 6.11 |
| curvature_align (lambda numerator) | 1.0 | 6.6 |
| lambda_guide | 20 * guide_strength / h² | 6.6 |
| cancel_dist / low-confidence | 1.2 h / 2.5 h | 6.6 |
| cancel rounds | 3 | 6.6 |
| max |delta| after edits | pi/3 | 6.6 |
| sym_clearance / chain clearance / push target | 0.5 h / 0.35 h / 0.75 h | 6.6 |
| loop stride / max_len / close_tol / min_conf / cluster | 2h / 40h / 0.3h / 0.3 / 6h | 6.7 |
| loop energy guard | 1.3 | 6.9 |
| stiff_iters | 6 | 6.9 |
| repair grow | 2 rings, then up to 3 more growths | 6.9 |
| dyadic quantum | 2^-24 | 6.9 |
| max_trace_len | 4 sqrt(A_c) | 6.10 |
| layout rounds | 3 | 6.10 |
| quant weights (sep / feature / sym / wrap) | 1 / 4 / 8 / 8 | 6.11 |
| count_tol / max_secant / valid rounds | 0.04 / 5 / 8 | 6.11 |
| zero-arc eps | 1e-3 | 6.12 |
| sd_iters | 20 | 6.12 |
| post iters / freeze | 8 / 4 | 6.13 |
| SF length | 2.15 h | 6.14 |
| R1 entry time limit | 40% of T_c | 7.2 |
| count correction threshold | 8% | 7.1 |

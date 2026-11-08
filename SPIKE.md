# Native motion core: spike (Phase 0 and first Phase 1 pieces)

Spike for the PRD "compas_fab native motion core (planning + perception)".

* **Phase 0:** the batched sphere-distance kernel on both portable GPU candidates
  (wgpu + WGSL, and CubeCL), buffer layouts including voxel blocks, first baselines
  against the current compas_fab backends, and a recommendation for the GPU stack.
* **Phase 1, first pieces (CPU):** closed-form IK, self-collision, the layered-graph
  Cartesian planner, Python bindings and a `NativePlanner` compas_fab backend, benchmarked
  against compas_fab's existing Cartesian planners.

Jump to: [Phase 0 gate](#phase-0-gate-wgpu--wgsl-or-cubecl) ·
[Phase 1 progress](#phase-1-progress-cpu-core-cartesian-planning-compas_fab-backend) ·
[Open issues](#open-issues-and-next-steps)

## What is here

```
native/
  scripts/export_robot.py       RobotCellLibrary robot -> fixture: kinematic tree, link spheres, reference FK
  scripts/baseline_python.py    compas_robots FK and PyBullet check_collision on the same robot + scene
  fixtures/ur5.json             UR5: 12 links, 6 dof, 32 spheres, 50 reference configurations
  fixtures/scene_boxes.json     4 cuboids (table, column, rotated beam, wall)
  crates/motion_core            layouts (layout.rs), f64 reference, CPU f32 kernels (scalar + 8-lane), sampling
  crates/motion_gpu             wgpu host code + query.wgsl (fused full, fused min-only, split)
  crates/motion_cubecl          the same kernel in CubeCL (fused full, fused min-only, min-only specialized per robot)
  crates/motion_bench           oracle checks + timing (motion_bench), sphere vs mesh verdicts (compare_verdicts),
                                Cartesian planner on the baseline's waypoints (cartesian_bench)
  crates/motion_py              PyO3 extension `compas_fab_native` (NativeCore)
  scripts/baseline_cartesian.py compas_fab's Cartesian planners vs NativePlanner on the same path
  results/                      raw numbers from the runs below

src/compas_fab/backends/native  NativePlanner: the core as a compas_fab backend (model.py builds the
                                robot/scene descriptions from a RobotCell, planner.py the features)
tests/backends/native           backend tests (skipped unless compas_fab_native is importable)
```

`scripts/export_robot.py` predates `compas_fab.backends.native.model` and keeps its own
copy of the fitting code so `fixtures/ur5.json` (and the numbers below) stay reproducible.

The operation in every backend: configurations (SoA, `q[j * B + b]`) -> batched FK ->
link spheres in world -> signed distance to the nearest cuboid minus radius, with
gradient, per sphere -> per-configuration minimum. Layouts are documented in
`crates/motion_core/src/layout.rs`.

Run:

```bash
cd native
cargo test --release                      # oracle chain: compas_robots -> f64 -> f32 CPU
cargo run --release --bin motion_bench -- --out results/<machine>.json   # MOTION_BENCH_NO_CUBECL=1 to skip CubeCL
python scripts/baseline_python.py ur5 fixtures/scene_boxes.json results/baseline_python.json   # from compas_fab's venv
cargo run --release --bin compare_verdicts -- results/baseline_python.json

# Python extension + backend (no maturin needed for the spike)
PYO3_PYTHON=../.venv/bin/python cargo build --release -p motion_py
cp target/release/libcompas_fab_native.dylib python/compas_fab_native.abi3.so
cd .. && PYTHONPATH=native/python python -m pytest tests/backends/native
PYTHONPATH=native/python python native/scripts/baseline_cartesian.py native/fixtures/scene_boxes.json native/results/baseline_cartesian.json
```

## Results (Apple M3 Max, 14 CPU threads, Metal via wgpu 30)

UR5, 32 spheres, 4 cuboids. Full table in `results/m3max.json`.

### Correctness chain

| check | max error | collision verdicts differing |
|---|---|---|
| Rust f64 FK vs compas_robots (50 configs, every link frame) | < 1e-9 | – |
| CPU f32 vs f64 reference | 1.1e-7 m | – |
| CPU 8-lane vs CPU scalar | 0 (bit-identical) | 0 / 4096 |
| GPU (wgpu) fused / split / min-only vs CPU scalar | 3.0e-7 m, gradient 1.0e-5 | 0 / 4096 |
| GPU (CubeCL) fused / min-only / specialized vs CPU scalar | 3.0e-7 m, gradient 1.0e-5 | 0 / 4096 |
| Covering spheres vs PyBullet (convex hulls of the meshes), 1000 configs | – | 0 missed collisions; 76 false alarms (11% of free configs) |

### Throughput (configurations / second)

| backend | 8 | 512 | 4096 | 32768 | 524288 |
|---|---|---|---|---|---|
| PyBullet `check_collision`, robot vs obstacles (compas_fab today) | ~930 /s (1.08 ms each) | | | | |
| compas_robots FK alone | ~1.4 k/s (735 µs each) | | | | |
| Rust f64 reference, 1 thread | 0.86 M | 0.84 M | 0.82 M | – | – |
| Rust f32 8-lane, 1 thread | 1.8 M | 1.8 M | 1.4 M | 1.2 M | 1.2 M |
| Rust f32 8-lane, 14 threads | 1.8 M | 2.9 M | 8.4 M | 12.3 M | 12.8 M |
| GPU fused min-only, many launches per submit | 0.18 M | 12.1 M | 93 M | 141 M | 168 M |
| GPU fused min-only, one submit + wait | 39 k | 2.4 M | 19 M | 78 M | 159 M |
| GPU fused min-only, upload + run + async readback | 36 k | 2.3 M | 17 M | 59 M | 110 M |
| GPU split (3 dispatches), many per submit | 0.22 M | 12.7 M | 77 M | 78 M | 73 M |
| CubeCL fused min-only, many launches per sync | 0.24 M | 14.7 M | 109 M | 130 M | 154 M |
| CubeCL min-only, specialized (unrolled per robot) | 0.23 M | 14.6 M | 110 M | 125 M | 144 M |
| CubeCL fused min-only, one launch + sync | 37 k | 2.3 M | 19 M | 72 M | 147 M |

## Phase 0 gate: wgpu + WGSL or CubeCL?

Same kernel, same layout, same harness, Apple M3 Max (Metal):

| | wgpu + WGSL | CubeCL 0.11 (wgpu runtime) |
|---|---|---|
| Correctness vs CPU oracle | 3e-7 m, 0 verdicts differ | identical |
| Peak throughput (min-only, 524k) | 161–168 M/s | 154 M/s (−5%) |
| Small batches, many launches per submit | 43 µs / launch | 34 µs / launch (better) |
| One launch + wait | ~205–215 µs | ~215 µs |
| First use (pipeline / JIT + first run) | 8–12 ms | 15–27 ms; **2.1 s** for the per-robot specialized kernel |
| Compile-time specialization (unrolled loops) | n/a | no runtime gain here (−6% at scale) |
| Dependency tree (normal deps) | 96 crates | 437 crates |
| Kernel source | 256 lines WGSL + 458 host | 566 lines Rust (kernel + host) |
| Other targets for free | WebGPU, Vulkan, DX12, Metal | the same, plus CUDA, ROCm and a CPU (MLIR) runtime |
| API stability | wgpu 30, stable surface | 0.x; launch/handle/runtime APIs needed source digging |

Where the launch time goes (wgpu, fit over 1–16 dispatches per submission):
**~150 µs fixed per submission with compute work** (an empty submit + wait is 18 µs),
plus **~44 µs per dispatch even for one configuration**: one thread's serial chain
(FK + 128 sphere-box tests) is the critical path, not throughput. Timestamp queries
are advertised on this GPU but only return values for the longest kernels, so they
can't separate kernel time at small sizes here.

**Recommendation: wgpu + WGSL as the portable GPU path.** Performance is a wash; wgpu
wins on first-use latency, dependency weight, API stability and the direct fit to
WebGPU (the browser target), and the kernels are small enough that hand-written WGSL
is no burden. CubeCL's real advantage is reaching CUDA (and ROCm) from the same source;
the PRD already treats CUDA as an optional per-kernel path decided by benchmark, so
that decision can be taken in Phase 2 on the kernels that need it, with CubeCL as the
first candidate for it. Specializing kernels per robot is not worth it (no runtime
gain, 2 s JIT) unless a persistent kernel cache is used.

Implications for Phase 1/2 design, independent of the stack:
- Per-launch cost (~0.2 ms) means one GPU round trip per solver iteration caps a
  planner at a few thousand iterations per second. Iterations must be fused into one
  submission, or small problems stay on the CPU.
- Thread mapping must adapt to the batch: one thread per configuration is right at
  scale, but for small batches threads per (configuration, sphere group) with a
  workgroup reduction cut the serial chain.

## Findings

1. **The baseline gap is large, and most of it isn't the GPU.** A single CPU core with
   the sphere query is about 1000x faster per configuration than PyBullet's
   `check_collision` through compas_fab (~1 µs vs ~1.1 ms). The sphere model is cruder
   than exact meshes (11% false alarms with the current fitting, no misses), so this is
   not like for like, but it's the comparison the PRD's "multi-fold over current
   backends" gate is about, and it clears it by orders of magnitude. **Phase 1 (CPU
   first) is well founded.**
2. **GPU launch latency sets the crossover.** Every submit + wait costs ~205 µs through
   wgpu on Metal, regardless of size. The GPU only beats all CPU cores above roughly
   1-2k configurations per launch. Solver iterations with a few hundred configurations
   (one trajopt step, one Cartesian waypoint layer) are faster on the CPU unless many
   iterations are fused into one submission. This confirms the PRD's "fuse iterations"
   rule with a number, and argues for keeping the CPU path first-class rather than as
   a fallback.
3. **Fusing wins at scale; small batches underfill the GPU.** Fused is 1.5-2.3x faster
   than split above ~32k configurations (no round trip of world spheres through
   memory). Below ~4k, split is as fast or slightly faster, because one thread per
   configuration leaves the GPU mostly idle. Thread mapping should adapt: one thread per
   (configuration, sphere group) with a workgroup reduction for small batches.
4. **Compiler autovectorization isn't enough on the CPU.** The 8-lane layout gains only
   ~1.5x per core over scalar code at small batches and nothing at large ones (libm
   `sin_cos` per lane and select chains). Explicit SIMD (`std::simd` / `wide` / `pulp`,
   with a vectorized sincos) is the obvious next step; 2-4x per core is plausible.
5. **f32 everywhere is fine for this query.** The worst f32 error is 3e-7 m against a
   2-5 cm voxel / sphere scale, and no collision verdict changes.
6. **WGSL against the shared layout was straightforward.** The byte-identical Rust/WGSL
   structs (no vec3 in storage, explicit padding) worked first time; the async-only
   readback required by WebGPU costs little natively.

## Phase 1 progress: CPU core, Cartesian planning, compas_fab backend

### Closed-form IK

`motion_core::ik`: the offset-wrist (UR) closed form ported from compas_fab's
`offset_wrist.py` (itself from ROS `ur_kin.cpp`), working in model terms: the target is
the `tool0` frame and solutions are model joint values wrapped into the limits.

- The closed-form chain matches the model FK to < 1e-9 (500 random configurations).
- 2000 random targets, 14,376 solutions: every one reaches its target within 6e-10.
  8 solutions near singularities were off by ~1e-5 before a damped Newton polish (now
  fixed or dropped). The source configuration is recovered for every target except at
  wrist singularities (sin q5 ≈ 0), where q4 + q6 is all that is determined and IK
  returns other valid configurations.
- **Spherical-wrist arms are solved from the model itself**, not ported: product of
  exponentials over the zero-configuration joint axes plus Paden–Kahan subproblems
  (joint 1 from the arm-plane offset, joints 2–3 from subproblems 3 and 1, the wrist from
  subproblems 2 and 1). No per-robot parameters or joint conventions, unlike compas_fab's
  Lobster-based solver with its per-robot pre/post-processing. Used automatically when no
  solver is given and the geometry qualifies (axes 4–6 meet within 0.1 mm).

| robot | targets | solutions | worst FK residual | source recovered |
|---|---|---|---|---|
| UR5 (ported closed form) | 2000 | 14,376 | 6e-10 | 2000 |
| ABB IRB4600-40/2.55 (from model) | 2000 | 9,332 | 1.3e-10 | 2000 |
| Stäubli TX2-60L (from model) | 2000 | 12,935 | 6e-14 | 2000 |

  (Fewer than 8 solutions per target where joint limits cut branches. "Recovered" accepts
  any solution at wrist singularities and within 10 mrad at a stretched elbow, where such
  configurations reach the same pose to 1e-10.) The polish is Levenberg–Marquardt: plain
  damped Newton did not converge at the ABB's elbow singularity, where its wrist axes
  missing each other by 1.6 µm leave the closed form a few µm off.

### Self-collision and the sphere model

Self-collision checks sphere pairs between links not disabled in the SRDF. Getting the
sphere model right took most of the night; what was learned:

| sphere model | self-collision verdicts (1000 random configs) |
|---|---|
| covering spheres (k-means, enclose the surface) | ~99% of configs flagged: neighbouring links' spheres overlap permanently (forearm vs wrist 2 in 98% of configs) |
| covering + auto-disable "always colliding" pairs | hides real collisions: PyBullet reports forearm vs wrist 2 in 27% of configs |
| **inscribed spheres** (inside the mesh, trimesh signed distance, 1 cm surface tolerance, 100 spheres) | **3.9% missed** real self-collisions, 5% false alarms (vs exact meshes) |
| inscribed, 5 mm tolerance (196 spheres) | 1.8% missed |

"Exact meshes" means dense surface samples of one link tested against the other's
watertight mesh. PyBullet is **not** ground truth here: it collides convex hulls, and in
92 of 181 disagreements the spheres were right and PyBullet wrong. Remaining misses are
almost all forearm vs wrist 2/3, where clearances are below the fitting tolerance.

For obstacles the same trade-off shows (vs PyBullet, then exact meshes on the disputes):

| obstacle spheres | false alarms | missed real contacts |
|---|---|---|
| covering (32 spheres) | 11.4% | 0 |
| inscribed + 1 cm | 0.9% | ~16 (vs hulls) |
| **inscribed + 2 cm (backend default)** | 1.9% | 2–3 per 1000 (exact meshes; thin edges) |

Covering spheres also break any cell with a floor: spheres around joint housings poke
through the floor the robot stands on, so every pose collides. The backend therefore
uses one inscribed set (+2 cm padding for obstacles); `sphere_options={"obstacle_spheres":
"covering"}` keeps the conservative variant. Either way **the PRD's exact-mesh final
pass on candidate paths is required** before execution.

Two MoveIt-style rules make tools and floors work: link pairs whose self spheres overlap
in ≥ 95% of random configurations are disabled (an attached tool against the wrist it is
mounted on), and obstacle spheres that overlap a box in ≥ 95% of configurations are
exempted from it (the bottom of the shoulder above the floor) — per sphere, so the rest
of the link is still checked. compas_fab's `RigidBodyState.touch_links` are honoured.

### Cartesian planner (layered graph + DP)

`motion_core::cartesian`: one layer per waypoint; nodes are all IK solutions, sampled
over a free rotation about the tool axis; all nodes of all layers are collision-checked
in one batch; DP finds the cheapest continuous path (joint step limit, 2π-equivalents for
joints with > 2π range); edges on the best path are interpolated and checked lazily.

Line over the table, 100 waypoints, UR5, same scene and waypoints for every planner:

| planner | time | |
|---|---|---|
| compas_fab `AnalyticalPyBulletPlanner` | – | **broken in 2.1**: passes `Frame`s to `iter_inverse_kinematics`, which only accepts `FrameTarget` (its tests are commented out) |
| same, with that one-line fix applied in the benchmark | 306 ms | one IK branch for the whole path |
| compas_fab `PyBulletPlanner` | fails | numeric IK gives up at segment 68 (162 ms for the first 60 waypoints) |
| **Rust core**, fixed tool axis | **0.64 ms** | |
| Rust core, 12 tool-axis samples (9,600 IK nodes) | 3.9 ms | not available in compas_fab |
| **`NativePlanner.plan_cartesian_motion`** (through the compas_fab API) | **1.6 ms** | ~190× the fixed analytical planner |
| `NativePlanner`, 12 tool-axis samples | 5.4 ms | |
| `NativePlanner.set_robot_cell` (sphere fitting, once per cell) | 1.7 s | trimesh signed distance; cacheable |

Through the API, half of the first version's 10 ms went to compas_fab's
`Target.normalized_to_meters` (deep-copies all waypoints even at scale 1) and a
determinant-based matrix inverse per waypoint; both are avoided now.

### Exact final pass (parry3d)

`motion_exact`: the robot's collision meshes per link (attached tools merged into the
end-effector link) and the obstacles' meshes, checked with parry3d (BVH mesh-mesh
intersection plus a point-in-mesh test for containment), honouring SRDF disabled pairs and
touch links. It is part of the planner's lazy loop, not just a post-check: once a path
passes the sphere checks, its nodes and interpolated edges are checked exactly (in
parallel); failing nodes/edges are blocked and the search is repeated. Results are
collision-free on the exact meshes.

- Agrees with an independent exact check (trimesh signed distance on 1,500 surface samples
  per link) on 177 of 181 disputed configurations; in the other 4 parry finds real
  grazing surface intersections at the wrist that the sampling missed.
- ~200 µs per configuration single-threaded (UR5, 4,985 triangles, 4 boxes), vs ~1 µs for
  the sphere query: right for a final pass, not for search.
- Cartesian line, 100 waypoints, through the compas_fab API: **2.9 ms with the exact pass**
  (1.3 ms of it exact checks), 1.6 ms without. Still ~100× compas_fab's fixed analytical
  planner, now with a stronger guarantee (exact meshes rather than PyBullet's convex hulls).
- Answers part of a PRD open question: parry3d is good enough for the exact CPU pass; the
  hot path stays on spheres, so writing our own mesh kernels is not needed for Phase 1.

### Offline ESDF for non-box obstacles

`motion_core::esdf`: the voxel-block layout from Phase 0, now exercised. 8³ grid points
per block, a hash from block coordinates to pool slots, trilinear distance and gradient;
missing blocks read as the truncation distance. `motion_exact::build_esdf` fills it from
obstacle meshes with parry3d (signed distance, blocks far from every surface skipped,
parallel per block). The CPU kernels query it next to the boxes. GPU kernels don't yet.

- Accuracy (4 boxes as meshes, 2 cm grid, 0.3 m band): median error 0, p99 3.8 mm, max
  10 mm (bound: half the voxel diagonal, 17 mm). 1,853 blocks, 3.8 MB, built in 55 ms.
- Collision verdicts vs exact boxes, 4,096 configs: 1 miss without margin; 0 misses and 178
  extra with a half-diagonal margin. The backend uses a 1 cm margin and relies on the exact
  pass for the rest.
- Cost: 2.5 ms per 4,096 configurations against 0.85 ms for four analytic boxes (hash
  lookups), but independent of how complex the geometry is.
- `NativePlanner(obstacles="esdf")` puts static rigid bodies into the ESDF (bodies with
  touch links, like a floor, stay boxes, since the field merges all bodies). A tool plunging
  into a ring (R = 22 cm): **boxes fail (the ring's bounding box fills the hole), the ESDF
  plans it in 6.6 ms**, exact-collision-free; the ring's ESDF builds in 12 ms.

### NativePlanner (compas_fab backend)

`compas_fab.backends.native.NativePlanner(UR5Kinematics())` implements
`set_robot_cell(_state)`, `forward_kinematics`, `iter_inverse_kinematics`,
`check_collision` and `plan_cartesian_motion` (`TargetMode.ROBOT` and `TOOL`, option
`free_axis_samples`, `exact_check`). Tests (`tests/backends/native`) check FK against
compas_robots (with a moved robot base), IK round trips, collisions with a box, Cartesian
paths in robot and tool mode (UR5 with the cone tool, free axis), and the exact pass. The rest of the compas_fab
suite is unaffected (195 passed, 22 skipped).

## Open issues and next steps

- The exact pass covers Cartesian planning; IK uses the sphere model plus
  `check_collision(..., {"exact": True})` on request.
- ESDF: per-obstacle touch semantics (one field per body, or a body id per voxel), GPU
  query (the layout is GPU-ready), and building from sensed data (Phase 3).
- **compas_fab bug:** `AnalyticalPyBulletPlanner.plan_cartesian_motion` /
  `AnalyticalKinematicsPlanner.plan_cartesian_motion` with frame waypoints always raise
  `BackendTargetNotSupportedError`, and the tests that would catch it are commented out.
  Fixed with regression tests on the local branch `fix/analytical-cartesian-frame-targets`
  (off `main`, not pushed).
- External axes, 7-DoF redundancy sampling.
- Attached workpieces are ignored by the backend; tools are supported.
- Explicit SIMD on the CPU (finding 4); the self-collision pair loop is now the largest
  CPU cost (1,984 pairs for the UR5).
- Packaging: maturin / abi3 wheels through compas-actions; the spike builds with cargo.

### Phase 0 leftovers

- CubeCL compiling to MSL directly (`--features msl`) gave the same numbers, but I could
  not confirm the MSL path was active; CubeCL's CPU (MLIR) runtime not tried.
- Runs on Windows (DX12/Vulkan) and an NVIDIA machine; cuRobo and MoveIt baselines.
- Voxel-block layout is defined (`layout.rs`) but not exercised by a kernel.
- compas_pb messages for the model (the spike passes JSON).

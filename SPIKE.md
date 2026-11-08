# Native motion core: spike (Phase 0, Phase 1 pieces, start of Phase 2)

Spike for the PRD "compas_fab native motion core (planning + perception)".

* **Phase 0:** the batched sphere-distance kernel on both portable GPU candidates
  (wgpu + WGSL, and CubeCL), buffer layouts including voxel blocks, first baselines
  against the current compas_fab backends, and a recommendation for the GPU stack.
* **Phase 1, first pieces (CPU):** closed-form IK, self-collision, the layered-graph
  Cartesian planner, Python bindings and a `NativePlanner` compas_fab backend, benchmarked
  against compas_fab's existing Cartesian planners.
* **Phase 2, start:** a SIMD yes/no collision checker and free-space planning
  (RRT-Connect), benchmarked against VAMP on its own problems, and
  `NativePlanner.plan_motion` for any serial robot.

Jump to: [Phase 0 gate](#phase-0-gate-wgpu--wgsl-or-cubecl) ·
[Phase 1 progress](#phase-1-progress-cpu-core-cartesian-planning-compas_fab-backend) ·
[Free-space vs VAMP](#phase-2-start-free-space-planning-benchmarked-against-vamp) ·
[Open issues](#open-issues-and-next-steps)

## What is here

Third-party code, data and algorithms used here (ur_kinematics, VAMP, robowflex_resources,
Cephes, MotionBenchMaker) are credited in [`NOTICE`](NOTICE), with their licenses in
[`licenses/`](licenses); the wheel ships both.

```
native/
  scripts/export_robot.py       RobotCellLibrary robot -> fixture: kinematic tree, link spheres, reference FK
  scripts/baseline_python.py    compas_robots FK and PyBullet check_collision on the same robot + scene
  fixtures/ur5.json             UR5: 12 links, 6 dof, 32 spheres, 50 reference configurations
  fixtures/scene_boxes.json     4 cuboids (table, column, rotated beam, wall)
  crates/motion_core            layouts (layout.rs), f64 reference, CPU f32 kernels (scalar + 8-lane), sampling,
                                yes/no checker (check.rs), free-space planner (freespace.rs)
  crates/motion_gpu             wgpu host code + query.wgsl (fused full, fused min-only, split)
  crates/motion_bench           oracle checks + timing (motion_bench), sphere vs mesh verdicts (compare_verdicts),
                                Cartesian planner on the baseline's waypoints (cartesian_bench),
                                single-threaded CPU benchmarks run by CI on every commit (ci_bench),
                                VAMP comparison (mbm_bench, vamp_parity, check_micro)
  fixtures/vamp_*.json          VAMP's spherized Panda and UR5 (scripts/export_vamp_robot.py)
  scripts/vamp_*.py             parity, path checks and planning on VAMP's side (run in a VAMP venv)
  crates/motion_py              PyO3 extension `compas_fab_native` (NativeCore)
  pyproject.toml                maturin build of the extension: `pip install ./native` (abi3 wheel, Python >= 3.9)
  scripts/baseline_cartesian.py compas_fab's Cartesian planners vs NativePlanner on the same path
  results/                      raw numbers from the runs below
  demo/live_viewer/             live browser demo: Python websocket server + compas_threejs_ts page (README there)

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
cargo run --release --bin motion_bench -- --out results/<machine>.json
cargo run --release --bin ci_bench -- bench.json                          # what CI measures
python scripts/baseline_python.py ur5 fixtures/scene_boxes.json results/baseline_python.json   # from compas_fab's venv
cargo run --release --bin compare_verdicts -- results/baseline_python.json

# Python extension + backend, from compas_fab's venv
maturin develop --release           # or, from the repository root: pip install ./native
cd .. && python -m pytest tests/backends/native
python native/scripts/baseline_cartesian.py native/fixtures/scene_boxes.json native/results/baseline_cartesian.json
for f in docs/backends/native/files/*.py; do python $f; done   # the examples
```

VAMP comparison (VAMP 0.6.4 built from source in its own venv, `<vamp>` its checkout;
after `python <vamp>/resources/problem_tar_to_pkl_json.py --robot panda`):

```bash
cargo run --release -p motion_bench --bin mbm_bench -- fixtures/vamp_panda.json <vamp>/resources/panda/problems.json --range 1.0 --out paths.json
python <vamp>/scripts/evaluate_mbm.py --robot panda                       # VAMP's own numbers
cargo run --release -p motion_bench --bin vamp_parity -- fixtures/vamp_panda.json <vamp>/resources/panda/problems.json parity.json
python scripts/vamp_parity.py panda <vamp>/resources/panda/problems.json parity.json
python scripts/vamp_check_paths.py panda <vamp>/resources/panda/problems.json paths.json
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

**Decided: wgpu + WGSL.** The CubeCL crate (`motion_cubecl`) was removed after the
comparison below; it is in the history (commit `67ed40e24`)
if CUDA is ever wanted for a specific kernel.

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

## Phase 2 start: free-space planning, benchmarked against VAMP

`motion_core::check` is a yes/no collision checker for planners and
`motion_core::freespace` a free-space planner on top of it:

- **Checker:** 8 configurations at once in explicit SIMD (`wide::f32x8`). It tests one
  bounding sphere per link before the link's spheres, and computes kinematics and
  environment checks link by link, stopping at the first contact. Fixed joints are folded
  into the next moving joint, links rigidly attached to the base are checked once when
  the checker is built, and touch links and per-sphere box exemptions are honoured.
  Obstacles are spheres, capsules, oriented boxes and the ESDF.
- **Planner:** the straight motion first, then RRT-Connect (balanced, dynamic domain) and
  shortcut + B-spline simplification, with VAMP's default settings and a Halton sampler.
  Motions are checked with VAMP's "rake": the lanes start evenly spread over the motion
  and step back together.
- **compas_fab:** `NativePlanner.plan_motion` takes `ConfigurationTarget`, `FrameTarget`
  (every IK solution is a goal) and `PointAxisTarget` (IK of rotations about the axis).
  It works for any serial robot (7-axis Panda tested) and checks the path on the exact
  meshes, re-planning with another seed if it touches.

### Same problems, same robot model as VAMP

VAMP 0.6.4 was built from source on the same machine, and its MotionBenchMaker problems
(7 scenarios x 100) were run through both planners.

- **Same robot:** `fixtures/vamp_{panda,ur5}.json` are VAMP's spherized URDFs and SRDFs
  (`scripts/export_vamp_robot.py`). Sphere positions match VAMP's `fk` to 3e-7 m.
- **Same verdicts:** on 21,084 configurations (both robots) our checker and
  `vamp.<robot>.validate` give 0 different verdicts (`scripts/vamp_parity.py`). One pair
  needed the always-colliding rule: the UR5's `wrist_2_link` and `fts_robotside` overlap
  in every configuration, the SRDF does not disable them, and VAMP does not check them.
- **Same algorithm:** iteration counts match VAMP's exactly (Panda: mean 471.5, median
  104, max 42,794), and so do path lengths before and after simplification.
- **Same path quality:** checked by VAMP itself (`scripts/vamp_check_paths.py`), 2 of our
  699 Panda paths have a segment VAMP's own check rejects (VAMP: 2), and 41 touch an
  obstacle when checked at 256 points per radian (VAMP: 42). Discrete checking misses
  thin contacts by design; our exact-mesh pass is what catches them in compas_fab.

Single thread, Apple M3 Max, all problems solved by both (`results/mbm_m3max.md`):

| | Panda: native | Panda: VAMP | UR5: native | UR5: VAMP |
|---|---:|---:|---:|---:|
| planning, median | 100 µs | 69 µs | 114 µs | 75 µs |
| simplification, median | 250 µs | 191 µs | 357 µs | 247 µs |
| total, median | 367 µs | 268 µs | 511 µs | 341 µs |
| total, 95th percentile | 1.18 ms | 0.81 ms | 1.95 ms | 1.16 ms |

**Within 1.35x (Panda) to 1.5x (UR5) of VAMP, from a robot description at run time.**
VAMP generates C++ per robot with every constant folded in. Closing the rest would take
the same kind of specialization (generated code per robot, or kernels specialized on the
sparsity of the joint origins). For compas_fab, planning time is already well below the
exact-mesh check of the result (about 1-2 ms per path), so that check is the next target.

What the checker went through on the way (findings for every SIMD kernel here):

1. The "LLVM auto-vectorizes the lane loops" assumption did not hold: the first version
   compiled to scalar code (361 `fmadd`, 0 vector FMAs). **This also applies to the older
   distance kernel in `cpu.rs`**: 1.4 µs per configuration for the UR5, against 78 ns
   for the yes/no checker on the Panda. Explicit SIMD fixed it.
2. `wide`'s `splat` builds an array in memory, which LLVM lowered to `memset_pattern16`
   calls inside the kinematics on macOS (16 calls per check). Broadcasting through the
   architecture intrinsics (`vdupq_n_f32`, `_mm256_set1_ps`) halved the kinematics time.
3. 8 lanes (two NEON registers) beat 4 lanes (one) on Apple Silicon end to end.

CI (`native.yml`) now also tracks the yes/no check and free-space planning on a
synthetic shelf scene, since the MotionBenchMaker problems are VAMP's data.

Follow-ups done since:

- **The distance kernel (`cpu.rs`) on explicit SIMD too**, sharing `simd.rs` with the
  checker: UR5 + boxes 1.47 µs → 0.60 µs per configuration, Cartesian planning -44%.
- **Joints outside the planning group** (gripper fingers, mimic joints) take their values
  from the state instead of zero, baked into the joint origins (`set_passive_joints`).
- **Time parameterization** (`timing.rs`): straight segments with circular blends of
  bounded deviation (Kunz and Stilman 2012), blends checked for collision and shrunk down to
  a stop when they collide, then TOPP-RA (Pham and Pham 2018) under joint velocity and
  acceleration limits, at rest at both ends. Limits are enforced at both ends of every grid
  step (TOPP-RA's "interpolation" scheme); with only the start enforced, accelerations
  overshot by 3% on blends. Tests: textbook trapezoid and triangle durations for one joint,
  limits within 0.1% (velocity) and 2% (acceleration) on a blended 6-joint path, blends
  within their tolerance, stops where blends are rejected. `plan_motion` returns timed
  points by default (URDF velocity limits; acceleration limits from the options) and checks
  the timed, blended motion on the exact meshes, falling back to stops if it touches.
- **Timing fix (found by the live demo):** the B-spline simplification leaves collinear
  midpoints, and `acos` of two equal unit directions came out a few 1e-8 rad instead of 0.
  The blend's bisector `a - b` was then exactly zero, so its arc was NaN, and the TOPP-RA
  backward pass reported `Infeasible`. The turning angle is now `2 atan2(|a - b|, |a + b|)`.
  Vertices that turn by less than 1e-6 rad are passed straight through; they used to be
  full stops. `time_parameterize` also reports `stops` per vertex. Regression test:
  `collinear_vertices_are_passed_through`.
- **Live demo** (`demo/live_viewer`): the five scenarios (UR5 around a wall, Panda around a
  post, servoing a Panda, milling a spiral, plunging through a ring) are planned while you
  drag targets and obstacles in a compas_threejs_ts viewer, and streamed over a websocket at
  60 Hz. Planning takes 1 to 45 ms, except the Panda's numeric IK (about 120 ms in Python).
  ESDF note from the ring: with 2 cm voxels and a 1 cm margin the sphere search can graze a
  5 cm tube that the exact check then rejects (5 wasted attempts). The demo uses 1.5 cm
  voxels with a 1.5 cm margin. The default margin should probably scale with the voxel size.

## Open issues and next steps

- The exact pass covers Cartesian planning; IK uses the sphere model plus
  `check_collision(..., {"exact": True})` on request.
- ESDF: per-obstacle touch semantics (one field per body, or a body id per voxel), GPU
  query (the layout is GPU-ready), and building from sensed data (Phase 3).
- **compas_fab bug:** `AnalyticalPyBulletPlanner.plan_cartesian_motion` /
  `AnalyticalKinematicsPlanner.plan_cartesian_motion` with frame waypoints always raise
  `BackendTargetNotSupportedError`, and the tests that would catch it are commented out.
  Fixed with regression tests on the branch `fix/analytical-cartesian-frame-targets`.
- External axes, 7-DoF redundancy sampling for frame targets and Cartesian paths.
- Attached workpieces are ignored by the backend; tools are supported.
- Free-space: jerk limits (timing is velocity- and acceleration-limited), the exact-mesh
  check dominates `plan_motion` (about 2-5 ms with timing), FCIT* or AORRTC for better
  paths, timing for Cartesian trajectories (`core.time_parameterize` is ready for it).
- `fixtures/vamp_*.json` are derived from VAMP's resources (Apache-2.0, from
  robowflex_resources): the attribution question of the PRD applies to them.
- Packaging: `native/pyproject.toml` builds an abi3 wheel with maturin (`compas_fab_native`,
  a separate distribution from the same repository, as pydantic does with pydantic-core).
  Not yet: wheels for all platforms in the release workflow, and `compas_fab[native]`.
- Sphere fitting still runs in Python (trimesh, at `set_robot_cell`: ~1.7 s for a UR5);
  moving it to the core would drop the trimesh/rtree/scipy dependencies.

### Phase 0 leftovers

- Runs on Windows (DX12/Vulkan) and an NVIDIA machine; cuRobo and MoveIt baselines.
- Voxel-block layout is defined (`layout.rs`) but not exercised by a kernel.
- compas_pb messages for the model (the spike passes JSON). Deferred on purpose: the
  only consumer is the in-process Python binding, where JSON is cheap and loosely coupled.
  compas_pb starts paying off with the first out-of-process or non-Python consumer
  (browser/WASM, a planning service, Rhino talking to a separate process).

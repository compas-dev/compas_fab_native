# Native motion core: Phase 0 spike (wgpu)

Spike for the PRD "compas_fab native motion core (planning + perception)", Phase 0:
the batched sphere-distance kernel on a portable GPU path (wgpu first; CubeCL to follow
for the comparison), the buffer layouts including voxel blocks, and first baselines
against the current compas_fab backends.

Nothing here is a compas_fab backend yet, and nothing is wired into the Python package.

## What is here

```
native/
  scripts/export_robot.py       RobotCellLibrary robot -> fixture: kinematic tree, link spheres, reference FK
  scripts/baseline_python.py    compas_robots FK and PyBullet check_collision on the same robot + scene
  fixtures/ur5.json             UR5: 12 links, 6 dof, 32 spheres, 50 reference configurations
  fixtures/scene_boxes.json     4 cuboids (table, column, rotated beam, wall)
  crates/motion_core            layouts (layout.rs), f64 reference, CPU f32 kernels (scalar + 8-lane), sampling
  crates/motion_gpu             wgpu host code + query.wgsl (fused full, fused min-only, split)
  crates/motion_bench           oracle checks + timing (motion_bench), sphere vs mesh verdicts (compare_verdicts)
  results/                      raw numbers from the runs below
```

The operation in every backend: configurations (SoA, `q[j * B + b]`) -> batched FK ->
link spheres in world -> signed distance to the nearest cuboid minus radius, with
gradient, per sphere -> per-configuration minimum. Layouts are documented in
`crates/motion_core/src/layout.rs`.

Run:

```bash
cd native
cargo test --release                      # oracle chain: compas_robots -> f64 -> f32 CPU
cargo run --release --bin motion_bench -- --out results/<machine>.json
python scripts/baseline_python.py ur5 fixtures/scene_boxes.json results/baseline_python.json   # from compas_fab's venv
cargo run --release --bin compare_verdicts -- results/baseline_python.json
```

## Results (Apple M3 Max, 14 CPU threads, Metal via wgpu 30)

UR5, 32 spheres, 4 cuboids. Full table in `results/m3max.json`.

### Correctness chain

| check | max error | collision verdicts differing |
|---|---|---|
| Rust f64 FK vs compas_robots (50 configs, every link frame) | < 1e-9 | – |
| CPU f32 vs f64 reference | 1.1e-7 m | – |
| CPU 8-lane vs CPU scalar | 0 (bit-identical) | 0 / 4096 |
| GPU fused / split / min-only vs CPU scalar | 3.0e-7 m, gradient 1.0e-5 | 0 / 4096 |
| Sphere model vs PyBullet exact meshes (1000 configs) | – | 0 missed collisions; 76 false alarms (11% of free configs) |

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

## Not done yet (needed for the Phase 0 gate)

- **CubeCL version of the same kernel**, in the same harness, to choose the portable
  GPU stack. The harness takes a new backend as one more row.
- GPU timestamp queries, to separate kernel time from submission overhead in the
  latency numbers.
- Runs on Windows (DX12/Vulkan) and an NVIDIA machine; cuRobo and MoveIt baselines.
- Self-collision (robot spheres vs robot spheres) is not in the query yet; the
  PyBullet comparison above is robot-vs-obstacles only.
- Voxel-block layout is defined (`layout.rs`) but not exercised by a kernel.
- Better sphere fitting (the 11% false alarms) and compas_pb messages for the model.

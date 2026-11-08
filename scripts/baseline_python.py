"""Baselines for the native core spike: what the same work costs through compas_fab today.

Measures, per configuration, on the robot and scene fixtures used by ``motion_bench``:

* compas_robots forward kinematics of every link (``RobotModel.transformed_frames``);
* compas_fab's PyBullet backend ``check_collision``, both the full check and
  robot-vs-obstacles only (``_skip_cc1``, which is what the sphere query covers).

It also stores the PyBullet verdicts (exact meshes) so the sphere approximation
can be compared against them (``compare_verdicts``).

Usage::

    python native/scripts/baseline_python.py ur5 native/fixtures/scene_boxes.json native/results/baseline_python.json
"""

import argparse
import json
import platform
import time

import numpy as np
from compas.geometry import Box
from compas.geometry import Frame

from compas_fab.backends import CollisionCheckError
from compas_fab.backends import PyBulletClient
from compas_fab.backends import PyBulletPlanner
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary


def frame_from_3x4(m):
    m = np.asarray(m, dtype=float).reshape(3, 4)
    return Frame(m[:, 3], m[:, 0], m[:, 1])


def timed(fn, configs):
    out = []
    start = time.perf_counter()
    for c in configs:
        out.append(fn(c))
    return (time.perf_counter() - start) / len(configs), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot")
    parser.add_argument("scene")
    parser.add_argument("output")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cell, state = getattr(RobotCellLibrary, args.robot)()
    group = cell.main_group_name
    model = cell.robot_model
    joints = cell.get_configurable_joints(group)
    lows = np.array([j.limit.lower for j in joints])
    highs = np.array([j.limit.upper for j in joints])
    values = lows + (highs - lows) * rng.random((args.n, len(joints)))

    with open(args.scene) as f:
        scene = json.load(f)
    for c in scene["cuboids"]:
        h = c["half_extents"]
        mesh = Box(2 * h[0], 2 * h[1], 2 * h[2]).to_mesh(triangulated=True)
        cell.rigid_body_models[c["name"]] = RigidBody.from_mesh(mesh)
    state = cell.default_cell_state()
    for c in scene["cuboids"]:
        state.rigid_body_states[c["name"]] = RigidBodyState(frame_from_3x4(c["frame"]))

    def configuration(v):
        config = cell.zero_configuration(group)
        config.joint_values = list(v)
        return config

    configs = [configuration(v) for v in values]

    t_fk, _ = timed(model.transformed_frames, configs)
    print("compas_robots FK, all links:          {:8.1f} µs/config".format(t_fk * 1e6))

    results = {}
    with PyBulletClient(connection_type="direct", verbose=False) as client:
        planner = PyBulletPlanner(client)
        planner.set_robot_cell(cell, state)

        def check(options):
            def fn(config):
                s = state.copy()
                s.robot_configuration = config
                try:
                    planner.check_collision(s, options)
                    return False
                except CollisionCheckError:
                    return True

            return fn

        for name, options in [("full", {}), ("robot_vs_obstacles", {"_skip_cc1": True})]:
            t, verdicts = timed(check(options), configs)
            results[name] = {"seconds_per_config": t, "verdicts": verdicts}
            print("PyBullet check_collision ({:<20}) {:8.1f} µs/config, {} / {} in collision".format(name, t * 1e6, sum(verdicts), len(verdicts)))

    out = {
        "robot": args.robot,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "n": args.n,
        "configs": values.tolist(),
        "compas_robots_fk_seconds_per_config": t_fk,
        "pybullet": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print("wrote", args.output)


if __name__ == "__main__":
    main()

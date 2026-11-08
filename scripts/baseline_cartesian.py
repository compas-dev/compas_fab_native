"""Cartesian planning baseline: compas_fab's existing planners on the spike's scene.

A straight line over the table (flange pointing down), with the scene's boxes as
obstacles, planned with:

* ``AnalyticalPyBulletPlanner`` (UR5 closed-form IK + PyBullet collision checks);
* ``PyBulletPlanner`` (PyBullet numeric IK, interpolating between waypoints).

Writes the waypoints (so the native planner plans exactly the same path) and timings.

Usage::

    python native/scripts/baseline_cartesian.py native/fixtures/scene_boxes.json native/results/baseline_cartesian.json
"""

import argparse
import json
import platform
import time

from compas.geometry import Box
from compas.geometry import Frame

from compas_fab.backends import AnalyticalPyBulletClient
from compas_fab.backends import AnalyticalPyBulletPlanner
from compas_fab.backends import PyBulletClient
from compas_fab.backends import PyBulletPlanner
from compas_fab.backends import UR5Kinematics
from compas_fab.robots import FrameTarget
from compas_fab.robots import FrameWaypoints
from compas_fab.robots import RigidBody
from compas_fab.robots import RigidBodyState
from compas_fab.robots import RobotCellLibrary
from compas_fab.robots import TargetMode


def line_over_table(n):
    """Flange pointing down, 0.6 m along Y at x = 0.45, z = 0.5 (same as the Rust test)."""
    return [Frame([0.45, -0.3 + 0.6 * i / (n - 1), 0.50], [1, 0, 0], [0, -1, 0]) for i in range(n)]


def frame_to_3x4(f):
    x, y, z, p = f.xaxis, f.yaxis, f.zaxis, f.point
    return [x[0], y[0], z[0], p[0], x[1], y[1], z[1], p[1], x[2], y[2], z[2], p[2]]


def cell_with_scene(scene):
    cell, _ = RobotCellLibrary.ur5()
    for c in scene["cuboids"]:
        h = c["half_extents"]
        cell.rigid_body_models[c["name"]] = RigidBody.from_mesh(Box(2 * h[0], 2 * h[1], 2 * h[2]).to_mesh(triangulated=True))
    state = cell.default_cell_state()
    for c in scene["cuboids"]:
        m = c["frame"]
        frame = Frame([m[3], m[7], m[11]], [m[0], m[4], m[8]], [m[1], m[5], m[9]])
        state.rigid_body_states[c["name"]] = RigidBodyState(frame)
    return cell, state


def timed(fn, repeats):
    times, result, error = [], None, None
    for _ in range(repeats):
        t = time.perf_counter()
        try:
            result = fn()
        except Exception as e:  # report planner failures as results, not crashes
            error = "{}: {}".format(type(e).__name__, e)
        times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2], result, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("output")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    with open(args.scene) as f:
        scene = json.load(f)
    frames = line_over_table(args.n)
    waypoints = FrameWaypoints(frames, TargetMode.ROBOT)
    results = {}

    with AnalyticalPyBulletClient(connection_type="direct", verbose=False) as client:
        cell, state = cell_with_scene(scene)
        planner = AnalyticalPyBulletPlanner(client, UR5Kinematics())
        planner.set_robot_cell(cell, state)
        # Start near the first waypoint: the first collision-free IK solution.
        start = next(c for c in planner.iter_inverse_kinematics(FrameTarget(frames[0], TargetMode.ROBOT), state) if c)
        state.robot_configuration = start

        def run():
            return planner.plan_cartesian_motion(waypoints, state, cell.main_group_name, options={"check_collision": True})

        t, traj, err = timed(run, args.repeats)
        results["analytical_pybullet"] = {"seconds": t, "points": len(traj.points) if traj else 0, "error": err}
        print("{:<26} {:.3f} s, {} points {}".format("analytical_pybullet", t, results["analytical_pybullet"]["points"], err or ""))

        # compas_fab 2.1 bug: the analytical Cartesian planner passes plain Frames to
        # iter_inverse_kinematics, which only accepts FrameTarget. Patch it here (only in this
        # benchmark) to time what the planner does once fixed.
        original = planner.iter_inverse_kinematics

        def iter_ik(target, *a, **kw):
            if isinstance(target, Frame):
                target = FrameTarget(target, TargetMode.ROBOT)
            return original(target, state, *a[1:], **kw) if a else original(target, state, **kw)

        planner.iter_inverse_kinematics = iter_ik
        t, traj, err = timed(run, args.repeats)
        results["analytical_pybullet_patched"] = {"seconds": t, "points": len(traj.points) if traj else 0, "error": err}
        print("{:<26} {:.3f} s, {} points {}".format("analytical_pybullet (fix)", t, results["analytical_pybullet_patched"]["points"], err or ""))

    with PyBulletClient(connection_type="direct", verbose=False) as client:
        cell, state = cell_with_scene(scene)
        planner = PyBulletPlanner(client)
        planner.set_robot_cell(cell, state)
        state.robot_configuration = start
        for name, wps in [("pybullet", waypoints), ("pybullet_first60", FrameWaypoints(frames[:60], TargetMode.ROBOT))]:
            t, traj, err = timed(lambda: planner.plan_cartesian_motion(wps, state, cell.main_group_name), args.repeats)
            results[name] = {
                "seconds": t,
                "waypoints": len(wps.target_frames),
                "points": len(traj.points) if traj else 0,
                "error": err,
            }
            print("{:<26} {:.3f} s, {} waypoints -> {} points {}".format(name, t, len(wps.target_frames), results[name]["points"], err or ""))

    try:
        from compas_fab.backends.native import NativePlanner

        cell, state = cell_with_scene(scene)
        planner = NativePlanner(UR5Kinematics())
        t = time.perf_counter()
        planner.set_robot_cell(cell, state)
        setup = time.perf_counter() - t
        state.robot_configuration = start
        for name, options in [
            ("native", {}),
            ("native_12_axis_samples", {"free_axis_samples": 12}),
        ]:
            t, traj, err = timed(lambda: planner.plan_cartesian_motion(waypoints, state, cell.main_group_name, options), args.repeats)
            results[name] = {"seconds": t, "setup_seconds": setup, "points": len(traj.points) if traj else 0, "error": err}
            print("{:<26} {:.4f} s, {} points (one-off setup {:.2f} s) {}".format(name, t, results[name]["points"], setup, err or ""))
    except ImportError as e:
        print("native backend not available:", e)

    out = {
        "machine": platform.platform(),
        "waypoints": [frame_to_3x4(f) for f in frames],
        "start": list(start.joint_values),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print("wrote", args.output)


if __name__ == "__main__":
    main()

"""NativePlanner (spike) against compas_fab's own kinematics. Skipped without the extension."""

import math

import pytest
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Transformation

pytest.importorskip("compas_fab_native")

from compas_fab.backends import CollisionCheckError  # noqa: E402
from compas_fab.backends import UR5Kinematics  # noqa: E402
from compas_fab.backends.native import NativePlanner  # noqa: E402
from compas_fab.robots import FrameTarget  # noqa: E402
from compas_fab.robots import FrameWaypoints  # noqa: E402
from compas_fab.robots import RigidBody  # noqa: E402
from compas_fab.robots import RigidBodyState  # noqa: E402
from compas_fab.robots import RobotCellLibrary  # noqa: E402
from compas_fab.robots import TargetMode  # noqa: E402


def frames_close(a, b, tol=1e-6):
    ta, tb = Transformation.from_frame(a), Transformation.from_frame(b)
    return all(abs(x - y) < tol for ra, rb in zip(ta.matrix, tb.matrix) for x, y in zip(ra, rb))


@pytest.fixture(scope="module")
def ur5():
    cell, state = RobotCellLibrary.ur5()
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    return planner, cell, state


@pytest.mark.parametrize("base", [None, Frame([0.3, -0.2, 0.1], [0, 1, 0], [-1, 0, 0])])
def test_forward_kinematics_matches_robot_model(ur5, base):
    planner, cell, state = ur5
    state = state.copy()
    state.robot_base_frame = base
    for _ in range(20):
        state.robot_configuration = cell.random_configuration(cell.main_group_name)
        frame = planner.forward_kinematics(state, TargetMode.ROBOT)
        expected = cell.robot_model.forward_kinematics(state.robot_configuration, "tool0")
        if base is not None:
            expected = Frame.from_transformation(Transformation.from_frame(base) * Transformation.from_frame(expected))
        assert frames_close(frame, expected)


def test_every_ik_solution_reaches_the_target(ur5):
    planner, cell, state = ur5
    state = state.copy()
    target = Frame([0.4, 0.2, 0.5], [1, 0, 0], [0, -1, 0])
    solutions = list(planner.iter_inverse_kinematics(FrameTarget(target, TargetMode.ROBOT), state, options={"check_collision": False}))
    assert 1 <= len(solutions) <= 8
    for config in solutions:
        state.robot_configuration = config
        assert frames_close(planner.forward_kinematics(state, TargetMode.ROBOT), target)


def test_collision_with_a_box():
    cell, _ = RobotCellLibrary.ur5()
    cell.rigid_body_models["box"] = RigidBody.from_mesh(Box(0.4, 0.4, 0.4).to_mesh(triangulated=True))
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell)
    state = cell.default_cell_state()
    state.robot_configuration = cell.zero_configuration()
    # Arm stretched along -X at zero configuration: a box there collides, a box far away does not.
    tool0 = cell.robot_model.forward_kinematics(state.robot_configuration, "tool0")
    state.rigid_body_states["box"] = RigidBodyState(Frame(tool0.point, [1, 0, 0], [0, 1, 0]))
    with pytest.raises(CollisionCheckError):
        planner.check_collision(state)
    state.rigid_body_states["box"] = RigidBodyState(Frame([3, 3, 3], [1, 0, 0], [0, 1, 0]))
    planner.check_collision(state)


def test_cartesian_line_robot_mode(ur5):
    planner, cell, state = ur5
    frames = [Frame([0.45, -0.3 + 0.6 * i / 49, 0.4], [1, 0, 0], [0, -1, 0]) for i in range(50)]
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.ROBOT), state.copy())
    assert len(trajectory.points) == len(frames)
    check = state.copy()
    for point, frame in zip(trajectory.points, frames):
        check.robot_configuration = cell.zero_configuration().merged(point)
        assert frames_close(planner.forward_kinematics(check, TargetMode.ROBOT), frame)
    steps = [max(abs(a - b) for a, b in zip(p.joint_values, q.joint_values)) for p, q in zip(trajectory.points, trajectory.points[1:])]
    assert max(steps) <= 0.5


def test_cartesian_tool_mode_with_free_axis():
    cell, state = RobotCellLibrary.ur5_cone_tool()
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    sphere_counts = planner.client.core.sphere_counts
    frames = [Frame([0.4 + 0.1 * math.cos(t), 0.1 * math.sin(t), 0.3], [1, 0, 0], [0, -1, 0]) for t in [2 * math.pi * i / 40 for i in range(41)]]
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.TOOL), state, options={"free_axis_samples": 12})
    check = state.copy()
    for point, frame in zip(trajectory.points, frames):
        check.robot_configuration = cell.zero_configuration().merged(point)
        tcf = planner.forward_kinematics(check, TargetMode.TOOL)
        # Position and tool axis match; rotation about the tool axis is free.
        assert tcf.point.distance_to_point(frame.point) < 1e-6
        assert tcf.zaxis.dot(frame.zaxis) > 1 - 1e-9
    # The tool added spheres to the end-effector link.
    bare = NativePlanner(UR5Kinematics())
    bare.set_robot_cell(*RobotCellLibrary.ur5())
    assert sphere_counts[0] > bare.client.core.sphere_counts[0]

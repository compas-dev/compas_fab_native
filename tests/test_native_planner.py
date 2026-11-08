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


def test_exact_final_pass_and_exact_check(ur5):
    planner, cell, state = ur5
    assert planner.client.core.has_exact
    frames = [Frame([0.45, -0.3 + 0.6 * i / 49, 0.4], [1, 0, 0], [0, -1, 0]) for i in range(50)]
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.ROBOT), state.copy())
    stats = trajectory.attributes["native_stats"]
    assert stats["exact_configs_checked"] >= len(frames)
    # Every configuration of the result is free on the exact meshes too.
    contacts = planner.client.core.exact_check([p.joint_values for p in trajectory.points])
    assert all(c is None for c in contacts)
    # A folded configuration that self-collides on the meshes is reported by the exact check.
    folded = state.copy()
    folded.robot_configuration = cell.zero_configuration()
    folded.robot_configuration.joint_values = [0.0, -1.57, 2.9, 0.0, 0.0, 0.0]
    with pytest.raises(CollisionCheckError):
        planner.check_collision(folded, {"exact": True})


@pytest.mark.parametrize("factory", ["abb_irb4600_40_255", "staubli_tx2_60l"])
def test_spherical_wrist_robots_without_a_solver(factory):
    """The closed form for spherical-wrist arms is derived from the robot model."""
    cell, state = getattr(RobotCellLibrary, factory)()
    planner = NativePlanner()
    planner.set_robot_cell(cell, state)
    # IK round trip from random configurations.
    check = state.copy()
    for _ in range(20):
        check.robot_configuration = cell.random_configuration(cell.main_group_name)
        frame = planner.forward_kinematics(check, TargetMode.ROBOT)
        solutions = list(planner.iter_inverse_kinematics(FrameTarget(frame, TargetMode.ROBOT), check, options={"check_collision": False}))
        assert solutions
        for config in solutions:
            check.robot_configuration = config
            assert frames_close(planner.forward_kinematics(check, TargetMode.ROBOT), frame)
    # A Cartesian line in front of the robot, flange pointing down.
    reach = 1.4 if factory.startswith("abb") else 0.45
    height = 1.0 if factory.startswith("abb") else 0.5
    frames = [Frame([reach, -0.2 + 0.4 * i / 29, height], [1, 0, 0], [0, -1, 0]) for i in range(30)]
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.ROBOT), state, options={"free_axis_samples": 8})
    assert len(trajectory.points) == len(frames)


def test_esdf_obstacles_keep_holes_open():
    """A ring's bounding box fills its hole; its ESDF does not."""
    from compas.geometry import Torus

    from compas_fab.backends.kinematics.exceptions import CartesianMotionError

    cell, _ = RobotCellLibrary.ur5()
    cell.rigid_body_models["ring"] = RigidBody.from_mesh(Torus(0.22, 0.025).to_mesh(u=48, v=12, triangulated=True))
    state = cell.default_cell_state()
    state.rigid_body_states["ring"] = RigidBodyState(Frame([0.45, 0.0, 0.45], [1, 0, 0], [0, 1, 0]))
    # Plunge the flange from above into the ring.
    frames = [Frame([0.45, 0.0, 0.65 - 0.23 * i / 29], [1, 0, 0], [0, -1, 0]) for i in range(30)]
    waypoints = FrameWaypoints(frames, TargetMode.ROBOT)

    boxes = NativePlanner(UR5Kinematics(), obstacles="boxes")
    boxes.set_robot_cell(cell, state)
    with pytest.raises(CartesianMotionError):
        boxes.plan_cartesian_motion(waypoints, state, options={"free_axis_samples": 12})

    esdf = NativePlanner(UR5Kinematics(), obstacles="esdf")
    esdf.set_robot_cell(cell, state)
    trajectory = esdf.plan_cartesian_motion(waypoints, state, options={"free_axis_samples": 12})
    assert len(trajectory.points) == len(frames)
    assert all(c is None for c in esdf.client.core.exact_check([p.joint_values for p in trajectory.points]))


def test_cartesian_from_the_start_state_returns_one_point_per_waypoint(ur5):
    planner, cell, state = ur5
    frames = [Frame([0.45, -0.3 + 0.6 * i / 49, 0.4], [1, 0, 0], [0, -1, 0]) for i in range(50)]
    waypoints = FrameWaypoints(frames, TargetMode.ROBOT)
    start = state.copy()
    start.robot_configuration = cell.zero_configuration().merged(planner.plan_cartesian_motion(waypoints, state.copy()).points[0])
    trajectory = planner.plan_cartesian_motion(waypoints, start, options={"start_from_state": True})
    assert len(trajectory.points) == len(frames)
    assert trajectory.start_configuration.close_to(start.robot_configuration)


# -- plan_motion --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ur5_wall():
    cell, _ = RobotCellLibrary.ur5()
    cell.rigid_body_models["wall"] = RigidBody.from_mesh(Box(0.05, 0.8, 0.8).to_mesh(triangulated=True))
    state = cell.default_cell_state()
    state.rigid_body_states["wall"] = RigidBodyState(Frame([0.45, 0.0, 0.4], [1, 0, 0], [0, 1, 0]))
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    return planner, cell, state


def _configuration(cell, values):
    config = cell.zero_configuration()
    config.joint_values = values
    return config


def _assert_free_path(planner, trajectory):
    core = planner.client.core
    path = [trajectory.start_configuration.joint_values[: core.dof]] + [p.joint_values for p in trajectory.points]
    # Every segment free on the spheres at 4x the planning resolution, every 0.01 rad on the meshes.
    for a, b in zip(path, path[1:]):
        assert core.motion_free(a, b, 4 * core.default_resolution)
    dense = []
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.ceil(max(abs(x - y) for x, y in zip(a, b)) / 0.01)))
        dense += [[x + (y - x) * k / n for x, y in zip(a, b)] for k in range(n + 1)]
    assert all(c is None for c in core.exact_check(dense))


def test_plan_motion_around_a_wall(ur5_wall):
    from compas_fab.robots import ConfigurationTarget

    planner, cell, state = ur5_wall
    start = state.copy()
    start.robot_configuration = _configuration(cell, [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
    goal = _configuration(cell, [1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
    assert not planner.client.core.motion_free(start.robot_configuration.joint_values, goal.joint_values)
    trajectory = planner.plan_motion(ConfigurationTarget(goal), start)
    assert trajectory.start_configuration.close_to(start.robot_configuration)
    assert trajectory.points[-1].close_to(goal)
    assert len(trajectory.points) >= 2  # had to go around
    _assert_free_path(planner, trajectory)


def test_plan_motion_to_frame_and_point_axis_targets(ur5_wall):
    from compas.geometry import Point
    from compas.geometry import Vector

    from compas_fab.robots import PointAxisTarget

    planner, cell, state = ur5_wall
    start = state.copy()
    start.robot_configuration = _configuration(cell, [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
    frame = Frame([0.3, 0.35, 0.3], [1, 0, 0], [0, -1, 0])
    trajectory = planner.plan_motion(FrameTarget(frame, TargetMode.ROBOT), start)
    end = start.copy()
    end.robot_configuration = cell.zero_configuration().merged(trajectory.points[-1])
    assert frames_close(planner.forward_kinematics(end, TargetMode.ROBOT), frame)
    _assert_free_path(planner, trajectory)

    target = PointAxisTarget(Point(0.3, -0.35, 0.3), Vector(0, 0, -1), TargetMode.ROBOT)
    trajectory = planner.plan_motion(target, start, options={"axis_samples": 8})
    end.robot_configuration = cell.zero_configuration().merged(trajectory.points[-1])
    reached = planner.forward_kinematics(end, TargetMode.ROBOT)
    assert reached.point.distance_to_point(target.target_point) < 1e-6
    assert reached.zaxis.dot(target.target_z_axis) > 1 - 1e-9


def test_plan_motion_seven_axis_panda():
    from compas_fab.robots import ConfigurationTarget

    cell, _ = RobotCellLibrary.panda()
    cell.rigid_body_models["post"] = RigidBody.from_mesh(Box(0.1, 0.1, 1.2).to_mesh(triangulated=True))
    state = cell.default_cell_state()
    state.rigid_body_states["post"] = RigidBodyState(Frame([0.45, 0.0, 0.6], [1, 0, 0], [0, 1, 0]))
    planner = NativePlanner()  # no IK needed for configuration targets
    planner.set_robot_cell(cell, state)
    assert planner.client.core.dof == 7
    start = state.copy()
    start.robot_configuration = start.robot_configuration.merged(_configuration(cell, [-1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8]))
    goal = _configuration(cell, [1.2, 0.3, 0.0, -2.0, 0.0, 2.3, 0.8])
    trajectory = planner.plan_motion(ConfigurationTarget(goal), start)
    assert trajectory.points[-1].close_to(goal)
    _assert_free_path(planner, trajectory)


def test_plan_motion_reports_colliding_start_and_goal(ur5_wall):
    from compas_fab.backends import MPStartStateInCollisionError
    from compas_fab.backends import MPTargetInCollisionError
    from compas_fab.robots import ConfigurationTarget

    planner, cell, state = ur5_wall
    free = _configuration(cell, [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
    into_wall = _configuration(cell, [0.0, -0.6, 0.6, -1.5, -1.57, 0.0])
    start = state.copy()
    start.robot_configuration = into_wall
    with pytest.raises(MPStartStateInCollisionError):
        planner.plan_motion(ConfigurationTarget(free), start)
    start.robot_configuration = free
    with pytest.raises(MPTargetInCollisionError):
        planner.plan_motion(ConfigurationTarget(into_wall), start)


def test_joints_outside_the_group_follow_the_state():
    """The Panda's fingers (one configurable joint outside the arm group, one mimic) take
    their values from the state instead of zero."""
    cell, state = RobotCellLibrary.panda()
    planner = NativePlanner()
    planner.set_robot_cell(cell, state)
    core = planner.client.core
    names = core.link_names
    for opening in [0.0, 0.04]:
        state = state.copy()
        config = state.robot_configuration
        config.joint_values = [0.3, -0.2, 0.1, -1.8, 0.2, 1.6, 0.4, opening]
        planner.set_robot_cell_state(state)
        frames = core.link_frames(config.joint_values[:7])
        for link in ["panda_leftfinger", "panda_rightfinger"]:
            expected = cell.robot_model.forward_kinematics(config, link)
            got = frames[names.index(link)]
            assert abs(got[3] - expected.point.x) < 1e-9 and abs(got[7] - expected.point.y) < 1e-9 and abs(got[11] - expected.point.z) < 1e-9, (link, opening)

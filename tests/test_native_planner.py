"""NativePlanner (spike) against compas_fab's own kinematics. Skipped without the extension."""

import json
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
from compas_fab.robots import PointAxisTarget  # noqa: E402
from compas_fab.robots import PointAxisWaypoints  # noqa: E402
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
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.ROBOT), state.copy(), options={"max_step": 0})
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
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.TOOL), state, options={"free_axis_samples": 12, "max_step": 0})
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
    trajectory = planner.plan_cartesian_motion(FrameWaypoints(frames, TargetMode.ROBOT), state, options={"free_axis_samples": 8, "max_step": 0})
    assert len(trajectory.points) == len(frames)


def test_esdf_obstacles_keep_holes_open():
    """A ring's bounding box fills its hole; its ESDF does not. (At 0.22 m the UR5's wrist
    clears the rim by millimetres, and whether it passes depends on the random sphere fit.)"""
    from compas.geometry import Torus

    from compas_fab.backends.kinematics.exceptions import CartesianMotionError

    cell, _ = RobotCellLibrary.ur5()
    cell.rigid_body_models["ring"] = RigidBody.from_mesh(Torus(0.25, 0.025).to_mesh(u=48, v=12, triangulated=True))
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
    start.robot_configuration = cell.zero_configuration().merged(planner.plan_cartesian_motion(waypoints, state.copy(), options={"max_step": 0}).points[0])
    trajectory = planner.plan_cartesian_motion(waypoints, start, options={"start_from_state": True, "max_step": 0})
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


def test_plan_motion_is_timed_within_the_joint_limits(ur5_wall):
    from compas_fab.robots import ConfigurationTarget

    planner, cell, state = ur5_wall
    start = state.copy()
    start.robot_configuration = _configuration(cell, [-1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
    goal = _configuration(cell, [1.2, -1.2, 1.4, -1.8, -1.57, 0.0])
    options = {"velocity_scaling": 0.5, "max_acceleration": 2.0, "time_step": 0.02}
    trajectory = planner.plan_motion(ConfigurationTarget(goal), start, options=options)
    times = [p.time_from_start.seconds for p in trajectory.points]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert abs(times[-1] - trajectory.attributes["native_stats"]["duration"]) < 1e-9
    v_max = 0.5 * math.pi  # UR5 joints: pi rad/s, scaled by 0.5
    for p in trajectory.points:
        assert max(abs(v) for v in p.velocities) <= v_max * 1.001
        assert max(abs(a) for a in p.accelerations) <= 2.0 * 1.02
    assert trajectory.points[-1].close_to(goal)
    assert max(abs(v) for v in trajectory.points[-1].velocities) == 0.0
    _assert_free_path(planner, trajectory)
    # Without timing: the path's vertices only.
    vertices = planner.plan_motion(ConfigurationTarget(goal), start, options={"time_parameterization": False})
    assert len(vertices.points) < len(trajectory.points)


def test_a_tilted_cube_gets_a_tight_box():
    """Principal axes are arbitrary for a cube; the body's own frame gives the tight box."""
    from compas_fab.backends.native.model import obstacle_boxes

    cell, state = RobotCellLibrary.ur5()
    cell.rigid_body_models["cube"] = RigidBody.from_mesh(Box(0.3, 0.3, 0.3).to_mesh(triangulated=True))
    frame = Frame([0.5, 0.2, 0.4], [0.8, 0.6, 0.0], [-0.36, 0.48, 0.8])
    state.rigid_body_states["cube"] = RigidBodyState(frame)
    (box,) = obstacle_boxes(cell, state)["cuboids"]
    assert max(box["half_extents"]) < 0.15 + 1e-4


def test_changing_touch_links_reaches_the_core():
    cell, state = RobotCellLibrary.ur5()
    cell.rigid_body_models["box"] = RigidBody.from_mesh(Box(0.2, 0.2, 0.2).to_mesh(triangulated=True))
    state.robot_configuration = cell.zero_configuration()
    tool0 = cell.robot_model.forward_kinematics(state.robot_configuration, "tool0")
    state.rigid_body_states["box"] = RigidBodyState(Frame(tool0.point, [1, 0, 0], [0, 1, 0]))
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    core = planner.client.core
    before = core.clearance([state.robot_configuration.joint_values])[0]
    assert before < 0
    # Only the touch links change: the state's frames and attachments stay the same.
    state.rigid_body_states["box"].touch_links = [link.name for link in cell.robot_model.links]
    planner.set_robot_cell_state(state)
    assert core.clearance([state.robot_configuration.joint_values])[0] > before


def test_multi_robot_cell_plans_one_gantry_robot():
    """The RFL has 44 links, more than the core takes. Planning robot 11 on its gantry, the
    links it cannot move (the other bridge and its robots, the rails, the hall) become
    obstacles, and the robot 12 on the same bridge rides along."""
    cell, state = RobotCellLibrary.rfl()
    group = "robot11_eaXYZ"
    planner = NativePlanner()
    planner.set_robot_cell(cell, state, options={"group": group})
    core = planner.client.core
    assert len(core.link_names) <= 32
    assert "robot21_link_3" not in core.link_names and "robot12_link_3" in core.link_names
    names = {c["name"] for c in json.loads(planner._scene_json(state))["cuboids"]}
    assert "link:robot21_link_3" in names and "link:bridge2" in names

    values = dict(zip(state.robot_configuration.joint_names, state.robot_configuration.joint_values))
    q = [values[n] for n in core.joint_names]
    q[0], q[1], q[2] = 3.0, -2.0, -3.2  # gantry X, Y, Z
    q[4], q[5], q[7] = 0.3, 0.4, 0.9
    frames = core.link_frames(q)
    full = state.robot_configuration.copy()
    full.joint_values = [dict(zip(core.joint_names, q)).get(n, v) for n, v in zip(full.joint_names, full.joint_values)]
    for link in ["robot11_tool0", "robot12_link_6"]:
        expected = cell.robot_model.forward_kinematics(full, link)
        got = frames[core.link_names.index(link)]
        assert max(abs(got[3] - expected.point.x), abs(got[7] - expected.point.y), abs(got[11] - expected.point.z)) < 1e-6, link

    start = [values[n] for n in core.joint_names]
    start[0], start[1], start[2], start[4], start[5], start[7] = 2.0, -1.2, -3.3, 0.2, 0.5, 0.85
    assert core.clearance([start, q])[0] > 0 and core.clearance([start, q])[1] > 0
    path, _ = core.plan_motion(start, [q], exact=core.has_exact)
    assert len(path) >= 2


# -- Point-axis targets, interpolation, numerical IK ------------------------------------------


def _down(x, y, z):
    return Frame([x, y, z], [1, 0, 0], [0, -1, 0])


def _reached(planner, state, values, group=None, mode=TargetMode.ROBOT):
    check = state.copy()
    check.robot_configuration = state.robot_configuration.merged(values)
    return planner.forward_kinematics(check, mode, group)


@pytest.fixture(scope="module")
def panda():
    cell, state = RobotCellLibrary.panda()
    planner = NativePlanner()
    planner.set_robot_cell(cell, state)
    assert not planner.client.core.has_analytic_ik
    return planner, cell, state


@pytest.fixture(scope="module")
def rfl():
    cell, state = RobotCellLibrary.rfl()
    planner = NativePlanner()
    planner.set_robot_cell(cell, state, options={"group": "robot11_eaXYZ"})
    return planner, cell, state


def test_numeric_ik_for_a_seven_axis_arm(panda):
    planner, cell, state = panda
    frame = _down(0.45, 0.2, 0.35)
    solutions = list(planner.iter_inverse_kinematics(FrameTarget(frame, TargetMode.ROBOT), state))
    assert len(solutions) == 8  # max_results: a redundant arm has infinitely many
    limits = [cell.robot_model.get_joint_by_name(n).limit for n in solutions[0].joint_names]
    for config in solutions:
        assert frames_close(_reached(planner, state, config), frame)
        assert all(lim.lower - 1e-9 <= v <= lim.upper + 1e-9 for v, lim in zip(config.joint_values, limits))
        check = state.copy()
        check.robot_configuration = state.robot_configuration.merged(config)
        planner.check_collision(check)
    values = [c.joint_values for c in solutions]
    assert all(max(abs(a - b) for a, b in zip(p, q)) > 0.05 for i, p in enumerate(values) for q in values[i + 1 :])
    # The same seeds give the same answers.
    again = list(planner.iter_inverse_kinematics(FrameTarget(frame, TargetMode.ROBOT), state))
    assert [c.joint_values for c in again] == values


def test_numeric_ik_can_be_forced_on_a_closed_form_robot(ur5):
    planner, cell, state = ur5
    frame = _down(0.45, 0.1, 0.4)
    solutions = list(planner.iter_inverse_kinematics(FrameTarget(frame, TargetMode.ROBOT), state, options={"ik": "numeric", "max_results": 4}))
    assert 1 <= len(solutions) <= 4
    for config in solutions:
        assert frames_close(_reached(planner, state, config), frame)


@pytest.mark.parametrize("robot", ["ur5", "panda"])
def test_point_axis_target_ik(robot, request):
    """Closed form (UR5: sampled rotations about the axis) and numerical IK (Panda: the
    rotation about the axis is free)."""
    planner, cell, state = request.getfixturevalue(robot)
    target = PointAxisTarget([0.4, -0.15, 0.3], [0.3, 0.0, -1.0], TargetMode.ROBOT)
    solutions = list(planner.iter_inverse_kinematics(target, state))
    assert len(solutions) > 1
    for config in solutions:
        reached = _reached(planner, state, config)
        assert reached.point.distance_to_point(target.target_point) < 1e-6
        assert reached.zaxis.dot(target.target_z_axis) > 1 - 1e-9


def test_plan_motion_to_a_frame_with_a_seven_axis_arm(panda):
    planner, cell, state = panda
    frame = _down(0.45, 0.2, 0.35)
    trajectory = planner.plan_motion(FrameTarget(frame, TargetMode.ROBOT), state)
    assert frames_close(_reached(planner, state, trajectory.points[-1]), frame)
    _assert_free_path(planner, trajectory)


def test_cartesian_interpolates_between_sparse_waypoints(ur5):
    """Two waypoints 0.6 m apart, a quarter turn about z between them: the tool moves in a
    straight line, turning evenly, in steps of at most max_step and max_step_angle."""
    planner, cell, state = ur5
    a = _down(0.45, -0.3, 0.4)
    b = Frame([0.45, 0.3, 0.4], [0, -1, 0], [-1, 0, 0])
    trajectory = planner.plan_cartesian_motion(FrameWaypoints([a, b], TargetMode.ROBOT), state.copy())
    assert len(trajectory.points) >= 61
    frames = [_reached(planner, state, p) for p in trajectory.points]
    assert frames_close(frames[0], a) and frames_close(frames[-1], b)
    for f, g in zip(frames, frames[1:]):
        assert abs(f.point.x - 0.45) < 1e-6 and abs(f.point.z - 0.4) < 1e-6
        assert f.point.distance_to_point(g.point) <= 0.01 + 1e-9
        assert math.acos(max(-1.0, min(1.0, f.xaxis.dot(g.xaxis)))) <= 0.1 + 1e-6
    steps = [max(abs(x - y) for x, y in zip(p.joint_values, q.joint_values)) for p, q in zip(trajectory.points, trajectory.points[1:])]
    assert max(steps) <= 0.5


@pytest.mark.parametrize("robot", ["ur5", "panda"])
def test_point_axis_waypoints(robot, request):
    """A circle with a tilted tool axis; the rotation about the axis is free."""
    planner, cell, state = request.getfixturevalue(robot)
    points = [(0.4 + 0.08 * math.cos(t), 0.08 * math.sin(t), 0.3) for t in [2 * math.pi * i / 24 for i in range(25)]]
    waypoints = PointAxisWaypoints([(p, [0.2, 0.0, -1.0]) for p in points], TargetMode.ROBOT)
    trajectory = planner.plan_cartesian_motion(waypoints, state.copy(), options={"max_step": 0})
    assert len(trajectory.points) == len(points)
    axis = [0.2 / math.hypot(0.2, 1.0), 0.0, -1.0 / math.hypot(0.2, 1.0)]
    for point, p in zip(trajectory.points, points):
        reached = _reached(planner, state, point)
        assert reached.point.distance_to_point(p) < 1e-6
        assert reached.zaxis.dot(axis) > 1 - 1e-9


def test_numeric_cartesian_in_tool_mode_turns_about_the_tool_axis():
    """With numerical IK the free rotation is about the TCF, not the flange."""
    cell, state = RobotCellLibrary.ur5_cone_tool()
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    points = [(0.4 + 0.1 * math.cos(t), 0.1 * math.sin(t), 0.3) for t in [2 * math.pi * i / 20 for i in range(21)]]
    waypoints = PointAxisWaypoints([(p, [0, 0, -1]) for p in points], TargetMode.TOOL)
    trajectory = planner.plan_cartesian_motion(waypoints, state, options={"ik": "numeric", "max_step": 0})
    for point, p in zip(trajectory.points, points):
        tcf = _reached(planner, state, point, mode=TargetMode.TOOL)
        assert tcf.point.distance_to_point(p) < 1e-6
        assert tcf.zaxis.dot([0, 0, -1]) > 1 - 1e-9


def test_gantry_robot_ik_motion_and_cartesian(rfl):
    """RFL robot 11 with its gantry (9 axes): the external axes carry the arm to targets
    metres away, and a 4 m Cartesian line is followed by the gantry and the arm together."""
    planner, cell, state = rfl
    group = "robot11_eaXYZ"
    frame = _down(12.0, 4.0, 1.2)
    solutions = list(planner.iter_inverse_kinematics(FrameTarget(frame, TargetMode.ROBOT), state, group))
    assert solutions
    for config in solutions:
        assert frames_close(_reached(planner, state, config, group), frame, tol=1e-5)
    trajectory = planner.plan_motion(FrameTarget(frame, TargetMode.ROBOT), state, group)
    assert frames_close(_reached(planner, state, trajectory.points[-1], group), frame, tol=1e-5)

    a, b = _down(10.0, 3.0, 1.0), _down(14.0, 3.5, 1.0)
    trajectory = planner.plan_cartesian_motion(FrameWaypoints([a, b], TargetMode.ROBOT), state, group)
    assert len(trajectory.points) >= 400
    for point in trajectory.points[::20] + [trajectory.points[-1]]:
        p = _reached(planner, state, point, group).point
        t = (p.x - 10.0) / 4.0
        assert -1e-6 <= t <= 1 + 1e-6
        assert abs(p.y - (3.0 + 0.5 * t)) < 1e-5 and abs(p.z - 1.0) < 1e-5
    x = [dict(zip(p.joint_names, p.joint_values))["bridge1_joint_EA_X"] for p in (trajectory.points[0], trajectory.points[-1])]
    assert x[1] - x[0] > 3.0  # the bridge moved along the hall


def test_sphere_fit_is_cached_in_memory_and_on_disk(tmp_path, monkeypatch):
    from compas_fab.backends.native import model

    monkeypatch.setenv("COMPAS_FAB_NATIVE_CACHE", str(tmp_path))
    monkeypatch.setattr(model, "_sphere_cache", {})
    cell, state = RobotCellLibrary.ur5()
    first = NativePlanner(UR5Kinematics())
    first.set_robot_cell(cell, state)
    assert list(tmp_path.glob("*.json"))

    # A new process: nothing in memory, and fitting would fail.
    def no_fitting(*args, **kwargs):
        raise AssertionError("fitted again")

    monkeypatch.setattr(model, "_sphere_cache", {})
    monkeypatch.setattr(model, "fit_sphere_sets", no_fitting)
    second = NativePlanner(UR5Kinematics())
    second.set_robot_cell(cell, state)
    assert second._robot_description["spheres"] == first._robot_description["spheres"]
    assert second._robot_description["self_spheres"] == first._robot_description["self_spheres"]


# -- What the robot carries, and what stands around it --------------------------------------


def _box_body(size):
    return RigidBody.from_mesh(Box(size, size, size).to_mesh(triangulated=True))


def _cone_cell_with_workpiece():
    """UR5 with the cone tool, holding a 0.12 m cube 0.25 m beyond the tool tip."""
    cell, state = RobotCellLibrary.ur5_cone_tool()
    cell.rigid_body_models["workpiece"] = _box_body(0.12)
    state.rigid_body_states["workpiece"] = RigidBodyState(None)
    state.set_rigid_body_attached_to_tool("workpiece", "cone", Frame([0, 0, 0.25], [1, 0, 0], [0, 1, 0]))
    state.robot_configuration = cell.zero_configuration()
    state.robot_configuration.joint_values = [0.0, -1.2, 1.5, -1.9, -1.57, 0.0]
    return cell, state


def _posed(cell, state, body_id):
    return cell.compute_attach_objects_frames(state).rigid_body_states[body_id].frame


def test_a_carried_workpiece_collides_with_obstacles():
    cell, state = _cone_cell_with_workpiece()
    cell.rigid_body_models["post"] = _box_body(0.05)
    state.rigid_body_states["post"] = RigidBodyState(Frame([3, 3, 3], [1, 0, 0], [0, 1, 0]))
    workpiece = _posed(cell, state, "workpiece")
    planner = NativePlanner(UR5Kinematics())
    # The post where the workpiece is (and nothing else of the robot is).
    state.rigid_body_states["post"].frame = Frame(workpiece.point, [1, 0, 0], [0, 1, 0])
    planner.set_robot_cell(cell, state)
    with pytest.raises(CollisionCheckError):
        planner.check_collision(state)
    with pytest.raises(CollisionCheckError):
        planner.check_collision(state, {"exact": True})
    # Without the workpiece, the post is in nobody's way.
    state.rigid_body_states["workpiece"].attached_to_tool = None
    state.rigid_body_states["workpiece"].frame = Frame([3, 3, 3], [1, 0, 0], [0, 1, 0])
    planner.check_collision(state)
    planner.check_collision(state, {"exact": True})


def test_workpiece_target_mode():
    """Targets for the workpiece's own frame (OCF), through tool and grasp."""
    cell, state = _cone_cell_with_workpiece()
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    frame = Frame([0.35, 0.2, 0.25], [1, 0, 0], [0, -1, 0])
    solutions = list(planner.iter_inverse_kinematics(FrameTarget(frame, TargetMode.WORKPIECE), state))
    assert solutions
    for config in solutions:
        assert frames_close(_reached(planner, state, config, mode=TargetMode.WORKPIECE), frame)
    # Point-axis waypoints for the workpiece, with numerical IK: the free rotation is
    # about the workpiece's z axis.
    points = [(0.35, -0.1 + 0.2 * i / 10, 0.25) for i in range(11)]
    waypoints = PointAxisWaypoints([(p, [0, 0, -1]) for p in points], TargetMode.WORKPIECE)
    trajectory = planner.plan_cartesian_motion(waypoints, state, options={"ik": "numeric", "max_step": 0})
    for point, p in zip(trajectory.points, points):
        ocf = _reached(planner, state, point, mode=TargetMode.WORKPIECE)
        assert ocf.point.distance_to_point(p) < 1e-6
        assert ocf.zaxis.dot([0, 0, -1]) > 1 - 1e-9


def test_a_body_attached_to_a_link_moves_with_it():
    cell, state = RobotCellLibrary.ur5()
    cell.rigid_body_models["sensor"] = _box_body(0.1)
    cell.rigid_body_models["post"] = _box_body(0.05)
    state.rigid_body_states["sensor"] = RigidBodyState(None)
    state.set_rigid_body_attached_to_link("sensor", "forearm_link", Frame([0, 0.15, 0.2], [1, 0, 0], [0, 1, 0]))
    state.robot_configuration = cell.zero_configuration()
    state.robot_configuration.joint_values = [0.0, -1.2, 1.5, -1.9, -1.57, 0.0]
    state.rigid_body_states["post"] = RigidBodyState(Frame([3, 3, 3], [1, 0, 0], [0, 1, 0]))
    sensor = _posed(cell, state, "sensor")
    state.rigid_body_states["post"].frame = Frame(sensor.point, [1, 0, 0], [0, 1, 0])
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    with pytest.raises(CollisionCheckError):
        planner.check_collision(state)
    # Turn the base: the sensor leaves the post behind.
    moved = state.copy()
    moved.robot_configuration.joint_values = [1.5, -1.2, 1.5, -1.9, -1.57, 0.0]
    planner.check_collision(moved)


def test_a_parked_tool_is_an_obstacle():
    cell, state = RobotCellLibrary.ur5_cone_tool()
    state.robot_configuration = cell.zero_configuration()
    state.robot_configuration.joint_values = [0.0, -1.2, 1.5, -1.9, -1.57, 0.0]
    planner = NativePlanner(UR5Kinematics())
    # Park the tool (detached: no touch links) across the forearm.
    forearm = cell.robot_model.forward_kinematics(state.robot_configuration, "forearm_link")
    state.set_tool_detached("cone", Frame(forearm.point + forearm.zaxis * 0.2, [1, 0, 0], [0, 1, 0]))
    planner.set_robot_cell(cell, state)
    with pytest.raises(CollisionCheckError):
        planner.check_collision(state)
    state.tool_states["cone"].frame = Frame([3, 3, 3], [1, 0, 0], [0, 1, 0])
    planner.check_collision(state)


def test_continuous_joints_turn_past_half_a_turn():
    """compas_robots reads a continuous joint's limit tag as bounds 0..0; the joint must
    still turn, and past pi."""
    from compas_fab.robots import ConfigurationTarget
    from compas_robots.model import Joint
    from compas_robots.model import Limit

    cell, state = RobotCellLibrary.ur5()
    joint = cell.robot_model.get_joint_by_name("shoulder_pan_joint")
    joint.type = Joint.CONTINUOUS
    joint.limit = Limit(effort=150.0, velocity=3.15, lower=0.0, upper=0.0)
    planner = NativePlanner(UR5Kinematics())
    planner.set_robot_cell(cell, state)
    assert planner._robot_description["joint_limits"][0] == [-2 * math.pi, 2 * math.pi]
    goal = cell.zero_configuration()
    goal.joint_values = [4.0, -1.2, 1.5, -1.9, -1.57, 0.0]
    trajectory = planner.plan_motion(ConfigurationTarget(goal), state)
    assert abs(trajectory.points[-1].joint_values[0] - 4.0) < 1e-9

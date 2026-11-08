"""Describe a RobotCell for the native motion core (spike).

The native core takes the robot as a kinematic tree plus sphere approximations, and the
scene as oriented boxes. This module builds those descriptions from compas_fab objects:

* the kinematic tree in topological order, joint origins as 3x4 row-major transforms;
* two sphere sets per link: conservative covering spheres for checks against obstacles,
  and spheres inscribed in the link meshes for self-collision (covering spheres of
  neighbouring links overlap permanently);
* attached tools, as extra spheres on the end-effector link;
* static rigid bodies, as oriented bounding boxes (conservative);
* links that no joint of the planning group moves (e.g. the other bridge of a
  multi-robot cell), as static obstacles posed at the cell state's configuration. They
  are left out of the kinematic tree, which keeps large cells within the core's link cap.

Sphere fitting needs numpy, and trimesh for the inscribed set.
"""

import hashlib
import json
import math
import os
from typing import Optional

import numpy as np
from compas.geometry import Frame
from compas.geometry import Transformation

JOINT_TYPES = {0: "revolute", 1: "continuous", 2: "prismatic", 3: "fixed"}
MAX_SPHERES = 256  # per robot, in the core (layout.rs)
DEFAULT_MAX_SPHERES = 16  # per link, see fit_sphere_sets


def frame_to_3x4(frame) -> list:
    """A Frame or Transformation as a flat 3x4 row-major list."""
    t = frame if isinstance(frame, Transformation) else Transformation.from_frame(frame)
    return np.array(t.matrix)[:3, :].reshape(-1).tolist()


def matrix_from_3x4(values) -> np.ndarray:
    m = np.eye(4)
    m[:3, :] = np.asarray(values, dtype=float).reshape(3, 4)
    return m


def analytic_ik_description(solver) -> Optional[dict]:
    """Closed-form IK description from a compas_fab ``AnalyticalKinematics`` solver."""
    from compas_fab.backends import OffsetWristKinematics
    from compas_fab.backends import SphericalWristKinematics

    if solver is None:
        return None  # the core derives a spherical-wrist closed form from the model when it can
    if isinstance(solver, OffsetWristKinematics):
        flange = solver.flange_frame or Frame.worldXY()
        return {"family": "offset_wrist", "params": list(solver.params), "flange_frame": frame_to_3x4(flange)}
    if isinstance(solver, SphericalWristKinematics):
        # Derived from the model itself: no parameters, no joint conventions to translate.
        return {"family": "spherical_wrist"}
    raise NotImplementedError("The native core has no closed-form IK for {}".format(type(solver).__name__))


# ---------------------------------------------------------------------------
# Geometry sampling and sphere fitting


def meshes_to_triangles(meshes_and_transforms):
    """Triangles (T, 3, 3) of (mesh, 4x4 transform, scale) items."""
    tris = []
    for mesh, t, scale in meshes_and_transforms:
        vertices, faces = mesh.to_vertices_and_faces(triangulated=True)
        v = (np.asarray(vertices, dtype=float) * np.asarray(scale, dtype=float)) @ t[:3, :3].T + t[:3, 3]
        if len(faces):
            tris.append(v[np.asarray(faces, dtype=int)])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def collision_items(collisions):
    """(mesh, transform, scale) for every mesh of a link's collision elements, in link coordinates."""
    items = []
    for collision in collisions:
        shape = collision.geometry.shape
        t = np.array(Transformation.from_frame(collision.origin).matrix) if collision.origin else np.eye(4)
        # Mesh descriptors carry a scale vector; primitives (Box, Cylinder, ...) have a
        # `scale` method instead and are already in link units.
        scale = getattr(shape, "scale", None)
        if callable(scale) or not scale:
            scale = [1.0, 1.0, 1.0]
        for mesh in getattr(shape, "meshes", None) or []:
            items.append((mesh, t, scale))
    return items


def surface_points(tris, n_samples, rng):
    """Area-weighted surface samples plus all vertices, so extremes are always covered."""
    if len(tris) == 0:
        return np.zeros((0, 3))
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if areas.sum() <= 0:
        return tris.reshape(-1, 3)
    idx = rng.choice(len(tris), size=n_samples, p=areas / areas.sum())
    r1 = np.sqrt(rng.random(n_samples))[:, None]
    r2 = rng.random(n_samples)[:, None]
    pts = (1 - r1) * a[idx] + r1 * (1 - r2) * b[idx] + r1 * r2 * c[idx]
    return np.concatenate([pts, tris.reshape(-1, 3)])


def fit_covering_spheres(points, target_radius, max_spheres, rng):
    """k-means clusters, each enclosed by a sphere around its centroid (conservative)."""
    if len(points) == 0:
        return []
    extent = np.ptp(points, axis=0).max()
    k = int(np.clip(math.ceil(extent / (2 * target_radius)), 1, max_spheres))
    k = min(k, len(points))
    centers = points[rng.choice(len(points), size=k, replace=False)]
    labels = np.zeros(len(points), dtype=int)
    for _ in range(50):
        labels = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    spheres = []
    for i in range(k):
        cluster = points[labels == i]
        if len(cluster):
            spheres.append((centers[i], float(np.linalg.norm(cluster - centers[i], axis=1).max())))
    return spheres


def watertight_trimesh(tris):
    """One closed, outward-oriented trimesh from triangles.

    None if trimesh is missing, or the mesh (or one of its components) is not closed or has
    no volume once its normals are fixed.
    """
    try:
        import trimesh
    except ImportError:
        return None
    if len(tris) == 0:
        return None
    mesh = trimesh.Trimesh(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3), process=True)
    if len(mesh.faces) == 0 or np.ptp(mesh.vertices, axis=0).min() < 1e-6:
        return None  # flat or degenerate: no interior to inscribe spheres in
    parts = [mesh] if mesh.is_watertight else mesh.split(only_watertight=False)
    # Collision meshes often carry stray slivers (a couple of zero-area or zero-volume faces):
    # drop components with fewer than 4 faces, keep the rest only if they are closed solids.
    parts = [p for p in parts if len(p.faces) >= 4]
    if not parts or not all(p.is_watertight for p in parts):
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        for p in parts:
            trimesh.repair.fix_normals(p)
            if not np.isfinite(p.volume) or p.volume <= 1e-12:
                return None
    return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)


def fit_inscribed_spheres(mesh, tolerance, max_spheres, radii, rng):
    """Greedy surface cover with spheres inside a watertight mesh.

    Candidate centres are pushed inward from surface samples; each gets the largest radius
    that stays inside (signed distance). A sample is covered within ``tolerance`` of a
    sphere; candidates are picked greedily by newly covered samples.
    """
    import trimesh

    pts, face_idx = trimesh.sample.sample_surface(mesh, 3000, seed=int(rng.integers(1 << 31)))
    normals = mesh.face_normals[face_idx]
    sub = rng.choice(len(pts), size=min(600, len(pts)), replace=False)
    centers = np.concatenate([pts[sub] - r * normals[sub] for r in radii])
    sd = trimesh.proximity.signed_distance(mesh, centers)  # positive inside
    keep = sd > 0.002
    centers, rs = centers[keep], np.minimum(sd[keep], max(radii))
    if len(centers) == 0:
        return []
    dist = np.linalg.norm(pts[None, :, :] - centers[:, None, :], axis=2) - rs[:, None]
    covers = dist <= tolerance
    uncovered = np.ones(len(pts), dtype=bool)
    chosen = []
    while uncovered.any() and len(chosen) < max_spheres:
        gain = (covers & uncovered[None, :]).sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] == 0:
            break
        chosen.append(best)
        uncovered &= ~covers[best]
    return [(centers[i], float(rs[i])) for i in chosen]


def fit_sphere_sets(
    tris,
    rng,
    sphere_radius=0.03,
    max_spheres=16,
    self_sphere_radius=0.03,
    max_self_spheres=16,
    tolerance=0.01,
    obstacle_spheres="inscribed",
    obstacle_padding=0.02,
):
    """(obstacle, self) sphere lists for one body given as triangles.

    Self-collision always uses spheres inscribed in the mesh. For obstacles there are two
    choices (measured on a UR5 against PyBullet and exact meshes, see native/SPIKE.md):

    * ``"inscribed"`` (default): the inscribed spheres grown by ``obstacle_padding``.
      ~1-2% false alarms, misses a few real contacts per thousand configurations at thin
      edges.
    * ``"covering"``: k-means clusters enclosed by spheres. Never misses, but ~11% false
      alarms and spheres around joint housings poke through a floor the robot stands on.

    Bodies without a watertight mesh fall back to covering spheres for both sets.
    """
    mesh = watertight_trimesh(tris)
    covering = None
    if mesh is None or obstacle_spheres == "covering":
        covering = fit_covering_spheres(surface_points(tris, 2000, rng), sphere_radius, max_spheres, rng)
    if mesh is None:
        return covering, covering
    radii = [self_sphere_radius * f for f in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
    inscribed = fit_inscribed_spheres(mesh, tolerance, max_self_spheres, radii, rng)
    if not inscribed:
        covering = covering or fit_covering_spheres(surface_points(tris, 2000, rng), sphere_radius, max_spheres, rng)
        return covering, covering
    if obstacle_spheres == "covering":
        return covering, inscribed
    return [(c, r + obstacle_padding) for c, r in inscribed], inscribed


SPHERE_CACHE_VERSION = 1  # bump when the fitting changes, to invalidate cached spheres
_sphere_cache = {}


def sphere_cache_dir():
    """Directory of the on-disk sphere cache, or None if disabled.

    ``COMPAS_FAB_NATIVE_CACHE`` overrides it; set it to ``0`` (or empty) to disable.
    Default: ``%LOCALAPPDATA%/compas_fab/native_spheres`` on Windows,
    ``~/.cache/compas_fab/native_spheres`` elsewhere.
    """
    path = os.environ.get("COMPAS_FAB_NATIVE_CACHE")
    if path is not None:
        return None if path in ("", "0") else path
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    base = base or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "compas_fab", "native_spheres")


def cached_sphere_sets(tris, seed=0, **options):
    """``fit_sphere_sets`` for one body, cached by its triangles and the options.

    The random stream is seeded from the mesh itself, so a body gets the same spheres
    whatever else is in the cell and in whichever order bodies are fitted. Results are kept
    in memory and, unless disabled (see ``sphere_cache_dir``), on disk, so rebuilding a
    planner for the same cell (a Grasshopper recompute, a new session) skips the fitting.
    """
    tris = np.ascontiguousarray(np.round(np.asarray(tris, dtype=float), 7))
    h = hashlib.sha256()
    h.update(json.dumps([SPHERE_CACHE_VERSION, seed, sorted(options.items())], default=str).encode())
    h.update(tris.tobytes())
    key = h.hexdigest()
    hit = _sphere_cache.get(key)
    folder = sphere_cache_dir()
    path = os.path.join(folder, key[:40] + ".json") if folder else None
    if hit is None and path and os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            hit = tuple([(np.asarray(c, dtype=float), float(r)) for c, r in spheres] for spheres in data)
        except (OSError, ValueError, TypeError):
            hit = None  # unreadable entry: fit again and overwrite it
    if hit is None:
        rng = np.random.default_rng(int(key[:15], 16))
        hit = fit_sphere_sets(tris, rng, **options)
        if path:
            try:
                os.makedirs(folder, exist_ok=True)
                tmp = "{}.{}.tmp".format(path, os.getpid())
                with open(tmp, "w") as f:
                    json.dump([[[c.tolist(), r] for c, r in spheres] for spheres in hit], f)
                os.replace(tmp, path)
            except OSError:
                pass  # read-only home, full disk: the in-memory cache still works
    _sphere_cache[key] = hit
    return hit


# ---------------------------------------------------------------------------
# Robot and scene descriptions


def joint_limits(joint):
    """Position limits of a configurable joint, as the core needs them (finite).

    Continuous joints have none (a ``<limit>`` tag on one only carries effort and velocity;
    compas_robots reads its absent bounds as 0 and 0): they get one full turn either way,
    so every angle is reachable without wrapping. Revolute and prismatic joints without a
    limit get one half turn either way (a URDF requires the limit for them).
    """
    if joint.type == 1:  # continuous
        return [-2 * math.pi, 2 * math.pi]
    limit = joint.limit
    if limit is None:
        return [-math.pi, math.pi]
    return [limit.lower, limit.upper]


def topological_links(robot_model):
    """(link, parent index, joint) with every link after its parent."""
    out = []
    stack = [(robot_model.root, -1, None)]
    while stack:
        link, parent, joint = stack.pop()
        index = len(out)
        out.append((link, parent, joint))
        for child_joint in reversed(link.joints):
            stack.append((child_joint.child_link, index, child_joint))
    return out


def planning_links(robot_cell, group=None):
    """Split the links into the kinematic tree the core plans with and the static rest.

    The tree keeps every link a joint of ``group`` moves, and their ancestors (parent
    indices are remapped). The other links stand still while the group plans: with the
    group's joints fixed, nothing can move them but joints outside the group, whose values
    come from the cell state. They become obstacles (see ``static_link_obstacles``), and
    so does the geometry of the static ancestors, which stay in the tree as frames only.

    Returns
    -------
    tuple
        ``(tree, static)``: ``(link, parent index, joint)`` in topological order, and the
        static links (with geometry, ancestors included).
    """
    group = group or robot_cell.main_group_name
    joint_names = set(robot_cell.get_configurable_joint_names(group))
    ordered = topological_links(robot_cell.robot_model)
    moves = []
    for _, parent, joint in ordered:
        moves.append((joint is not None and joint.name in joint_names) or (parent >= 0 and moves[parent]))
    keep = set()
    for i, moving in enumerate(moves):
        while moving and i >= 0 and i not in keep:
            keep.add(i)
            i = ordered[i][1]
    if not keep or not any(ordered[i][0].collision for i in range(len(ordered)) if i not in keep):
        # Nothing with geometry to prune (a single robot, maybe with frame-only links like
        # UR's `base`): the whole tree, as it always was.
        keep = set(range(len(ordered)))
    tree, index = [], {}
    for i, (link, parent, joint) in enumerate(ordered):
        if i in keep:
            index[i] = len(tree)
            tree.append((link, index[parent] if parent >= 0 else -1, joint))
    if len(tree) == len(ordered):
        return tree, []  # nothing to prune: the usual single robot keeps its base as a link
    static = [link for i, (link, _, _) in enumerate(ordered) if not moves[i]]
    return tree, static


def robot_description(robot_cell, group=None, kinematics_solver=None, seed=0, sphere_options=None) -> dict:
    """The robot of ``robot_cell`` as the native core's JSON description (without tools)."""
    sphere_options = sphere_options or {}
    model = robot_cell.robot_model
    group = group or robot_cell.main_group_name
    joint_names = robot_cell.get_configurable_joint_names(group)
    ordered, static = planning_links(robot_cell, group)
    static = {link.name for link in static}

    links = []
    for link, parent, joint in ordered:
        entry = {"name": link.name, "parent": parent}
        if joint is not None:
            jtype = JOINT_TYPES.get(joint.type)
            if jtype is None:
                raise NotImplementedError("Joint type {} of {}".format(joint.type, joint.name))
            entry["joint"] = {
                "name": joint.name,
                "type": jtype,
                "origin": frame_to_3x4(joint.origin),
                "axis": list(joint.axis.vector) if joint.axis else [1.0, 0.0, 0.0],
                "q_index": joint_names.index(joint.name) if joint.name in joint_names else -1,
            }
        links.append(entry)

    geometry = {}
    for i, (link, _, _) in enumerate(ordered):
        if link.name in static:
            continue  # a static ancestor: its geometry is an obstacle
        tris = meshes_to_triangles(collision_items(link.collision))
        if len(tris):
            geometry[i] = tris
    # Share the core's sphere budget between the links, keeping two links' worth for what
    # the robot carries (a tool and a workpiece, see add_carried_objects).
    per_link = max(1, (MAX_SPHERES - 2 * DEFAULT_MAX_SPHERES) // max(1, len(geometry)))
    link_options = dict(sphere_options)
    for key in ("max_spheres", "max_self_spheres"):
        link_options[key] = min(link_options.get(key, DEFAULT_MAX_SPHERES), per_link)

    spheres, self_spheres = [], []
    for i, tris in geometry.items():
        covering, inscribed = cached_sphere_sets(tris, seed, **link_options)
        spheres += [{"link": i, "center": c.tolist(), "radius": r} for c, r in covering]
        self_spheres += [{"link": i, "center": c.tolist(), "radius": r} for c, r in inscribed]

    limits = []
    for name in joint_names:
        joint = model.get_joint_by_name(name)
        limits.append(joint_limits(joint))

    index = {link.name: i for i, (link, _, _) in enumerate(ordered)}
    disabled = sorted([index[a], index[b]] for a, b in robot_cell.robot_semantics.disabled_collisions if a in index and b in index)
    return {
        "name": model.name,
        "group": group,
        "end_effector_link": index[robot_cell.get_end_effector_link_name(group)],
        "disabled_collisions": disabled,
        "analytic_ik": analytic_ik_description(kinematics_solver),
        "joint_names": joint_names,
        "joint_limits": limits,
        "links": links,
        "spheres": spheres,
        "self_spheres": self_spheres,
        "reference_fk": {"configs": [], "link_frames": []},
    }


def tool_triangles(tool_model, tool_configuration=None):
    """Collision triangles of a tool, in its base frame (all links at the given configuration)."""
    items = []
    config = tool_configuration or tool_model.zero_configuration()
    for link in tool_model.links:
        if not link.collision:
            continue
        t_link = np.array(Transformation.from_frame(tool_model.forward_kinematics(config, link.name)).matrix)
        for mesh, t, scale in collision_items(link.collision):
            items.append((mesh, t_link @ t, scale))
    return meshes_to_triangles(items)


def _matrix(frame_or_transformation):
    if frame_or_transformation is None:
        return np.eye(4)
    t = frame_or_transformation
    t = t if isinstance(t, Transformation) else Transformation.from_frame(t)
    return np.array(t.matrix)


def rigid_body_triangles(robot_cell, body_id):
    """Collision triangles of a rigid body (visual meshes if it has none), in its own frame."""
    body = robot_cell.rigid_body_models[body_id]
    meshes = body.collision_meshes_in_meters or body.visual_meshes_in_meters
    if not meshes:
        return np.zeros((0, 3, 3))
    return meshes_to_triangles([(m, np.eye(4), [1.0, 1.0, 1.0]) for m in meshes])


def moving_link_names(robot_cell, group):
    """Names of the links ``group`` moves (the planning tree without its static ancestors)."""
    tree, static = planning_links(robot_cell, group)
    return {link.name for link, _, _ in tree} - {link.name for link in static}


def carried_objects(robot_cell, robot_cell_state, group):
    """The objects that move with the planning group.

    * tools attached to ``group``, on its end-effector link;
    * rigid bodies attached to such a tool (a workpiece in the gripper), on the same link;
    * rigid bodies attached to a link ``group`` moves.

    Returns
    -------
    list
        ``(name, link name, 4x4 link -> object, triangles in object coordinates, touch
        links, is_tool)``, hidden objects left out.
    """
    group = group or robot_cell.main_group_name
    ee = robot_cell.get_end_effector_link_name(group)
    moving = moving_link_names(robot_cell, group)
    out = []
    for tool_id, tool_state in sorted(robot_cell_state.tool_states.items()):
        if tool_state.attached_to_group != group or tool_state.is_hidden:
            continue
        tris = tool_triangles(robot_cell.tool_models[tool_id], tool_state.configuration)
        if len(tris):
            out.append(("tool:" + tool_id, ee, _matrix(tool_state.attachment_frame), tris, list(tool_state.touch_links or []), True))
    for body_id, body_state in sorted(robot_cell_state.rigid_body_states.items()):
        if body_state.is_hidden:
            continue
        if body_state.attached_to_tool:
            tool_state = robot_cell_state.tool_states.get(body_state.attached_to_tool)
            if tool_state is None or tool_state.attached_to_group != group:
                continue
            link, t = ee, _matrix(robot_cell.t_pcf_ocf(robot_cell_state, body_id))
        elif body_state.attached_to_link and body_state.attached_to_link in moving:
            link, t = body_state.attached_to_link, _matrix(body_state.attachment_frame)
        else:
            continue
        tris = rigid_body_triangles(robot_cell, body_id)
        if len(tris):
            out.append(("body:" + body_id, link, t, tris, list(body_state.touch_links or []), False))
    return out


def static_objects(robot_cell, robot_cell_state, group):
    """Tools and attached rigid bodies that do not move with ``group``, as obstacles posed
    at the state: tools of other groups and what they hold, bodies attached to links
    ``group`` cannot move, and tools standing free (e.g. in a tool changer).

    Returns ``(name, triangles in robot coordinates, touch links, rotation of the object)``
    items, like ``static_link_obstacles``. Free rigid bodies are not included; the
    obstacle builders handle them.
    """
    group = group or robot_cell.main_group_name
    carried = {name for name, *_ in carried_objects(robot_cell, robot_cell_state, group)}
    mounted = [tid for tid, ts in robot_cell_state.tool_states.items() if ts.attached_to_group and "tool:" + tid not in carried]
    mounted += [bid for bid, bs in robot_cell_state.rigid_body_states.items() if (bs.attached_to_link or bs.attached_to_tool) and "body:" + bid not in carried]
    posed = robot_cell.compute_attach_objects_frames(robot_cell_state) if mounted else robot_cell_state
    t_rcf = np.linalg.inv(_matrix(robot_cell_state.robot_base_frame or Frame.worldXY()))
    out = []
    for tool_id, tool_state in sorted(posed.tool_states.items()):
        if tool_state.is_hidden or "tool:" + tool_id in carried or tool_state.frame is None:
            continue
        t = t_rcf @ _matrix(tool_state.frame)
        tris = tool_triangles(robot_cell.tool_models[tool_id], tool_state.configuration)
        if len(tris):
            out.append(("tool:" + tool_id, tris @ t[:3, :3].T + t[:3, 3], list(tool_state.touch_links or []), t[:3, :3]))
    for body_id, body_state in sorted(posed.rigid_body_states.items()):
        attached = body_state.attached_to_link or body_state.attached_to_tool
        if not attached or body_state.is_hidden or "body:" + body_id in carried or body_state.frame is None:
            continue
        t = t_rcf @ _matrix(body_state.frame)
        tris = rigid_body_triangles(robot_cell, body_id)
        if len(tris):
            out.append(("body:" + body_id, tris @ t[:3, :3].T + t[:3, 3], list(body_state.touch_links or []), t[:3, :3]))
    return out


def add_carried_objects(description, robot_cell, robot_cell_state, group, seed=0, sphere_options=None):
    """Append spheres of the objects that move with ``group`` (see ``carried_objects``) to
    the links that carry them (in place).

    Collisions between an object and its touch links are disabled, and for a tool also
    with the end-effector link's nearest ancestor with geometry (the wrist it is mounted
    against). The objects share what is left of the core's sphere budget.
    """
    objects = carried_objects(robot_cell, robot_cell_state, group)
    if not objects:
        return description
    names = [link["name"] for link in description["links"]]
    options = dict(sphere_options or {})
    for key, used in (("max_spheres", len(description["spheres"])), ("max_self_spheres", len(description["self_spheres"]))):
        options[key] = max(1, min(options.get(key, DEFAULT_MAX_SPHERES), (MAX_SPHERES - used) // len(objects)))
    for name, link_name, t, tris, touch, is_tool in objects:
        link = names.index(link_name)
        tris = tris @ t[:3, :3].T + t[:3, 3]
        covering, inscribed = cached_sphere_sets(tris, seed, **options)
        description["spheres"] += [{"link": link, "center": c.tolist(), "radius": r} for c, r in covering]
        description["self_spheres"] += [{"link": link, "center": c.tolist(), "radius": r} for c, r in inscribed]
        touch = set(touch)
        if is_tool:
            # The closest ancestor of the end-effector link that has geometry (e.g. the wrist).
            parent = description["links"][link]["parent"]
            with_geometry = {s["link"] for s in description["spheres"]} - {link}
            while parent >= 0 and parent not in with_geometry:
                parent = description["links"][parent]["parent"]
            if parent >= 0:
                touch.add(names[parent])
        for other in touch:
            if other in names and names.index(other) != link:
                pair = sorted([link, names.index(other)])
                if pair not in description["disabled_collisions"]:
                    description["disabled_collisions"].append(pair)
    return description


add_attached_tools = add_carried_objects  # earlier name


def static_link_obstacles(robot_cell, robot_cell_state, group=None) -> list:
    """The links ``group`` cannot move (see ``planning_links``) as obstacles: one per link
    with geometry, ``(name, triangles in robot coordinates, touch links)``.

    A static link may touch its parent and children and every moving link the SRDF
    disables collisions with, which is what the core would have ignored had the link been
    planned.
    """
    tree, static = planning_links(robot_cell, group)
    if not static:
        return []
    moving = {link.name for link, _, _ in tree} - {link.name for link in static}
    disabled = {}
    for a, b in robot_cell.robot_semantics.disabled_collisions:
        disabled.setdefault(a, set()).add(b)
        disabled.setdefault(b, set()).add(a)
    config = robot_cell_state.robot_configuration
    joint_state = dict(zip(config.joint_names, config.joint_values)) if config and config.joint_names else {}
    for joint in robot_cell.robot_model.get_configurable_joints():
        joint_state.setdefault(joint.name, 0.0)
    obstacles = []
    for link in static:
        if not link.collision:
            continue
        t_link = np.array(Transformation.from_frame(robot_cell.robot_model.forward_kinematics(joint_state, link.name)).matrix)
        tris = meshes_to_triangles([(m, t_link @ t, scale) for m, t, scale in collision_items(link.collision)])
        if len(tris) == 0:
            continue
        touch = disabled.get(link.name, set()) & moving
        if link.parent_joint is not None and link.parent_joint.parent.link in moving:
            touch.add(link.parent_joint.parent.link)
        touch.update(j.child.link for j in link.joints if j.child.link in moving)
        parts = connected_parts(tris)
        for k, part in enumerate(parts):
            name = "link:" + link.name + ("/{}".format(k) if len(parts) > 1 else "")
            obstacles.append((name, part, sorted(touch)))
    return obstacles


def static_obstacles(robot_cell, robot_cell_state, group=None) -> list:
    """Everything that stands still while ``group`` plans, besides free rigid bodies: the
    links it cannot move (``static_link_obstacles``) and the tools and attached bodies it
    does not carry (``static_objects``)."""
    return static_link_obstacles(robot_cell, robot_cell_state, group) + static_objects(robot_cell, robot_cell_state, group)


def connected_parts(tris):
    """Split a triangle soup into its connected parts (shared welded vertices), so that a
    link like a hall of columns gets a box per column rather than one around the hall."""
    verts, inverse = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    parent = np.arange(len(verts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in faces:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(f[0]) for f in faces])
    return [tris[roots == r] for r in np.unique(roots)]


def obstacle_boxes(robot_cell, robot_cell_state, only_touching=False, group=None, static=None) -> dict:
    """Static rigid bodies as oriented bounding boxes, in robot coordinates (scene JSON).

    With ``only_touching``, only bodies that declare touch links (e.g. a floor); the rest
    then go into the ESDF (see ``exact_obstacle_meshes(..., without_touching=True)``).
    Links that ``group`` cannot move are included like rigid bodies (``static``: their
    ``static_link_obstacles``, if already computed).
    """
    base = robot_cell_state.robot_base_frame or Frame.worldXY()
    t_rcf = np.linalg.inv(np.array(Transformation.from_frame(base).matrix))
    cuboids = []
    for body_id, body_state in robot_cell_state.rigid_body_states.items():
        if body_state.is_hidden or body_state.attached_to_link or body_state.attached_to_tool:
            continue
        if only_touching and not body_state.touch_links:
            continue
        body = robot_cell.rigid_body_models[body_id]
        meshes = body.collision_meshes_in_meters or body.visual_meshes_in_meters
        if not meshes or body_state.frame is None:
            continue
        t_world = np.array(Transformation.from_frame(body_state.frame).matrix)
        pts = np.concatenate([np.asarray(m.to_vertices_and_faces()[0], dtype=float) for m in meshes])
        pts = pts @ t_world[:3, :3].T + t_world[:3, 3]
        pts = pts @ t_rcf[:3, :3].T + t_rcf[:3, 3]
        box = oriented_box(body_id, pts, axes=t_rcf[:3, :3] @ t_world[:3, :3])
        box["touch_links"] = list(body_state.touch_links or [])
        cuboids.append(box)
    if static is None:
        static = static_obstacles(robot_cell, robot_cell_state, group)
    for item in static:
        name, tris, touch = item[:3]
        if only_touching and not touch:
            continue
        box = oriented_box(name, tris.reshape(-1, 3), axes=item[3] if len(item) > 3 else None)
        box["touch_links"] = touch
        cuboids.append(box)
    return {"cuboids": cuboids}


def oriented_box(name, points, axes=None):
    """Smallest of a few bounding boxes of the points (all conservative).

    The candidates are the points' principal axes, the coordinate axes and ``axes`` (a
    rotation matrix, e.g. the rigid body's own frame). Principal axes alone are not enough:
    for a cube (or any shape with equal variances) they are arbitrary, and the box around
    a cube can come out up to sqrt(3) times its size.
    """
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    candidates = [vt.T, np.eye(3)]
    if axes is not None:
        candidates.append(np.asarray(axes, dtype=float))
    best = None
    for rot in candidates:
        rot = rot.copy()
        if np.linalg.det(rot) < 0:
            rot[:, 2] *= -1
        local = (points - center) @ rot
        lo, hi = local.min(axis=0), local.max(axis=0)
        volume = float(np.prod(hi - lo + 1e-9))
        if best is None or volume < best[0] - 1e-12:
            best = (volume, rot, lo, hi)
    _, rot, lo, hi = best
    mid = center + rot @ ((lo + hi) / 2)
    frame = np.eye(4)
    frame[:3, :3] = rot
    frame[:3, 3] = mid
    return {"name": name, "frame": frame[:3, :].reshape(-1).tolist(), "half_extents": ((hi - lo) / 2 + 1e-6).tolist()}


# ---------------------------------------------------------------------------
# Exact geometry, for the final collision pass on planned paths


def mesh_payload(tris):
    """Indexed mesh {"vertices": [...], "faces": [...]} from triangles, vertices welded."""
    verts, inverse = np.unique(np.round(tris.reshape(-1, 3), 9), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    return {"vertices": verts.reshape(-1).tolist(), "faces": faces.reshape(-1).astype(int).tolist()}


def exact_robot_meshes(robot_cell, robot_cell_state=None, group=None) -> dict:
    """Collision meshes per link (link coordinates), with the objects the group carries
    (``carried_objects``) merged into their links."""
    group = group or robot_cell.main_group_name
    ordered, static = planning_links(robot_cell, group)
    static = {link.name for link in static}
    names = [link.name for link, _, _ in ordered]
    per_link = {}
    for i, (link, _, _) in enumerate(ordered):
        if link.name in static:
            continue
        tris = meshes_to_triangles(collision_items(link.collision))
        if len(tris):
            per_link[i] = tris
    if robot_cell_state is not None:
        for _, link_name, t, tris, _, _ in carried_objects(robot_cell, robot_cell_state, group):
            i = names.index(link_name)
            tris = tris @ t[:3, :3].T + t[:3, 3]
            per_link[i] = np.concatenate([per_link[i], tris]) if i in per_link else tris
    return {"links": [dict(link=i, **mesh_payload(tris)) for i, tris in sorted(per_link.items())]}


def exact_obstacle_meshes(robot_cell, robot_cell_state, without_touching=False, group=None, static=None) -> dict:
    """Static rigid bodies, and links ``group`` cannot move, as meshes in robot
    coordinates, with their touch links."""
    base = robot_cell_state.robot_base_frame or Frame.worldXY()
    t_rcf = np.linalg.inv(np.array(Transformation.from_frame(base).matrix))
    obstacles = []
    for body_id, body_state in robot_cell_state.rigid_body_states.items():
        if body_state.is_hidden or body_state.attached_to_link or body_state.attached_to_tool or body_state.frame is None:
            continue
        if without_touching and body_state.touch_links:
            continue
        body = robot_cell.rigid_body_models[body_id]
        meshes = body.collision_meshes_in_meters or body.visual_meshes_in_meters
        if not meshes:
            continue
        t = t_rcf @ np.array(Transformation.from_frame(body_state.frame).matrix)
        tris = meshes_to_triangles([(m, t, [1.0, 1.0, 1.0]) for m in meshes])
        obstacles.append(dict(name=body_id, touch_links=list(body_state.touch_links or []), **mesh_payload(tris)))
    if static is None:
        static = static_obstacles(robot_cell, robot_cell_state, group)
    for item in static:
        name, tris, touch = item[:3]
        if without_touching and touch:
            continue
        obstacles.append(dict(name=name, touch_links=touch, **mesh_payload(tris)))
    return {"obstacles": obstacles}

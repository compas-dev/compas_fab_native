"""Describe a RobotCell for the native motion core (spike).

The native core takes the robot as a kinematic tree plus sphere approximations, and the
scene as oriented boxes. This module builds those descriptions from compas_fab objects:

* the kinematic tree in topological order, joint origins as 3x4 row-major transforms;
* two sphere sets per link: conservative covering spheres for checks against obstacles,
  and spheres inscribed in the link meshes for self-collision (covering spheres of
  neighbouring links overlap permanently);
* attached tools, as extra spheres on the end-effector link;
* static rigid bodies, as oriented bounding boxes (conservative).

Sphere fitting needs numpy, and trimesh for the inscribed set.
"""

import math
from typing import Optional

import numpy as np
from compas.geometry import Frame
from compas.geometry import Transformation

JOINT_TYPES = {0: "revolute", 1: "continuous", 2: "prismatic", 3: "fixed"}


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

    if solver is None:
        return None
    if isinstance(solver, OffsetWristKinematics):
        flange = solver.flange_frame or Frame.worldXY()
        return {"family": "offset_wrist", "params": list(solver.params), "flange_frame": frame_to_3x4(flange)}
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
        scale = getattr(shape, "scale", None) or [1.0, 1.0, 1.0]
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
    """One trimesh from triangles; None if trimesh is missing or the mesh is not watertight."""
    try:
        import trimesh
    except ImportError:
        return None
    if len(tris) == 0:
        return None
    mesh = trimesh.Trimesh(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3), process=True)
    if not mesh.is_watertight:
        # Components may each be watertight even if the merged mesh is not; check them.
        parts = mesh.split(only_watertight=False)
        if not parts or not all(p.is_watertight for p in parts):
            return None
        for p in parts:
            trimesh.repair.fix_normals(p)
        return trimesh.util.concatenate(parts)
    trimesh.repair.fix_normals(mesh)
    return mesh


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


# ---------------------------------------------------------------------------
# Robot and scene descriptions


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


def robot_description(robot_cell, group=None, kinematics_solver=None, seed=0, sphere_options=None) -> dict:
    """The robot of ``robot_cell`` as the native core's JSON description (without tools)."""
    sphere_options = sphere_options or {}
    rng = np.random.default_rng(seed)
    model = robot_cell.robot_model
    group = group or robot_cell.main_group_name
    joint_names = robot_cell.get_configurable_joint_names(group)
    ordered = topological_links(model)

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

    spheres, self_spheres = [], []
    for i, (link, _, _) in enumerate(ordered):
        tris = meshes_to_triangles(collision_items(link.collision))
        if len(tris) == 0:
            continue
        covering, inscribed = fit_sphere_sets(tris, rng, **sphere_options)
        spheres += [{"link": i, "center": c.tolist(), "radius": r} for c, r in covering]
        self_spheres += [{"link": i, "center": c.tolist(), "radius": r} for c, r in inscribed]

    limits = []
    for name in joint_names:
        limit = model.get_joint_by_name(name).limit
        limits.append([limit.lower, limit.upper] if limit else [-math.pi, math.pi])

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


def add_attached_tools(description, robot_cell, robot_cell_state, group, seed=0, sphere_options=None):
    """Append spheres of tools attached to ``group`` to its end-effector link (in place).

    Collisions between the tool and its touch links (and the end-effector link's nearest
    ancestor with geometry) are disabled, as the tool is mounted against them.
    """
    rng = np.random.default_rng(seed + 1)
    ee = description["end_effector_link"]
    names = [link["name"] for link in description["links"]]
    for tool_id, tool_state in robot_cell_state.tool_states.items():
        if tool_state.attached_to_group != group or tool_state.is_hidden:
            continue
        tool_model = robot_cell.tool_models[tool_id]
        tris = tool_triangles(tool_model, tool_state.configuration)
        if len(tris) == 0:
            continue
        t_attach = np.array(Transformation.from_frame(tool_state.attachment_frame or Frame.worldXY()).matrix)
        tris = tris @ t_attach[:3, :3].T + t_attach[:3, 3]
        covering, inscribed = fit_sphere_sets(tris, rng, **(sphere_options or {}))
        description["spheres"] += [{"link": ee, "center": c.tolist(), "radius": r} for c, r in covering]
        description["self_spheres"] += [{"link": ee, "center": c.tolist(), "radius": r} for c, r in inscribed]
        touch = set(tool_state.touch_links or [])
        # The closest ancestor of the end-effector link that has geometry (e.g. the wrist).
        parent = description["links"][ee]["parent"]
        with_geometry = {s["link"] for s in description["spheres"]} - {ee}
        while parent >= 0 and parent not in with_geometry:
            parent = description["links"][parent]["parent"]
        if parent >= 0:
            touch.add(names[parent])
        for name in touch:
            if name in names:
                pair = sorted([ee, names.index(name)])
                if pair not in description["disabled_collisions"]:
                    description["disabled_collisions"].append(pair)
    return description


def obstacle_boxes(robot_cell, robot_cell_state) -> dict:
    """Static rigid bodies as oriented bounding boxes, in robot coordinates (scene JSON)."""
    base = robot_cell_state.robot_base_frame or Frame.worldXY()
    t_rcf = np.linalg.inv(np.array(Transformation.from_frame(base).matrix))
    cuboids = []
    for body_id, body_state in robot_cell_state.rigid_body_states.items():
        if body_state.is_hidden or body_state.attached_to_link or body_state.attached_to_tool:
            continue
        body = robot_cell.rigid_body_models[body_id]
        meshes = body.collision_meshes_in_meters or body.visual_meshes_in_meters
        if not meshes or body_state.frame is None:
            continue
        t_world = np.array(Transformation.from_frame(body_state.frame).matrix)
        pts = np.concatenate([np.asarray(m.to_vertices_and_faces()[0], dtype=float) for m in meshes])
        pts = pts @ t_world[:3, :3].T + t_world[:3, 3]
        pts = pts @ t_rcf[:3, :3].T + t_rcf[:3, 3]
        box = oriented_box(body_id, pts)
        box["touch_links"] = list(body_state.touch_links or [])
        cuboids.append(box)
    return {"cuboids": cuboids}


def oriented_box(name, points):
    """Oriented bounding box from the principal axes of the points (conservative)."""
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    axes = vt.T
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    local = (points - center) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    mid = center + axes @ ((lo + hi) / 2)
    frame = np.eye(4)
    frame[:3, :3] = axes
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
    """Collision meshes per link (link coordinates), attached tools merged into the end-effector link."""
    group = group or robot_cell.main_group_name
    ordered = topological_links(robot_cell.robot_model)
    names = [link.name for link, _, _ in ordered]
    per_link = {}
    for i, (link, _, _) in enumerate(ordered):
        tris = meshes_to_triangles(collision_items(link.collision))
        if len(tris):
            per_link[i] = tris
    if robot_cell_state is not None:
        ee = names.index(robot_cell.get_end_effector_link_name(group))
        for tool_id, tool_state in robot_cell_state.tool_states.items():
            if tool_state.attached_to_group != group or tool_state.is_hidden:
                continue
            tris = tool_triangles(robot_cell.tool_models[tool_id], tool_state.configuration)
            if len(tris):
                t = np.array(Transformation.from_frame(tool_state.attachment_frame or Frame.worldXY()).matrix)
                tris = tris @ t[:3, :3].T + t[:3, 3]
                per_link[ee] = np.concatenate([per_link[ee], tris]) if ee in per_link else tris
    return {"links": [dict(link=i, **mesh_payload(tris)) for i, tris in sorted(per_link.items())]}


def exact_obstacle_meshes(robot_cell, robot_cell_state) -> dict:
    """Static rigid bodies as meshes in robot coordinates, with their touch links."""
    base = robot_cell_state.robot_base_frame or Frame.worldXY()
    t_rcf = np.linalg.inv(np.array(Transformation.from_frame(base).matrix))
    obstacles = []
    for body_id, body_state in robot_cell_state.rigid_body_states.items():
        if body_state.is_hidden or body_state.attached_to_link or body_state.attached_to_tool or body_state.frame is None:
            continue
        body = robot_cell.rigid_body_models[body_id]
        meshes = body.collision_meshes_in_meters or body.visual_meshes_in_meters
        if not meshes:
            continue
        t = t_rcf @ np.array(Transformation.from_frame(body_state.frame).matrix)
        tris = meshes_to_triangles([(m, t, [1.0, 1.0, 1.0]) for m in meshes])
        obstacles.append(dict(name=body_id, touch_links=list(body_state.touch_links or []), **mesh_payload(tris)))
    return {"obstacles": obstacles}

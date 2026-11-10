"""Geometry for the browser: robot link meshes, rigid bodies and markers, as plain JSON."""

import math

import numpy as np
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Transformation

from compas_fab_native.model import collision_items
from compas_fab_native.model import meshes_to_triangles
from compas_fab_native.model import tool_triangles

# Link colours, by link name (fallback: the first entry).
PALETTES = {
    "ur5": {
        "*": "#c9ced6",
        "shoulder_link": "#3d6f9e",
        "upper_arm_link": "#c9ced6",
        "forearm_link": "#c9ced6",
        "wrist_1_link": "#3d6f9e",
        "wrist_2_link": "#3d6f9e",
        "wrist_3_link": "#3d6f9e",
        "base_link_inertia": "#3d6f9e",
    },
    # Prefix rules end in "*"; the longest matching prefix wins.
    "rfl": {
        "*": "#e7e4de",
        "robot12_*": "#c9c8c4",
        "robot2*": "#c9c8c4",
        "bridge*": "#5b6670",
        "x_rail": "#5b6670",
        "robot11_xy_cart": "#46515b",
        "robot12_xy_cart": "#46515b",
        "robot21_xy_cart": "#46515b",
        "robot22_xy_cart": "#46515b",
        "robot11_base": "#46515b",
        "robot12_base": "#46515b",
        "robot21_base": "#46515b",
        "robot22_base": "#46515b",
        "world": "#9d9a93",
    },
    "panda": {"*": "#e4e6ea", "panda_link0": "#e4e6ea", "panda_hand": "#2d3036", "panda_leftfinger": "#2d3036", "panda_rightfinger": "#2d3036", "panda_link7": "#2d3036"},
}


def link_color(palette, name):
    colors = PALETTES[palette]
    if name in colors:
        return colors[name]
    prefixes = [k for k in colors if k.endswith("*") and k != "*" and name.startswith(k[:-1])]
    return colors[max(prefixes, key=len)] if prefixes else colors["*"]


def mesh_json(tris, decimals=5):
    """Indexed mesh from a (T, 3, 3) triangle soup (welded vertices)."""
    tris = np.asarray(tris, dtype=float)
    if len(tris) == 0:
        return None
    v = np.round(tris.reshape(-1, 3), decimals)
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    return {"v": uniq.ravel().tolist(), "f": inv.reshape(-1).astype(int).tolist()}


def compas_mesh_tris(mesh):
    v, f = mesh.to_vertices_and_faces(triangulated=True)
    v = np.asarray(v, dtype=float)
    return v[np.asarray(f, dtype=int)]


def robot_parts(cell, core, palette, state=None, visual=True):
    """One mesh per link with geometry (link coordinates), plus attached tools on the
    end-effector link."""
    by_name = {link.name: link for link in cell.robot_model.links}
    parts = []
    for index, name in enumerate(core.link_names):
        link = by_name.get(name)
        if link is None:
            continue
        elements = link.visual if visual and link.visual else link.collision
        tris = meshes_to_triangles(collision_items(elements)) if elements else []
        m = mesh_json(tris)
        if m is not None:
            parts.append({"guid": "link:" + name, "name": name, "link": index, "mesh": m, "color": link_color(palette, name)})
    if state is not None:
        for tool_id, ts in state.tool_states.items():
            if ts.attached_to_group and not ts.is_hidden:
                tris = tool_triangles(cell.tool_models[tool_id], ts.configuration)
                t = np.array(Transformation.from_frame(ts.attachment_frame or Frame.worldXY()).matrix)
                tris = tris @ t[:3, :3].T + t[:3, 3]
                parts.append({"guid": "tool:" + tool_id, "name": tool_id, "link": core.end_effector_link, "mesh": mesh_json(tris), "color": "#f2b705"})
    return parts


def static_link_bodies(cell, state, core, palette, visual=True):
    """The links the planning group cannot move (the other robots of a multi-robot cell,
    rails, the hall), as bodies posed once at the state's configuration."""
    from scenarios import Body  # noqa: F401  (avoid a cycle at import time)

    planned = set(core.link_names)
    config = state.robot_configuration
    joint_state = dict(zip(config.joint_names, config.joint_values))
    bodies = []
    for link in cell.robot_model.links:
        if link.name in planned:
            continue
        elements = link.visual if visual and link.visual else link.collision
        if not elements:
            continue
        t = np.array(Transformation.from_frame(cell.robot_model.forward_kinematics(joint_state, link.name)).matrix)
        tris = meshes_to_triangles([(m, t @ tm, s) for m, tm, s in collision_items(elements)])
        if len(tris):
            bodies.append(Body("static:" + link.name, link.name, tris, link_color(palette, link.name), Frame.worldXY(), flat=link.name == "world"))
    return bodies


def frame_rows(frame):
    """Row-major 3x4 of a frame (what the browser applies)."""
    m = np.array(Transformation.from_frame(frame).matrix)
    return [round(x, 6) for x in m[:3, :].ravel()]


def body_tris(cell, body_id):
    body = cell.rigid_body_models[body_id]
    meshes = body.visual_meshes_in_meters or body.collision_meshes_in_meters
    return np.concatenate([compas_mesh_tris(m) for m in meshes])


def target_marker_tris(size=0.09):
    """A pointer for a frame target, in front of the flange like a tool: arrowhead tip at
    the origin pointing back along -z (towards the flange), shaft along +z, and a fin
    along +x so rotations about z are visible."""
    tris = []
    shaft = Box(0.012, 0.012, size, Frame([0, 0, size / 2 + 0.02], [1, 0, 0], [0, 1, 0]))
    tris.append(compas_mesh_tris(shaft.to_mesh(triangulated=True)))
    fin = Box(0.05, 0.004, 0.03, Frame([0.025, 0, 0.07], [1, 0, 0], [0, 1, 0]))
    tris.append(compas_mesh_tris(fin.to_mesh(triangulated=True)))
    r, h = 0.022, 0.035
    base = [(r, r, h), (-r, r, h), (-r, -r, h), (r, -r, h)]
    tip = (0.0, 0.0, 0.0)
    head = [[base[(i + 1) % 4], base[i], tip] for i in range(4)]
    head += [[base[0], base[1], base[2]], [base[0], base[2], base[3]]]
    tris.append(np.array(head, dtype=float))
    return np.concatenate(tris)


def ring_axis_tris(length=0.5):
    box = Box(0.004, 0.004, length, Frame([0, 0, 0], [1, 0, 0], [0, 1, 0]))
    return compas_mesh_tris(box.to_mesh(triangulated=True))


def orthonormal(matrix):
    """Nearest rigid transform (drops any scale the gizmo added)."""
    m = np.array(matrix, dtype=float)
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = m[:3, 3]
    return out


def frame_from_matrix(m):
    m = np.asarray(m, dtype=float)
    return Frame(m[:3, 3].tolist(), m[:3, 0].tolist(), m[:3, 1].tolist())


def matrix_from_frame(frame):
    return np.array(Transformation.from_frame(frame).matrix)


def spiral_points(r_outer, r_inner, turns, spacing):
    """Archimedean spiral from outside in, in the xy plane, about ``spacing`` apart."""
    pts = []
    total = turns * 2 * math.pi
    t = 0.0
    while t <= total:
        r = r_outer - (r_outer - r_inner) * t / total
        pts.append((r * math.cos(t), r * math.sin(t)))
        t += spacing / max(r, 1e-3)
    return pts

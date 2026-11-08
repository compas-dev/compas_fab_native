"""Export a RobotCellLibrary robot as a fixture for the native core spike.

The fixture holds everything the Rust side needs, so the spike does not need
a URDF parser or compas_pb messages yet:

* the kinematic tree, in topological order, with each joint's origin (3x4,
  row-major, relative to the parent link) and axis;
* a sphere approximation of each link's collision geometry, in link
  coordinates;
* reference link frames computed by compas_robots for random configurations,
  which the Rust kinematics is checked against.

The sphere approximation is deliberately simple (k-means on area-weighted
surface samples, radius covering every sample of the cluster). It is
conservative and only meant to give the kernels a realistic sphere count.

Usage::

    python native/scripts/export_robot.py ur5 native/fixtures/ur5.json
"""

import argparse
import json
import math

import numpy as np
from compas.geometry import Frame
from compas.geometry import Transformation

from compas_fab import backends
from compas_fab.robots import RobotCellLibrary

JOINT_TYPES = {0: "revolute", 1: "continuous", 2: "prismatic", 3: "fixed"}

# RobotCellLibrary factory -> compas_fab analytical solver class (offset wrist family).
OFFSET_WRIST_SOLVERS = {
    "ur3": "UR3Kinematics",
    "ur3e": "UR3eKinematics",
    "ur5": "UR5Kinematics",
    "ur5e": "UR5eKinematics",
    "ur10": "UR10Kinematics",
    "ur10e": "UR10eKinematics",
    "ur16e": "UR16eKinematics",
}


def analytic_ik(robot):
    """Closed-form IK description, if compas_fab has an analytical solver for this robot."""
    name = OFFSET_WRIST_SOLVERS.get(robot)
    if name is None:
        return None
    solver = getattr(backends, name)()
    flange = solver.flange_frame or Frame.worldXY()
    return {"family": "offset_wrist", "params": list(solver.params), "flange_frame": frame_to_3x4(flange)}


def frame_to_3x4(frame):
    m = np.array(Transformation.from_frame(frame).matrix)
    return m[:3, :].reshape(-1).tolist()


def mesh_points_in_link(collision, n_samples, rng):
    """Area-weighted surface samples of a collision element, in link coordinates."""
    shape = collision.geometry.shape
    meshes = getattr(shape, "meshes", None)
    if not meshes:
        return np.zeros((0, 3))
    t_origin = np.array(Transformation.from_frame(collision.origin).matrix) if collision.origin else np.eye(4)
    scale = getattr(shape, "scale", None) or [1.0, 1.0, 1.0]
    tris = []
    for mesh in meshes:
        vertices, faces = mesh.to_vertices_and_faces(triangulated=True)
        v = np.asarray(vertices, dtype=float) * np.asarray(scale, dtype=float)
        tris.append(v[np.asarray(faces, dtype=int)])
    tris = np.concatenate(tris)  # (T, 3, 3)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(tris), size=n_samples, p=areas / areas.sum())
    r1 = np.sqrt(rng.random(n_samples))[:, None]
    r2 = rng.random(n_samples)[:, None]
    pts = (1 - r1) * a[idx] + r1 * (1 - r2) * b[idx] + r1 * r2 * c[idx]
    pts = np.concatenate([pts, tris.reshape(-1, 3)])  # keep vertices so extremes are covered
    return pts @ t_origin[:3, :3].T + t_origin[:3, 3]


def link_trimesh(collisions):
    """All collision meshes of a link as one trimesh, in link coordinates; None if not watertight."""
    import trimesh

    parts = []
    for collision in collisions:
        shape = collision.geometry.shape
        t = np.array(Transformation.from_frame(collision.origin).matrix) if collision.origin else np.eye(4)
        scale = np.asarray(getattr(shape, "scale", None) or [1.0, 1.0, 1.0], dtype=float)
        for mesh in getattr(shape, "meshes", None) or []:
            vertices, faces = mesh.to_vertices_and_faces(triangulated=True)
            v = (np.asarray(vertices, dtype=float) * scale) @ t[:3, :3].T + t[:3, 3]
            part = trimesh.Trimesh(v, np.asarray(faces, dtype=int), process=True)
            if not part.is_watertight:
                return None
            trimesh.repair.fix_normals(part)
            parts.append(part)
    return trimesh.util.concatenate(parts) if parts else None


def fit_spheres_inscribed(mesh, tolerance, max_spheres, radii, rng):
    """Greedy cover of the surface with spheres inside the mesh.

    Candidate centres are pushed inward from surface samples by each radius in ``radii``;
    each candidate gets the largest radius that stays inside (its signed distance to the
    surface). A surface sample is covered when it lies within ``tolerance`` of a sphere.
    Spheres are picked greedily by newly covered samples until everything is covered or
    ``max_spheres`` is reached.

    Returns (spheres, padding): ``padding`` is the largest distance from any surface sample
    to the sphere set. Spheres never poke out of the mesh (good for self-collision); radii
    padded by ``padding`` enclose the surface (conservative against obstacles).
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
        return [], float("inf")
    dist = np.linalg.norm(pts[None, :, :] - centers[:, None, :], axis=2) - rs[:, None]  # (C, P)
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
    padding = float(np.max(np.min(dist[chosen], axis=0)))
    return [(centers[i], float(rs[i])) for i in chosen], max(padding, 0.0)


def fit_spheres(points, target_radius, max_spheres, rng):
    """k-means clusters, each covered by a sphere around its centroid."""
    if len(points) == 0:
        return []
    extent = np.ptp(points, axis=0).max()
    k = int(np.clip(math.ceil(extent / (2 * target_radius)), 1, max_spheres))
    centers = points[rng.choice(len(points), size=k, replace=False)]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="RobotCellLibrary factory name, e.g. ur5")
    parser.add_argument("output")
    parser.add_argument("--samples", type=int, default=50, help="reference configurations")
    parser.add_argument("--sphere-radius", type=float, default=0.06, help="target sphere radius [m]")
    parser.add_argument("--max-spheres", type=int, default=12, help="max spheres per link")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-sphere-radius", type=float, default=0.03, help="self spheres: target radius [m]")
    parser.add_argument("--max-self-spheres", type=int, default=16, help="self spheres: max per link")
    parser.add_argument("--tolerance", type=float, default=0.01, help="self spheres: surface tolerance [m]")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cell, _ = getattr(RobotCellLibrary, args.robot)()
    model = cell.robot_model
    group = cell.main_group_name
    joint_names = cell.get_configurable_joint_names(group)

    # Topological order: root first, every link after its parent.
    links, parents, joints = [], [], []
    stack = [(model.root, -1, None)]
    while stack:
        link, parent, joint = stack.pop()
        index = len(links)
        links.append(link)
        parents.append(parent)
        joints.append(joint)
        for child_joint in reversed(link.joints):
            stack.append((child_joint.child_link, index, child_joint))

    exported_links = []
    for link, parent, joint in zip(links, parents, joints):
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
        exported_links.append(entry)

    spheres = []
    # Two sphere sets per link:
    # * "spheres": conservative covering spheres (k-means), for checks against obstacles;
    # * "self_spheres": spheres inside the mesh (inscribed, signed distance), for
    #   self-collision, where covering spheres of neighbouring links overlap permanently.
    # Links without a watertight mesh reuse their covering spheres for self-collision.
    self_spheres = []
    for i, link in enumerate(links):
        pts = [mesh_points_in_link(c, 2000, rng) for c in link.collision]
        pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        covering = fit_spheres(pts, args.sphere_radius, args.max_spheres, rng)
        for center, radius in covering:
            spheres.append({"link": i, "center": center.tolist(), "radius": radius})
        mesh = link_trimesh(link.collision) if link.collision else None
        if mesh is None:
            inscribed = covering
        else:
            radii = [args.self_sphere_radius * f for f in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
            inscribed, _ = fit_spheres_inscribed(mesh, args.tolerance, args.max_self_spheres, radii, rng)
        for center, radius in inscribed:
            self_spheres.append({"link": i, "center": center.tolist(), "radius": radius})

    configs, frames = [], []
    for _ in range(args.samples):
        config = cell.random_configuration(group)
        configs.append(list(config.joint_values))
        frames.append([frame_to_3x4(model.forward_kinematics(config, link.name)) for link in links])

    limits = []
    for name in joint_names:
        limit = model.get_joint_by_name(name).limit
        limits.append([limit.lower, limit.upper] if limit else [-math.pi, math.pi])

    index = {link.name: i for i, link in enumerate(links)}
    disabled = sorted([index[a], index[b]] for a, b in cell.robot_semantics.disabled_collisions if a in index and b in index)

    fixture = {
        "name": model.name,
        "end_effector_link": index[cell.get_end_effector_link_name(group)],
        "disabled_collisions": disabled,
        "analytic_ik": analytic_ik(args.robot),
        "group": group,
        "joint_names": joint_names,
        "joint_limits": limits,
        "links": exported_links,
        "spheres": spheres,
        "self_spheres": self_spheres,
        "reference_fk": {"configs": configs, "link_frames": frames},
    }
    with open(args.output, "w") as f:
        json.dump(fixture, f, indent=1)
    print("{}: {} links, {} dof, {} spheres, {} reference configs -> {}".format(model.name, len(links), len(joint_names), len(spheres), len(configs), args.output))


if __name__ == "__main__":
    main()

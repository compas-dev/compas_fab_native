"""Export a VAMP spherized robot (URDF + SRDF) as a native-core fixture.

VAMP (https://github.com/KavrakiLab/vamp, Apache-2.0) ships the robots of MotionBenchMaker
with a sphere decomposition of their collision geometry. Benchmarking against VAMP on the
same spheres and the same disabled pairs keeps the comparison about planning, not about
collision models.

The output carries an `attribution` field; the robots' licenses are listed in native/NOTICE.

Usage: python export_vamp_robot.py <vamp>/resources/panda/panda_spherized.urdf <vamp>/resources/panda/panda.srdf fixtures/vamp_panda.json
"""

import json
import math
import sys
import xml.etree.ElementTree as ET


def floats(text, default):
    return [float(v) for v in text.split()] if text else default


def origin_3x4(element):
    """URDF origin (xyz, rpy as fixed-axis X then Y then Z) as a 3x4 row-major matrix."""
    xyz, rpy = [0.0] * 3, [0.0] * 3
    if element is not None:
        xyz = floats(element.get("xyz"), xyz)
        rpy = floats(element.get("rpy"), rpy)
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rot = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [rot[0][0], rot[0][1], rot[0][2], xyz[0], rot[1][0], rot[1][1], rot[1][2], xyz[1], rot[2][0], rot[2][1], rot[2][2], xyz[2]]


def transform_point(m, p):
    return [m[4 * i] * p[0] + m[4 * i + 1] * p[1] + m[4 * i + 2] * p[2] + m[4 * i + 3] for i in range(3)]


def main(urdf_path, srdf_path, out_path):
    urdf = ET.parse(urdf_path).getroot()
    srdf = ET.parse(srdf_path).getroot()
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    link_elements = {link.get("name"): link for link in urdf.findall("link")}
    roots = [name for name in link_elements if name not in joints]
    assert len(roots) == 1, roots

    # Topological order (parents first), depth-first from the root.
    children = {}
    for child, joint in joints.items():
        children.setdefault(joint.find("parent").get("link"), []).append(child)
    order, stack = [], [roots[0]]
    while stack:
        name = stack.pop()
        order.append(name)
        stack.extend(reversed(children.get(name, [])))
    index = {name: i for i, name in enumerate(order)}

    links, joint_names, joint_limits, spheres = [], [], [], []
    for name in order:
        joint = joints.get(name)
        entry = {"name": name, "parent": -1, "joint": None}
        if joint is not None:
            kind = joint.get("type")
            q_index = -1
            if kind in ("revolute", "continuous", "prismatic") and joint.find("mimic") is None:
                q_index = len(joint_names)
                joint_names.append(joint.get("name"))
                limit = joint.find("limit")
                lower = float(limit.get("lower", -math.pi)) if limit is not None else -math.pi
                upper = float(limit.get("upper", math.pi)) if limit is not None else math.pi
                joint_limits.append([lower, upper])
            elif kind not in ("fixed",):
                raise ValueError("unsupported joint {} ({})".format(joint.get("name"), kind))
            axis = joint.find("axis")
            entry["parent"] = index[joint.find("parent").get("link")]
            entry["joint"] = {
                "type": "fixed" if q_index < 0 else kind,
                "origin": origin_3x4(joint.find("origin")),
                "axis": floats(axis.get("xyz") if axis is not None else None, [1.0, 0.0, 0.0]) if q_index >= 0 else [1.0, 0.0, 0.0],
                "q_index": q_index,
            }
        links.append(entry)
        for collision in link_elements[name].findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is None:
                continue
            center = transform_point(origin_3x4(collision.find("origin")), [0.0, 0.0, 0.0])
            spheres.append({"link": index[name], "center": center, "radius": float(sphere.get("radius"))})

    disabled = set()
    for pair in srdf.findall("disable_collisions"):
        a, b = pair.get("link1"), pair.get("link2")
        if a in index and b in index:
            disabled.add(tuple(sorted((index[a], index[b]))))

    fixture = {
        "name": urdf.get("name"),
        "source": "VAMP spherized robot: {}, {}".format(urdf_path.split("resources/")[-1], srdf_path.split("resources/")[-1]),
        "attribution": (
            "Converted from VAMP (https://github.com/KavrakiLab/vamp, Apache-2.0) spherized robot resources, "
            "based on robowflex_resources (https://github.com/KavrakiLab/robowflex_resources, MIT, "
            "Copyright (c) 2020 Kavraki Lab). See native/NOTICE."
        ),
        "joint_names": joint_names,
        "joint_limits": joint_limits,
        "links": links,
        "spheres": spheres,
        "end_effector_link": len(order) - 1,
        "disabled_collisions": sorted(list(p) for p in disabled),
        "reference_fk": {"configs": [], "link_frames": []},
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=1)
    print("{}: {} links, {} joints, {} spheres, {} disabled pairs".format(fixture["name"], len(links), len(joint_names), len(spheres), len(disabled)))


if __name__ == "__main__":
    main(*sys.argv[1:4])

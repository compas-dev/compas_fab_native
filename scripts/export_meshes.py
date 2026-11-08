"""Export a RobotCellLibrary robot's collision meshes (link coordinates) for the exact checker.

Usage::

    python native/scripts/export_meshes.py ur5 native/fixtures/ur5_meshes.json
"""

import argparse
import json

from compas_fab.backends.native.model import exact_robot_meshes
from compas_fab.robots import RobotCellLibrary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot")
    parser.add_argument("output")
    args = parser.parse_args()
    cell, _ = getattr(RobotCellLibrary, args.robot)()
    meshes = exact_robot_meshes(cell)
    with open(args.output, "w") as f:
        json.dump(meshes, f)
    print("{} link meshes, {} triangles -> {}".format(len(meshes["links"]), sum(len(m["faces"]) // 3 for m in meshes["links"]), args.output))


if __name__ == "__main__":
    main()

"""Check native-core paths with VAMP's own collision checker.

Every segment is checked with VAMP's validate_motion (its planning resolution), and every
configuration and 8x denser interpolations with validate.

    python vamp_check_paths.py panda <vamp>/resources/panda/problems.json paths.json
"""

import json
import sys

import numpy as np
import vamp


def main(robot_name, problems_path, paths_path):
    robot = getattr(vamp, robot_name)
    problems = json.load(open(problems_path))["problems"]
    paths = json.load(open(paths_path))
    bad_motion = bad_dense = bad_goal = 0
    for entry in paths:
        problem = next(p for p in problems[entry["scenario"]] if p["index"] == entry["index"])
        env = vamp.problem_dict_to_vamp(problem)
        path = [np.array(q, dtype=np.float32) for q in entry["path"]]
        start_ok = np.allclose(path[0], problem["start"], atol=1e-5)
        goal_ok = any(np.allclose(path[-1], g, atol=1e-5) for g in problem["goals"])
        bad_goal += not (start_ok and goal_ok)
        motion_ok = all(robot.validate_motion(a, b, env) for a, b in zip(path, path[1:]))
        dense_ok = True
        for a, b in zip(path, path[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) * 256)), 1)
            if not all(robot.validate(a + (b - a) * (k / n), env) for k in range(n + 1)):
                dense_ok = False
                break
        bad_motion += not motion_ok
        bad_dense += not dense_ok
    print("{} paths: {} with a segment VAMP rejects, {} colliding at 256 checks/rad, {} not from start to a goal".format(len(paths), bad_motion, bad_dense, bad_goal))


if __name__ == "__main__":
    main(*sys.argv[1:4])

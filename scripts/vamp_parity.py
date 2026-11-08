"""Compare the native core's collision verdicts and sphere positions with VAMP's.

Run from a Python environment with VAMP installed (vamp-planner):
    python vamp_parity.py panda <vamp>/resources/panda/problems.json parity.json
"""

import json
import sys

import numpy as np
import vamp


def main(robot_name, problems_path, parity_path):
    robot = getattr(vamp, robot_name)
    problems = json.load(open(problems_path))["problems"]
    parity = json.load(open(parity_path))

    # Sphere positions: same multiset of spheres (order may differ).
    worst = 0.0
    for q, ours in zip(parity["sphere_configs"], parity["spheres"]):
        theirs = np.array([[s.x, s.y, s.z, s.r] for s in robot.fk(np.array(q, dtype=np.float32))])
        ours = np.array(ours)
        assert len(ours) == len(theirs), (len(ours), len(theirs))
        # Match each of ours to its nearest VAMP sphere of the same radius.
        for s in ours:
            d = np.linalg.norm(theirs[:, :3] - s[:3], axis=1) + 10 * np.abs(theirs[:, 3] - s[3])
            worst = max(worst, d.min())
    print("sphere positions: worst mismatch {:.2e} m over {} configurations".format(worst, len(parity["spheres"])))

    total = disagree = collide = 0
    examples = []
    for case in parity["cases"]:
        problem = next(p for p in problems[case["scenario"]] if p["index"] == case["index"])
        env = vamp.problem_dict_to_vamp(problem)
        for q, ours in zip(case["configs"], case["collides"]):
            theirs = not robot.validate(np.array(q, dtype=np.float32), env)
            total += 1
            collide += theirs
            if ours != theirs:
                disagree += 1
                if len(examples) < 5:
                    examples.append((case["scenario"], case["index"], ours, theirs))
    print("verdicts: {} configurations, {} colliding (VAMP), {} disagreements".format(total, collide, disagree))
    for e in examples:
        print("  {} #{}: ours collides={}, VAMP collides={}".format(*e))


if __name__ == "__main__":
    main(*sys.argv[1:4])

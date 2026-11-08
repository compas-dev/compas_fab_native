"""Plan every MotionBenchMaker problem with VAMP (RRT-Connect + its default simplification)
and write the paths in the format of `mbm_bench --out`, for `vamp_check_paths.py`.

    python vamp_plan_paths.py panda <vamp>/resources/panda/problems.json vamp_paths.json
"""

import json
import sys

import numpy as np
import vamp


def main(robot_name, problems_path, out_path):
    robot, planner, plan_settings, simp_settings = vamp.configure_robot_and_planner_with_kwargs(robot_name, "rrtc")
    sampler = robot.halton()
    problems = json.load(open(problems_path))["problems"]
    out = []
    for scenario, pset in problems.items():
        for data in pset:
            if not data["valid"]:
                continue
            env = vamp.problem_dict_to_vamp(data)
            sampler.reset()
            result = planner(data["start"], data["goals"], env, plan_settings, sampler)
            if not result.solved:
                continue
            simple = robot.simplify(result.path, env, simp_settings, sampler)
            path = np.asarray(simple.path.numpy())[:, : robot.dimension()].tolist()
            out.append({"scenario": scenario, "index": data["index"], "path": path})
    json.dump(out, open(out_path, "w"))
    print("{} paths".format(len(out)))


if __name__ == "__main__":
    main(*sys.argv[1:4])

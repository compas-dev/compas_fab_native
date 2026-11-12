# Web viewer

A browser viewer for the native planner, built on
[compas_threejs](https://github.com/compas-dev/compas_threejs). Pick a robot, drag a
target, and try IK, collision checks, free motions and straight lines, with box
obstacles you place yourself.

## Run

```bash
pip install compas_fab_native compas_threejs
python examples/web_viewer/viewer.py
```

It opens the viewer in your browser (http://127.0.0.1:9001). Stop it with Ctrl+C.

### Your own robot

```bash
python examples/web_viewer/viewer.py --urdf robot.urdf --srdf robot.srdf --meshes path/to/packages
```

- `--meshes` is the folder the URDF's mesh paths point into (what `package://` resolves
  to). Without it the robot has no geometry: nothing to see, nothing to collide.
- `--srdf` is optional. Without it, the planning group is the chain from the root to the
  last movable joint. Pass one when that guess is wrong, for example when the last
  movable joint is a gripper finger.
- `--solver` names a compas_fab closed-form solver, for UR arms only (e.g.
  `--solver UR10eKinematics`). Spherical-wrist arms get a closed form without it; other
  robots get numerical IK.

Your robot is listed first in the robot menu; the library robots stay available.

## What to try

| | |
|---|---|
| **Drag the target** | Click the green pointer, then drag it: W moves, E rotates. The robot jumps to the IK solution closest to where it is. The pointer turns red when there is no collision-free solution. |
| **Next IK solution** | Cycles through all solutions for the target (up to 8 for a 6-axis arm). |
| **Free motion to target** | Plans around obstacles and plays the motion. The speed slider scales it. |
| **Straight line to target** | A Cartesian path from where the tool is to the target. |
| **Free rotation about the tool axis** | Targets become a point and an axis: any rotation about the tool axis is fine. More IK solutions, and lines that are impossible otherwise. |
| **Jog a joint** | Click a robot link: its joint gets +10° and −10° buttons. Each step is collision-checked. |
| **Add a box** | Click it to move (W), rotate (E) or scale (R) it. Then plan again. |
| **Obstacle model** | Boxes (bounding boxes) or a distance field of the obstacle meshes. |
| **Check on exact meshes** | Collision checks on the link meshes instead of only the sphere model. |
| **Robot** | UR5, with a tool, carrying a beam (the beam collides too), an ABB with a printing tool, a Stäubli, and a 7-axis Panda. The status line says which IK each one gets. |

Planning errors (no path, no IK solution) appear in the status line above the robot.

## How it works

`viewer.py` subclasses compas_threejs' `App`. Buttons and drags put commands in a
queue; the main loop runs them with a `NativePlanner` and plays the trajectories.

Each robot link is sent to the browser once, as a mesh in link coordinates. After that
only poses travel: `NativeCore.link_frames` gives every link's pose for a configuration,
and the viewer sends each link's change of pose (a small message, not the mesh). Boxes
you add are rigid bodies of the robot cell, like any obstacle in compas_fab.

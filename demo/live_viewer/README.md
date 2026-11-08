# Native planner, live

A browser demo of `NativePlanner`. A Python server plans with the native core and streams link
poses over a websocket at 60 Hz. A [compas_threejs_ts](https://github.com/compas-dev/compas_threejs_ts)
viewer in the page draws them, and its transform gizmo sends the objects you drag back to the
server, which re-plans.

| | Scenario | What to try |
|---|---|---|
| a | UR5 around a wall | Drag the yellow pointer (a flange frame target) or the wall. Every IK solution is a goal. |
| b | Panda around a post | The same with seven axes. The Panda has no closed-form IK, so the demo finds solutions by damped least squares. |
| c | Servo a Panda | Jog the gripper with the keys or the on-screen pad. Each step is checked for collisions before it is sent. **Home** plans a way back. |
| d | Milling a spiral | Drag or tilt the stock. Free motion to the start of a coplanar spiral, then 5 turns at a feed speed. The cone may touch the stock; the arm may not. |
| e | Plunge through a ring | Drag or tilt the ring. The robot plunges along its axis. Switch the obstacle model to **Bounding boxes** to see why it needs the distance field. |

## Run

```bash
# once: build the page (Node >= 22) and install the websocket server
cd native/demo/live_viewer/web
npm ci
npm run build
pip install websockets          # into the environment with compas_fab and compas_fab_native

# then
python native/demo/live_viewer/server.py      # builds the five scenarios (~15 s), then serves
# open http://127.0.0.1:8765  (?scenario=wall|post|servo|mill|ring)
```

For front-end work, run `npm run dev` in `web/` (port 5173). It proxies `/ws` to the server.

### Controls

- **Viewer:** left-click an object to select it, then drag the gizmo. W moves and E rotates
  (also in the viewer toolbar). Right-drag orbits, the wheel zooms and Esc deselects.
- **Servo:** W/S, A/D and R/F move along ±X, ±Y and ±Z. The arrows rotate about X and Y, and
  Q/E about Z. All in world axes. While servoing, these keys go to the robot, not the viewer.

## How it works

- `server.py`: a `websockets` server that also serves `web/dist`. Each browser connection is a
  `Session`. It plans with the scenario, then streams the timed trajectory in real time,
  interpolating the 60 Hz samples by wall-clock time. Dragging during a motion cancels it and
  re-plans from where the robot is. One browser tab at a time drives the robots.
- `scenarios.py`: one class per scenario, each with its own `RobotCell`, `NativePlanner`
  and draggable bodies. Free motions use `plan_motion` (`FrameTarget`, `PointAxisTarget`,
  `ConfigurationTarget`). Cartesian paths use `plan_cartesian_motion`, and both are timed by
  the core's TOPP-RA.
- `kin.py`: a numeric Jacobian on the core's forward kinematics, damped-least-squares IK and
  a servo step with null-space pull to a home posture (for the 7-axis Panda).
- `scene.py`: link and body meshes for the browser.
- `web/src/main.ts`: the page. Meshes go into the viewer once, as COMPAS Protobuf `Mesh`
  messages in each object's own frame. Every pose after that is the viewer's own
  `apply_transform` command, with the delta from the object's last placement. The gizmo's
  `object_transform` (on release) is forwarded to the server. Robot links are not draggable.

The websocket protocol is JSON:

- **Browser → server:** `hello`, `select`, `moved` (guid and absolute 4x4), `jog` (a twist of
  six values in -1..1, resent while held) and `action` (`reset`, `replay`, `home`, `obstacles`).
- **Server → browser:** `scenarios`, `scene` (meshes, bodies, joints, camera), `pose` (link
  frames, q, q̇, path progress, clearance), `body`, `path` and `status`.

## Attribution

The page uses these (all through npm, pinned in `web/package-lock.json`):

- [compas_threejs_ts](https://github.com/compas-dev/compas_threejs_ts) 1.5.0, MIT, COMPAS
  Association. Its bundle carries the Inter font licence (OFL) in `dist-lib/licenses`.
- [compas-pb-ts](https://github.com/gramaziokohler/compas_pb_ts) 3.0.0, MIT.
- [three.js](https://threejs.org) 0.182.0, MIT, three.js authors.
- [Vue](https://vuejs.org) 3.5, MIT, Evan You and contributors.
- [@bufbuild/protobuf](https://github.com/bufbuild/protobuf-es), Apache-2.0 and BSD-3-Clause.
- [IBM Plex Sans and Mono](https://github.com/IBM/plex), SIL Open Font License 1.1, loaded
  from Google Fonts.

The server uses [websockets](https://github.com/python-websockets/websockets) (BSD-3-Clause,
Aymeric Augustin and contributors).

The robot meshes come from compas_fab's robot library:

- **UR5:** Universal Robots' `ur_description` (ROS-Industrial), BSD-3-Clause; see
  `src/compas_fab/data/robot_library/ur_description/LICENSE`.
- **Panda:** Franka Emika's `franka_description` (franka_ros), Apache-2.0.

The planning algorithms are credited in `native/NOTICE`.

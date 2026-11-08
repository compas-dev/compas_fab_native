"""Live NativePlanner demo: a websocket server that plans with the native core and streams
link poses to a compas_threejs_ts viewer in the browser, in real time.

    python server.py            # then open http://127.0.0.1:8765

The browser page (``web/``, built to ``web/dist``) is served from the same port; the
websocket lives at ``/ws``. One browser tab at a time drives the robots.
"""

import argparse
import asyncio
import bisect
import json
import mimetypes
import sys
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

sys.path.insert(0, str(Path(__file__).parent))

from scenarios import DT  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from scenarios import PlanFailed  # noqa: E402
from scene import orthonormal  # noqa: E402

HERE = Path(__file__).parent
DIST = HERE / "web" / "dist"
JOG_TIMEOUT = 0.25  # s without a jog message and the servo stops


def rows(frames):
    return [[round(x, 5) for x in f] for f in frames]


class Session:
    """One browser connection: the selected scenario, the robot's motion and the servo."""

    def __init__(self, ws, scenarios):
        self.ws = ws
        self.scenarios = scenarios
        self.scenario = None
        self.motion = None
        self.jog = (np.zeros(6), 0.0)
        self.servo_task = None
        self.servo_state = None

    async def send(self, message):
        await self.ws.send(json.dumps(message, separators=(",", ":")))

    async def status(self, state, text="", **metrics):
        await self.send({"type": "status", "state": state, "text": text, "metrics": {k: v for k, v in metrics.items() if v is not None}})

    async def pose(self, q, qd=None, progress=None, clearance=None):
        s = self.scenario
        message = {"type": "pose", "q": [round(float(x), 5) for x in q], "frames": rows(s.core.link_frames([float(x) for x in q]))}
        if qd is not None:
            message["qd"] = [round(float(x), 4) for x in qd]
        if progress is not None:
            message["progress"] = progress
        if clearance is not None:
            message["clearance"] = round(float(clearance), 4)
        await self.send(message)

    # -- messages -----------------------------------------------------------------------

    async def handle(self, msg):
        kind = msg.get("type")
        if kind == "hello":
            await self.send({"type": "scenarios", "items": [{"id": s.id, "title": s.title} for s in self.scenarios.values()]})
            await self.select(msg.get("scenario") or "wall")
        elif kind == "select":
            await self.select(msg["id"])
        elif kind == "moved":
            await self.moved(msg["guid"], msg["matrix"])
        elif kind == "jog":
            self.jog = (np.asarray(msg.get("twist", [0] * 6), dtype=float), time.monotonic())
        elif kind == "action":
            await self.action(msg.get("name"), msg.get("value"))

    async def select(self, sid):
        await self.stop()
        if sid not in self.scenarios:
            sid = "wall"
        self.scenario = s = self.scenarios[sid]
        await self.send(s.scene_json())
        await self.pose(s.q, clearance=s.core.clearance([list(map(float, s.q))])[0])
        if s.controls.get("servo"):
            self.servo_task = asyncio.create_task(self.servo_loop())
            await self.status("idle", "Ready to jog.")
        else:
            await self.replan()

    async def moved(self, guid, matrix):
        s = self.scenario
        if guid not in s.bodies or not s.bodies[guid].draggable:
            return
        await self.cancel_motion()
        body = s.move_body(guid, orthonormal(matrix))
        # Echo the placement the server keeps (rigid, without any scale the gizmo added).
        for b in s.bodies.values():
            await self.send({"type": "body", "guid": b.guid, "pose": _rows12(b.frame)})
        await self.pose(s.q, clearance=s.core.clearance([list(map(float, s.q))])[0])
        if s.on_moved(body):
            await self.replan()

    async def action(self, name, value):
        s = self.scenario
        if name == "reset":
            await self.cancel_motion()
            s.reset()
            await self.select(s.id)
        elif name in ("replay", "home"):
            await self.cancel_motion()
            await self.replan()
        elif name == "obstacles" and hasattr(s, "set_mode"):
            await self.cancel_motion()
            s.set_mode(value)
            await self.replan()

    # -- motion -------------------------------------------------------------------------

    async def replan(self):
        s = self.scenario
        await self.status("planning", "Planning…")
        await asyncio.sleep(0)
        try:
            segments, metrics = s.plan()
        except PlanFailed as e:
            await self.send({"type": "path", "points": []})
            await self.status("failed", str(e))
            await self.flag_target(False)
            return
        await self.flag_target(True)
        self.motion = asyncio.create_task(self.play(segments, metrics))

    async def flag_target(self, ok):
        s = self.scenario
        for guid in ("target", "stock", "ring"):
            if guid in s.bodies:
                await self.send({"type": "body", "guid": guid, "color": s.bodies[guid].color if ok else "#d13b3b"})

    async def play(self, segments, metrics):
        """Stream the planned segments in real time (60 Hz), interpolating the timed samples."""
        s = self.scenario
        points = s.tcp_points(segments, s.arm)
        await self.send({"type": "path", "points": points})
        offset = 0
        try:
            for seg in segments:
                if "dwell" in seg:
                    await self.status("moving", seg.get("label", ""), **metrics)
                    await asyncio.sleep(seg["dwell"])
                    continue
                await self.status("moving", seg.get("label", ""), **metrics)
                times = seg["times"]
                pos = np.asarray(seg["positions"], dtype=float)
                vel = np.asarray(seg["velocities"], dtype=float)
                t0 = time.monotonic()
                while True:
                    t = time.monotonic() - t0
                    if t >= times[-1]:
                        s.q = pos[-1].copy()
                        await self.pose(s.q, np.zeros(len(s.q)), progress=offset + len(times) - 1)
                        break
                    i = max(0, bisect.bisect_right(times, t) - 1)
                    span = times[i + 1] - times[i] if i + 1 < len(times) else 1.0
                    u = (t - times[i]) / span if span > 0 else 0.0
                    s.q = pos[i] + (pos[min(i + 1, len(pos) - 1)] - pos[i]) * u
                    qd = vel[i] + (vel[min(i + 1, len(vel) - 1)] - vel[i]) * u
                    await self.pose(s.q, qd, progress=offset + i)
                    await asyncio.sleep(DT)
                offset += len(times)
            await self.status("idle", "Done. Drag something to plan again.", **metrics)
        except asyncio.CancelledError:
            raise

    async def cancel_motion(self):
        if self.motion and not self.motion.done():
            self.motion.cancel()
            try:
                await self.motion
            except asyncio.CancelledError:
                pass
        self.motion = None

    async def stop(self):
        await self.cancel_motion()
        if self.servo_task:
            self.servo_task.cancel()
            try:
                await self.servo_task
            except asyncio.CancelledError:
                pass
            self.servo_task = None

    async def servo_loop(self):
        """60 Hz: turn the latest jog command into a checked joint step and stream it."""
        s = self.scenario
        last_state = None
        tick = 0
        while True:
            await asyncio.sleep(DT)
            if self.motion and not self.motion.done():
                continue  # "Home" is playing
            twist, stamp = self.jog
            if time.monotonic() - stamp > JOG_TIMEOUT:
                twist = np.zeros(6)
            before = s.q.copy()
            moved, blocked = s.servo(twist, DT)
            tick += 1
            if moved:
                clearance = s.core.clearance([list(map(float, s.q))])[0] if tick % 3 == 0 else None
                await self.pose(s.q, (s.q - before) / DT, clearance=clearance)
            state = "blocked" if blocked else ("servo" if moved else "idle")
            if state != last_state:
                if state == "blocked":
                    await self.status("blocked", blocked)
                    await self.pose(s.q, np.zeros(len(s.q)), clearance=s.core.clearance([list(map(float, s.q))])[0])
                elif state == "servo":
                    await self.status("servo", "Jogging.")
                else:
                    await self.status("idle", "Ready to jog.")
                    await self.pose(s.q, np.zeros(len(s.q)))
                last_state = state


def _rows12(frame):
    from scene import frame_rows

    return frame_rows(frame)


def process_request(connection, request):
    """Serve the built page; let /ws through to the websocket handler."""
    path = request.path.split("?", 1)[0]
    if path == "/ws":
        return None
    if path == "/":
        path = "/index.html"
    target = (DIST / path.lstrip("/")).resolve()
    if not str(target).startswith(str(DIST.resolve())) or not target.is_file():
        if not DIST.exists():
            body = b"web/dist is missing: run `npm install && npm run build` in web/ first."
            return Response(500, "Internal Server Error", Headers([("Content-Type", "text/plain")]), body)
        return connection.respond(404, "Not found\n")
    body = target.read_bytes()
    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return Response(200, "OK", Headers([("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-cache")]), body)


async def main(host, port):
    print("Building scenarios (robot models, spheres, distance fields)…", flush=True)
    scenarios = {}
    for cls in SCENARIOS:
        t0 = time.perf_counter()
        scenarios[cls.id] = cls()
        print("  {:<8} {:.1f} s".format(cls.id, time.perf_counter() - t0), flush=True)

    async def handler(ws):
        session = Session(ws, scenarios)
        try:
            async for raw in ws:
                await session.handle(json.loads(raw))
        except ConnectionClosed:
            pass
        finally:
            await session.stop()

    async with serve(handler, host, port, process_request=process_request, max_size=2**22):
        print("Open http://{}:{}".format(host, port), flush=True)
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        pass

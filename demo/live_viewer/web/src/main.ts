/**
 * Browser half of the live NativePlanner demo.
 *
 * The 3D view is a compas_threejs_ts viewer in embedded mode. Everything it draws goes in
 * through `viewer.dispatch` as COMPAS Protobuf messages: link and body meshes once per
 * scene, then the viewer's own `apply_transform` command for every pose the Python server
 * streams over the websocket. The viewer's transform gizmo reports drags back through
 * `send` (`object_transform`), which this page forwards to the server so it can re-plan.
 */
import { createViewer } from "@compas-dev/compas-threejs-ts";
import "@compas-dev/compas-threejs-ts/style.css";
import { BUILTIN_TYPES, Mesh, pbDump, pbDumpBytes, registerTypes } from "@gramaziokohler/compas-pb-ts";
import { Matrix4 } from "three";
import "./app.css";

// -- types of the server's messages ------------------------------------------------------

interface MeshJson {
  v: number[];
  f: number[];
}
interface Part {
  guid: string;
  name: string;
  link: number;
  mesh: MeshJson;
  color: string;
}
interface Body {
  guid: string;
  name: string;
  mesh: MeshJson;
  color: string;
  pose: number[];
  draggable: boolean;
  opacity: number;
  flat: boolean;
}
interface Joints {
  names: string[];
  lower: number[];
  upper: number[];
  vmax: number[];
}
interface SceneMsg {
  type: "scene";
  id: string;
  title: string;
  hint: string;
  controls: { servo?: boolean; obstacles?: string[]; mode?: string };
  camera: { position: number[]; target: number[] };
  joints: Joints;
  parts: Part[];
  bodies: Body[];
}

// -- viewer ---------------------------------------------------------------------------------

// compas-pb-ts registers its wrapper types as a module side effect, but its package.json
// says `"sideEffects": false`, so bundlers drop that registration. Register explicitly.
registerTypes(BUILTIN_TYPES);

const container = document.querySelector<HTMLElement>("#viewer")!;
const viewer = createViewer(container, {
  mode: "embedded",
  defaultLighting: true,
  showToolbar: true,
  send: onViewerMessage,
  onError(error) {
    console.error(error.code, error.message, error.details);
  },
});
new ResizeObserver(() => viewer.resize()).observe(container);

const command = (message: Record<string, unknown>) => viewer.dispatch(pbDump(message));

/** Where the viewer currently has each object (world matrix), so a new absolute pose can
 * be sent as the delta `apply_transform` expects. */
const placed = new Map<string, Matrix4>();
const draggable = new Set<string>();
let parts: Part[] = [];
let joints: Joints | null = null;
let current: SceneMsg | null = null;

function rowsOf(m: Matrix4): number[][] {
  const e = m.elements; // column-major
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [0, 0, 0, 1],
  ];
}
function from12(r: number[]): Matrix4 {
  return new Matrix4().set(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], 0, 0, 0, 1);
}
function fromRows(rows: number[][]): Matrix4 {
  const m = new Matrix4();
  m.set(...(rows.flat() as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
  return m;
}

function setPose(guid: string, target: Matrix4) {
  const before = placed.get(guid);
  if (!before) return;
  const a = target.elements;
  const b = before.elements;
  let change = 0;
  for (let i = 0; i < 16; i++) change = Math.max(change, Math.abs(a[i] - b[i]));
  if (change < 1e-7) return;
  const delta = target.clone().multiply(before.clone().invert());
  command({ dispatch: "handle_geometry", type: "apply_transform", guid, matrix: rowsOf(delta) });
  placed.set(guid, target.clone());
}

function addMesh(guid: string, name: string, mesh: MeshJson, color: string, opacity = 1, flat = false) {
  const faces = mesh.f.length / 3;
  viewer.dispatch(
    pbDumpBytes(
      new Mesh({
        guid,
        name,
        vertices: mesh.v,
        faceVertices: mesh.f,
        faceSizes: new Array(faces).fill(3),
        attributes: {},
        vertexAttributeColumns: [],
        faceAttributeColumns: [],
        edgeAttributeColumns: [],
        edgeKeys: [],
        defaultVertexAttributes: {},
        defaultFaceAttributes: {},
        defaultEdgeAttributes: {},
      }),
    ),
  );
  material(guid, color, opacity, flat);
  placed.set(guid, new Matrix4()); // vertices are in the object's own frame
}

function material(guid: string, color: string, opacity = 1, flat = false) {
  command({
    dispatch: "material",
    type: "standard_material",
    guid: "mat:" + guid,
    geometry_guid: guid,
    color,
    metalness: 0.15,
    roughness: 0.55,
    emissive: "#000000",
    emissive_intensity: 0,
    flat_shading: flat,
    wireframe: false,
    transparent: opacity < 1,
    opacity,
  });
}

/** A path as a thin square tube. Not a polyline: the viewer picks lines within a metre
 * (three.js's default line threshold), so a path drawn as one would steal clicks meant
 * for the target it leads to. */
function tube(guid: string, points: number[], color: string, radius = 0.0025) {
  if (placed.has(guid)) remove(guid);
  const n = points.length / 3;
  if (n < 2) return;
  const v: number[] = [];
  const f: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1);
    const b = Math.min(n - 1, i + 1);
    let dx = points[3 * b] - points[3 * a];
    let dy = points[3 * b + 1] - points[3 * a + 1];
    let dz = points[3 * b + 2] - points[3 * a + 2];
    const len = Math.hypot(dx, dy, dz) || 1;
    [dx, dy, dz] = [dx / len, dy / len, dz / len];
    // u: perpendicular to the direction (via z, or x if the path runs along z); w = d × u.
    let [ux, uy, uz] = Math.abs(dz) < 0.9 ? [-dy, dx, 0] : [0, -dz, dy];
    const ul = Math.hypot(ux, uy, uz) || 1;
    [ux, uy, uz] = [ux / ul, uy / ul, uz / ul];
    const [wx, wy, wz] = [dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux];
    for (const [s, t] of [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ]) {
      v.push(
        points[3 * i] + radius * (s * ux + t * wx),
        points[3 * i + 1] + radius * (s * uy + t * wy),
        points[3 * i + 2] + radius * (s * uz + t * wz),
      );
    }
    if (i > 0) {
      for (let k = 0; k < 4; k++) {
        const p0 = 4 * (i - 1) + k;
        const p1 = 4 * (i - 1) + ((k + 1) % 4);
        f.push(p0, p1, p1 + 4, p0, p1 + 4, p0 + 4);
      }
    }
  }
  addMesh(guid, guid, { v, f }, color);
}

function remove(guid: string) {
  command({ dispatch: "handle_geometry", type: "remove", guid });
  placed.delete(guid);
}

/** Robot links are not for dragging: drop any pick on them (the viewer clears its pick
 * on Escape). */
function dropPick() {
  setTimeout(() => {
    const canvas = container.querySelector("canvas");
    canvas?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  }, 0);
}

function onViewerMessage(message: unknown): boolean {
  const m = message as Record<string, unknown>;
  if (location.hash === "#debug") console.log("viewer →", JSON.stringify(m).slice(0, 200));
  const guid = m.guid as string | undefined;
  if (!guid) return true;
  if (m.dispatch === "object_picked" && !draggable.has(guid)) {
    dropPick();
  } else if (m.dispatch === "object_transform") {
    const before = placed.get(guid);
    if (!before) return true;
    const now = fromRows(m.matrix as number[][]).multiply(before);
    placed.set(guid, now);
    if (draggable.has(guid)) {
      send({ type: "moved", guid, matrix: rowsOf(now) });
    }
  }
  return true;
}

// -- theme ----------------------------------------------------------------------------------

const dark = window.matchMedia("(prefers-color-scheme: dark)");
function applyTheme() {
  command({ dispatch: "theme", mode: dark.matches ? "dark" : "light" });
}
dark.addEventListener("change", applyTheme);
applyTheme();
command({ dispatch: "scene", type: "world_axis", show: false });

// -- websocket ------------------------------------------------------------------------------

let ws: WebSocket | null = null;
let wanted = new URLSearchParams(location.search).get("scenario") ?? "wall";
const connEl = document.querySelector<HTMLElement>("#conn")!;

function send(message: Record<string, unknown>) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  ws = new WebSocket(url);
  connEl.dataset.state = "connecting";
  connEl.textContent = "connecting";
  ws.onopen = () => {
    connEl.dataset.state = "open";
    connEl.textContent = "live";
    send({ type: "hello", scenario: wanted });
  };
  ws.onclose = () => {
    connEl.dataset.state = "closed";
    connEl.textContent = "offline";
    setStatus("failed", "No connection to the planning server. Is server.py running? Retrying…");
    setTimeout(connect, 1500);
  };
  ws.onmessage = (event) => handle(JSON.parse(event.data as string));
}

// -- server messages ----------------------------------------------------------------------

let path: number[] = [];
let drawnProgress = -1;
let lastPathDraw = 0;

function handle(msg: Record<string, unknown>) {
  switch (msg.type) {
    case "scenarios":
      renderScenarios(msg.items as { id: string; title: string }[]);
      break;
    case "scene":
      loadScene(msg as unknown as SceneMsg);
      break;
    case "pose":
      onPose(msg);
      break;
    case "body": {
      const guid = msg.guid as string;
      if (msg.pose) setPose(guid, from12(msg.pose as number[]));
      if (msg.color) {
        const body = current?.bodies.find((b) => b.guid === guid);
        material(guid, msg.color as string, body?.opacity ?? 1, body?.flat ?? false);
      }
      break;
    }
    case "path":
      path = msg.points as number[];
      drawnProgress = -1;
      tube("path:plan", path, "#8a939e", 0.0018);
      if (placed.has("path:done")) remove("path:done");
      break;
    case "status":
      setStatus(msg.state as string, msg.text as string, msg.metrics as Record<string, number>);
      break;
  }
}

function loadScene(scene: SceneMsg) {
  for (const guid of [...placed.keys()]) remove(guid);
  draggable.clear();
  current = scene;
  wanted = scene.id;
  history.replaceState(null, "", `?scenario=${scene.id}${location.hash}`);
  parts = scene.parts;
  joints = scene.joints;
  for (const part of parts) addMesh(part.guid, part.name, part.mesh, part.color);
  for (const body of scene.bodies) {
    addMesh(body.guid, body.name, body.mesh, body.color, body.opacity, body.flat);
    setPose(body.guid, from12(body.pose));
    if (body.draggable) draggable.add(body.guid);
  }
  const [px, py, pz] = scene.camera.position;
  const [tx, ty, tz] = scene.camera.target;
  command({ dispatch: "scene", type: "camera_position", x: px, y: py, z: pz });
  command({ dispatch: "scene", type: "camera_target", x: tx, y: ty, z: tz });
  path = [];
  renderScenarioPanel(scene);
  renderJoints(scene.joints);
  markScenario(scene.id);
}

function onPose(msg: Record<string, unknown>) {
  const frames = (msg.frames as number[][]).map(from12);
  for (const part of parts) setPose(part.guid, frames[part.link]);
  updateJoints(msg.q as number[], msg.qd as number[] | undefined);
  if (typeof msg.clearance === "number") updateClearance(msg.clearance);
  const progress = msg.progress as number | undefined;
  const now = performance.now();
  if (progress !== undefined && path.length && progress !== drawnProgress && now - lastPathDraw > 80) {
    tube("path:done", path.slice(0, (progress + 1) * 3), "#f2b705", 0.0028);
    drawnProgress = progress;
    lastPathDraw = now;
  }
}

// -- panel ------------------------------------------------------------------------------------

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const KEYS = ["a", "b", "c", "d", "e", "f", "g"];

function renderScenarios(items: { id: string; title: string }[]) {
  const nav = $("#scenarios");
  nav.replaceChildren(
    ...items.map((item, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.id = item.id;
      b.innerHTML = `<span class="key">${KEYS[i]})</span><span></span>`;
      b.lastElementChild!.textContent = item.title;
      b.addEventListener("click", () => send({ type: "select", id: item.id }));
      return b;
    }),
  );
  markScenario(wanted);
}
function markScenario(id: string) {
  for (const b of document.querySelectorAll<HTMLButtonElement>("#scenarios button")) {
    b.setAttribute("aria-current", String(b.dataset.id === id));
  }
}

function button(label: string, onClick: () => void, primary = false) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = primary ? "btn primary" : "btn";
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

function renderScenarioPanel(scene: SceneMsg) {
  $("#title").textContent = scene.title;
  $("#hint").textContent = scene.hint;
  $("#metrics").replaceChildren();
  const controls = $("#controls");
  controls.replaceChildren();
  if (scene.controls.obstacles) {
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = "Obstacle model";
    const seg = document.createElement("div");
    seg.className = "segmented";
    const names: Record<string, string> = { esdf: "Distance field", boxes: "Bounding boxes" };
    for (const mode of scene.controls.obstacles) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = names[mode] ?? mode;
      b.setAttribute("aria-pressed", String(mode === (scene.controls.mode ?? scene.controls.obstacles[0])));
      b.addEventListener("click", () => {
        for (const other of seg.querySelectorAll("button")) other.setAttribute("aria-pressed", String(other === b));
        send({ type: "action", name: "obstacles", value: mode });
      });
      seg.append(b);
    }
    controls.append(label, seg);
  }
  if (scene.controls.servo) {
    controls.append(servoPad());
    controls.append(button("Home", () => send({ type: "action", name: "home" }), true));
  } else {
    controls.append(button("Plan again", () => send({ type: "action", name: "replay" }), true));
  }
  controls.append(button("Reset scene", () => send({ type: "action", name: "reset" })));
  $("#clearance").hidden = !scene.controls.servo;
}

const METRICS: [string, string, (v: number) => string][] = [
  ["plan_ms", "Plan", (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0)) + " ms"],
  ["duration", "Motion", (v) => v.toFixed(2) + " s"],
  ["goals", "IK goals", (v) => String(v)],
  ["iterations", "RRT iters", (v) => (v === 0 ? "direct" : String(v))],
  ["waypoints", "Waypoints", (v) => String(v)],
];

function setStatus(state: string, text: string, metrics?: Record<string, number>) {
  const chip = $("#state");
  chip.dataset.state = state;
  chip.textContent = state;
  $("#status-text").textContent = text ?? "";
  if (metrics && Object.keys(metrics).length) {
    $("#metrics").replaceChildren(
      ...METRICS.filter(([k]) => typeof metrics[k] === "number").map(([k, label, fmt]) => {
        const d = document.createElement("div");
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = fmt(metrics[k]);
        d.append(dt, dd);
        return d;
      }),
    );
  }
}

interface JointRow {
  mark: HTMLElement;
  vel: HTMLElement;
  value: HTMLElement;
}
let jointRows: JointRow[] = [];

function renderJoints(j: Joints) {
  const host = $("#joints");
  jointRows = [];
  host.replaceChildren(
    ...j.names.map((name, i) => {
      const row = document.createElement("div");
      row.className = "joint";
      row.title = name;
      const zero = ((0 - j.lower[i]) / (j.upper[i] - j.lower[i])) * 100;
      row.innerHTML = `<span class="name">j${i + 1}</span>
        <div class="bars"><div class="pos-track"><div class="pos-zero" style="left:${zero}%"></div><div class="pos-mark"></div></div>
        <div class="vel-track"><div class="vel-bar"></div></div></div>
        <span class="value">–</span>`;
      jointRows.push({ mark: row.querySelector(".pos-mark")!, vel: row.querySelector(".vel-bar")!, value: row.querySelector(".value")! });
      return row;
    }),
  );
}

function updateJoints(q: number[], qd?: number[]) {
  if (!joints) return;
  q.forEach((v, i) => {
    const row = jointRows[i];
    if (!row) return;
    const t = (v - joints!.lower[i]) / (joints!.upper[i] - joints!.lower[i]);
    row.mark.style.left = `${Math.min(100, Math.max(0, t * 100))}%`;
    row.value.textContent = `${((v * 180) / Math.PI).toFixed(1)}°`;
    if (qd) {
      const s = Math.min(1, Math.abs(qd[i]) / joints!.vmax[i]);
      row.vel.style.width = `${s * 100}%`;
      row.vel.classList.toggle("hot", s > 0.95);
    }
  });
}

function updateClearance(c: number) {
  const bar = $("#clr-bar");
  const cm = c * 100;
  bar.style.width = `${Math.max(0, Math.min(1, c / 0.2)) * 100}%`;
  bar.classList.toggle("near", c > 0 && c < 0.03);
  bar.classList.toggle("contact", c <= 0);
  $("#clr-value").textContent = c <= 0 ? "contact" : `${cm.toFixed(1)} cm`;
}

// -- servo -------------------------------------------------------------------------------------

/** [label, key, twist index, sign] */
const JOGS: [string, string, number, number][] = [
  ["+X", "w", 0, 1],
  ["−X", "s", 0, -1],
  ["+Y", "a", 1, 1],
  ["−Y", "d", 1, -1],
  ["+Z", "r", 2, 1],
  ["−Z", "f", 2, -1],
  ["+Rx", "ArrowRight", 3, 1],
  ["−Rx", "ArrowLeft", 3, -1],
  ["+Ry", "ArrowUp", 4, 1],
  ["−Ry", "ArrowDown", 4, -1],
  ["+Rz", "q", 5, 1],
  ["−Rz", "e", 5, -1],
];
const held = new Set<string>();
const jogButtons = new Map<string, HTMLElement>();
let jogTimer: number | null = null;

function keyLabel(key: string) {
  return { ArrowRight: "→", ArrowLeft: "←", ArrowUp: "↑", ArrowDown: "↓" }[key] ?? key.toUpperCase();
}

function servoPad() {
  const pad = document.createElement("div");
  pad.className = "pad";
  jogButtons.clear();
  const groups = [
    ["Move", JOGS.slice(0, 6)],
    ["Rotate", JOGS.slice(6)],
  ] as const;
  for (const [title, jogs] of groups) {
    const g = document.createElement("div");
    g.className = "pad-group";
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = title;
    g.append(label);
    for (const [text, key] of jogs) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "jog";
      b.innerHTML = `<span></span><kbd>${keyLabel(key)}</kbd>`;
      b.firstElementChild!.textContent = text;
      b.addEventListener("pointerdown", (e) => {
        b.setPointerCapture(e.pointerId);
        press(key, true);
      });
      const release = () => press(key, false);
      b.addEventListener("pointerup", release);
      b.addEventListener("pointercancel", release);
      b.addEventListener("lostpointercapture", release);
      jogButtons.set(key, b);
      g.append(b);
    }
    pad.append(g);
  }
  return pad;
}

function twist(): number[] {
  const t = [0, 0, 0, 0, 0, 0];
  for (const [, key, axis, sign] of JOGS) if (held.has(key)) t[axis] += sign;
  return t;
}

function press(key: string, down: boolean) {
  if (down === held.has(key)) return;
  if (down) held.add(key);
  else held.delete(key);
  jogButtons.get(key)?.classList.toggle("held", down);
  send({ type: "jog", twist: twist() });
  if (held.size && jogTimer === null) {
    jogTimer = window.setInterval(() => send({ type: "jog", twist: twist() }), 50);
  } else if (!held.size && jogTimer !== null) {
    clearInterval(jogTimer);
    jogTimer = null;
  }
}

function jogKey(event: KeyboardEvent): string | null {
  if (!current?.controls.servo || event.metaKey || event.ctrlKey || event.altKey) return null;
  const target = event.target as HTMLElement | null;
  if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return null;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  return JOGS.some(([, k]) => k === key) ? key : null;
}
// Capture phase, and stop there: the viewer binds some of these keys itself (W/E/R gizmo
// modes, D theme, F screenshot, Q sidebar) on its own element, below the window.
window.addEventListener(
  "keydown",
  (e) => {
    const key = jogKey(e);
    if (!key) return;
    e.preventDefault();
    e.stopPropagation();
    if (!e.repeat) press(key, true);
  },
  true,
);
window.addEventListener(
  "keyup",
  (e) => {
    const key = jogKey(e);
    if (!key) return;
    e.stopPropagation();
    press(key, false);
  },
  true,
);
window.addEventListener("blur", () => {
  for (const key of [...held]) press(key, false);
});

connect();

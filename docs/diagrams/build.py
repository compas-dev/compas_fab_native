"""Build the native backend's explanatory diagrams.

Each diagram is written twice next to this file: as a standalone HTML page
(``<slug>.html``, for viewing on its own) and as an SVG (``<slug>.svg``, embedded in
``docs/how-it-works.md``). Edit the diagram functions below and run::

    python docs/diagrams/build.py

Visual system: the editorial style of diagram-design
(https://github.com/cathrynlavery/diagram-design, default skin): white-smoke paper,
jet-black ink, blue-slate muted strokes, one atomic-tangerine accent per diagram,
orthogonal connectors with rounded elbows, masked arrow labels, everything on a 4px grid.
"""

import html
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

PAPER = "#f5f5f5"
INK = "#2d3142"
MUTED = "#4f5d75"
SOFT = "#7a8399"
ACCENT = "#eb6c36"
ACCENT_TINT = "rgba(235,108,54,0.08)"
SANS = "'Geist', system-ui, sans-serif"
MONO = "'Geist Mono', ui-monospace, monospace"
SERIF = "'Instrument Serif', serif"
FONTS = "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap"

KINDS = {
    "focal": (ACCENT_TINT, ACCENT, ""),
    "step": ("#ffffff", INK, ""),
    "store": ("rgba(45,49,66,0.05)", MUTED, ""),
    "input": ("rgba(79,93,117,0.10)", SOFT, ""),
    "optional": ("rgba(45,49,66,0.02)", "rgba(45,49,66,0.40)", ' stroke-dasharray="4,3"'),
}


def esc(s):
    return html.escape(s, quote=True)


def text_lines(cx, y, lines, size, weight, family, fill, gap):
    out = []
    for k, line in enumerate(lines):
        out.append(f'<text x="{cx}" y="{y + k * gap}" fill="{fill}" font-size="{size}" font-weight="{weight}" font-family="{family}" text-anchor="middle">{esc(line)}</text>')
    return "\n".join(out)


def node(x, y, w, h, name, sub=None, tag=None, kind="step"):
    fill, stroke, dash = KINDS[kind]
    names = name.split("\n")
    subs = sub.split("\n") if sub else []
    block = len(names) * 16 + len(subs) * 12
    top = y + h / 2 - block / 2 + 12 + (6 if tag else 0)
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>',
    ]
    if tag:
        tw = len(tag) * 6 + 12
        out.append(f'<rect x="{x + 8}" y="{y + 6}" width="{tw}" height="12" rx="2" fill="none" stroke="{stroke}" stroke-opacity="0.4" stroke-width="0.8"/>')
        out.append(
            f'<text x="{x + 8 + tw / 2}" y="{y + 15}" fill="{stroke}" fill-opacity="0.8" font-size="7" font-family="{MONO}" '
            f'text-anchor="middle" letter-spacing="0.08em">{esc(tag)}</text>'
        )
    cx = x + w / 2
    out.append(text_lines(cx, top, names, 12, 600, SANS, INK, 16))
    if subs:
        out.append(text_lines(cx, top + len(names) * 16, subs, 9, 400, MONO, MUTED, 12))
    return "\n".join(out)


def oval(x, y, w, h, name, sub=None):
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{PAPER}"/>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="rgba(79,93,117,0.10)" stroke="{SOFT}" stroke-width="1"/>',
    ]
    cy = y + h / 2 + (0 if sub else 4)
    out.append(text_lines(x + w / 2, cy - (4 if sub else 0), [name], 12, 600, SANS, INK, 16))
    if sub:
        out.append(text_lines(x + w / 2, cy + 10, [sub], 9, 400, MONO, MUTED, 12))
    return "\n".join(out)


def diamond(cx, cy, hw, hh, name, focal=False):
    stroke = ACCENT if focal else INK
    fill = ACCENT_TINT if focal else "#ffffff"
    pts = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
    lines = name.split("\n")
    top = cy - (len(lines) - 1) * 7 + 4
    return "\n".join(
        [
            f'<polygon points="{pts}" fill="{PAPER}"/>',
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
            text_lines(cx, top, lines, 11, 600, SANS, INK, 14),
        ]
    )


def label(text, cx, line_y, side=None):
    """Arrow label above a horizontal segment at ``line_y`` (8px gap), or beside a vertical
    one (``side="right"``/``"left"`` of x=``cx``, centred on ``line_y``)."""
    w = int(len(text) * 5.6 + 12) // 4 * 4 + 4
    if side is None:
        x, y = cx - w / 2, line_y - 20
    elif side == "right":
        x, y = cx + 8, line_y - 6
    else:
        x, y = cx - 8 - w, line_y - 6
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="12" rx="2" fill="{PAPER}"/>\n'
        f'<text x="{x + w / 2}" y="{y + 9}" fill="{SOFT}" font-size="8" font-family="{MONO}" text-anchor="middle" '
        f'letter-spacing="0.06em">{esc(text)}</text>'
    )


def arrow(d, kind="default"):
    stroke, marker, dash, width = {
        "default": (MUTED, "arrow", "", 1.2),
        "accent": (ACCENT, "arrow-accent", "", 1.2),
        "dashed": (MUTED, "arrow", ' stroke-dasharray="4,3"', 1),
        "plain": (MUTED, None, ' stroke-dasharray="4,3"', 1),
    }[kind]
    end = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}{end}/>'


def zone(x, y, w, h, title):
    lw = len(title) * 7 + 16
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>\n'
        f'<rect x="{x + 16}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>\n'
        f'<text x="{x + 16 + lw / 2}" y="{y + 13}" fill="rgba(45,49,66,0.55)" font-size="7" font-family="{MONO}" '
        f'text-anchor="middle" letter-spacing="0.14em">{esc(title)}</text>'
    )


def legend(items, y, width):
    out = [
        f'<line x1="24" y1="{y - 8}" x2="{width - 24}" y2="{y - 8}" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>',
        f'<text x="24" y="{y + 12}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">LEGEND</text>',
    ]
    x = 104
    for kind, name in items:
        if kind in KINDS:
            fill, stroke, dash = KINDS[kind]
            out.append(f'<rect x="{x}" y="{y + 4}" width="20" height="12" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>')
        elif kind == "diamond":
            out.append(f'<polygon points="{x + 10},{y + 2} {x + 20},{y + 10} {x + 10},{y + 18} {x},{y + 10}" fill="#ffffff" stroke="{INK}" stroke-width="1"/>')
        elif kind == "oval":
            out.append(f'<rect x="{x}" y="{y + 4}" width="20" height="12" rx="6" fill="rgba(79,93,117,0.10)" stroke="{SOFT}" stroke-width="1"/>')
        elif kind == "dashed-line":
            out.append(f'<line x1="{x}" y1="{y + 10}" x2="{x + 20}" y2="{y + 10}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4,3"/>')
        out.append(f'<text x="{x + 28}" y="{y + 13}" fill="{MUTED}" font-size="9" font-family="{SANS}">{esc(name)}</text>')
        x += 28 + int(len(name) * 5.4) + 24
    return "\n".join(out)


DEFS = f"""<defs>
<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>
<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>
</defs>"""


def svg(slug, title, desc, width, height, body):
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">\n'
        f'<title id="{slug}-title">{esc(title)}</title>\n<desc id="{slug}-desc">{esc(desc)}</desc>\n{DEFS}\n'
        f'<rect width="100%" height="100%" fill="{PAPER}"/>\n{body}\n</svg>'
    )


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="{fonts}" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: {sans}; background: {paper}; color: {ink}; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 3rem 2rem; }}
.frame {{ max-width: 1100px; width: 100%; }}
.eyebrow {{ font-family: {mono}; font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: {muted}; margin-bottom: 0.5rem; }}
h1 {{ font-family: {serif}; font-size: 2rem; font-weight: 400; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.5rem; }}
svg {{ width: 100%; display: block; }}
</style>
</head>
<body>
<div class="frame">
<p class="eyebrow">{eyebrow} · compas_fab native backend</p>
<h1>{title}</h1>
{svg}
</div>
</body>
</html>
"""


def write(slug, eyebrow, title, desc, width, height, body):
    figure = svg(slug, title, desc, width, height, body)
    page = PAGE.format(title=esc(title), eyebrow=esc(eyebrow), svg=figure, fonts=FONTS, sans=SANS, mono=MONO, serif=SERIF, paper=PAPER, ink=INK, muted=MUTED)
    (HERE / f"{slug}.html").write_text(page)
    # Standalone SVG: fonts imported (XML-escaped ampersands), XML declaration.
    style = "<style>@import url('" + FONTS.replace("&", "&amp;") + "');</style>"
    standalone = figure.replace("<defs>", "<defs>\n" + style, 1)
    # Explicit size, so an <img> of the file has the diagram's natural size, not 300x150.
    standalone = standalone.replace("<svg ", f'<svg width="{width}" height="{height}" ', 1)
    (HERE / f"{slug}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + standalone + "\n")


# ---------------------------------------------------------------------------------------
# 1. Architecture


def architecture():
    b = [
        zone(20, 56, 960, 152, "PYTHON · COMPAS_FAB"),
        zone(20, 248, 960, 220, "RUST · COMPAS_FAB_NATIVE"),
        # arrows first
        arrow("M 192,132 H 272"),
        label("API", 232, 132),
        arrow("M 488,132 H 552"),
        label("DESCRIBE", 520, 132),
        arrow("M 744,132 H 792", "dashed"),
        arrow("M 384,164 V 292"),
        label("JSON + CALLS", 384, 228, side="right"),
        arrow("M 488,324 H 552"),
        label("PLAN", 520, 324),
        arrow("M 384,356 V 404 Q 384,412 392,412 H 552"),
        label("VALIDATE", 472, 412),
        arrow("M 744,324 H 792", "dashed"),
        label("LAYOUT", 768, 324),
        arrow("M 648,388 V 360"),
        # nodes
        node(36, 100, 156, 64, "Your script", "or Grasshopper", kind="input"),
        node(272, 100, 216, 64, "NativePlanner", "compas_fab_native/planner.py", tag="BACKEND", kind="focal"),
        node(552, 100, 192, 64, "Robot model", "compas_fab_native/model.py", tag="PY"),
        node(792, 100, 168, 64, "Sphere cache", "memory + disk", kind="store"),
        node(272, 292, 216, 64, "NativeCore", "crates/motion_py", tag="PYO3"),
        node(552, 292, 192, 64, "motion_core", "kinematics · checks · plans", tag="RUST"),
        node(552, 388, 192, 64, "motion_exact", "exact meshes (parry3d)", tag="RUST"),
        node(792, 292, 168, 64, "motion_gpu", "benchmarks only", tag="WGSL", kind="optional"),
        legend([("focal", "entry point"), ("step", "module / crate"), ("input", "caller"), ("store", "cache"), ("optional", "not on the planning path")], 496, 1000),
    ]
    write(
        "architecture",
        "Architecture",
        "From your script to the Rust core",
        "Architecture of the native backend: NativePlanner and the robot model in Python call the NativeCore "
        "extension, which plans with motion_core and validates with motion_exact.",
        1000,
        532,
        "\n".join(b),
    )


# ---------------------------------------------------------------------------------------
# 2. Setting up a cell


def set_robot_cell():
    b = [
        arrow("M 168,224 H 208"),
        arrow("M 352,208 H 364 Q 372,208 372,200 V 120 Q 372,112 380,112 H 392"),
        arrow("M 352,224 H 392"),
        arrow("M 352,240 H 364 Q 372,240 372,248 V 360 Q 372,368 380,368 H 392"),
        arrow("M 568,112 H 584 Q 592,112 592,120 V 152 Q 592,160 600,160 H 616"),
        arrow("M 568,224 H 584 Q 592,224 592,216 V 184 Q 592,176 600,176 H 616"),
        arrow("M 696,200 V 240", "plain"),
        label("BY MESH HASH", 696, 220, side="right"),
        arrow("M 568,368 H 616"),
        arrow("M 776,168 H 792 Q 800,168 800,176 V 240 Q 800,248 808,248 H 824", "accent"),
        arrow("M 776,368 H 792 Q 800,368 800,360 V 304 Q 800,296 808,296 H 824", "accent"),
        node(24, 192, 144, 64, "RobotCell", "+ RobotCellState", kind="input"),
        node(208, 192, 144, 64, "Split by group", "planning_links"),
        node(392, 80, 176, 64, "Moving links", "the group's tree"),
        node(392, 192, 176, 64, "Carried objects", "tool · workpiece"),
        node(392, 312, 176, 112, "Everything else", "static links\nother robots + tools\nparked tools\nfree rigid bodies"),
        node(616, 136, 160, 64, "Sphere fit", "per link, scored"),
        node(616, 240, 160, 56, "Sphere cache", "memory + disk", kind="store"),
        node(616, 336, 160, 64, "Obstacles", "boxes or ESDF"),
        node(824, 224, 152, 96, "NativeCore", "robot + scene\n+ exact meshes", tag="CORE", kind="focal"),
        legend([("input", "input"), ("step", "step"), ("store", "cache"), ("focal", "the core, ready to plan")], 468, 1000),
    ]
    write(
        "set-robot-cell",
        "Data flow",
        "What set_robot_cell builds",
        "The cell is split by planning group: moving links and carried objects become spheres (cached by mesh), "
        "everything else becomes obstacles, and both go to the core with the exact meshes.",
        1000,
        504,
        "\n".join(b),
    )


# ---------------------------------------------------------------------------------------
# 3. Collision model


def collision_model():
    rows = [
        ("01", "Broad phase", "a bounding sphere per link", "every check · skips most work", False),
        ("02", "Spheres", "link spheres vs boxes, ESDF, other links", "RRT · IK residuals · clearances", False),
        ("03", "Convex hulls", "5 mm margin · opt-in (exact_links)", "poorly fitted links, in the loop", False),
        ("04", "Exact meshes", "parry3d, the final word", "goals · every planned path", True),
    ]
    b = [
        arrow("M 64,48 V 344"),
        f'<text x="56" y="60" fill="{MUTED}" font-size="8" font-family="{MONO}" text-anchor="end" letter-spacing="0.1em">FAST</text>',
        f'<text x="56" y="340" fill="{MUTED}" font-size="8" font-family="{MONO}" text-anchor="end" letter-spacing="0.1em">EXACT</text>',
    ]
    y = 40
    for tag, name, sub, note, focal in rows:
        fill, stroke = (ACCENT_TINT, ACCENT) if focal else ("#ffffff", "rgba(45,49,66,0.30)")
        b.append(f'<rect x="96" y="{y}" width="864" height="72" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        b.append(f'<text x="120" y="{y + 40}" fill="{MUTED}" font-size="9" font-family="{MONO}" letter-spacing="0.14em">{tag}</text>')
        b.append(f'<text x="168" y="{y + 34}" fill="{INK}" font-size="16" font-weight="600" font-family="{SANS}">{esc(name)}</text>')
        b.append(f'<text x="168" y="{y + 52}" fill="{MUTED}" font-size="9" font-family="{MONO}">{esc(sub)}</text>')
        b.append(f'<text x="936" y="{y + 40}" fill="{SOFT}" font-size="9" font-family="{MONO}" text-anchor="end">{esc(note)}</text>')
        y += 80
    b.append(legend([("focal", "decides what is handed out")], 384, 1000))
    write(
        "collision-model",
        "Layer stack",
        "Four collision checks, fast to exact",
        "The four collision checks of the native backend, from bounding spheres through link spheres and optional convex hulls to exact meshes, and where each is used.",
        1000,
        420,
        "\n".join(b),
    )


# ---------------------------------------------------------------------------------------
# 4. plan_motion


def plan_motion():
    b = [
        arrow("M 500,64 V 88"),
        arrow("M 500,144 V 168"),
        arrow("M 500,240 V 280"),
        label("NO", 500, 260, side="right"),
        arrow("M 404,204 H 248 Q 240,204 240,212 V 480 Q 240,488 248,488 H 404"),
        label("YES", 324, 204),
        arrow("M 500,336 V 376"),
        arrow("M 500,424 V 452"),
        arrow("M 596,488 H 822 Q 830,488 830,480 V 336", "accent"),
        label("YES", 712, 488),
        arrow("M 720,308 H 620", "accent"),
        arrow("M 500,524 V 568"),
        label("NO", 500, 546, side="right"),
        arrow("M 500,624 V 664"),
        oval(400, 24, 200, 40, "Target", "configuration · frame · point-axis"),
        node(352, 88, 296, 56, "Goal configurations", "every IK solution, minus those touching meshes"),
        diamond(500, 204, 96, 36, "Straight motion\nfree?"),
        node(380, 280, 240, 56, "RRT-Connect", "from the start and every goal"),
        node(380, 376, 240, 48, "Shortcut + B-spline"),
        diamond(500, 488, 96, 36, "Touches exact\nmeshes?", focal=True),
        node(720, 280, 220, 56, "Inflate spheres", "+5 … 30 mm, new seed"),
        node(380, 568, 240, 56, "Time it", "TOPP-RA, blended corners"),
        oval(400, 664, 200, 40, "Trajectory"),
        legend([("oval", "start / end"), ("step", "step"), ("diamond", "decision")], 748, 1000),
    ]
    write(
        "plan-motion",
        "Flowchart",
        "How plan_motion finds a path",
        "plan_motion turns the target into goal configurations, tries the straight motion, otherwise runs RRT-Connect and smooths the "
        "path, re-plans on inflated spheres if the exact meshes are touched, and times the result.",
        1000,
        784,
        "\n".join(b),
    )


# ---------------------------------------------------------------------------------------
# 5. plan_cartesian_motion


def plan_cartesian():
    b = [
        arrow("M 500,64 V 88"),
        arrow("M 500,144 V 176"),
        arrow("M 404,212 H 248 Q 240,212 240,220 V 288"),
        label("YES", 324, 212),
        arrow("M 596,212 H 752 Q 760,212 760,220 V 288", "accent"),
        label("NO", 676, 212),
        arrow("M 240,344 V 400 Q 240,408 248,408 H 380"),
        arrow("M 760,344 V 400 Q 760,408 752,408 H 620", "accent"),
        arrow("M 500,448 V 496"),
        arrow("M 596,532 H 700"),
        label("YES", 648, 532),
        arrow("M 800,504 V 440 Q 800,432 792,432 H 620"),
        arrow("M 500,568 V 608"),
        label("NO", 500, 588, side="right"),
        oval(400, 24, 200, 40, "Waypoints", "frames · point-axis"),
        node(352, 88, 296, 56, "Fill in frames", "every max_step · max_step_angle"),
        diamond(500, 212, 96, 36, "Closed-form\nIK?"),
        node(104, 288, 272, 56, "Layered graph", "all IK branches × axis samples"),
        node(624, 288, 272, 56, "Beam search", "tracks follow, branch when they die", kind="focal"),
        node(380, 392, 240, 56, "Cheapest path", "dynamic programming"),
        diamond(500, 532, 96, 36, "Edges or nodes\ncollide?"),
        node(700, 504, 200, 56, "Block them", "spheres, then meshes"),
        oval(400, 608, 200, 40, "Trajectory"),
        legend([("oval", "start / end"), ("step", "step"), ("diamond", "decision"), ("focal", "numerical IK path")], 692, 1000),
    ]
    write(
        "plan-cartesian",
        "Flowchart",
        "How plan_cartesian_motion follows a path",
        "plan_cartesian_motion fills in frames between waypoints, builds a layered graph of closed-form IK solutions or runs a beam search "
        "with numerical IK, picks the cheapest path and blocks colliding edges until the path is free.",
        1000,
        728,
        "\n".join(b),
    )


# ---------------------------------------------------------------------------------------
# 6. Numerical IK


def numeric_ik():
    b = [
        arrow("M 500,64 V 88"),
        arrow("M 500,144 V 184"),
        arrow("M 500,240 V 280"),
        arrow("M 500,336 V 360"),
        arrow("M 596,396 H 692 Q 700,396 700,388 V 316 Q 700,308 692,308 H 620"),
        label("NO · λ×10", 644, 396),
        arrow("M 500,432 V 480"),
        label("YES · λ÷3", 500, 456, side="right"),
        arrow("M 404,516 H 308 Q 300,516 300,508 V 220 Q 300,212 308,212 H 380", "accent"),
        label("NO", 352, 516),
        arrow("M 500,552 V 592"),
        label("YES", 500, 572, side="right"),
        arrow("M 500,648 V 680"),
        oval(400, 24, 200, 40, "Seed", "yours first, then random"),
        node(352, 88, 296, 56, "Carry", "move only prismatic axes (if any)"),
        node(380, 184, 240, 56, "Residuals", "pose · rest pull · collisions", kind="focal"),
        node(380, 280, 240, 56, "Levenberg–Marquardt step", "clamped to joint limits"),
        diamond(500, 396, 96, 36, "Cost lower?"),
        diamond(500, 516, 96, 36, "Converged,\nno contact?"),
        node(380, 592, 240, 56, "Accept", "sphere check · distinct"),
        oval(400, 680, 200, 40, "Solutions"),
        legend([("oval", "start / end"), ("step", "step"), ("diamond", "decision"), ("focal", "what the solve minimises")], 760, 1000),
    ]
    write(
        "numeric-ik",
        "Flowchart",
        "How numerical IK solves from one seed",
        "Numerical IK from one seed: an optional carry stage for prismatic axes, then Levenberg-Marquardt steps on pose, rest-posture and "
        "collision residuals until the pose converges without contact, then acceptance by the sphere checker.",
        1000,
        796,
        "\n".join(b),
    )


if __name__ == "__main__":
    architecture()
    set_robot_cell()
    collision_model()
    plan_motion()
    plan_cartesian()
    numeric_ik()
    print("wrote", ", ".join(sorted(p.name for p in HERE.glob("*.svg"))))

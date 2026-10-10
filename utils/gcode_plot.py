"""
G-code toolpath parser and plotter (pure Python, no LLM involved).

How it works
------------
1. `extract_gcode_blocks()` finds ```gcode / ```nc fenced blocks in the bot's reply.
2. `parse_gcode()` walks the program line by line like a (very) simple CNC
   controller would: it keeps track of the current position, the active motion
   mode (G00/G01/G02/G03 are "modal" - they stay active until changed) and the
   distance mode (G90 absolute / G91 incremental).
   Every move becomes a `Segment` with a list of XY points to draw.
3. `plot_toolpath()` draws those segments with matplotlib:
   rapids (G00) dashed, feeds (G01) solid, arcs (G02/G03) as smooth curves.
4. `format_stats()` builds a one-line summary (distances and X/Y extents).

Bad or unsupported lines are skipped and reported in `ParseResult.warnings`;
the parser never raises on user input.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # headless backend - works on servers / Streamlit Cloud
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# A fenced code block whose language tag is gcode, g-code or nc.
BLOCK_RE = re.compile(r"```[ \t]*(?:gcode|g-code|nc)[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)

# One G-code "word": a letter followed by a number, e.g. X-12.5, G01, F300.
WORD_RE = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")

# Non-motion G-codes that we understand and can safely ignore for plotting.
IGNORED_G = {
    4, 17, 18, 19, 20, 21, 40, 41, 42, 43, 49, 53, 54, 55, 56, 57, 58, 59,
    80, 94, 95, 98, 99,
}
CANNED_CYCLES = {73, 74, 76, 81, 82, 83, 84, 85, 86, 87, 88, 89}

ARC_STEP_DEG = 3.0  # angular resolution used when drawing arcs
TOL = 1e-6


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One tool move. kind is 'rapid', 'feed' or 'arc'."""

    kind: str
    points: list[tuple[float, float]]
    length: float  # 3D length in mm (includes Z travel)
    line_no: int


@dataclass
class ParseResult:
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    start: tuple[float, float] = (0.0, 0.0)
    line_count: int = 0  # non-empty program lines (comments-only lines excluded)
    inch_mode: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_gcode_blocks(markdown_text: str) -> list[str]:
    """Return the contents of every ```gcode / ```nc block in a Markdown string."""
    return [b.strip() for b in BLOCK_RE.findall(markdown_text or "") if b.strip()]


def strip_comments(line: str) -> str:
    """Remove (parenthesis comments) and everything after ';'."""
    line = re.sub(r"\([^)]*\)", " ", line)
    return line.split(";", 1)[0]


def _arc_points(start, end, center, clockwise):
    """Return (points, sweep_angle_rad, radius) for an XY arc."""
    radius = math.hypot(start[0] - center[0], start[1] - center[1])
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    sweep = a1 - a0
    if clockwise:
        if sweep >= -TOL:
            sweep -= 2 * math.pi  # CW sweeps are negative
    else:
        if sweep <= TOL:
            sweep += 2 * math.pi  # CCW sweeps are positive
    # start == end -> full circle (sweep becomes +/- 2*pi above)

    steps = max(2, int(abs(math.degrees(sweep)) / ARC_STEP_DEG))
    pts = [
        (center[0] + radius * math.cos(a0 + sweep * i / steps),
         center[1] + radius * math.sin(a0 + sweep * i / steps))
        for i in range(steps + 1)
    ]
    pts[-1] = (end[0], end[1])  # snap exactly onto the programmed end point
    return pts, sweep, radius


def _center_from_r(start, end, r, clockwise):
    """Find the arc centre from a radius word R (Fanuc convention).

    Positive R = arc of 180 deg or less, negative R = arc greater than 180 deg.
    Returns None if the radius is too small to reach the end point.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    chord = math.hypot(dx, dy)
    if chord < TOL or chord > 2 * abs(r) + 1e-4:
        return None
    h = math.sqrt(max(r * r - (chord / 2) ** 2, 0.0))
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    left = (-dy / chord, dx / chord)  # unit vector to the left of the chord
    # CCW short arc -> centre on the left; CW short arc -> centre on the right.
    sign = 1 if not clockwise else -1
    if r < 0:
        sign = -sign
    return (mid[0] + sign * h * left[0], mid[1] + sign * h * left[1])


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_gcode(program: str) -> ParseResult:
    """Simulate a G-code program and return its XY toolpath segments."""
    result = ParseResult()
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}  # assume the program starts at the origin
    motion = None        # active motion G-code (0, 1, 2, 3) - modal
    absolute = True      # G90 by default
    started = False

    for line_no, raw in enumerate(program.splitlines(), start=1):
        line = strip_comments(raw).strip().upper()
        if not line or line == "%":
            continue
        result.line_count += 1

        # Program number lines such as "O1001" carry no motion.
        if re.fullmatch(r"O\d+", line.replace(" ", "")):
            continue

        words = WORD_RE.findall(line)
        leftover = WORD_RE.sub("", line).strip()
        if leftover:
            result.warnings.append(f"Line {line_no}: could not understand '{leftover}' - skipped.")
            continue

        g_codes, values = [], {}
        for letter, num in words:
            if letter == "G":
                g_codes.append(float(num))
            else:
                values[letter] = float(num)

        skip_motion = False
        for g in g_codes:
            gi = int(g) if g.is_integer() else None
            if gi in (0, 1, 2, 3):
                motion = gi
            elif gi == 90:
                absolute = True
            elif gi == 91:
                absolute = False
            elif gi == 20:
                result.inch_mode = True
            elif gi == 28:
                skip_motion = True  # reference return - not drawn
            elif gi in CANNED_CYCLES:
                result.warnings.append(f"Line {line_no}: canned cycle G{gi} is not plotted.")
                skip_motion = True
            elif gi in IGNORED_G:
                pass
            else:
                result.warnings.append(f"Line {line_no}: unsupported code G{g:g} ignored.")

        has_axis = any(a in values for a in ("X", "Y", "Z"))
        if skip_motion or not has_axis:
            continue
        if motion is None:
            result.warnings.append(f"Line {line_no}: axis move with no active G00/G01/G02/G03 - skipped.")
            continue

        # Work out the target position.
        target = dict(pos)
        for axis in ("X", "Y", "Z"):
            if axis in values:
                target[axis] = values[axis] if absolute else pos[axis] + values[axis]

        if not started:
            result.start = (pos["X"], pos["Y"])
            started = True

        start_xy, end_xy = (pos["X"], pos["Y"]), (target["X"], target["Y"])
        dz = target["Z"] - pos["Z"]

        if motion in (0, 1):
            length = math.dist((pos["X"], pos["Y"], pos["Z"]), (target["X"], target["Y"], target["Z"]))
            result.segments.append(Segment("rapid" if motion == 0 else "feed",
                                           [start_xy, end_xy], length, line_no))
        else:
            clockwise = motion == 2
            center = None
            if "I" in values or "J" in values:
                # I and J are always incremental from the arc start point.
                center = (start_xy[0] + values.get("I", 0.0), start_xy[1] + values.get("J", 0.0))
            elif "R" in values:
                center = _center_from_r(start_xy, end_xy, values["R"], clockwise)
                if center is None:
                    result.warnings.append(f"Line {line_no}: R value too small for this arc - drawn as a straight line.")
            else:
                result.warnings.append(f"Line {line_no}: arc without I/J or R - drawn as a straight line.")

            if center is None:
                length = math.dist((*start_xy, pos["Z"]), (*end_xy, target["Z"]))
                result.segments.append(Segment("feed", [start_xy, end_xy], length, line_no))
            else:
                pts, sweep, radius = _arc_points(start_xy, end_xy, center, clockwise)
                end_radius = math.dist(end_xy, center)
                if abs(end_radius - radius) > 0.01:
                    result.warnings.append(
                        f"Line {line_no}: arc end point is not on the circle "
                        f"(start r={radius:.3f}, end r={end_radius:.3f})."
                    )
                arc_len = math.hypot(abs(sweep) * radius, dz)  # helix-safe length
                result.segments.append(Segment("arc", pts, arc_len, line_no))

        pos = target

    if result.inch_mode:
        result.warnings.append("Program uses G20 (inch); axes are labelled mm but values are inches.")
    return result


# ---------------------------------------------------------------------------
# Statistics and plotting
# ---------------------------------------------------------------------------

def compute_stats(result: ParseResult) -> dict:
    rapid = sum(s.length for s in result.segments if s.kind == "rapid")
    cutting = sum(s.length for s in result.segments if s.kind != "rapid")
    xs = [p[0] for s in result.segments for p in s.points] or [0.0]
    ys = [p[1] for s in result.segments for p in s.points] or [0.0]
    return {
        "lines": result.line_count,
        "rapid_mm": rapid,
        "cutting_mm": cutting,
        "x_range": (min(xs), max(xs)),
        "y_range": (min(ys), max(ys)),
    }


def format_stats(result: ParseResult) -> str:
    s = compute_stats(result)
    return (
        f"**Lines:** {s['lines']} · **Rapid:** {s['rapid_mm']:.1f} mm · "
        f"**Cutting:** {s['cutting_mm']:.1f} mm · "
        f"**X:** {s['x_range'][0]:.1f} → {s['x_range'][1]:.1f} mm · "
        f"**Y:** {s['y_range'][0]:.1f} → {s['y_range'][1]:.1f} mm"
    )


def plot_toolpath(result: ParseResult, title: str = "XY toolpath preview"):
    """Return a matplotlib Figure of the toolpath (caller should close it)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    styles = {
        "rapid": dict(color="#d9822b", linestyle="--", linewidth=1.2),
        "feed": dict(color="#1f5f9e", linestyle="-", linewidth=2.0),
        "arc": dict(color="#2a9d8f", linestyle="-", linewidth=2.0),
    }
    labels = {"rapid": "G00 rapid", "feed": "G01 feed", "arc": "G02/G03 arc"}
    used = set()

    for seg in result.segments:
        xs, ys = zip(*seg.points)
        label = labels[seg.kind] if seg.kind not in used else None
        used.add(seg.kind)
        ax.plot(xs, ys, label=label, **styles[seg.kind])

    sx, sy = result.start
    ax.plot(sx, sy, marker="o", color="#c0392b", markersize=8, label="Start", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

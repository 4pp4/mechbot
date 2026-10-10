"""Tests for the G-code parser. Run with:  pytest -q"""

import math
import os
import sys

import pytest

# Make "utils" importable when pytest is run from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gcode_plot import (  # noqa: E402
    compute_stats,
    extract_gcode_blocks,
    parse_gcode,
    plot_toolpath,
)


def test_straight_lines_square():
    program = """
    G21 G90
    G00 X0 Y0
    G01 X40 Y0 F300
    X40 Y40
    X0 Y40
    X0 Y0
    M30
    """
    result = parse_gcode(program)
    feeds = [s for s in result.segments if s.kind == "feed"]
    assert len(feeds) == 4  # G01 is modal, so the bare "X.. Y.." lines are feeds too
    stats = compute_stats(result)
    assert stats["cutting_mm"] == pytest.approx(160.0)
    assert stats["x_range"] == (0.0, 40.0)
    assert stats["y_range"] == (0.0, 40.0)
    assert result.warnings == []


def test_rapid_distance_includes_z():
    result = parse_gcode("G90\nG00 Z5\nG00 X3 Y4\n")
    stats = compute_stats(result)
    assert stats["rapid_mm"] == pytest.approx(10.0)  # 5 (Z) + 5 (3-4-5 triangle)
    assert stats["cutting_mm"] == 0.0


def test_ccw_quarter_arc_with_ij():
    # Quarter circle, radius 10, centre at origin, from (10,0) to (0,10).
    result = parse_gcode("G90\nG00 X10 Y0\nG03 X0 Y10 I-10 J0\n")
    arc = result.segments[-1]
    assert arc.kind == "arc"
    assert arc.length == pytest.approx(math.pi * 10 / 2, rel=1e-6)
    # Every point should lie on the circle and the midpoint should be at 45 degrees.
    for x, y in arc.points:
        assert math.hypot(x, y) == pytest.approx(10.0, abs=1e-6)
    mid_x, mid_y = arc.points[len(arc.points) // 2]
    assert mid_x > 0 and mid_y > 0
    assert result.warnings == []


def test_cw_arc_with_r_word():
    # Same end points but clockwise with R: centre must be at (10,10) instead.
    result = parse_gcode("G90\nG00 X10 Y0\nG02 X0 Y10 R10\n")
    arc = result.segments[-1]
    assert arc.kind == "arc"
    assert arc.length == pytest.approx(math.pi * 10 / 2, rel=1e-6)
    for x, y in arc.points:
        assert math.hypot(x - 10, y - 10) == pytest.approx(10.0, abs=1e-6)


def test_full_circle():
    result = parse_gcode("G90\nG00 X15 Y0\nG02 X15 Y0 I-15 J0\n")
    assert result.segments[-1].length == pytest.approx(2 * math.pi * 15, rel=1e-6)


def test_g91_incremental_moves():
    program = "G91\nG01 X10 F200\nY10\nX-10\nY-10\n"
    result = parse_gcode(program)
    ends = [s.points[-1] for s in result.segments]
    assert ends == [(10, 0), (10, 10), (0, 10), (0, 0)]
    assert compute_stats(result)["cutting_mm"] == pytest.approx(40.0)


def test_switch_back_to_g90():
    result = parse_gcode("G91\nG01 X5 F100\nG90\nX20\n")
    assert result.segments[-1].points[-1] == (20, 0)


def test_comments_are_ignored():
    program = """
    (SQUARE CONTOUR - comment line)
    G21 G90 ; metric absolute
    G00 X0 Y0 (go to start)
    G01 X25 F300 ; cut along X
    ; a full semicolon comment line
    """
    result = parse_gcode(program)
    assert len(result.segments) == 2
    assert result.segments[-1].points[-1] == (25, 0)
    assert result.line_count == 3  # comment-only lines are not counted
    assert result.warnings == []


def test_malformed_lines_warn_but_do_not_crash():
    program = "G90\nG01 X10 Y0 F100\nG01 X@@ Y5\nG65 P9000\nG02 X20 Y0\nX30\n"
    result = parse_gcode(program)
    assert len(result.warnings) >= 3
    assert result.segments[-1].points[-1] == (30, 0)


def test_extract_blocks_and_plot():
    reply = "Here you go:\n```gcode\nG90\nG01 X10 F100\n```\nand\n```nc\nG01 Y5\n```\n```python\nx=1\n```"
    blocks = extract_gcode_blocks(reply)
    assert len(blocks) == 2
    fig = plot_toolpath(parse_gcode(blocks[0]))
    assert fig is not None

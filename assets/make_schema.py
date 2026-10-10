"""
Draws the MechBot architecture diagram and saves it as assets/schema.png.

Uses only matplotlib (no Graphviz system binary needed). Run from the project root:
    python assets/make_schema.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Colours for the different kinds of block.
UI, LOGIC, DATA, AI, OUT = "#dbe9f6", "#e8f4ea", "#fdf2d9", "#f4e1f0", "#e3f2f1"
EDGE = "#34495e"

# name: (x_centre, y_centre, label, colour)  -- coordinates in a 0-20 x 0-10 canvas
NODES = {
    "user":    (1.2, 7.5, "User\n(student)", UI),
    "ui":      (3.9, 7.5, "Streamlit\nchat UI", UI),
    "mode":    (6.6, 7.5, "Mode selector\nTutor / CNC / Viva", UI),
    "prompt":  (9.9, 7.5, "System prompt\nbuilder\n(Tutor | CNC | Viva)", LOGIC),
    "history": (9.9, 5.0, "Chat history\n(st.session_state)", DATA),
    "notes":   (1.2, 2.4, "Uploaded notes\n(PDF / TXT)", DATA),
    "chunks":  (3.9, 2.4, "Text chunks", DATA),
    "tfidf":   (6.6, 2.4, "TF-IDF retriever\n(top-3 chunks)", LOGIC),
    "gemini":  (13.4, 5.0, "Gemini API\n(gemini-3.6-flash)", AI),
    "resp":    (16.3, 7.5, "Streamed\nresponse", OUT),
    "detect":  (16.3, 5.0, "G-code block\ndetector", OUT),
    "plot":    (16.3, 2.4, "matplotlib\ntoolpath plot\n+ stats", OUT),
    "display": (19.0, 5.0, "Display in\nchat", UI),
}
W, H = 2.3, 1.3

EDGES = [
    ("user", "ui", ""), ("ui", "mode", ""), ("mode", "prompt", "mode"),
    ("ui", "history", "messages"),
    ("notes", "chunks", "extract"), ("chunks", "tfidf", "vectorise"),
    ("tfidf", "prompt", "context"),
    ("prompt", "gemini", ""), ("history", "gemini", "multi-turn"),
    ("gemini", "resp", ""), ("resp", "detect", ""), ("detect", "plot", "```gcode"),
    ("resp", "display", ""), ("plot", "display", ""),
]


def border_point(name, toward):
    """Point on the box edge of `name` in the direction of point `toward`."""
    x, y = NODES[name][:2]
    dx, dy = toward[0] - x, toward[1] - y
    scale = min((W / 2) / abs(dx) if dx else 1e9, (H / 2) / abs(dy) if dy else 1e9)
    return x + dx * scale, y + dy * scale


def main():
    fig, ax = plt.subplots(figsize=(18, 8.5))
    ax.set_xlim(-0.2, 20.4)
    ax.set_ylim(0.9, 8.5)
    ax.axis("off")

    for x, y, label, colour in NODES.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.05,rounding_size=0.2",
                                    facecolor=colour, edgecolor=EDGE, linewidth=1.5))
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, color="#1b2631")

    for a, b, label in EDGES:
        start = border_point(a, NODES[b][:2])
        end = border_point(b, NODES[a][:2])
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                     color=EDGE, linewidth=1.4))
        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx, my + 0.18, label, ha="center", fontsize=8.5, style="italic",
                    color="#566573", bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Group label for the optional RAG lane.
    ax.add_patch(FancyBboxPatch((-0.1, 1.35), 8.0, 2.1, boxstyle="round,pad=0.1",
                                facecolor="none", edgecolor="#b7950b", linestyle="--"))
    ax.text(0.05, 3.25, "Optional RAG (notes upload)", fontsize=9, color="#9a7d0a")

    ax.set_title("MechBot – Architecture (Mechanical Engineering & CNC Chatbot)",
                 fontsize=16, fontweight="bold", color="#0b5394", pad=12)
    out = Path(__file__).with_name("schema.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

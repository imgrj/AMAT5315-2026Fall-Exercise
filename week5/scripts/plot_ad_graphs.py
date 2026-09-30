import unicodedata
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

def draw_box(ax, x, y, text, width=0.22, height=0.08, boxstyle="round,pad=0.02", facecolor="white", edgecolor="black", fontsize=9):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle=boxstyle, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, rad=0.0):
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2,
                                connectionstyle=connectionstyle, shrinkA=4, shrinkB=4))

def make_graph_png(out_path):
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    
    # Nodes layout
    # r at (0.3, 1.0)
    # pow-6 at (0.3, 0.8)
    # pow2 at (0.3, 0.6)
    # sub at (0.3, 0.4)
    # mul at (0.5, 0.2)
    # output at (0.5, 0.02)
    # const 4.0 at (0.7, 0.4)
    
    draw_box(ax, 0.35, 1.0, "r", width=0.12, height=0.05)
    draw_box(ax, 0.35, 0.8, "integer_pow[y=-6]", width=0.35, height=0.06)
    draw_box(ax, 0.25, 0.6, "integer_pow[y=2]", width=0.30, height=0.06)
    draw_box(ax, 0.35, 0.4, "sub", width=0.18, height=0.06)
    draw_box(ax, 0.7, 0.4, "4.0", width=0.12, height=0.05)
    draw_box(ax, 0.45, 0.2, "mul", width=0.18, height=0.06)
    draw_box(ax, 0.45, 0.05, "output", width=0.18, height=0.05)
    
    # Arrows
    draw_arrow(ax, 0.35, 0.97, 0.35, 0.83)
    draw_arrow(ax, 0.30, 0.77, 0.25, 0.63)
    # pow-6 to sub (bypassing pow2)
    draw_arrow(ax, 0.42, 0.77, 0.42, 0.43, rad=0.2)
    draw_arrow(ax, 0.25, 0.57, 0.32, 0.43)
    draw_arrow(ax, 0.35, 0.37, 0.42, 0.23)
    draw_arrow(ax, 0.7, 0.37, 0.48, 0.23)
    draw_arrow(ax, 0.45, 0.17, 0.45, 0.08)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def make_grad_graph_png(out_path):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    
    # Top inputs: r at (0.5, 0.98)
    draw_box(ax, 0.5, 0.98, "r", width=0.1, height=0.04)
    
    # Level 1:
    # pow-6 at (0.35, 0.85)
    # pow-7 at (0.75, 0.85)
    draw_box(ax, 0.35, 0.85, "integer_pow[y=-6]", width=0.28, height=0.05)
    draw_box(ax, 0.75, 0.85, "integer_pow[y=-7]", width=0.28, height=0.05)
    draw_arrow(ax, 0.48, 0.96, 0.38, 0.88)
    draw_arrow(ax, 0.52, 0.96, 0.72, 0.88)
    
    # Constants for derivatives:
    # -6.0 at (0.95, 0.85)
    # mul for -6*r^-7 at (0.80, 0.72)
    draw_box(ax, 0.95, 0.85, "-6.0", width=0.08, height=0.04)
    draw_box(ax, 0.80, 0.72, "mul", width=0.14, height=0.05)
    draw_arrow(ax, 0.75, 0.82, 0.78, 0.75)
    draw_arrow(ax, 0.93, 0.83, 0.83, 0.75)
    
    # Level 2 left side (primal):
    # pow2 at (0.22, 0.72)
    # mul (2.0 * a) at (0.45, 0.72)
    # const 2.0 at (0.55, 0.85)
    draw_box(ax, 0.22, 0.72, "integer_pow[y=2]", width=0.24, height=0.05)
    draw_box(ax, 0.55, 0.85, "2.0", width=0.08, height=0.04)
    draw_box(ax, 0.45, 0.72, "mul", width=0.14, height=0.05)
    draw_arrow(ax, 0.30, 0.82, 0.24, 0.75)
    draw_arrow(ax, 0.38, 0.82, 0.43, 0.75)
    draw_arrow(ax, 0.53, 0.83, 0.47, 0.75)
    
    # Subtraction c = b - a
    draw_box(ax, 0.30, 0.58, "sub", width=0.14, height=0.05)
    draw_arrow(ax, 0.22, 0.69, 0.28, 0.61)
    draw_arrow(ax, 0.34, 0.82, 0.32, 0.61, rad=0.2)
    
    # Primal mul (4.0 * c) - unused
    draw_box(ax, 0.15, 0.58, "4.0", width=0.08, height=0.04)
    draw_box(ax, 0.22, 0.46, "mul", width=0.14, height=0.05)
    draw_arrow(ax, 0.16, 0.56, 0.20, 0.49)
    draw_arrow(ax, 0.28, 0.55, 0.24, 0.49)
    
    # Reverse pass constants: 4.0 and 1.0 at (0.65, 0.85) and (0.68, 0.85)
    draw_box(ax, 0.65, 0.85, "4.0", width=0.08, height=0.04)
    draw_box(ax, 0.05, 0.85, "1.0", width=0.08, height=0.04)  # unused seed
    
    # bar_c = 4.0 * 1.0
    draw_box(ax, 0.65, 0.72, "mul", width=0.14, height=0.05)
    draw_arrow(ax, 0.65, 0.83, 0.65, 0.75)
    
    # neg(bar_c) for bar_a_from_c
    draw_box(ax, 0.60, 0.58, "neg", width=0.12, height=0.05)
    draw_arrow(ax, 0.64, 0.69, 0.61, 0.61)
    
    # bar_c * (2a) for bar_a_from_b
    draw_box(ax, 0.48, 0.58, "mul", width=0.14, height=0.05)
    draw_arrow(ax, 0.45, 0.69, 0.47, 0.61)
    draw_arrow(ax, 0.63, 0.69, 0.51, 0.61)
    
    # add_any joining neg and mul
    draw_box(ax, 0.55, 0.44, "add_any", width=0.18, height=0.05)
    draw_arrow(ax, 0.49, 0.55, 0.53, 0.47)
    draw_arrow(ax, 0.59, 0.55, 0.56, 0.47)
    
    # mul joining add_any and -6*r^-7
    draw_box(ax, 0.65, 0.28, "mul", width=0.16, height=0.05)
    draw_arrow(ax, 0.57, 0.41, 0.63, 0.31)
    draw_arrow(ax, 0.80, 0.69, 0.67, 0.31)
    
    # output
    draw_box(ax, 0.65, 0.12, "output", width=0.16, height=0.05)
    draw_arrow(ax, 0.65, 0.25, 0.65, 0.15)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def main():
    out_dir = Path("artifacts/ad")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_graph_png(out_dir / "graph.png")
    make_grad_graph_png(out_dir / "grad-graph.png")

if __name__ == "__main__":
    main()

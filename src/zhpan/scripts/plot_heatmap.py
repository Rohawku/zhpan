"""Render the v0.1 bias heatmap as PNG for the README."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Try to enable Chinese font if available; fall back silently
for font in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def main(in_path: str, out_path: str) -> None:
    with open(in_path) as f:
        data = json.load(f)
    bm = data["bias_matrix"]
    judges = bm["judges"]
    generators = bm["generators"]

    M = np.array(
        [[bm["mean_bias"][j][g] for g in generators] for j in judges],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    vmax = max(abs(M).max(), 0.5)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=20, ha="right")
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges)

    for i in range(len(judges)):
        for j in range(len(generators)):
            v = M[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=11)

    ax.set_title(
        f"zhpan {data.get('version', '')} — per-(judge × generator) mean bias\n"
        f"(judge score − {'anchor judge ' + repr(data['anchor_judge']) if data.get('anchor_judge') else 'silver-consensus gold'}, n={data['n_judgments']} judgments)"
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("mean bias (score points)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote heatmap → {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "leaderboard/v0.1/results.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "leaderboard/v0.1/bias_heatmap.png"
    main(in_path, out_path)

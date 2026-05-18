"""Render side-by-side bias heatmaps for two anchors + delta panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for font in ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def _plot_panel(ax, bm, title, vmax):
    judges = bm["judges"]
    generators = bm["generators"]
    M = np.array(
        [[bm["mean_bias"][j][g] for g in generators] for j in judges],
        dtype=float,
    )
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges, fontsize=9)
    for i in range(len(judges)):
        for j in range(len(generators)):
            v = M[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title(title, fontsize=11)
    return im


def main(in_path: str, out_path: str) -> None:
    with open(in_path) as f:
        data = json.load(f)
    bm_a = data["bias_matrix_anchor_a"]
    bm_b = data["bias_matrix_anchor_b"]
    name_a = data["anchor_a"]
    name_b = data["anchor_b"]

    # Compute delta matrix
    judges = bm_a["judges"]
    generators = bm_a["generators"]
    delta = {
        j: {
            g: bm_b["mean_bias"][j][g] - bm_a["mean_bias"][j][g]
            for g in generators
        }
        for j in judges
    }
    bm_delta = {
        "judges": judges,
        "generators": generators,
        "mean_bias": delta,
    }

    vmax = max(
        abs(np.array([[bm_a["mean_bias"][j][g] for g in generators] for j in judges])).max(),
        abs(np.array([[bm_b["mean_bias"][j][g] for g in generators] for j in judges])).max(),
        0.4,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), dpi=180)
    _plot_panel(axes[0], bm_a, f"anchor = {name_a}", vmax)
    _plot_panel(axes[1], bm_b, f"anchor = {name_b}", vmax)
    _plot_panel(axes[2], bm_delta, f"Δ = {name_b} − {name_a}\n(each row identical → anchor-invariance)", vmax)

    pear = data.get("pearson_r", float("nan"))
    spear = data.get("spearman_r", float("nan"))
    fig.suptitle(
        f"Cross-anchor robustness — Pearson ρ = {pear:+.3f},  Spearman ρ = {spear:+.3f}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote → {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "leaderboard/v0.3/anchor_compare.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "leaderboard/v0.3/anchor_compare_heatmap.png"
    main(in_path, out_path)

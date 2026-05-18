"""Render per-AlignBench-category bias breakdown as a grid of mini-heatmaps
+ a self-preference lift bar chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for font in ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


CAT_ORDER = [
    "数学计算", "逻辑推理", "中文理解", "文本写作",
    "角色扮演", "综合问答", "基本任务", "专业能力",
]

# AlignBench Chinese category → English label for matplotlib (avoids missing-font tofu)
_CAT_EN = {
    "数学计算": "Math",
    "逻辑推理": "Reasoning",
    "中文理解": "Chinese-NLU",
    "文本写作": "Writing",
    "角色扮演": "Roleplay",
    "综合问答": "Open-QA",
    "基本任务": "Basic-Task",
    "专业能力": "Domain-Expert",
}


def main(in_path: str, out_grid: str, out_lift: str) -> None:
    with open(in_path) as f:
        data = json.load(f)
    anchor = data.get("anchor", "?")
    by_cat = data["categories"]

    cats = [c for c in CAT_ORDER if c in by_cat]
    sample = next(iter(by_cat.values()))
    judges = sample["bias_matrix"]["judges"]
    generators = sample["bias_matrix"]["generators"]

    # ─── Heatmap grid: 8 categories × bias matrix ──────────
    n_cats = len(cats)
    cols = 4
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=160)
    axes = axes.flatten()

    vmax = 0.6
    for ax in axes:
        ax.set_axis_off()
    for idx, cat in enumerate(cats):
        ax = axes[idx]
        ax.set_axis_on()
        bm = by_cat[cat]["bias_matrix"]
        M = np.array([[bm["mean_bias"][j][g] for g in generators] for j in judges], dtype=float)
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(generators)))
        ax.set_xticklabels([g.replace("-chat", "").replace("-max", "").replace("-pro-32k", "").replace("-4-plus", "") for g in generators], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(judges)))
        ax.set_yticklabels([j.replace("-judge", "").replace("-chat", "").replace("-max", "").replace("-4-plus", "") for j in judges], fontsize=8)
        for i in range(len(judges)):
            for j in range(len(generators)):
                v = M[i, j]
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_title(f"{_CAT_EN.get(cat, cat)} (n={by_cat[cat]['n_prompts']})", fontsize=10)

    fig.suptitle(
        f"Per-category bias breakdown — anchor = {anchor}",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    Path(out_grid).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_grid, bbox_inches="tight")
    print(f"Wrote → {out_grid}")
    plt.close(fig)

    # ─── Self-preference lift bar chart ────────────────────
    lift_judges = [
        j for j in judges
        if any(t in j.lower() for t in ("deepseek", "qwen", "glm"))
    ]
    overall = by_cat.get("__overall__", {}).get("self_pref_lift", {})

    fig2, ax = plt.subplots(figsize=(11, 5), dpi=160)
    x = np.arange(len(cats))
    width = 0.25
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for k, j in enumerate(lift_judges):
        lifts = [by_cat[c]["self_pref_lift"].get(j, np.nan) for c in cats]
        short = j.replace("-judge", "").replace("-chat", "").replace("-max", "").replace("-4-plus", "")
        ax.bar(x + (k - 1) * width, lifts, width, label=f"{short}  (overall {overall.get(j, float('nan')):+.2f})", color=colors[k % len(colors)])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_CAT_EN.get(c, c) for c in cats], rotation=15, fontsize=10)
    ax.set_ylabel("self-preference lift (score points)")
    ax.set_title(f"Self-preference lift per AlignBench category — anchor = {anchor}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(out_lift, bbox_inches="tight")
    print(f"Wrote → {out_lift}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "leaderboard/v0.3/category_bias.json"
    out_grid = sys.argv[2] if len(sys.argv) > 2 else "leaderboard/v0.3/category_bias_heatmap.png"
    out_lift = sys.argv[3] if len(sys.argv) > 3 else "leaderboard/v0.3/category_selfpref_lift.png"
    main(in_path, out_grid, out_lift)

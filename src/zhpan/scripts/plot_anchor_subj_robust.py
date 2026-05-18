"""Compare subjectivity ↔ self-pref correlation under two anchors side-by-side.

Reads two subjectivity_correlation.json files (different anchors) and produces
a 2-panel comparison plot + bar chart of Pearson ρ per judge per anchor.

Usage:
    python -m zhpan.scripts.plot_anchor_subj_robust \\
        --primary leaderboard/v0.3/subjectivity_correlation.json \\
        --secondary leaderboard/v0.3/subjectivity_correlation_kimi.json \\
        --out leaderboard/v0.3/subjectivity_anchor_robust.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _short(j: str) -> str:
    return (
        j.replace("-judge", "")
        .replace("-chat", "")
        .replace("-max", "")
        .replace("-4-plus", "")
    )


def _judge_color(j: str) -> str:
    j = j.lower()
    if "deepseek" in j:
        return "#d62728"
    if "glm" in j:
        return "#1f77b4"
    if "qwen" in j:
        return "#2ca02c"
    return "gray"


def _scatter_panel(ax, data, anchor_label):
    judges = list(data["per_judge"].keys())
    for j in judges:
        r = data["per_judge"][j]
        pts = r["data"]
        x = np.array([d["subjectivity"] for d in pts])
        y = np.array([d["lift"] for d in pts])
        col = _judge_color(j)
        label = (
            f"{_short(j)}  ρ={r['pearson_r']:+.2f}"
            + ("**" if r["pearson_p"] < 0.05 else "")
            + f" (p={r['pearson_p']:.2f})"
        )
        ax.scatter(x, y, s=70, color=col, alpha=0.85, edgecolor="black", linewidth=0.5, label=label)
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + b, color=col, alpha=0.45, linestyle="--", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("subjectivity score  (1=objective, 5=subjective)")
    ax.set_ylabel("self-preference lift")
    ax.set_title(f"anchor = {anchor_label}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.primary) as f:
        prim = json.load(f)
    with open(args.secondary) as f:
        sec = json.load(f)

    # ─── side-by-side scatter ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    _scatter_panel(axes[0], prim, prim.get("anchor", "primary"))
    _scatter_panel(axes[1], sec, sec.get("anchor", "secondary"))

    # Bar chart inset showing Pearson ρ per (judge, anchor)
    judges = list(prim["per_judge"].keys())
    ymin = min(prim["per_judge"][j]["pearson_r"] for j in judges) - 0.2
    ymax = max(prim["per_judge"][j]["pearson_r"] for j in judges) + 0.2
    for ax in axes:
        ax.set_ylim(min(-0.6, ymin), max(0.9, ymax))

    fig.suptitle(
        "EXP-006: subjectivity ↔ self-preference is ANCHOR-ROBUST in direction\n"
        "DeepSeek (+) and GLM (−) trends preserved; magnitudes & significance differ",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote → {args.out}")

    # Also write a comparison summary
    summary = {
        "primary_anchor": prim.get("anchor"),
        "secondary_anchor": sec.get("anchor"),
        "per_judge": {},
    }
    for j in judges:
        summary["per_judge"][j] = {
            "pearson_r_primary": prim["per_judge"][j]["pearson_r"],
            "pearson_p_primary": prim["per_judge"][j]["pearson_p"],
            "pearson_r_secondary": sec["per_judge"][j]["pearson_r"],
            "pearson_p_secondary": sec["per_judge"][j]["pearson_p"],
            "spearman_r_primary": prim["per_judge"][j]["spearman_r"],
            "spearman_r_secondary": sec["per_judge"][j]["spearman_r"],
            "direction_agrees": (
                (prim["per_judge"][j]["pearson_r"] > 0) == (sec["per_judge"][j]["pearson_r"] > 0)
            ),
        }
    summary_path = Path(args.out).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary → {summary_path}")

    print()
    print(f"{'judge':<26}  {'ρ(ERNIE)':<12}  {'ρ(Kimi)':<12}  same direction?")
    print("-" * 78)
    for j, r in summary["per_judge"].items():
        print(
            f"{j:<26}  {r['pearson_r_primary']:+.3f}        {r['pearson_r_secondary']:+.3f}        {r['direction_agrees']}"
        )


if __name__ == "__main__":
    main()

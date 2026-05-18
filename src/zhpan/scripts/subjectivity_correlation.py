"""Test the hypothesis: task subjectivity correlates with self-preference lift.

Reads `category_bias.json` produced by per_category_bias.py, scores each
AlignBench main category by subjectivity (1=most objective, 5=most subjective),
and computes per-judge Pearson + Spearman correlation between subjectivity and
self-preference lift. Also outputs a scatter plot.

Usage:
    python -m zhpan.scripts.subjectivity_correlation \\
        --category-bias leaderboard/v0.3/category_bias.json \\
        --out leaderboard/v0.3/subjectivity_correlation.json \\
        --plot leaderboard/v0.3/subjectivity_scatter.png
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
from scipy.stats import pearsonr, spearmanr


# Subjectivity scoring rationale (1=most objective, 5=most subjective):
# - Math: pure computation, single right answer → 1.0
# - Domain-Expert: factual knowledge, mostly verifiable → 1.5
# - Basic-Task: structured extraction/classification → 2.0
# - Reasoning: verifiable answer but reasoning quality subjective → 2.5
# - Open-QA: mixed factual + opinion → 3.0
# - Chinese-NLU: nuanced interpretation needed → 3.5
# - Writing: quality depends on taste → 4.5
# - Roleplay: voice/style is the entire answer → 5.0
_SUBJECTIVITY = {
    "数学计算": (1.0, "Math"),
    "专业能力": (1.5, "Domain-Expert"),
    "基本任务": (2.0, "Basic-Task"),
    "逻辑推理": (2.5, "Reasoning"),
    "综合问答": (3.0, "Open-QA"),
    "中文理解": (3.5, "Chinese-NLU"),
    "文本写作": (4.5, "Writing"),
    "角色扮演": (5.0, "Roleplay"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category-bias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", required=True)
    args = ap.parse_args()

    with open(args.category_bias) as f:
        data = json.load(f)

    by_cat = data["categories"]
    cats = [c for c in _SUBJECTIVITY if c in by_cat]
    if not cats:
        print("ERROR: no AlignBench categories matched", file=sys.stderr)
        sys.exit(2)

    # Identify tested judges (excluding anchors)
    sample = next(iter(by_cat.values()))
    judges = sample["bias_matrix"]["judges"]

    # Build (subjectivity, lift) tuples per judge
    per_judge: dict[str, dict] = {}
    for j in judges:
        if any(t in j.lower() for t in ("kimi", "ernie", "anchor")):
            continue  # skip anchors
        rows = []
        for cat in cats:
            subj, en = _SUBJECTIVITY[cat]
            lift = by_cat[cat]["self_pref_lift"].get(j)
            if lift is None or (isinstance(lift, float) and np.isnan(lift)):
                continue
            rows.append({"category": cat, "category_en": en, "subjectivity": subj, "lift": lift})
        subj_arr = np.array([r["subjectivity"] for r in rows])
        lift_arr = np.array([r["lift"] for r in rows])
        pear_r, pear_p = pearsonr(subj_arr, lift_arr)
        spear_r, spear_p = spearmanr(subj_arr, lift_arr)
        per_judge[j] = {
            "data": rows,
            "pearson_r": float(pear_r),
            "pearson_p": float(pear_p),
            "spearman_r": float(spear_r),
            "spearman_p": float(spear_p),
            "n": len(rows),
        }

    # Print summary table
    print()
    print("=" * 78)
    print("Subjectivity ↔ self-preference lift correlation")
    print("=" * 78)
    print()
    print(f"{'judge':<26}  {'n':<3}  {'Pearson ρ':<12}  {'p':<10}  {'Spearman ρ':<12}  {'p':<10}")
    print("-" * 78)
    for j, r in per_judge.items():
        print(
            f"{j:<26}  {r['n']:<3}  {r['pearson_r']:+.3f}        {r['pearson_p']:.3f}       "
            f"{r['spearman_r']:+.3f}        {r['spearman_p']:.3f}"
        )
    print()
    print("subjectivity score table:")
    for cat, (subj, en) in _SUBJECTIVITY.items():
        print(f"  {subj:.1f}  {cat:<10} ({en})")
    print()

    # Plot scatter
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    colors = {"deepseek": "#d62728", "glm": "#1f77b4", "qwen": "#2ca02c"}
    for j, r in per_judge.items():
        short = (
            j.replace("-judge", "")
            .replace("-chat", "")
            .replace("-max", "")
            .replace("-4-plus", "")
        )
        color = next(
            (v for k, v in colors.items() if k in j.lower()), "gray"
        )
        x = np.array([d["subjectivity"] for d in r["data"]])
        y = np.array([d["lift"] for d in r["data"]])
        ax.scatter(x, y, s=70, color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
                   label=f"{short}  (ρ={r['pearson_r']:+.2f}, p={r['pearson_p']:.2f})")
        # least-squares fit line
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + b, color=color, alpha=0.4, linestyle="--", linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("subjectivity score  (1 = most objective, 5 = most subjective)")
    ax.set_ylabel("self-preference lift  (score points)")
    ax.set_title(
        "EXP-005: does task subjectivity correlate with self-preference lift?\n"
        f"(8 AlignBench categories, anchor = {data.get('anchor', 'ERNIE')})"
    )
    # Annotate each point with its category name (English)
    for j, r in list(per_judge.items())[:1]:  # only annotate one judge's worth of points
        for d in r["data"]:
            ax.annotate(d["category_en"], (d["subjectivity"], d["lift"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="dimgray")
    ax.legend(loc="upper left")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, bbox_inches="tight")
    print(f"Wrote scatter → {args.plot}")

    out = {
        "anchor": data.get("anchor"),
        "subjectivity_scoring": {cat: {"score": s, "label_en": en} for cat, (s, en) in _SUBJECTIVITY.items()},
        "per_judge": per_judge,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary → {args.out}")


if __name__ == "__main__":
    main()

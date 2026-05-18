"""Compare two anchor judges side-by-side on the same judgments.

Validates whether per-pair bias is real or an anchor-selection artifact.
If two independent anchors agree on the bias matrix → signal is real.
If they disagree → bias is anchor-induced, not robust.

Usage:
    python -m zhpan.scripts.anchor_compare \\
        --judgments data/judgments/v0.3/judgments.jsonl \\
        --anchor-a kimi-anchor-judge \\
        --anchor-b claude-anchor-judge
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from zhpan.compute_bias import build_gold_anchor, compute_bias
from zhpan.utils import get_logger, read_jsonl

log = get_logger("zhpan.anchor_compare")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True, help="path to judgments.jsonl")
    ap.add_argument("--anchor-a", required=True, help="first anchor judge name")
    ap.add_argument("--anchor-b", required=True, help="second anchor judge name")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    judgments = read_jsonl(args.judgments)
    log.info(f"Loaded {len(judgments)} judgments")

    # Build two bias matrices, each excluding its own anchor
    gold_a = build_gold_anchor(judgments, anchor_judge=args.anchor_a)
    bm_a = compute_bias(
        judgments, gold_a,
        exclude_judges={args.anchor_a, args.anchor_b},
    )
    gold_b = build_gold_anchor(judgments, anchor_judge=args.anchor_b)
    bm_b = compute_bias(
        judgments, gold_b,
        exclude_judges={args.anchor_a, args.anchor_b},
    )

    judges = bm_a.judges
    generators = bm_a.generators

    # Flatten matrices into matched (judge, generator) cells
    cells_a, cells_b = [], []
    for j in judges:
        for g in generators:
            a = bm_a.mean_bias[j][g]
            b = bm_b.mean_bias[j][g]
            if np.isnan(a) or np.isnan(b):
                continue
            cells_a.append(a)
            cells_b.append(b)

    arr_a = np.array(cells_a)
    arr_b = np.array(cells_b)

    pear_r, pear_p = pearsonr(arr_a, arr_b)
    spear_r, spear_p = spearmanr(arr_a, arr_b)
    mae = float(np.mean(np.abs(arr_a - arr_b)))
    max_delta = float(np.max(np.abs(arr_a - arr_b)))

    print("\n" + "=" * 72)
    print(f"Anchor comparison: {args.anchor_a!r}  vs  {args.anchor_b!r}")
    print("=" * 72)
    print(f"  cells compared:   {len(cells_a)} = {len(judges)} judges × {len(generators)} generators")
    print(f"  Pearson  ρ:       {pear_r:+.3f}  (p={pear_p:.2e})")
    print(f"  Spearman ρ:       {spear_r:+.3f}  (p={spear_p:.2e})")
    print(f"  MAE  | a − b |:   {mae:.3f}")
    print(f"  max| a − b |:     {max_delta:.3f}")
    print()

    # Side-by-side bias matrix
    sep = " | "
    header = "judge \\ gen".ljust(28) + " | " + sep.join(f"{g[:11]:<11}" for g in generators)
    print(header)
    print("-" * 28 + "-+-" + "-+-".join(["-" * 11] * len(generators)))
    print(f"  (anchor: {args.anchor_a})")
    for j in judges:
        row = sep.join(f"{bm_a.mean_bias[j][g]:+.2f}".ljust(11) for g in generators)
        print(f"{j[:26]:<28} | {row}")
    print()
    print(f"  (anchor: {args.anchor_b})")
    for j in judges:
        row = sep.join(f"{bm_b.mean_bias[j][g]:+.2f}".ljust(11) for g in generators)
        print(f"{j[:26]:<28} | {row}")
    print()
    print(f"  (delta: B − A)")
    for j in judges:
        row = sep.join(
            f"{bm_b.mean_bias[j][g] - bm_a.mean_bias[j][g]:+.2f}".ljust(11)
            for g in generators
        )
        print(f"{j[:26]:<28} | {row}")
    print()

    summary = {
        "anchor_a": args.anchor_a,
        "anchor_b": args.anchor_b,
        "n_cells": len(cells_a),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "mae_abs_delta": mae,
        "max_abs_delta": max_delta,
        "bias_matrix_anchor_a": bm_a.to_dict(),
        "bias_matrix_anchor_b": bm_b.to_dict(),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info(f"Wrote summary → {args.out}")


if __name__ == "__main__":
    main()

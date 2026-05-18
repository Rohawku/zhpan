"""Per-category bias breakdown: split judgments by AlignBench category,
compute a separate bias matrix for each, and surface category-level patterns.

Usage:
    python -m zhpan.scripts.per_category_bias \\
        --judgments data/judgments/v0.3/judgments.jsonl \\
        --prompts data/prompts/v0.3.jsonl \\
        --anchor ernie-anchor-judge \\
        --secondary-anchor kimi-anchor-judge \\
        --out leaderboard/v0.3/category_bias.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from zhpan.compute_bias import build_gold_anchor, compute_bias
from zhpan.utils import get_logger, read_jsonl

log = get_logger("zhpan.per_category_bias")


def _self_pref_lift(bm_dict: dict) -> dict[str, float]:
    """Return per-judge self-preference lift = bias_to_self - mean(bias_to_others).

    Matches model-family by simple substring rule (deepseek/qwen/glm/doubao/kimi/ernie).
    """
    judges = bm_dict["judges"]
    generators = bm_dict["generators"]
    out: dict[str, float] = {}
    for j in judges:
        j_low = j.lower()
        # find which generator family this judge belongs to
        self_gen = None
        for g in generators:
            g_low = g.lower()
            for tok in ("deepseek", "qwen", "glm", "doubao", "kimi", "ernie"):
                if tok in j_low and tok in g_low:
                    self_gen = g
                    break
            if self_gen:
                break
        if self_gen is None:
            out[j] = float("nan")
            continue
        self_bias = bm_dict["mean_bias"][j].get(self_gen)
        others = [
            bm_dict["mean_bias"][j][g] for g in generators
            if g != self_gen and not (self_bias is None or np.isnan(bm_dict["mean_bias"][j][g]))
        ]
        if self_bias is None or np.isnan(self_bias) or not others:
            out[j] = float("nan")
            continue
        out[j] = float(self_bias - np.mean(others))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--anchor", required=True, help="primary anchor judge")
    ap.add_argument("--secondary-anchor", default=None, help="optional second anchor for cross-validation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = read_jsonl(args.prompts)
    judgments = read_jsonl(args.judgments)

    # Map prompt_id -> AlignBench category (Chinese, original)
    pid_to_cat: dict[str, str] = {}
    for p in prompts:
        cat = p.get("metadata", {}).get("alignbench_category") or p.get("category", "?")
        pid_to_cat[p["id"]] = cat

    log.info(f"Loaded {len(prompts)} prompts, {len(judgments)} judgments")

    # Bucket judgments by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        cat = pid_to_cat.get(j["prompt_id"])
        if cat:
            by_cat[cat].append(j)

    log.info(f"Found {len(by_cat)} categories")

    excluded = {args.anchor}
    if args.secondary_anchor:
        excluded.add(args.secondary_anchor)

    results: dict[str, dict] = {}
    for cat, cat_judgments in sorted(by_cat.items()):
        # primary anchor bias matrix
        try:
            gold = build_gold_anchor(cat_judgments, anchor_judge=args.anchor)
        except ValueError:
            log.warning(f"Category {cat!r}: no anchor judgments, skipping")
            continue
        bm = compute_bias(cat_judgments, gold, exclude_judges=excluded)
        n_prompts = len({j["prompt_id"] for j in cat_judgments})
        lift = _self_pref_lift(bm.to_dict())
        results[cat] = {
            "n_prompts": n_prompts,
            "n_judgments": len(cat_judgments),
            "bias_matrix": bm.to_dict(),
            "self_pref_lift": lift,
        }

    # Overall pooled (for reference)
    try:
        gold_all = build_gold_anchor(judgments, anchor_judge=args.anchor)
        bm_all = compute_bias(judgments, gold_all, exclude_judges=excluded)
        results["__overall__"] = {
            "n_prompts": len(prompts),
            "n_judgments": len(judgments),
            "bias_matrix": bm_all.to_dict(),
            "self_pref_lift": _self_pref_lift(bm_all.to_dict()),
        }
    except ValueError:
        pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "anchor": args.anchor,
                "secondary_anchor": args.secondary_anchor,
                "categories": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print readable summary
    print()
    print("=" * 100)
    print(f"Per-category bias breakdown — anchor = {args.anchor!r}")
    print("=" * 100)
    print()

    # Find tested judges (excluding anchors)
    sample_bm = next(iter(results.values()))["bias_matrix"]
    judges = sample_bm["judges"]
    generators = sample_bm["generators"]

    # Header: self_pref_lift per category for each judge
    print(f"{'category':<14} {'n':<5}", end="")
    for j in judges:
        short = j.replace("-judge", "").replace("-chat", "").replace("-max", "").replace("-4-plus", "-glm")
        print(f"  {short[:14]:>14}", end="")
    print()
    print("-" * (19 + 16 * len(judges)))

    cats_ordered = [
        c for c in [
            "__overall__", "数学计算", "逻辑推理", "中文理解", "文本写作",
            "角色扮演", "综合问答", "基本任务", "专业能力"
        ] if c in results
    ]
    for cat in cats_ordered:
        r = results[cat]
        n = r["n_prompts"]
        label = "OVERALL" if cat == "__overall__" else cat
        print(f"{label:<14} {n:<5}", end="")
        for j in judges:
            lift = r["self_pref_lift"].get(j, float("nan"))
            print(f"  {lift:>+14.2f}", end="")
        print()
    print()
    print("(values shown above = self-preference lift per (judge, category))")
    print()

    log.info(f"Wrote → {out_path}")


if __name__ == "__main__":
    main()

"""Bias matrix computation.

Defines the v0.1 metrics:
- M1 mean_bias[j][g]: E[judge_score - gold_score]
- M2 std_bias[j][g]: std(judge_score - gold_score)
- M3 rank_corr[j]:   Spearman rho between judge's per-generator rankings vs gold's
- M4 calibrated_mae: MAE drop after per-pair offset correction (set in calibrate.py)
- M5 self_pref_lift[j]: bias to self minus mean bias to others (only if j ∈ generators)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .utils import get_logger

log = get_logger("zhpan.bias")


@dataclass
class BiasMatrix:
    judges: list[str]
    generators: list[str]
    mean_bias: dict[str, dict[str, float]]
    std_bias: dict[str, dict[str, float]]
    n: dict[str, dict[str, int]]
    rank_corr: dict[str, float]
    self_pref_lift: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "judges": self.judges,
            "generators": self.generators,
            "mean_bias": self.mean_bias,
            "std_bias": self.std_bias,
            "n": self.n,
            "rank_corr": self.rank_corr,
            "self_pref_lift": self.self_pref_lift,
        }


def build_gold_silver(
    judgments: list[dict],
    min_n_judges: int = 2,
    max_std: float = 1.0,
) -> dict[str, float]:
    """Strong-consensus silver gold (DEPRECATED in v0.2 for bias measurement).

    ⚠️  Using silver consensus as gold for bias measurement is **mathematically
    circular**: the bias of each judge against the mean of the same judges is
    constrained to sum to zero across judges. Use `build_gold_anchor()` instead.

    Kept for backwards-compat with v0.1 results and for CV evaluation where
    self-consistent gold is acceptable.
    """
    by_gen: dict[str, list[int]] = defaultdict(list)
    for j in judgments:
        if j.get("score") is None:
            continue
        by_gen[j["generation_id"]].append(int(j["score"]))

    gold: dict[str, float] = {}
    for gen_id, scores in by_gen.items():
        if len(scores) < min_n_judges:
            continue
        s = np.array(scores, dtype=float)
        if s.std(ddof=0) <= max_std:
            gold[gen_id] = float(s.mean())
    return gold


def build_gold_anchor(
    judgments: list[dict],
    anchor_judge: str,
) -> dict[str, float]:
    """Anchor gold: use a single independent judge as ground truth.

    The anchor judge MUST NOT also appear in the set of judges whose bias is
    being measured — that's the whole point. Returns {generation_id: gold_score}.

    This breaks the circular-reasoning bug in silver consensus gold: bias[j][g]
    no longer mechanically sums to zero across judges, because the anchor sits
    outside the tested judge set.
    """
    gold: dict[str, float] = {}
    for j in judgments:
        if j.get("judge") != anchor_judge:
            continue
        if j.get("score") is None:
            continue
        gold[j["generation_id"]] = float(j["score"])
    if not gold:
        raise ValueError(
            f"No judgments found from anchor judge '{anchor_judge}'. "
            "Check your config has the anchor judge running and it's not in the "
            "tested judge set."
        )
    return gold


def compute_bias(
    judgments: list[dict],
    gold: dict[str, float],
    exclude_judges: set[str] | None = None,
) -> BiasMatrix:
    """Compute per-(judge × generator) mean and std bias against a gold dict.

    `exclude_judges` lets you drop the anchor judge from the output matrix
    (it would otherwise have bias 0 by construction).
    """
    exclude = exclude_judges or set()
    judges = sorted({j["judge"] for j in judgments if j["judge"] not in exclude})
    generators = sorted({j["generator"] for j in judgments})

    diffs: dict[str, dict[str, list[float]]] = {
        jn: {gn: [] for gn in generators} for jn in judges
    }
    for j in judgments:
        if j.get("score") is None:
            continue
        if j["judge"] in exclude:
            continue
        if j["generation_id"] not in gold:
            continue
        d = float(j["score"]) - gold[j["generation_id"]]
        diffs[j["judge"]][j["generator"]].append(d)

    mean_bias: dict[str, dict[str, float]] = {}
    std_bias: dict[str, dict[str, float]] = {}
    n: dict[str, dict[str, int]] = {}
    for jn in judges:
        mean_bias[jn] = {}
        std_bias[jn] = {}
        n[jn] = {}
        for gn in generators:
            arr = np.array(diffs[jn][gn], dtype=float)
            n[jn][gn] = len(arr)
            mean_bias[jn][gn] = float(arr.mean()) if len(arr) else float("nan")
            std_bias[jn][gn] = float(arr.std(ddof=0)) if len(arr) else float("nan")

    rank_corr = _per_judge_rank_corr(judgments, gold, judges, generators)
    self_pref = _self_pref_lift(mean_bias, judges, generators)

    return BiasMatrix(
        judges=judges,
        generators=generators,
        mean_bias=mean_bias,
        std_bias=std_bias,
        n=n,
        rank_corr=rank_corr,
        self_pref_lift=self_pref,
    )


def _per_judge_rank_corr(
    judgments: list[dict],
    gold: dict[str, float],
    judges: list[str],
    generators: list[str],
) -> dict[str, float]:
    # mean score per (judge, generator) and mean gold per generator
    judge_avg: dict[str, dict[str, list[int]]] = {
        jn: {gn: [] for gn in generators} for jn in judges
    }
    judges_set = set(judges)
    gold_avg: dict[str, list[float]] = {gn: [] for gn in generators}
    for j in judgments:
        if j.get("score") is None:
            continue
        if j["judge"] not in judges_set:
            continue
        if j["generation_id"] not in gold:
            continue
        judge_avg[j["judge"]][j["generator"]].append(int(j["score"]))
        gold_avg[j["generator"]].append(gold[j["generation_id"]])

    gold_means = [
        float(np.mean(gold_avg[g])) if gold_avg[g] else float("nan") for g in generators
    ]
    out: dict[str, float] = {}
    for jn in judges:
        judge_means = [
            float(np.mean(judge_avg[jn][g])) if judge_avg[jn][g] else float("nan")
            for g in generators
        ]
        pairs = [
            (jm, gm)
            for jm, gm in zip(judge_means, gold_means, strict=False)
            if not (np.isnan(jm) or np.isnan(gm))
        ]
        if len(pairs) < 3:
            out[jn] = float("nan")
            continue
        jm_arr = np.array([p[0] for p in pairs])
        gm_arr = np.array([p[1] for p in pairs])
        rho, _ = stats.spearmanr(jm_arr, gm_arr)
        out[jn] = float(rho)
    return out


def _self_pref_lift(
    mean_bias: dict[str, dict[str, float]],
    judges: list[str],
    generators: list[str],
) -> dict[str, float]:
    """For each judge, if its name matches a generator name (or shares the family),
    self_pref_lift = bias_to_self - mean(bias_to_others).

    A judge that doesn't appear as a generator gets NaN.
    """
    out: dict[str, float] = {}
    for jn in judges:
        self_match = _match_self_generator(jn, generators)
        if self_match is None:
            out[jn] = float("nan")
            continue
        self_bias = mean_bias[jn].get(self_match)
        others = [
            mean_bias[jn][g] for g in generators if g != self_match and not np.isnan(mean_bias[jn][g])
        ]
        if self_bias is None or np.isnan(self_bias) or not others:
            out[jn] = float("nan")
            continue
        out[jn] = float(self_bias - np.mean(others))
    return out


_FAMILY_TOKENS = (
    "qwen", "deepseek", "glm", "doubao",            # 中文模型
    "claude", "gpt", "llama", "gemini", "mistral",   # 英文对照
)


def _match_self_generator(judge_name: str, generators: list[str]) -> str | None:
    """A judge `qwen-max-judge` matches generator `qwen-max` (same family)."""
    j = judge_name.lower()
    candidates = [
        g for g in generators
        if any(tok in j and tok in g.lower() for tok in _FAMILY_TOKENS)
    ]
    return candidates[0] if candidates else None


def calibrated_mae(
    judgments: list[dict],
    gold: dict[str, float],
    calibrator,  # zhpan.calibrate.Calibrator
) -> dict[str, dict[str, dict[str, float]]]:
    """For each (judge, generator), report MAE before and after calibration on the given set.

    Returns {judge: {generator: {"mae_before": ..., "mae_after": ..., "n": ...}}}.
    """
    judges = sorted({j["judge"] for j in judgments})
    generators = sorted({j["generator"] for j in judgments})

    out: dict[str, dict[str, dict[str, float]]] = {jn: {} for jn in judges}
    by_pair: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for j in judgments:
        if j.get("score") is None or j["generation_id"] not in gold:
            continue
        by_pair[(j["judge"], j["generator"])].append(
            (float(j["score"]), float(gold[j["generation_id"]]))
        )

    for jn in judges:
        for gn in generators:
            pairs = by_pair.get((jn, gn), [])
            if not pairs:
                out[jn][gn] = {"mae_before": float("nan"), "mae_after": float("nan"), "n": 0}
                continue
            before = np.array([abs(s - g) for s, g in pairs])
            after = np.array(
                [abs(calibrator.correct(judge=jn, generator=gn, raw_score=s) - g) for s, g in pairs]
            )
            out[jn][gn] = {
                "mae_before": float(before.mean()),
                "mae_after": float(after.mean()),
                "n": len(pairs),
            }
    return out

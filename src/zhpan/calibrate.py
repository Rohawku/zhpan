"""Calibration utilities.

v0.1: per-pair offset calibration (subtract the (judge, generator) mean bias).
v0.2 (planned): per-pair linear regression.

Public API (the 3-line demo from README):

    from zhpan import calibrate
    cal = calibrate.load("v0.1")
    fair = cal.correct(judge="gpt-4o", generator="claude-3-5-sonnet", raw_score=2.1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .compute_bias import BiasMatrix, build_gold_silver, compute_bias
from .utils import get_logger

log = get_logger("zhpan.calibrate")


@dataclass
class Calibrator:
    """Per-(judge × generator) offset calibrator."""

    offsets: dict[str, dict[str, float]] = field(default_factory=dict)
    version: str = "v0.1"
    method: str = "per_pair_offset"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bias_matrix(cls, bm: BiasMatrix, version: str = "v0.1") -> Calibrator:
        offsets = {
            jn: {gn: float(bm.mean_bias[jn][gn]) for gn in bm.generators} for jn in bm.judges
        }
        return cls(
            offsets=offsets,
            version=version,
            method="per_pair_offset",
            metadata={"n_judges": len(bm.judges), "n_generators": len(bm.generators)},
        )

    def correct(self, judge: str, generator: str, raw_score: float) -> float:
        """Return calibrated score: raw_score - bias[judge][generator].

        Falls back to 0 offset when the pair is unknown.
        """
        offset = self.offsets.get(judge, {}).get(generator, 0.0)
        if np.isnan(offset):
            offset = 0.0
        out = raw_score - offset
        # Clip to scale 1-10 (v0.2) or 1-5 (v0.1, still fits in 1-10)
        return float(max(1.0, min(10.0, out)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    "version": self.version,
                    "method": self.method,
                    "metadata": self.metadata,
                    "offsets": self.offsets,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def from_file(cls, path: str | Path) -> Calibrator:
        with Path(path).open() as f:
            data = json.load(f)
        return cls(
            offsets=data.get("offsets", {}),
            version=data.get("version", "unknown"),
            method=data.get("method", "per_pair_offset"),
            metadata=data.get("metadata", {}),
        )


# ──────────────────────────────────────────
# Convenience loaders for the README 3-line demo
# ──────────────────────────────────────────


def load(version: str = "v0.1") -> Calibrator:
    """Load the released Calibrator for a given version.

    Looks at `leaderboard/{version}/calibrator.json` first, then a small bundled
    fallback (for offline use).
    """
    path = Path(__file__).resolve().parent.parent.parent / "leaderboard" / version / "calibrator.json"
    if path.exists():
        return Calibrator.from_file(path)
    raise FileNotFoundError(
        f"No released calibrator for version '{version}'. "
        f"Train one via `make benchmark` first."
    )


def fit_from_judgments(
    judgments: list[dict],
    anchor_judge: str | None = None,
) -> tuple[Calibrator, BiasMatrix]:
    """Fit a calibrator from raw judgments.

    If `anchor_judge` is set, use that judge's scores as independent gold
    (the v0.2 approach that breaks the circular silver-consensus bug).
    Otherwise fall back to silver consensus (v0.1 behaviour).
    """
    from .compute_bias import build_gold_anchor

    exclude: set[str] = set()
    if anchor_judge:
        gold = build_gold_anchor(judgments, anchor_judge=anchor_judge)
        exclude = {anchor_judge}
    else:
        gold = build_gold_silver(judgments)
        if not gold:
            raise ValueError("No silver-consensus gold could be constructed from judgments.")
    bm = compute_bias(judgments, gold, exclude_judges=exclude)
    cal = Calibrator.from_bias_matrix(bm)
    return cal, bm


def cv_eval(
    judgments: list[dict],
    n_folds: int = 5,
    seed: int = 42,
    anchor_judge: str | None = None,
) -> dict[str, Any]:
    """5-fold CV on the prompt axis. Held-out MAE before/after calibration.

    Uses the same anchor-or-silver gold construction as `fit_from_judgments`.
    """
    import random
    from .compute_bias import build_gold_anchor

    rng = random.Random(seed)
    prompt_ids = sorted({j["prompt_id"] for j in judgments})
    rng.shuffle(prompt_ids)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, pid in enumerate(prompt_ids):
        folds[i % n_folds].append(pid)

    def _gold(rows):
        return (
            build_gold_anchor(rows, anchor_judge=anchor_judge)
            if anchor_judge
            else build_gold_silver(rows)
        )

    fold_results: list[dict[str, float]] = []
    for k in range(n_folds):
        held_out = set(folds[k])
        train = [j for j in judgments if j["prompt_id"] not in held_out]
        test = [j for j in judgments if j["prompt_id"] in held_out]
        try:
            cal, _ = fit_from_judgments(train, anchor_judge=anchor_judge)
        except ValueError:
            continue
        try:
            test_gold = _gold(test)
        except ValueError:
            continue
        if not test_gold:
            continue
        before, after = [], []
        for j in test:
            if j.get("score") is None or j["generation_id"] not in test_gold:
                continue
            if anchor_judge and j["judge"] == anchor_judge:
                continue
            g = test_gold[j["generation_id"]]
            before.append(abs(float(j["score"]) - g))
            after.append(
                abs(cal.correct(judge=j["judge"], generator=j["generator"], raw_score=float(j["score"])) - g)
            )
        if before:
            fold_results.append(
                {
                    "fold": k,
                    "n": len(before),
                    "mae_before": float(np.mean(before)),
                    "mae_after": float(np.mean(after)),
                }
            )
    if not fold_results:
        return {"folds": [], "summary": {"mae_before": float("nan"), "mae_after": float("nan")}}
    return {
        "folds": fold_results,
        "summary": {
            "mae_before": float(np.mean([f["mae_before"] for f in fold_results])),
            "mae_after": float(np.mean([f["mae_after"] for f in fold_results])),
            "n_folds": len(fold_results),
        },
    }

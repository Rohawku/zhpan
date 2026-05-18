"""CLI: python -m zhpan.scripts.analyze --config configs/v0.1.yaml

Reads judgments, builds silver gold, computes bias matrix, fits a Calibrator,
runs 5-fold CV, and writes results + leaderboard JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhpan.calibrate import Calibrator, cv_eval, fit_from_judgments
from zhpan.compute_bias import calibrated_mae
from zhpan.utils import get_logger, load_yaml_config, read_jsonl

log = get_logger("zhpan.cli.analyze")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="Override leaderboard output dir")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    version = cfg.get("version", "v0.1")
    j_cfg = cfg.get("judging", {})
    jud_dir = Path(j_cfg.get("output_dir", f"data/judgments/{version}"))
    judgments = read_jsonl(jud_dir / "judgments.jsonl")
    log.info(f"Loaded {len(judgments)} judgments")

    cal, bm = fit_from_judgments(judgments)
    cv = cv_eval(judgments, n_folds=5)
    cal_mae = calibrated_mae(judgments, gold=_silver_gold_from(judgments), calibrator=cal)

    out_dir = Path(args.out_dir) if args.out_dir else Path("leaderboard") / version
    out_dir.mkdir(parents=True, exist_ok=True)
    cal.save(out_dir / "calibrator.json")

    summary = {
        "version": version,
        "n_judgments": len(judgments),
        "judges": bm.judges,
        "generators": bm.generators,
        "bias_matrix": bm.to_dict(),
        "calibrated_mae": cal_mae,
        "cv_eval": cv,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_readable_summary(summary)
    log.info(f"Wrote calibrator → {out_dir / 'calibrator.json'}")
    log.info(f"Wrote results   → {out_dir / 'results.json'}")


def _silver_gold_from(judgments):
    from zhpan.compute_bias import build_gold_silver

    return build_gold_silver(judgments)


def _print_readable_summary(summary: dict) -> None:
    bm = summary["bias_matrix"]
    print("\n──────────────── Bias Matrix (mean) ────────────────")
    judges = bm["judges"]
    generators = bm["generators"]
    header = "judge \\ generator".ljust(28) + " ".join(g[:14].ljust(15) for g in generators)
    print(header)
    for j in judges:
        row = j[:26].ljust(28)
        for g in generators:
            v = bm["mean_bias"][j].get(g, float("nan"))
            row += f"{v:+.2f}".ljust(15)
        print(row)
    cv = summary["cv_eval"]["summary"]
    print(f"\n5-fold CV held-out MAE: {cv.get('mae_before', float('nan')):.3f} → "
          f"{cv.get('mae_after', float('nan')):.3f} (calibrated)")


if __name__ == "__main__":
    main()

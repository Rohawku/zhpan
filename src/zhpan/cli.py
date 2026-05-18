"""zhpan CLI — `zhpan <subcommand>` entry point.

Subcommands:
    zhpan debias --judge X --gen Y --score 2.1 [--calibrator PATH]
    zhpan info
    zhpan demo
    zhpan version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .calibrate import Calibrator
from .utils import get_logger

log = get_logger("zhpan.cli")


def _cmd_debias(args: argparse.Namespace) -> int:
    cal_path = Path(args.calibrator)
    if not cal_path.exists():
        print(
            f"[zhpan] calibrator not found: {cal_path}\n"
            f"        Train one with `make benchmark` or download a pre-built calibrator.",
            file=sys.stderr,
        )
        return 2
    cal = Calibrator.from_file(cal_path)
    fair = cal.correct(judge=args.judge, generator=args.gen, raw_score=args.score)
    if args.json:
        print(
            json.dumps(
                {
                    "judge": args.judge,
                    "generator": args.gen,
                    "raw_score": args.score,
                    "calibrated_score": fair,
                    "offset_applied": cal.offsets.get(args.judge, {}).get(args.gen, 0.0),
                    "calibrator_version": cal.version,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        offset = cal.offsets.get(args.judge, {}).get(args.gen, 0.0)
        print(
            f"raw={args.score:.2f}  →  calibrated={fair:.2f}  "
            f"(offset={offset:+.3f}, judge={args.judge}, gen={args.gen})"
        )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(f"zhpan version {__version__}")
    print("Subcommands:")
    print("  debias   — apply per-(judge × generator) calibration to a raw score")
    print("  demo     — run the offline mock pipeline end-to-end")
    print("  version  — print version and exit")
    print()
    print("Project: https://github.com/USERNAME/zhpan")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    import subprocess

    print("[zhpan] running offline demo pipeline ...")
    cmds = [
        ["python", "-m", "zhpan.scripts.build_prompts", "--use-curated", "--out", "data/prompts/demo.jsonl"],
        ["python", "-m", "zhpan.scripts.run_generate", "--config", "configs/demo.yaml"],
        ["python", "-m", "zhpan.scripts.run_judge", "--config", "configs/demo.yaml"],
        ["python", "-m", "zhpan.scripts.analyze", "--config", "configs/demo.yaml"],
    ]
    for cmd in cmds:
        log.info(f"$ {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="zhpan",
        description="zhpan — Debias Chinese LLM-as-a-Judge in 3 lines.",
    )
    ap.add_argument("--version", action="version", version=f"zhpan {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    p_debias = sub.add_parser("debias", help="Apply per-pair calibration to a raw judge score")
    p_debias.add_argument("--judge", required=True, help="judge model name (e.g. qwen-max-judge)")
    p_debias.add_argument("--gen", required=True, help="generator model name (e.g. deepseek-chat)")
    p_debias.add_argument("--score", required=True, type=float, help="raw judge score (1-5)")
    p_debias.add_argument(
        "--calibrator",
        default="leaderboard/v0.1/calibrator.json",
        help="path to a calibrator JSON",
    )
    p_debias.add_argument("--json", action="store_true", help="emit JSON")
    p_debias.set_defaults(func=_cmd_debias)

    p_info = sub.add_parser("info", help="Show package info & available commands")
    p_info.set_defaults(func=_cmd_info)

    p_demo = sub.add_parser("demo", help="Run the offline mock pipeline (no API keys needed)")
    p_demo.set_defaults(func=_cmd_demo)

    p_version = sub.add_parser("version", help="Print version and exit")
    p_version.set_defaults(func=_cmd_version)

    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

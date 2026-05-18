"""CLI: python -m zhpan.scripts.run_judge --config configs/v0.1.yaml"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from zhpan.generate import GenerationRecord
from zhpan.judge import run_judging
from zhpan.models import ModelSpec
from zhpan.prompts import load_prompts
from zhpan.utils import Budget, DiskCache, get_logger, load_yaml_config, read_jsonl

log = get_logger("zhpan.cli.judge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    prompts = load_prompts(cfg["prompts"]["path"])
    judges = [ModelSpec.from_dict(j) for j in cfg["judges"]]

    gen_dir = Path(cfg.get("generation", {}).get("output_dir", "data/generations/v0.1"))
    gens_raw = read_jsonl(gen_dir / "generations.jsonl")
    generations = [
        GenerationRecord(
            generation_id=g["generation_id"],
            prompt_id=g["prompt_id"],
            generator=g["generator"],
            response=g["response"],
            in_tokens=g.get("in_tokens", 0),
            out_tokens=g.get("out_tokens", 0),
            error=g.get("error"),
            metadata=g.get("metadata"),
        )
        for g in gens_raw
    ]

    j_cfg = cfg.get("judging", {})
    cache = DiskCache(
        root=cfg.get("cache_root", "data/cache"),
        enabled=j_cfg.get("cache", True),
    )
    budget = Budget(cap_usd=float(cfg.get("budget", {}).get("max_usd", 50)))

    asyncio.run(
        run_judging(
            prompts=prompts,
            generations=generations,
            judges=judges,
            out_dir=Path(j_cfg.get("output_dir", "data/judgments/v0.1")),
            concurrency=int(j_cfg.get("concurrency", 5)),
            cache=cache,
            budget=budget,
        )
    )


if __name__ == "__main__":
    main()

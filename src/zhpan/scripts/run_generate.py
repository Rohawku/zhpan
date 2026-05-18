"""CLI: python -m zhpan.scripts.run_generate --config configs/v0.1.yaml"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from zhpan.generate import run_generation
from zhpan.models import ModelSpec
from zhpan.prompts import load_prompts
from zhpan.utils import Budget, DiskCache, get_logger, load_yaml_config

log = get_logger("zhpan.cli.generate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Estimate cost & exit without API calls")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    prompts = load_prompts(cfg["prompts"]["path"])
    generators = [ModelSpec.from_dict(g) for g in cfg["generators"]]
    gen_cfg = cfg.get("generation", {})

    if args.dry_run:
        n_calls = len(prompts) * len(generators)
        # rough estimate: 200 in-tokens + 400 out-tokens per call, $5/M average
        est_usd = n_calls * (200 + 400) / 1_000_000 * 5.0
        print(json.dumps(
            {
                "prompts": len(prompts),
                "generators": len(generators),
                "calls": n_calls,
                "estimated_usd": round(est_usd, 4),
            },
            indent=2,
        ))
        return

    cache = DiskCache(
        root=cfg.get("cache_root", "data/cache"),
        enabled=gen_cfg.get("cache", True),
    )
    budget = Budget(cap_usd=float(cfg.get("budget", {}).get("max_usd", 50)))

    asyncio.run(
        run_generation(
            prompts=prompts,
            generators=generators,
            out_dir=Path(gen_cfg.get("output_dir", "data/generations/v0.1")),
            concurrency=int(gen_cfg.get("concurrency", 5)),
            cache=cache,
            budget=budget,
        )
    )


if __name__ == "__main__":
    main()

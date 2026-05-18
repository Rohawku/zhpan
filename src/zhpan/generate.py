"""Generation pipeline: N prompts × M generators → data/generations/."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from .models import BaseClient, ModelSpec, make_client
from .prompts import Prompt
from .utils import Budget, DiskCache, append_jsonl, get_logger

log = get_logger("zhpan.generate")


@dataclass
class GenerationRecord:
    generation_id: str
    prompt_id: str
    generator: str
    response: str
    in_tokens: int
    out_tokens: int
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "prompt_id": self.prompt_id,
            "generator": self.generator,
            "response": self.response,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "error": self.error,
            "metadata": self.metadata or {},
        }


async def _one(
    client: BaseClient,
    prompt: Prompt,
    semaphore: asyncio.Semaphore,
) -> GenerationRecord:
    async with semaphore:
        gen_id = f"{prompt.id}__{client.spec.name}"
        try:
            result = await client.complete(
                [
                    {"role": "system", "content": "你是一位有帮助的中文 AI 助手。"},
                    {"role": "user", "content": prompt.prompt},
                ]
            )
            quality_hint = result.raw.get("quality_hint")  # mock-only; harmless otherwise
            return GenerationRecord(
                generation_id=gen_id,
                prompt_id=prompt.id,
                generator=client.spec.name,
                response=result.text,
                in_tokens=result.in_tokens,
                out_tokens=result.out_tokens,
                metadata={
                    "vendor": client.spec.vendor,
                    "model": client.spec.model,
                    "quality_hint": quality_hint,
                },
            )
        except Exception as e:
            log.warning(f"Generation failed for {gen_id}: {e}")
            return GenerationRecord(
                generation_id=gen_id,
                prompt_id=prompt.id,
                generator=client.spec.name,
                response="",
                in_tokens=0,
                out_tokens=0,
                error=str(e),
            )


async def run_generation(
    prompts: list[Prompt],
    generators: list[ModelSpec],
    out_dir: str | Path,
    *,
    concurrency: int = 5,
    cache: DiskCache | None = None,
    budget: Budget | None = None,
) -> list[GenerationRecord]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"
    if out_path.exists():
        out_path.unlink()

    semaphore = asyncio.Semaphore(concurrency)
    clients = [make_client(g, cache=cache, budget=budget) for g in generators]

    tasks = [_one(client, p, semaphore) for p in prompts for client in clients]
    log.info(f"Submitting {len(tasks)} generation tasks ({len(prompts)} × {len(generators)})")

    results: list[GenerationRecord] = await tqdm_asyncio.gather(*tasks, desc="generate")
    for r in results:
        append_jsonl(out_path, r.to_dict())
    log.info(f"Wrote {len(results)} generations → {out_path}")
    if budget is not None:
        log.info(budget.report())
    return results

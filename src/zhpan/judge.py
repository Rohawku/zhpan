"""Judge pipeline: score generations with one or more judge models on a 1-5 rubric."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from .generate import GenerationRecord
from .models import BaseClient, ModelSpec, make_client
from .prompts import Prompt
from .utils import Budget, DiskCache, append_jsonl, get_logger

log = get_logger("zhpan.judge")


_RUBRIC_V0_1 = """\
你是一位严谨的评测员，需要对 AI 助手针对用户提问给出的回答进行质量打分。

请按 1-5 分整数打一个综合质量分：
  1 = 无效（错误、跑题或有害）
  2 = 较差（存在明显错误或严重遗漏）
  3 = 一般（大致正确但有瑕疵）
  4 = 良好（准确、清晰、有帮助，仅有小问题）
  5 = 优秀（准确、清晰、有洞见、结构良好）

判分维度：正确性、有用性、清晰度、完整性，以及（如适用）推理质量。
不要因回答较长或较短而加减分，除非长度本身严重影响了质量。

请严格按以下两行格式输出，不要写其他内容：
评分理由：<一句话简要说明>
评分：<1-5 的整数>
"""


_SCORE_REGEX = re.compile(r"(?:score|评分)\s*[:：]\s*([1-5])", re.IGNORECASE)


@dataclass
class JudgmentRecord:
    judgment_id: str
    generation_id: str
    prompt_id: str
    generator: str
    judge: str
    score: int | None
    reasoning: str
    raw_text: str
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgment_id": self.judgment_id,
            "generation_id": self.generation_id,
            "prompt_id": self.prompt_id,
            "generator": self.generator,
            "judge": self.judge,
            "score": self.score,
            "reasoning": self.reasoning,
            "raw_text": self.raw_text,
            "error": self.error,
            "metadata": self.metadata or {},
        }


def _parse_score(text: str) -> tuple[int | None, str]:
    m = _SCORE_REGEX.search(text)
    score = int(m.group(1)) if m else None
    reasoning = ""
    for line in text.splitlines():
        low = line.lower().lstrip()
        if low.startswith("reasoning") or line.lstrip().startswith("评分理由"):
            reasoning = re.split(r"[:：]", line, maxsplit=1)[-1].strip()
            break
    if not reasoning:
        reasoning = text.strip().splitlines()[0][:200] if text.strip() else ""
    return score, reasoning


def _build_judge_messages(
    prompt: Prompt, generation: GenerationRecord, *, mock_meta: bool
) -> list[dict[str, str]]:
    user_block = (
        f"### 用户提问\n{prompt.prompt}\n\n"
        f"### AI 回答\n{generation.response}\n\n"
        "请按上述评分标准对该回答打分。"
    )
    if mock_meta:
        # Embed quality hint so the MockClient can simulate per-pair bias.
        # Real APIs will ignore this line as benign context.
        q = (generation.metadata or {}).get("quality_hint")
        meta_line = f"[BENCH_META] gen={generation.generator}"
        if q is not None:
            meta_line += f" q={q}"
        user_block = meta_line + "\n" + user_block
    return [
        {"role": "system", "content": _RUBRIC_V0_1},
        {"role": "user", "content": user_block},
    ]


async def _one_judgment(
    judge_client: BaseClient,
    prompt: Prompt,
    generation: GenerationRecord,
    semaphore: asyncio.Semaphore,
    mock_meta: bool,
) -> JudgmentRecord:
    async with semaphore:
        jid = f"{generation.generation_id}__judged_by__{judge_client.spec.name}"
        if generation.error or not generation.response.strip():
            return JudgmentRecord(
                judgment_id=jid,
                generation_id=generation.generation_id,
                prompt_id=prompt.id,
                generator=generation.generator,
                judge=judge_client.spec.name,
                score=None,
                reasoning="",
                raw_text="",
                error="generation_missing",
            )
        try:
            messages = _build_judge_messages(prompt, generation, mock_meta=mock_meta)
            result = await judge_client.complete(
                messages, temperature=0.0, max_tokens=200
            )
            score, reasoning = _parse_score(result.text)
            return JudgmentRecord(
                judgment_id=jid,
                generation_id=generation.generation_id,
                prompt_id=prompt.id,
                generator=generation.generator,
                judge=judge_client.spec.name,
                score=score,
                reasoning=reasoning,
                raw_text=result.text,
                metadata={
                    "vendor": judge_client.spec.vendor,
                    "model": judge_client.spec.model,
                },
            )
        except Exception as e:
            log.warning(f"Judge failed for {jid}: {e}")
            return JudgmentRecord(
                judgment_id=jid,
                generation_id=generation.generation_id,
                prompt_id=prompt.id,
                generator=generation.generator,
                judge=judge_client.spec.name,
                score=None,
                reasoning="",
                raw_text="",
                error=str(e),
            )


async def run_judging(
    prompts: list[Prompt],
    generations: list[GenerationRecord],
    judges: list[ModelSpec],
    out_dir: str | Path,
    *,
    concurrency: int = 5,
    cache: DiskCache | None = None,
    budget: Budget | None = None,
) -> list[JudgmentRecord]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judgments.jsonl"
    if out_path.exists():
        out_path.unlink()

    prompts_by_id = {p.id: p for p in prompts}
    mock_meta = any(j.vendor == "mock" for j in judges)

    semaphore = asyncio.Semaphore(concurrency)
    judge_clients = [make_client(j, cache=cache, budget=budget) for j in judges]

    tasks = []
    for gen in generations:
        if gen.prompt_id not in prompts_by_id:
            continue
        for jc in judge_clients:
            tasks.append(
                _one_judgment(jc, prompts_by_id[gen.prompt_id], gen, semaphore, mock_meta)
            )

    log.info(f"Submitting {len(tasks)} judge tasks ({len(generations)} × {len(judges)})")
    results: list[JudgmentRecord] = await tqdm_asyncio.gather(*tasks, desc="judge")
    for r in results:
        append_jsonl(out_path, r.to_dict())
    log.info(f"Wrote {len(results)} judgments → {out_path}")
    if budget is not None:
        log.info(budget.report())
    return results

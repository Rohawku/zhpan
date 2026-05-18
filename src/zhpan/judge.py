"""Judge pipeline: score generations with one or more judge models.

v0.2: 1-10 overall score, 7-dimension breakdown to combat ceiling effect.
"""

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


_RUBRIC_V0_2 = """\
你是一位严谨的评测员，需要对 AI 助手针对用户提问给出的回答进行质量打分。

请先在 7 个分项维度上各打 1-10 分，再综合给出 1-10 的整数总分。

7 个分项维度（各 1-10，先 1 行）：
  D1 正确性：事实/计算/逻辑是否正确
  D2 推理质量：步骤是否清晰、推导是否站得住
  D3 完整性：是否覆盖了题目要求的所有要点
  D4 切题度：是否真正回答了问题、避免跑题或转移话题
  D5 清晰度：表达是否易读、结构是否合理
  D6 深度：是否给出有洞见的解释，而非套话
  D7 安全/伦理：是否符合中文社会语境的安全与伦理要求

为了拉开分辨率，请使用 1-10 的全部区间：
  9-10 = 接近无可挑剔
  7-8  = 良好但有可指出的小问题
  5-6  = 中等，明显有可改进之处
  3-4  = 较差，有重要错误或缺陷
  1-2  = 很差或基本无效

严格按下面 9 行格式输出，每行一个数字或一句话，不要写多余内容：
D1: <1-10>
D2: <1-10>
D3: <1-10>
D4: <1-10>
D5: <1-10>
D6: <1-10>
D7: <1-10>
评分理由：<一句话简要说明>
评分：<1-10 的整数总分>
"""


_SCORE_REGEX = re.compile(r"(?:score|评分)\s*[:：]\s*(\d{1,2})", re.IGNORECASE)
_DIM_REGEX = re.compile(r"^\s*D([1-7])\s*[:：]\s*(\d{1,2})", re.MULTILINE | re.IGNORECASE)


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


def _parse_score(text: str) -> tuple[int | None, str, dict[str, int]]:
    """Returns (overall_1to10, reasoning, dim_scores). dim_scores: {D1..D7: int}."""
    m = _SCORE_REGEX.search(text)
    score = int(m.group(1)) if m else None
    if score is not None and not (1 <= score <= 10):
        score = None
    dims: dict[str, int] = {}
    for dim_match in _DIM_REGEX.finditer(text):
        idx = int(dim_match.group(1))
        val = int(dim_match.group(2))
        if 1 <= val <= 10:
            dims[f"D{idx}"] = val
    reasoning = ""
    for line in text.splitlines():
        low = line.lower().lstrip()
        if low.startswith("reasoning") or line.lstrip().startswith("评分理由"):
            reasoning = re.split(r"[:：]", line, maxsplit=1)[-1].strip()
            break
    if not reasoning:
        reasoning = text.strip().splitlines()[0][:200] if text.strip() else ""
    return score, reasoning, dims


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
        {"role": "system", "content": _RUBRIC_V0_2},
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
                messages, temperature=0.0, max_tokens=400
            )
            score, reasoning, dims = _parse_score(result.text)
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
                    "dim_scores": dims,
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

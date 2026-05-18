"""Basic tests — pure-Python, no API calls. Run: make test"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zhpan.calibrate import Calibrator, cv_eval, fit_from_judgments
from zhpan.compute_bias import build_gold_silver, compute_bias
from zhpan.generate import run_generation
from zhpan.judge import run_judging
from zhpan.models import ModelSpec
from zhpan.prompts import Prompt, load_prompts
from zhpan.utils import write_jsonl


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────


def _demo_prompts() -> list[Prompt]:
    return [
        Prompt(id=f"t-{i}", category="reasoning", source="test", prompt=f"prompt {i}?", metadata={})
        for i in range(6)
    ]


def _mock_specs() -> tuple[list[ModelSpec], list[ModelSpec]]:
    generators = [
        ModelSpec(name=n, vendor="mock", model="mock")
        for n in ("mock-qwen", "mock-deepseek", "mock-glm")
    ]
    judges = [
        ModelSpec(name=n, vendor="mock", model="mock")
        for n in ("mock-judge-qwen", "mock-judge-deepseek", "mock-judge-glm")
    ]
    return generators, judges


# ──────────────────────────────────────────
# Tests
# ──────────────────────────────────────────


def test_prompts_roundtrip(tmp_path: Path) -> None:
    rows = [
        {"id": "x-1", "category": "reasoning", "source": "test", "prompt": "what?", "metadata": {}},
        {"id": "x-2", "category": "coding", "source": "test", "prompt": "code?", "metadata": {}},
    ]
    p = tmp_path / "p.jsonl"
    write_jsonl(p, rows)
    loaded = load_prompts(p)
    assert len(loaded) == 2
    assert loaded[0].id == "x-1"


def test_calibrator_save_load(tmp_path: Path) -> None:
    offsets = {"judge_a": {"gen_x": 1.5, "gen_y": -0.5}}
    cal = Calibrator(offsets=offsets, version="v0.1", method="per_pair_offset")
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = Calibrator.from_file(path)
    assert loaded.offsets["judge_a"]["gen_x"] == 1.5
    assert loaded.correct("judge_a", "gen_x", raw_score=4.0) == 2.5  # 4 - 1.5
    # Clipping to [1, 5]
    assert loaded.correct("judge_a", "gen_x", raw_score=0.5) == 1.0
    assert loaded.correct("unknown", "unknown", raw_score=3.5) == 3.5


def test_end_to_end_mock_pipeline(tmp_path: Path) -> None:
    prompts = _demo_prompts()
    generators, judges = _mock_specs()

    # 1. Generate
    gens = asyncio.run(
        run_generation(
            prompts=prompts,
            generators=generators,
            out_dir=tmp_path / "gens",
            concurrency=4,
        )
    )
    assert len(gens) == len(prompts) * len(generators)
    assert all(g.response for g in gens)

    # 2. Judge
    judgments = asyncio.run(
        run_judging(
            prompts=prompts,
            generations=gens,
            judges=judges,
            out_dir=tmp_path / "judgments",
            concurrency=4,
        )
    )
    assert len(judgments) == len(gens) * len(judges)
    judgment_dicts = [j.to_dict() for j in judgments]
    assert sum(1 for j in judgment_dicts if j["score"] is not None) > 0

    # 3. Build silver gold + bias matrix
    gold = build_gold_silver(judgment_dicts, min_n_judges=2, max_std=2.0)
    assert len(gold) > 0
    bm = compute_bias(judgment_dicts, gold)
    assert set(bm.judges) == {j.name for j in judges}
    assert set(bm.generators) == {g.name for g in generators}

    # 4. Fit calibrator + check non-trivial bias
    cal, bm2 = fit_from_judgments(judgment_dicts)
    assert any(
        abs(cal.offsets[jn][gn]) > 0.1
        for jn in bm2.judges
        for gn in bm2.generators
    ), "Expected some non-trivial per-pair bias from mock vendors"

    # 5. CV improves MAE on held out
    cv = cv_eval(judgment_dicts, n_folds=3)
    s = cv["summary"]
    if s["n_folds"] >= 2:
        # Calibration should not make things much worse.
        assert s["mae_after"] <= s["mae_before"] + 0.5


def test_self_pref_lift_metric() -> None:
    # synthetic: qwen-judge gives +1 to qwen-gen, -1 to deepseek-gen
    judgments = []
    for i in range(10):
        gen_id = f"p-{i}__qwen-gen"
        judgments.append(
            {
                "judgment_id": f"{gen_id}__judged_by__qwen-judge",
                "generation_id": gen_id,
                "prompt_id": f"p-{i}",
                "generator": "qwen-gen",
                "judge": "qwen-judge",
                "score": 5,
            }
        )
        gen_id2 = f"p-{i}__deepseek-gen"
        judgments.append(
            {
                "judgment_id": f"{gen_id2}__judged_by__qwen-judge",
                "generation_id": gen_id2,
                "prompt_id": f"p-{i}",
                "generator": "deepseek-gen",
                "judge": "qwen-judge",
                "score": 3,
            }
        )

    gold = {j["generation_id"]: 4.0 for j in judgments}
    bm = compute_bias(judgments, gold)
    # qwen-judge -> qwen-gen +1, qwen-judge -> deepseek-gen -1 → lift = +2
    assert bm.self_pref_lift["qwen-judge"] == pytest.approx(2.0, abs=0.01)

"""Model wrappers — unified interface for generator and judge calls across vendors.

Vendors supported (v0.1 focus = Chinese frontier models):
- dashscope   (阿里 Qwen — DashScope OpenAI-compatible endpoint)
- deepseek    (DeepSeek — chat + reasoner)
- zhipu       (智谱 GLM-4 — OpenAI-compatible)
- doubao      (字节豆包 — Volcengine Ark, OpenAI-compatible)
- openai      (OpenAI — cross-lingual control)
- anthropic   (Anthropic — cross-lingual control)
- mock        (offline canned responses for tests / demo)
"""

from __future__ import annotations

import asyncio
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .utils import Budget, DiskCache, get_logger

log = get_logger("zhpan.models")


# ──────────────────────────────────────────
# Cost table (USD per 1M tokens) — approximate
# ──────────────────────────────────────────

_COST_PER_M_TOK = {
    # Chinese frontier
    "qwen-max": {"in": 2.80, "out": 11.20},
    "qwen-plus": {"in": 0.11, "out": 0.45},
    "qwen-turbo": {"in": 0.04, "out": 0.16},
    "deepseek-chat": {"in": 0.27, "out": 1.10},
    "deepseek-reasoner": {"in": 0.55, "out": 2.19},
    "glm-4-plus": {"in": 7.00, "out": 7.00},
    "glm-4-air": {"in": 0.07, "out": 0.07},
    "glm-4": {"in": 1.40, "out": 1.40},
    "doubao-pro-32k": {"in": 0.11, "out": 0.28},
    "doubao-lite-32k": {"in": 0.04, "out": 0.08},
    # Moonshot Kimi (used as anchor judge)
    "moonshot-v1-8k": {"in": 1.65, "out": 1.65},
    "moonshot-v1-32k": {"in": 3.30, "out": 3.30},
    "moonshot-v1-128k": {"in": 8.30, "out": 8.30},
    "kimi-latest": {"in": 0.275, "out": 1.65},
    # Cross-lingual control
    "gpt-4o": {"in": 2.5, "out": 10.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.6},
    "claude-3-5-sonnet": {"in": 3.0, "out": 15.0},
    "claude-3-5-haiku": {"in": 0.8, "out": 4.0},
    # Mock — free
    "mock": {"in": 0.0, "out": 0.0},
}


def _estimate_cost(model_key: str, in_tokens: int, out_tokens: int) -> float:
    # Mock vendors are always free
    if model_key.startswith("mock"):
        return 0.0
    # Strip "-judge" suffix so "qwen-max-judge" inherits qwen-max pricing.
    base_key = model_key[:-6] if model_key.endswith("-judge") else model_key
    base = _COST_PER_M_TOK.get(base_key, {"in": 1.0, "out": 1.0})
    return (in_tokens * base["in"] + out_tokens * base["out"]) / 1_000_000


# ──────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────


@dataclass
class ModelSpec:
    name: str
    vendor: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(
            name=d["name"],
            vendor=d["vendor"],
            model=d["model"],
            params=d.get("params", {}),
        )


@dataclass
class CompletionResult:
    text: str
    in_tokens: int
    out_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────
# Base client
# ──────────────────────────────────────────


class BaseClient(ABC):
    def __init__(
        self,
        spec: ModelSpec,
        cache: DiskCache | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.spec = spec
        self.cache = cache
        self.budget = budget

    async def complete(self, messages: list[dict[str, str]], **overrides: Any) -> CompletionResult:
        params = {**self.spec.params, **overrides}
        cache_key = {
            "vendor": self.spec.vendor,
            "model": self.spec.model,
            "messages": messages,
            "params": params,
        }
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return CompletionResult(
                    text=hit["text"],
                    in_tokens=hit.get("in_tokens", 0),
                    out_tokens=hit.get("out_tokens", 0),
                    raw=hit.get("raw", {}),
                )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1.5, min=1, max=30),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
            reraise=True,
        ):
            with attempt:
                result = await self._call(messages, params)

        if self.budget is not None:
            self.budget.charge(
                label=f"{self.spec.vendor}/{self.spec.name}",
                usd=_estimate_cost(self.spec.name, result.in_tokens, result.out_tokens),
            )

        if self.cache is not None:
            self.cache.put(
                cache_key,
                {
                    "text": result.text,
                    "in_tokens": result.in_tokens,
                    "out_tokens": result.out_tokens,
                    "raw": result.raw,
                },
            )
        return result

    @abstractmethod
    async def _call(
        self, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> CompletionResult: ...


# ──────────────────────────────────────────
# OpenAI-compatible shared base
# ──────────────────────────────────────────


class _OpenAICompatibleClient(BaseClient):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"

    async def _call(
        self, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> CompletionResult:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env var {self.api_key_env}")
        body = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
        }
        if "top_p" in params:
            body["top_p"] = params["top_p"]
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return CompletionResult(
            text=text,
            in_tokens=usage.get("prompt_tokens", 0),
            out_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )


class OpenAIClient(_OpenAICompatibleClient):
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"


class DeepSeekClient(_OpenAICompatibleClient):
    base_url = "https://api.deepseek.com"
    api_key_env = "DEEPSEEK_API_KEY"


class DashScopeClient(_OpenAICompatibleClient):
    """阿里 Qwen via DashScope OpenAI-compatible endpoint."""

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"


class ZhipuClient(_OpenAICompatibleClient):
    """智谱 GLM-4 series, OpenAI-compatible."""

    base_url = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env = "ZHIPU_API_KEY"


class DoubaoClient(_OpenAICompatibleClient):
    """字节豆包 via Volcengine Ark OpenAI-compatible endpoint."""

    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    api_key_env = "ARK_API_KEY"


class MoonshotClient(_OpenAICompatibleClient):
    """Moonshot Kimi — used as independent anchor judge in zhpan v0.2+."""

    base_url = "https://api.moonshot.cn/v1"
    api_key_env = "MOONSHOT_API_KEY"


class TogetherClient(_OpenAICompatibleClient):
    base_url = "https://api.together.xyz/v1"
    api_key_env = "TOGETHER_API_KEY"


# ──────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────


class AnthropicClient(BaseClient):
    async def _call(
        self, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> CompletionResult:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing env var ANTHROPIC_API_KEY")

        system_text = ""
        user_msgs: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_msgs.append(m)

        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": user_msgs,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
        }
        if system_text:
            body["system"] = system_text

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return CompletionResult(
            text=text,
            in_tokens=usage.get("input_tokens", 0),
            out_tokens=usage.get("output_tokens", 0),
            raw=data,
        )


# ──────────────────────────────────────────
# Mock — offline tests + demo
# ──────────────────────────────────────────


class MockClient(BaseClient):
    """Deterministic-ish canned responses keyed by model + prompt hash.

    Per-model 'style' gives a baseline quality; per-pair bias injected by judges.
    """

    _STYLE = {
        # name -> (base_quality, verbosity, voice)
        "mock-qwen": (4.1, 1.1, "让我一步步来分析这个问题。"),
        "mock-deepseek": (4.0, 1.2, "我会从基本原理出发推理。"),
        "mock-glm": (3.8, 1.0, "好的，我来回答。"),
        "mock-doubao": (3.6, 0.9, "简单来说，"),
        "mock-judge-qwen": (0.0, 0.0, "judge"),
        "mock-judge-deepseek": (0.0, 0.0, "judge"),
        "mock-judge-glm": (0.0, 0.0, "judge"),
    }

    _JUDGE_BIAS: dict[str, dict[str, float]] = {
        "mock-judge-qwen": {
            "mock-qwen": +0.6,
            "mock-deepseek": -0.2,
            "mock-glm": -0.3,
            "mock-doubao": -0.1,
        },
        "mock-judge-deepseek": {
            "mock-qwen": -0.3,
            "mock-deepseek": +0.5,
            "mock-glm": -0.4,
            "mock-doubao": -0.2,
        },
        "mock-judge-glm": {
            "mock-qwen": -0.2,
            "mock-deepseek": -0.3,
            "mock-glm": +0.7,
            "mock-doubao": -0.4,
        },
    }

    async def _call(
        self, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> CompletionResult:
        await asyncio.sleep(0)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        seed = abs(hash((self.spec.name, last_user))) % (2**31)
        rng = random.Random(seed)
        if self.spec.name.startswith("mock-judge-"):
            return self._judge_response(messages, rng)
        return self._generator_response(last_user, rng)

    def _generator_response(self, prompt: str, rng: random.Random) -> CompletionResult:
        base_q, verb, voice = self._STYLE.get(self.spec.name, (3.5, 1.0, "回答："))
        n_sentences = max(1, int(rng.gauss(3 * verb, 1)))
        body = "".join(
            [voice]
            + [f"第{i+1}段：针对该问题给出要点。" for i in range(n_sentences)]
        )
        return CompletionResult(
            text=body,
            in_tokens=len(prompt) // 4,
            out_tokens=len(body) // 4,
            raw={"mock": True, "quality_hint": base_q},
        )

    def _judge_response(
        self, messages: list[dict[str, str]], rng: random.Random
    ) -> CompletionResult:
        prompt_text = "\n".join(m["content"] for m in messages)
        gen_name = ""
        quality_hint = 3.5
        for line in prompt_text.splitlines():
            if line.startswith("[BENCH_META]"):
                for tok in line.replace("[BENCH_META]", "").strip().split():
                    if tok.startswith("gen="):
                        gen_name = tok[4:]
                    elif tok.startswith("q="):
                        try:
                            quality_hint = float(tok[2:])
                        except ValueError:
                            pass
                break
        bias = self._JUDGE_BIAS.get(self.spec.name, {}).get(gen_name, 0.0)
        noise = rng.gauss(0, 0.3)
        score = max(1.0, min(5.0, quality_hint + bias + noise))
        text = (
            f"评分理由：回答基本切题，逻辑清晰，质量尚可。\n"
            f"评分：{round(score)}"
        )
        return CompletionResult(
            text=text,
            in_tokens=80,
            out_tokens=20,
            raw={"mock": True, "raw_score": score, "bias_applied": bias},
        )


# ──────────────────────────────────────────
# Factory
# ──────────────────────────────────────────


_REGISTRY: dict[str, type[BaseClient]] = {
    "dashscope": DashScopeClient,
    "deepseek": DeepSeekClient,
    "zhipu": ZhipuClient,
    "doubao": DoubaoClient,
    "moonshot": MoonshotClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "together": TogetherClient,
    "mock": MockClient,
}


def make_client(
    spec: ModelSpec,
    cache: DiskCache | None = None,
    budget: Budget | None = None,
) -> BaseClient:
    if spec.vendor not in _REGISTRY:
        raise ValueError(f"Unknown vendor '{spec.vendor}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[spec.vendor](spec=spec, cache=cache, budget=budget)

"""Verify each configured vendor responds with a 1-prompt smoke test.

Cost: each call < ¥0.001. Total: ~¥0.005. Used to confirm API keys + model names
before running the full benchmark.
"""

import asyncio

from zhpan.models import ModelSpec, make_client


PROBES: list[ModelSpec] = [
    ModelSpec(name="qwen-max", vendor="dashscope", model="qwen-max-2025-01-25",
              params={"temperature": 0.0, "max_tokens": 30}),
    ModelSpec(name="deepseek-chat", vendor="deepseek", model="deepseek-chat",
              params={"temperature": 0.0, "max_tokens": 30}),
    ModelSpec(name="glm-4-plus", vendor="zhipu", model="glm-4-plus",
              params={"temperature": 0.0, "max_tokens": 30}),
    ModelSpec(name="doubao-pro-32k", vendor="doubao", model="doubao-1-5-pro-32k-250115",
              params={"temperature": 0.0, "max_tokens": 30}),
]


async def probe_one(spec: ModelSpec) -> dict:
    client = make_client(spec)
    try:
        result = await client.complete(
            [{"role": "user", "content": "请用一句话回答：你好。"}],
        )
        return {
            "spec": spec.name,
            "vendor": spec.vendor,
            "model": spec.model,
            "ok": True,
            "preview": result.text[:80].replace("\n", " "),
            "in_tokens": result.in_tokens,
            "out_tokens": result.out_tokens,
        }
    except Exception as e:
        return {
            "spec": spec.name,
            "vendor": spec.vendor,
            "model": spec.model,
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


async def main() -> None:
    print("Probing each vendor with a single tiny call …\n")
    results = await asyncio.gather(*[probe_one(s) for s in PROBES])
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        print(f"{mark}  {r['spec']:<18} vendor={r['vendor']:<10} model={r['model']}")
        if r["ok"]:
            print(f"     in_tok={r['in_tokens']} out_tok={r['out_tokens']}")
            print(f"     preview: {r['preview']!r}\n")
        else:
            print(f"     ERROR: {r['error']}\n")


if __name__ == "__main__":
    asyncio.run(main())

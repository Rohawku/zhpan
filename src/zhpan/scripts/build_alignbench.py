"""Build the v0.3 prompt set from THUDM/AlignBench (v1.1 release, 683 prompts).

Samples N prompts (default 150) balanced across the 8 AlignBench categories,
maps AlignBench categories to zhpan's CATEGORIES, and preserves the original
category + reference answer + supporting evidence URLs in metadata.

Source license: AlignBench is released under Apache-2.0
(https://github.com/THUDM/AlignBench).

Usage:
    python -m zhpan.scripts.build_alignbench --out data/prompts/v0.3.jsonl --n 150
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

from zhpan.utils import get_logger, write_jsonl

log = get_logger("zhpan.build_alignbench")


# AlignBench Chinese category → zhpan CATEGORIES
_CAT_MAP = {
    "专业能力": "general_qa",
    "角色扮演": "writing",
    "数学计算": "math",
    "逻辑推理": "reasoning",
    "文本写作": "writing",
    "基本任务": "extraction",
    "中文理解": "multilingual",
    "综合问答": "general_qa",
}


def _fetch_alignbench() -> list[dict]:
    """Resolve and download data_v1.1_release.jsonl via GitHub API."""
    with httpx.Client(timeout=30, follow_redirects=True) as c:
        meta = c.get(
            "https://api.github.com/repos/THUDM/AlignBench/contents/data/data_v1.1_release.jsonl",
            headers={"User-Agent": "zhpan-build-alignbench"},
        )
        meta.raise_for_status()
        download_url = meta.json()["download_url"]
        log.info(f"Downloading AlignBench v1.1 from {download_url}")
        r = c.get(download_url, headers={"User-Agent": "zhpan-build-alignbench"})
        r.raise_for_status()
    rows = [json.loads(line) for line in r.text.strip().split("\n")]
    log.info(f"Fetched {len(rows)} AlignBench prompts")
    return rows


def _balanced_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cat = r.get("category", "其他")
        by_cat[cat].append(r)

    cats = sorted(by_cat.keys())
    base = n // len(cats)
    remainder = n - base * len(cats)

    chosen: list[dict] = []
    for i, cat in enumerate(cats):
        target = base + (1 if i < remainder else 0)
        items = by_cat[cat][:]
        rng.shuffle(items)
        chosen.extend(items[:target])
    return chosen


def build(out_path: Path, n: int, seed: int) -> int:
    raw = _fetch_alignbench()
    sampled = _balanced_sample(raw, n=n, seed=seed)
    out_rows: list[dict] = []
    for r in sampled:
        cn_cat = r.get("category", "其他")
        zhpan_cat = _CAT_MAP.get(cn_cat, "general_qa")
        qid = r.get("question_id")
        out_rows.append(
            {
                "id": f"alignbench-{qid}",
                "category": zhpan_cat,
                "source": "alignbench-v1.1",
                "prompt": r["question"],
                "metadata": {
                    "lang": "zh",
                    "alignbench_category": cn_cat,
                    "alignbench_subcategory": r.get("subcategory"),
                    "reference": r.get("reference"),
                    "evidences": r.get("evidences"),
                    "alignbench_question_id": qid,
                },
            }
        )
    write_jsonl(out_path, out_rows)
    return len(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build zhpan v0.3 prompt set from AlignBench")
    ap.add_argument("--out", default="data/prompts/v0.3.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = build(Path(args.out), n=args.n, seed=args.seed)
    log.info(f"Wrote {n} prompts to {args.out}")
    print(json.dumps({"out": args.out, "n": n}, indent=2))


if __name__ == "__main__":
    main()

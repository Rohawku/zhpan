"""Prompt loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import read_jsonl

CATEGORIES = (
    "reasoning",
    "coding",
    "writing",
    "math",
    "extraction",
    "safety",
    "multilingual",
    "general_qa",
)


@dataclass
class Prompt:
    id: str
    category: str
    source: str
    prompt: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prompt:
        return cls(
            id=d["id"],
            category=d["category"],
            source=d.get("source", "unknown"),
            prompt=d["prompt"],
            metadata=d.get("metadata", {}),
        )


def load_prompts(path: str | Path) -> list[Prompt]:
    rows = read_jsonl(path)
    out: list[Prompt] = []
    seen_ids: set[str] = set()
    for r in rows:
        p = Prompt.from_dict(r)
        if p.id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {p.id}")
        if p.category not in CATEGORIES:
            raise ValueError(
                f"Unknown category '{p.category}' in {p.id}. Allowed: {CATEGORIES}"
            )
        seen_ids.add(p.id)
        out.append(p)
    return out

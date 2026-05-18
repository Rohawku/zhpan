"""Shared utilities: config loading, logging, cache, budget tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────

_LOG_LEVEL = os.environ.get("ZHPAN_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(h)
    logger.setLevel(_LOG_LEVEL)
    return logger


log = get_logger("zhpan")


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────
# Cache (disk-based, content-addressed)
# ──────────────────────────────────────────


@dataclass
class DiskCache:
    """SHA256(payload) → JSON file. Avoids re-paying for identical API calls."""

    root: Path
    enabled: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key[:2]}" / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        p = self._path(self._key(payload))
        if not p.exists():
            return None
        try:
            with p.open() as f:
                return json.load(f)
        except Exception:
            return None

    def put(self, payload: dict[str, Any], value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        p = self._path(self._key(payload))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(value, f, ensure_ascii=False)


# ──────────────────────────────────────────
# Budget guard
# ──────────────────────────────────────────


@dataclass
class Budget:
    """Track cumulative USD spend across a run; abort if over cap."""

    cap_usd: float = 50.0
    spent_usd: float = 0.0
    per_call: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Budget:
        cap = float(os.environ.get("ZHPAN_BUDGET_USD", "30"))
        return cls(cap_usd=cap)

    def charge(self, label: str, usd: float) -> None:
        self.spent_usd += usd
        self.per_call.append({"label": label, "usd": usd})
        if self.spent_usd > self.cap_usd:
            raise BudgetExceeded(
                f"Budget exceeded: spent ${self.spent_usd:.4f} > cap ${self.cap_usd:.2f}"
            )

    def report(self) -> str:
        return f"Spent ${self.spent_usd:.4f} / cap ${self.cap_usd:.2f} ({len(self.per_call)} calls)"


class BudgetExceeded(RuntimeError):
    pass


# ──────────────────────────────────────────
# JSONL helpers
# ──────────────────────────────────────────


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# data/gold/

This directory contains gold-standard annotations used to measure judge bias.

## Files (planned)

- `v0.1_silver.jsonl` — strong-consensus silver gold (3-judge agreement, filtered)
- `v0.1_human.jsonl` — author-annotated subset (n=20-100)

## Format

```json
{
  "generation_id": "v0.1-001-claude-3-5-sonnet",
  "prompt_id": "v0.1-001",
  "generator": "claude-3-5-sonnet",
  "gold_score": 4.5,
  "gold_source": "silver_consensus" | "human",
  "annotator": "auto" | "author",
  "metadata": {"agreement_std": 0.3}
}
```

## How gold is constructed

See [docs/methodology.md](../../docs/methodology.md) Section 5.

## Reproducibility

- Silver gold: deterministic, regenerable from `data/judgments/`
- Human gold: shipped as-is in git

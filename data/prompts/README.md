# data/prompts/

This directory contains versioned prompt sets used by the benchmark.

## Files (planned)

- `v0.1.jsonl` — 100 prompts, balanced across 8 categories. **Sources**: MT-Bench + AlpacaEval.

## Format

Each line is a JSON object:

```json
{
  "id": "v0.1-001",
  "category": "reasoning",
  "source": "mt-bench",
  "source_id": "...",
  "prompt": "...",
  "metadata": {"difficulty": "medium", "lang": "en"}
}
```

## Licensing

- MT-Bench: Apache 2.0 (see lmsys/MT-Bench)
- AlpacaEval: Apache 2.0 (see tatsu-lab/alpaca_eval)

Both allow redistribution with attribution. We include only the prompt text (no scoring), and cite source in every record.

## How to regenerate

```bash
python -m zhpan.scripts.build_prompts --source mt-bench,alpaca-eval --n 100 --out data/prompts/v0.1.jsonl
```

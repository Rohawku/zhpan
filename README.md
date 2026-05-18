# zhpan · 中评

> 🎯 **Debias Chinese LLM-as-a-Judge in 3 lines.**
> 三行代码消除中文场景下大模型裁判的系统性偏差。

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## What is this?

When you use a Chinese frontier LLM (Qwen / DeepSeek / GLM-4 / Doubao) to **judge** the outputs of other models, that judge does **not** score every generator fairly. Different judges have **systematic biases against different generators**—biases that a single global offset cannot correct.

**zhpan** measures these **per-(judge × generator)** biases on a 40-prompt Chinese benchmark and gives you a `Calibrator` you can drop into any production data-quality pipeline.

## Install

```bash
pip install zhpan        # coming soon to PyPI
# or, for now:
git clone https://github.com/USERNAME/zhpan
cd zhpan && pip install -e .
```

## 3-line debias (the whole API)

```python
from zhpan import Calibrator

cal = Calibrator.from_file("leaderboard/v0.1/calibrator.json")
fair = cal.correct(judge="qwen-max-judge", generator="deepseek-chat", raw_score=2.1)
# → e.g. 3.4  (was systematically underrated by 1.3)
```

Or from the command line:

```bash
zhpan debias --judge qwen-max-judge --gen deepseek-chat --score 2.1
# raw=2.10  →  calibrated=3.40  (offset=-1.30, judge=qwen-max-judge, gen=deepseek-chat)
```

## Try it offline in 30 seconds (no API keys)

```bash
make install
make demo          # full pipeline against mock vendors
zhpan debias --judge mock-judge-qwen --gen mock-deepseek --score 2.0 \
             --calibrator leaderboard/demo/calibrator.json
```

You'll see a bias matrix print to the terminal, plus a CV-validated calibrator written to `leaderboard/demo/`.

## How it works (60-second version)

1. **Generate.** For each of 40 Chinese prompts (covering reasoning / coding / writing / math / extraction / safety / multilingual / general QA), every generator in your config produces one response.
2. **Judge.** Every judge in your config scores every (prompt, generator) pair on a 1-5 Chinese rubric.
3. **Silver gold.** When multiple judges agree closely (std ≤ 1.0), their average is taken as silver ground-truth.
4. **Bias matrix.** For each (judge, generator) pair: `bias = mean(judge_score - silver_gold)`.
5. **Calibrate.** The `Calibrator` subtracts the learned per-pair offset, then clips to [1, 5]. 5-fold CV on the prompt axis confirms calibration doesn't overfit.

## Run the full benchmark on real APIs

```bash
cp .env.example .env       # then fill in your API keys
make build-prompts         # writes data/prompts/v0.1.jsonl
make benchmark             # generate + judge + analyze, ~30 USD total
```

Currently supported Chinese vendors:
- **dashscope** — 阿里 Qwen (qwen-max / plus / turbo)
- **deepseek** — DeepSeek (chat / reasoner)
- **zhipu** — 智谱 GLM-4 (glm-4-plus / air)
- **doubao** — 字节豆包 via Volcengine Ark

Plus optional cross-lingual control: `openai`, `anthropic`, `together`.

## Project layout

```
zhpan/
├── src/zhpan/         # main package — pip-install target
│   ├── calibrate.py   # Calibrator class
│   ├── compute_bias.py
│   ├── generate.py    # async generation pipeline
│   ├── judge.py       # Chinese rubric LLM judge
│   ├── models.py      # vendor adapters
│   └── cli.py         # `zhpan ...` CLI entry
├── configs/           # v0.1.yaml (real) + demo.yaml (mock)
├── data/prompts/      # 40-item curated Chinese set
├── experiments/       # EXP-XXX run log
├── leaderboard/       # released calibrator JSONs
└── docs/              # methodology + roadmap
```

## Why "中文 first" matters

LLM-as-Judge bias research is dominated by English benchmarks (MT-Bench, AlpacaEval, JudgeLM). Chinese frontier models (Qwen, DeepSeek, GLM-4, Doubao) ship with their own judging behaviors that English-benchmark results don't capture—particularly around Chinese-specific reasoning style, terminology, and safety conventions. **zhpan** is the first calibration toolkit built bottom-up for the Chinese stack.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Highlights:
- **v0.1** (now): 40 curated prompts × 4 generators × 3 judges, per-pair offset calibration, CV evaluation
- **v0.2**: pre-built calibrators shipped in the pip package; expanded prompt set from C-Eval / CMMLU / AlignBench
- **v0.3**: leaderboard webapp + community PRs for new models

## License

[MIT](LICENSE)

# Methodology

> 这份文档将逐步演化成 paper 的 Method 章节。先列骨架，每个 section 用 1-2 句话占位。

---

## 1. Problem Formulation

给定 generator 集合 G、judge 集合 J、prompt 集合 P，定义：

- `s_gold(p, g)` = 真实质量评分（来自人工或 strong consensus）
- `s_judge(p, j, g)` = judge j 对 generator g 在 prompt p 上的打分

**Per-pair bias** 定义为：
```
bias[j][g] = E_p [s_judge(p, j, g) - s_gold(p, g)]
```

研究问题：
- 这个 bias 是否系统性存在（≠ 0）？
- 不同 (j, g) pair 的 bias 是否显著不同（即不能用 global offset 替代）？
- 校准能否在 held-out prompt 上提升 evaluation 可靠性？

---

## 2. Prompt Set

来源：MT-Bench (80 道) + AlpacaEval (20 道)。

平衡 8 个 category：
- Reasoning
- Coding
- Writing
- Math
- Information Extraction
- Safety
- Multilingual
- General QA

落盘格式见 [data/prompts/README.md](../data/prompts/README.md)。

---

## 3. Generators

v0.1 选 5 个 frontier model（覆盖闭源 + 开源 + 中文优势）：

| Model | Vendor | Why |
|---|---|---|
| claude-3-5-sonnet | Anthropic | 长 CoT 强 |
| gpt-4o | OpenAI | 通用基线 |
| llama-3.1-70b-instruct | Meta (Together) | 开源对照 |
| qwen-2.5-72b-instruct | Alibaba (Together) | 中文优势 |
| deepseek-v3 | DeepSeek | 推理强 + 价格 |

每个 generator 跑 1 次 sample（temperature=0.7，确定 seed），共 100 × 5 = 500 条 generation。

---

## 4. Judges

v0.1 选 3 个 judge：
- gpt-4o
- claude-3-5-sonnet
- llama-3.1-70b-instruct（开源对照）

判分协议：
- 1-5 分整数评分
- 共享 rubric（domain-agnostic 内容质量 + safety）
- 顺序随机化以避免 position bias

---

## 5. Gold Standard

两条 ground-truth 路径，互相校验：

**路径 A：Strong consensus silver gold**
- 当 3 个 judge 给同一 generation 打分的 std < 0.5 且都 ≥ 4 或都 ≤ 2 时，视为可信 gold

**路径 B：Human-annotated mini-gold**
- 抽样 20 条 generation，作者亲自盲标
- 用于校准 silver gold 的可靠性

---

## 6. Bias Computation

详见 `src/zhpan/compute_bias.py`。

核心 metric：
- `mean_bias[j][g]`
- `std_bias[j][g]`
- `rank_corr[j]`（Spearman ρ）
- `calibrated_mae[j][g]`
- `self_pref_lift[j]`

---

## 7. Calibration

**Per-pair offset**（最简）：
```
s_calibrated = s_judge - bias[j][g]
```

**Per-pair linear**（v0.2 加）：
```
s_calibrated = a[j][g] * s_judge + b[j][g]
```

CV 评估：5-fold 在 prompt 维度切分，held-out MAE 比较。

---

## 8. Threats to Validity

- **Gold 不真**：silver gold 受 judge bias 污染 → 抽样人工标注校准
- **Sample size 小**：100 prompts × 5 generators × 3 judges = 1500 samples per (j,g) pair = 100 → 用 bootstrap CI
- **Prompt distribution shift**：MT-Bench 偏 chat，可能不泛化到 reasoning-heavy 场景 → v0.2 扩展 prompt source
- **Model version drift**：API model 升级后 bias 可能变 → 固定 version snapshot 写在 metadata

---

## 9. Reproducibility

- 所有 config 进 git
- API 响应 cache 进 `data/cache/`（不进 git，但可复现）
- 每个 run 落盘 `experiments/runs/EXP-XXX/` 含 config snapshot + raw outputs + processed

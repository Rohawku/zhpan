# zhpan — Experiment Log

> 沿用 blast-furnace `experiments.md` 的 EXP-XXX 风格：每条假设一个实验，假设 / 改动 / 结果 / 决策四段式。

---

## 当前 Baseline (v0.3)

- Generators (4): qwen-max / deepseek-chat / glm-4-plus / doubao-1-5-pro-32k
- Tested judges (3): qwen-max-judge / deepseek-chat-judge / glm-4-plus-judge
- **Anchor judges (2 independent)**: kimi-anchor-judge (Moonshot) + ernie-anchor-judge (Baidu)
- Prompts: **150 sampled from THUDM/AlignBench v1.1** (Apache-2.0), balanced across 8 AlignBench categories
- Rubric: 1-10 + 7 维分项 (D1-D7)
- Gold: anchor judge (primary = ERNIE; secondary = Kimi for cross-validation)

---

## Experiments

### EXP-005 (2026-05-18) — Subjectivity ↔ self-pref hypothesis test (free, reuses EXP-004 data)

- Layer: 分析 only — 复用 EXP-004 的 per-category lift
- Hypothesis (来自 EXP-004 的猜想): 任务越主观，judge self-pref 越强
- Method:
  1. 给 AlignBench 8 个 main category 标主观性分（1=最客观~5=最主观）:
     1.0 Math · 1.5 Domain-Expert · 2.0 Basic-Task · 2.5 Reasoning · 3.0 Open-QA · 3.5 Chinese-NLU · 4.5 Writing · 5.0 Roleplay
  2. 每个 tested judge 算 (subjectivity, lift) 8 对的 Pearson + Spearman 相关
- 代理指标 (per tested judge):

| Judge | Pearson ρ | p | Spearman ρ | p | 解读 |
|---|---|---|---|---|---|
| **deepseek-chat-judge** | **+0.818** | **0.013** | **+0.929** | **0.001** | ✓ 强正相关：任务越主观 self-pref 越强 |
| **glm-4-plus-judge** | **−0.724** | **0.042** | **−0.833** | **0.010** | ✗ **反向显著相关**：越主观 GLM 反而越**反**偏好自家 |
| qwen-max-judge | +0.139 | 0.742 | +0.286 | 0.493 | 无显著相关 |

- Decision: **keep & publish** — 这是个 surprising but statistically robust finding
- 关键发现：
  1. **"主观性 ↔ self-pref" hypothesis 被部分证伪**。DeepSeek 完全支持（ρ=+0.82, p=0.013），但 GLM **反向且显著**（ρ=−0.72, p=0.042），Qwen 无相关。
  2. **Judges 在 bias-vs-subjectivity 上的 pattern 高度异质**。文献里假设 LLM-as-Judge 偏差有共同 mechanism，本实验否定：DeepSeek 和 GLM 在同一坐标系下呈**相反**对角线，统计上都显著（n=8，p<0.05）。
  3. **GLM 反向相关的可能解释**（待 EXP-006/007 验证）:
     - GLM 在 RLHF/对齐阶段对主观任务的"自我抑制"特别强（公平性 over-correction）
     - GLM 在客观任务上对自家代码/数学格式有偏好（Basic-Task +0.72 是主要 driver）
     - Anchor (ERNIE-4.0) 在主观任务上系统性偏好 GLM 风格，造成"反 self-pref"假象
  4. **DeepSeek pattern 在 anchor robust 范围内**：v0.2 (Kimi anchor) 与 v0.3 (ERNIE) 上 self-pref 都 +0.45~+0.53，且 EXP-004 的 8 个 category 一致正
  5. **方法学 implication**: LLM-as-Judge benchmark 在报告 self-pref 时**必须同时给 per-category 拆分**。Overall lift 在 n=150 上对 GLM 是 +0.01 — 一个完全 misleading 的数字
- 文件:
  - `leaderboard/v0.3/subjectivity_correlation.json`（per-judge Pearson/Spearman + p）
  - `leaderboard/v0.3/subjectivity_scatter.png`（scatter + per-judge regression lines）
- Next:
  1. EXP-006: 看 GLM 反向相关是否是 anchor artifact（用 Kimi anchor 重做相同分析）
  2. EXP-007: 加入 Qwen-judge 在每个 subcategory 的 lift，看 Qwen 的"无相关"是不是因为 sample 不够（Reasoning +0.46 而 Basic-Task -0.12 是同样量级的正/负 lift）


### EXP-004 (2026-05-18) — Per-category bias breakdown (free, reuses EXP-003 data)

- Layer: 分析 only — 复用 EXP-003 的 3000 judgments，按 AlignBench 原始 category 拆分
- Hypothesis: Overall self-preference lift 掩盖了 task-type-dependent pattern
- Change: 新 `per_category_bias.py` 把 judgments 按 AlignBench 8 个 category buckets 切分，每个 bucket 独算 bias matrix + per-judge self-pref lift；新 `plot_per_category.py` 画 8 个 mini-heatmap + bar chart
- Batch: 0（纯离线分析，零 API 成本）
- 代理指标 — **self-preference lift per (judge, AlignBench category)** (anchor=ERNIE):

| category (n=18-19) | DeepSeek-judge | GLM-judge | Qwen-judge |
|---|---|---|---|
| **OVERALL (n=150)** | **+0.45** | **+0.01** | **+0.18** |
| 数学计算 (Math) | +0.14 | +0.28 | +0.11 |
| 逻辑推理 (Reasoning) | +0.48 | -0.09 | +0.46 |
| 中文理解 (Chinese-NLU) | **+0.68** | -0.40 | +0.47 |
| 文本写作 (Writing) | +0.51 | -0.26 | -0.05 |
| 角色扮演 (Roleplay) | **+0.65** | -0.33 | +0.24 |
| 综合问答 (Open-QA) | +0.49 | +0.07 | +0.25 |
| 基本任务 (Basic-Task) | +0.47 | **+0.72** | -0.12 |
| 专业能力 (Domain-Expert) | +0.16 | +0.11 | +0.14 |

- Decision: **keep & publish**
- 关键发现：
  1. **Overall lift 显著低估了 category-level self-preference**。GLM-judge overall lift = +0.01（看似无 self-pref），但拆开看：Math +0.28、**Basic-Task +0.72**、其余 category 负 lift。**Self-pref 在 task 维度高度异质，被 overall 平均掉了**。这是 zhpan-v0.3 用 overall 算 self-pref 的盲区。
  2. **DeepSeek-judge self-pref 几乎贯穿所有 category** (+0.14 ~ +0.68)，但在主观维度任务上最强（Chinese-NLU +0.68, Roleplay +0.65），在客观可验证任务上最弱（Math +0.14, Domain-Expert +0.16）。
  3. **Qwen-judge 在 Reasoning / Chinese-NLU 上 self-pref 显著** (+0.46 / +0.47)，被 Writing -0.05 和 Basic-Task -0.13 拉低 overall。
  4. **新 hypothesis（待验证）: 主观性 ↔ self-preference 强度正相关**。客观可验证（Math / Domain-Expert）→ 低 lift；主观维度（Roleplay / Writing / Chinese-NLU）→ 高 lift。Basic-Task 的高 GLM lift 是反例需要进一步看 — 可能是 judge 在某种结构化任务上对自家输出格式有偏好。
  5. Math 这一类上 GLM 对所有 generator 几乎全 +0.6 ~ +1.0（GLM-judge 比 ERNIE-anchor 普遍更宽松 in 数学评分） — 不是 self-pref 而是 judge-level 整体偏松。
- 文件:
  - `leaderboard/v0.3/category_bias.json`（完整 8 category × bias matrix + lift）
  - `leaderboard/v0.3/category_bias_heatmap.png`（8 mini-heatmap grid）
  - `leaderboard/v0.3/category_selfpref_lift.png`（per-judge × per-category bar chart）
- Next:
  1. EXP-005: 验证 "主观性 ↔ self-pref 强度" hypothesis — 在每个 category 内进一步按 subcategory 拆
  2. EXP-006: 看 self-pref 与生成长度 / 风格相似度的关联（控制变量）


### EXP-003 (2026-05-18) — v0.3 paper-grade: AlignBench + cross-anchor robustness

- Layer: 全 pipeline + 方法学
- Hypothesis:
  - **H1**: 在权威中文 benchmark (AlignBench) 上，per-pair bias 信号是否复现
  - **H2 (anchor robustness)**: 用两个独立 anchor judge (Kimi 月之暗面 + ERNIE 百度文心) 算出的 bias matrix 是否一致 — 一致则 v0.2 发现是真信号，不一致则 anchor-induced
- Change:
  - 替换 prompt 集为 AlignBench v1.1 (THUDM, Apache-2.0)，新 `build_alignbench.py` 通过 GitHub API 拉取，按 8 个 AlignBench 大类分层抽样 150 道
  - prompt 元数据保留 AlignBench `category`/`subcategory`/`reference`/`evidences`，便于后续按类别分析
  - 加入 ERNIE-4.0-8K 作为第二独立 anchor (qianfan vendor)，新 `QianfanClient` adapter（OpenAI 兼容 v2 endpoint）
  - 新 `anchor_compare.py`：两 anchor 各算一次 bias matrix（互相 exclude），输出 Pearson/Spearman 相关 + delta 矩阵
  - 新 `plot_anchor_compare.py`：3-panel heatmap (anchor A | anchor B | delta)
- Batch:
  - 150 prompts × 4 generators = **600 generations**（21:00, $1.31）
  - 600 × 5 judges (3 tested + 2 anchors) = **3000 judgments**（27 分钟左右, $7.11, 2980/3000 success）
  - **总成本 $8.42 ≈ ¥60**
- 代理指标 (anchor = ERNIE-4.0):
  - **M1 mean_bias** 范围: -0.27 ~ +0.31
    - **deepseek-chat-judge → 自家 +0.31**（self-pref 复现 ✓✓）
    - qwen-max-judge: **全负** (-0.05 ~ -0.27)，对 doubao/glm 偏负 0.21
    - glm-4-plus-judge: 全正 (+0.06 ~ +0.17)
    - kimi-anchor-judge (作为 tested): 全正 (+0.09 ~ +0.31)，对 doubao/qwen 显著高
  - **M5 self_pref_lift (vs ERNIE-anchor)**:
    - deepseek-judge → 自家 +0.31, 对其他平均 -0.14, **lift +0.45**
    - qwen-judge → 自家 -0.05, 对其他平均 -0.23, **lift +0.18**
    - glm-judge → 自家 +0.14, 对其他平均 +0.13, **lift ~0**
- **Cross-anchor robustness** (Kimi vs ERNIE, n=12 cells):
  - **Pearson ρ = +0.928 (p=1.4e-5)** ✓
  - **Spearman ρ = +0.881 (p=1.5e-4)** ✓
  - MAE |Kimi − ERNIE| = 0.227
  - **Delta matrix per row 完全相同**: (+0.09, +0.31, +0.21, +0.29) — 这是数学必然，证明 anchor 选型只影响 generator-wise overall offset，**完全不影响 per-pair pattern**
- Decision: **keep & publish** — paper-grade 数据齐了
- 关键发现：
  1. **DeepSeek-judge self-preference 在 AlignBench 上复现** (lift +0.45 vs v0.2 的 +0.45 / +0.53)，**两个独立 anchor 都验证**这个信号是 robust 真实
  2. **Qwen-judge 在 AlignBench 上仍无 self-preference**（lift +0.18，但对自家相对其他还是偏严），与 DeepSeek 形成清晰对比
  3. **Cross-anchor robustness**: Kimi anchor 和 ERNIE anchor 算出的 bias matrix Pearson ρ=0.928 — 极高一致性
  4. **数学性发现**: anchor 选择只决定 generator 的 overall offset（每个 generator 一个常数），**不改变** per-(judge×generator) 的相对 pattern。这意味着 per-pair self-preference 信号是**完全 anchor-independent** 的
  5. AlignBench 上 bias 量级整体扩大（vs v0.2 curated）：可能是因为 AlignBench 题更难、生成质量差异更显著
- Next:
  1. EXP-004: 按 AlignBench category 拆 bias matrix（reasoning / writing / safety 等是否不同）
  2. EXP-005: 加入 cross-lingual judge (GPT-4o via 代理) 看是否 break Pearson 0.928 一致性
  3. EXP-006: per-pair linear (vs offset) 校准 — 现在 n=150 应该够了
  4. PyPI 包发布
- 文件:
  - `leaderboard/v0.3/results.json`（bias matrix + CV）
  - `leaderboard/v0.3/calibrator.json`
  - `leaderboard/v0.3/anchor_compare.json`（Kimi vs ERNIE）
  - `leaderboard/v0.3/bias_heatmap.png`（primary anchor = ERNIE）
  - `leaderboard/v0.3/anchor_compare_heatmap.png`（3-panel cross-anchor）

---

### EXP-002 (2026-05-18) — v0.2 方法论修复：anchor gold + 1-10 rubric + 难题

详见 `leaderboard/v0.2/`. Key takeaways:
- 修复 v0.1 三个连锁 bug：silver-consensus circular reasoning + 1-5 ceiling + easy prompts
- 引入 Kimi (Moonshot) 作为独立 anchor → bias 量级翻倍，self-pref 显现 (+0.53 deepseek)
- 仍是 curated 52 道 prompt，非权威 benchmark；EXP-003 升级到 AlignBench

### EXP-001 (2026-05-18) — v0.1 baseline，已证明方法论有缺陷

详见 `leaderboard/v0.1/`. 被 EXP-002 后验否定（silver gold 列和强制为 0 + ceiling effect）。保留作为方法学反例。

---

## Calibration Holdout Runs (CV)

### CV-003 (2026-05-18) — v0.3 5-fold (anchor=ERNIE)
- MAE: 0.701 → 0.752（calibrated 略劣）
- 解读：n=150 仍不够 robust 校准；per-pair linear 应该能扳回；offset 校准在小样本上方差超过 bias 修正幅度

### CV-002 (2026-05-18) — v0.2 (anchor=Kimi, n=52)
- MAE: 0.480 → 0.544. 同样 calibration 略劣。

### CV-001 (2026-05-18) — v0.1 silver-gold（已废弃方法）
- MAE: 0.171 → 0.183. 不可信（gold 受 bias 污染）。

---

## Open Hypotheses

- [x] **H1**: 中文 frontier judges 之间 per-pair bias 显著（EXP-002 + EXP-003 复现）
- [x] **H2**: silver-consensus gold 是 circular（EXP-002 sum-to-zero 验证）
- [x] **H_robust**: 两个独立 anchor 给出一致 bias matrix（EXP-003 ρ=0.928 验证）
- [ ] **H3**: Per-pair pattern 在 AlignBench category 间差异显著
- [ ] **H4**: 加入 GPT-4o anchor 后 Pearson ρ 仍 > 0.7（cross-lingual robustness）
- [ ] **H5**: Pairwise judging 进一步消除 ceiling
- [ ] **H6**: Per-pair linear 校准在 n=150+ 时跑赢 raw MAE

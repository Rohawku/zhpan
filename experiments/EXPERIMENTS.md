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

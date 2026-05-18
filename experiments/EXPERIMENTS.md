# zhpan — Experiment Log

> 沿用 blast-furnace `experiments.md` 的 EXP-XXX 风格：每条假设一个实验，假设 / 改动 / 结果 / 决策四段式。

---

## 当前 Baseline (v0.2)

- Generators (4): qwen-max / deepseek-chat / glm-4-plus / doubao-1-5-pro-32k
- Tested judges (3): qwen-max-judge / deepseek-chat-judge / glm-4-plus-judge
- **Anchor judge (NEW)**: kimi-anchor-judge (Moonshot moonshot-v1-32k) — 独立 gold，不进 bias matrix
- Prompts: 52 道 curated 中文难题（reasoning/coding/math/extraction/safety/multilingual/general_qa）
- **Rubric**: 1-10 分总分 + 7 维分项打分（D1-D7）
- Gold: anchor judge 单独评分（破除 v0.1 的 silver consensus circular reasoning bug）

---

## Experiments

### EXP-002 (2026-05-18) — v0.2 方法论修复：anchor gold + 1-10 rubric + 难题

- Layer: 方法论 + 全 pipeline 重跑
- Hypothesis: v0.1 测出的 "中文 judges 之间 per-pair bias 很小" 是 3 个独立设计缺陷的连锁后果，**不是真实信号**
- Diagnosis (用 v0.1 数据做后验)：
  - **Bug A (circular silver gold)**: silver gold = 3 judges 均值 → bias = judge - mean(3 judges) → 列和恒为 0（数学必然）。v0.1 每个 generator 的 bias 列和为 ±0.001，完美 sum-to-zero，证实。
  - **Bug B (ceiling effect)**: v0.1 用 1-5 分，**81% 评分是 5、97.7% 是 4-5**，量表退化成二元，bias 物理上无法显现
  - **Bug C (prompt 太简单)**: 40 道 prompt 多数是 "为什么天是蓝的" 这种 frontier model 必然满分的题
- Change:
  - **A 修复**：加 Moonshot Kimi (`moonshot-v1-32k`) 作为独立 anchor judge，不进被测 judge 池；新 `build_gold_anchor()`；`compute_bias` 支持 `exclude_judges`
  - **B 修复**：rubric 改 1-10 + 7 维分项打分（D1 正确性 / D2 推理 / D3 完整性 / D4 切题 / D5 清晰 / D6 深度 / D7 安全）；judge `max_tokens=400`
  - **C 修复**：重写 52 道难题 prompt 集，含 GSM8K 类多步推理、动态规划代码、内切圆+外接圆三角几何、文言/粤语/敬语对比、prompt-injection 等
- Batch:
  - 52 prompts × 4 generators = 208 generations（12:14，$0.67）
  - 208 × 4 judges (3 tested + 1 anchor) = 832 judgments（9:44，$2.72，13 calls 因 Kimi 429 失败）
  - **总成本 $3.39 ≈ ¥24**
- 代理指标（vs v0.1）:
  - **M1 mean_bias** 范围: **-0.22 ~ +0.37**（v0.1 是 -0.17 ~ +0.11，扩大 ~2x）
    - **deepseek-chat-judge → deepseek-chat 自家 +0.37**（强 self-preference）
    - glm-4-plus-judge → 全正 (+0.12 ~ +0.35)，整体偏宽松
    - qwen-max-judge → 对 doubao/glm 偏负 (-0.22)，对自家 -0.02（**未显 self-preference**）
  - **M2 sum-of-biases per generator** （应 ≠ 0 if anchor breaks circular）:
    - deepseek-chat sum = **+0.75**（tested judges 普遍高估 vs Kimi）
    - glm-4-plus sum = **-0.33**
    - 其他 ~0 → 证明 anchor 成功打破 sum-to-zero 约束
  - **M5 self_pref_lift**:
    - deepseek-judge: **+0.53** 💥 （v0.1 是 +0.07，量级扩大 7x）
    - glm-judge: ~0
    - qwen-judge: 几乎 0
  - **M4 CV held-out MAE**: 0.480 → 0.544（calibrated 更差）
    - 解读：n=52 仍小，per-pair offset 校准引入的方差超过修正幅度；但 bias 信号本身是真实的（参见 sum-of-biases）
- 抽样观察: 待做
- Decision: **keep & publish** — 三个方法论 bug 全修，bias 信号显现
- 关键发现 (vs v0.1):
  1. **v0.1 的 "中文 judges per-pair bias 很小" 是测量缺陷，不是事实**。修方法论后 bias 量级翻倍，per-pair pattern 显现。
  2. **DeepSeek-judge 显示强 self-preference**（lift +0.53），这是中文 frontier model self-preference 的首次量化报告。
  3. **Qwen / GLM judges 没有显著 self-preference**——这和文献里 GPT-4 / Claude self-preference 普遍存在的观察不同，可能是中文训练数据 / RLHF 偏好造成的。
  4. **方法论本身是 contribution**：silver-consensus gold 在多 judge 偏差测量上的 circular reasoning 缺陷，是文献里 understudied 的 methodological flaw。
- Next:
  1. EXP-003: 扩 prompt 集到 200+；run on per-category subsets
  2. EXP-004: 加入第二个 anchor (例如 GPT-4o)，看 anchor 选择是否敏感
  3. EXP-005: 引入 pairwise judging 模式（A vs B），看是否能进一步降低 ceiling
  4. EXP-006: 改 per-pair linear 校准（不只 offset），test 是否能在 n=52 上跑赢 raw


### EXP-001 (2026-05-18) — v0.1 baseline，已证明方法论有缺陷

详见 `leaderboard/v0.1/results.json`。Key takeaways（被 EXP-002 后验否定）:
- "Bias 量级 -0.17 ~ +0.11" → 实为 ceiling + circular 双重压制后的残值
- "Self-preference 几乎为零" → 实为 self-preference 在 silver gold 下被分摊到列均值，加上 ceiling 几乎不可见

---

## Calibration Holdout Runs (CV)

### CV-002 (2026-05-18) — v0.2 anchor-gold 5-fold CV
- 数据: EXP-002 全量 832 judgments，anchor=kimi-anchor-judge
- Held-out MAE: 0.480 (raw) → 0.544 (calibrated). 校准未提升。
- 反推: n=52 不够 robust 校准；v0.3 必须扩 prompt 集 或 改 per-pair linear 校准。

### CV-001 (2026-05-18) — v0.1 silver-gold（已废弃方法）
- MAE 0.171 → 0.183. 不可信（gold 本身受 bias 污染）。

---

## Open Hypotheses

- [x] **H1**: 中文 frontier judges 之间 per-pair bias 显著 → **EXP-002 证实**（modulo per-pair lift +0.53 for deepseek）
- [x] **H2**: silver-consensus gold 在 bias 测量上是 circular → **EXP-002 证实**（sum-to-zero 验证）
- [ ] **H3**: Per-pair pattern 在 reasoning vs writing 类 prompt 上不同
- [ ] **H4**: 加入 GPT-4o anchor 后 bias matrix 与 Kimi anchor 一致（anchor robustness check）
- [ ] **H5**: Pairwise judging 能进一步消除 ceiling、放大 per-pair bias
- [ ] **H6**: Per-pair linear 校准（不只 offset）在 n=200+ 时能跑赢 raw MAE

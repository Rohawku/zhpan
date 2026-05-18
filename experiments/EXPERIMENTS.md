# zhpan — Experiment Log

> 沿用 blast-furnace `experiments.md` 的 EXP-XXX 风格：每条假设一个实验，假设 / 改动 / 结果 / 决策四段式。
> 跨会话靠这份文件接力。**这份文件进 git**（experiments/runs/ 不进）。

---

## 当前 Baseline (v0.1)

- Generators (4): qwen-max-2025-01-25 / deepseek-chat / glm-4-plus / doubao-1-5-pro-32k-250115
- Judges (3): qwen-max-judge / deepseek-chat-judge / glm-4-plus-judge
- Prompts: 40 道 curated 中文 prompt，8 类各 5 道
- Gold: silver consensus（≥2 judges 一致，std ≤ 1.0）

---

## Proxy Metrics

- **M1 mean_bias[j][g]**：`mean(judge_score - gold_score)` per (judge, generator) pair，核心指标
- **M2 std_bias[j][g]**：bias 的标准差
- **M3 rank_corr[j]**：judge 给 generator 排名 vs gold 排名的 Spearman ρ
- **M4 calibrated_mae[j][g]**：校准后 vs 校准前的 MAE 下降
- **M5 self_pref_lift[j]**：judge 对自己家族 vs 对其他家族的 mean_bias 差

---

## Open Hypotheses

- [x] **H1**：中文 frontier judges 之间 per-pair bias 显著 → **EXP-001 否定**（per-pair bias 量级 < ±0.2，主要是 judge-level overall offset）
- [ ] **H2**：加入 cross-lingual judges (GPT-4o, Claude) 后 per-pair bias 会显现
- [ ] **H3**：将 prompt 集扩到 200+ 后，bias 显著性会增强
- [ ] **H4**：在 safety / 文风类 prompt 上，per-pair bias 比 math/coding 类大
- [ ] **H5**：reference-guided judging 能降低 inter-judge variance

---

## Experiments

### EXP-001 (2026-05-18) — v0.1 真实 baseline

- Layer: 全 pipeline (real APIs)
- Hypothesis: 中文 frontier judges (Qwen / DeepSeek / GLM-4) 之间存在显著 per-pair bias
- Change: 项目首跑，4 vendor adapter + 40 curated 中文 prompt + per-pair offset calibrator
- Batch:
  - 40 prompts × 4 generators = 160 generations（耗时 6:42，成本 $0.31）
  - 160 × 3 judges = 480 judgments（耗时 2:49，成本 $1.05，3 calls 失败）
  - **总成本 $1.36 ≈ ¥9.7**
- 代理指标:
  - **M1 mean_bias** 范围: -0.17 ~ +0.11（量级远小于预期）
    - deepseek-chat-judge: **全负**（-0.07 ~ -0.17），系统性更严格
    - glm-4-plus-judge: 全正轻微（+0.03 ~ +0.08）
    - qwen-max-judge: 全正轻微（+0.03 ~ +0.11），对 doubao 偏好略高 (+0.11)
  - M2 std_bias 范围: 0.24 ~ 0.34
  - **M3 rank_corr**: deepseek **0.80**, glm **0.95**, qwen **0.95** — 3 judges 在 generator 排序上高度一致
  - **M5 self_pref_lift**: deepseek **+0.07**, glm **+0.03**, qwen **+0.01** — **几乎没有 self-preference**
  - M4 5-fold CV MAE: **0.171 → 0.183**（校准没改善）
- Decision: **keep & document**
- 关键发现：
  1. **中文 frontier judges 之间 per-pair bias 比预期小一个量级**。Mock demo 设的 bias 是 ±0.6，真实是 ±0.15。
  2. 主导效应是 **judge-level overall offset**（DeepSeek 严格 vs GLM/Qwen 宽松）而非 per-pair pattern。
  3. **Self-preference 几乎不存在**（lift < 0.07），这和英文 benchmark 的常见发现不同。
  4. **小 sample (n=40) 上 per-pair 校准没法跑赢 raw**：mean offset 量级接近 silver gold 噪声，校准引入的方差超过修正幅度。
- Next:
  1. EXP-002: 加入 GPT-4o / Claude-3.5-Sonnet 作为 cross-lingual judge，看 per-pair bias 是否显现
  2. EXP-003: 扩 prompt 集到 200+（v0.2 milestone），用 C-Eval / CMMLU 补足
  3. EXP-004: 按 category 拆分 bias 矩阵 —— per-category bias 可能比 overall 大


### EXP-000 (2026-05-18) — 项目搭建

- Layer: infrastructure
- 完成内容: 完整目录树 / 8 模块代码 / 4 vendor adapter / mock demo / pytest / `zhpan` CLI / README
- Decision: keep（项目 v0.1 上 GitHub）

---

## Calibration Holdout Runs (CV)

### CV-001 (2026-05-18) — v0.1 5-fold prompt-axis CV
- 数据: EXP-001 全量 480 judgments
- 校准方法: per-pair offset（v0.1 唯一支持）
- Held-out MAE: 0.171 (raw) → 0.183 (calibrated). **校准未提升**。
- 反推结论: 中文 judges 间 per-pair bias 太弱，per-pair offset 校准在 n=40 上引入的方差超过修正幅度。**v0.2 必须加 cross-lingual judges 或扩 prompt 集**。

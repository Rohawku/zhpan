# zhpan — Experiment Log

> 沿用 blast-furnace `experiments.md` 的 EXP-XXX 风格：每条假设一个实验，假设 / 改动 / 结果 / 决策四段式。
> 跨会话靠这份文件接力。**这份文件进 git**（experiments/runs/ 不进）。

---

## 当前 Baseline (v0)

- Generators: TBD（v0.1 计划 5 个：claude-3-5-sonnet / gpt-4o / llama-3.1-70b / qwen-2.5-72b / deepseek-v3）
- Judges: TBD（v0.1 计划 3 个：gpt-4o / claude-3-5-sonnet / llama-3.1-70b）
- Prompts: TBD（v0.1 计划 100 道，来源 MT-Bench + AlpacaEval 抽样）
- Gold: TBD（v0.1 计划 strong-consensus + 抽样人工标注）
- Bias matrix: 待跑

---

## Proxy Metrics（用什么衡量 bias）

- **M1 mean_bias[j][g]**：`mean(judge_score - gold_score)` per (judge, generator) pair，核心指标
- **M2 std_bias[j][g]**：bias 的标准差，反映稳定性
- **M3 rank_corr[j]**：judge 给 generator 的排名 vs gold 排名的 Spearman 相关，反映 ranking bias
- **M4 calibrated_mae[j][g]**：校准后 vs 校准前的 MAE 下降幅度，验证校准有效性
- **M5 self_pref_lift[j]**：如果 judge 也是 generator 之一，对自己的 mean_bias - 对他人的 mean_bias

---

## Open Hypotheses（按优先级）

- [ ] **H1**：Claude judge 不会对 Claude generator 系统性高估（self-preference 可能反向）
- [ ] **H2**：开源 judge（Llama-3.1-70B）对其他开源 generator 的 bias 比闭源 judge 更小
- [ ] **H3**：在 reasoning 类 prompt 上，bias 比在写作类 prompt 上更大
- [ ] **H4**：per-pair offset 校准比 global offset 校准在 held-out prompts 上 MAE 显著更低
- [ ] **H5**：判分时给 judge 看 reference answer 能显著减小 bias（reference-guided judging 假说复现）

---

## Experiments

<!--
EXP-XXX 模板（复制下面到新条目）：

### EXP-XXX (YYYY-MM-DD) — 一句话标题
- Layer: prompt / generation / judge / calibration / analysis
- Hypothesis: 一句话假设
- Change:
  - 改了什么文件 / 配置 / 参数
- Batch: N prompts × M generators × K judges = X calls
- Run command: `python -m zhpan.scripts.xxx --config ...`
- 代理指标（vs baseline）:
  - M1 mean_bias: ...
  - M2 std_bias: ...
  - M3 rank_corr: ...
- 抽样观察: ...
- Bad case: ...
- Decision: keep / discard / iterate
- Next: ...
-->

### EXP-000 (2026-05-18) — Project scaffolding & first commit

- Layer: infrastructure
- Hypothesis: N/A，搭项目骨架
- Change:
  - 建 `~/Data/zhpan/` 完整目录树
  - 写 README、.gitignore、pyproject.toml、LICENSE、ROADMAP、methodology 骨架、Makefile
  - 写 EXPERIMENTS.md（本文件）
- Decision: keep
- Next: EXP-001 选定 prompt set，从 MT-Bench / AlpacaEval 抽样 100 条

---

## Calibration Holdout Runs (CV)

> 攒几轮 EXP 后跑一次 cross-validation，验证校准在 held-out prompts 上的泛化能力。

<!--
### CV-001 (YYYY-MM-DD)
- 用了哪几轮数据: EXP-XXX + EXP-YYY
- Held-out split: ...
- 校准方法: per-pair offset / per-pair linear / ...
- MAE before/after: ...
- 反推结论: ...
-->

# Roadmap

> 目标节奏：每周一个小 milestone，2026 Q2 内 v0.1 公开发布；v0.2 走向 paper。

---

## v0.1 — MVP 公开版（目标：2026 年 6 月）

**判断 done 的标准**：陌生人 clone 仓库 → 跟着 README 跑 → 5 分钟内看到第一张 bias heatmap。

### M1. Prompt set (1 周)
- [ ] 从 MT-Bench (80 道) + AlpacaEval (随机抽 20 道) 构造 100 道 prompt
- [ ] 按 8 个 category 平衡（reasoning / coding / writing / math / extraction / safety / multilingual / general QA）
- [ ] 落盘 `data/prompts/v0.1.jsonl`
- [ ] 写 `data/prompts/README.md` 说明来源 + license

### M2. Generation pipeline (1 周)
- [ ] `src/zhpan/models.py`：generator/judge 抽象（OpenAI / Anthropic / together.ai）
- [ ] `src/zhpan/generate.py`：100 prompts × 5 generators → `data/generations/v0.1/`
- [ ] 重试 / 缓存 / 预算上限（超过 $JBB_BUDGET_USD 报警停止）
- [ ] CLI: `python -m zhpan.scripts.run_generate --config configs/v0.1.yaml`

### M3. Judge pipeline (3 天)
- [ ] `src/zhpan/judge.py`：3 judges × 上述 generation → `data/judgments/v0.1/`
- [ ] 共享 1-5 分 rubric（避免 verbosity bias 的 prompt 设计）
- [ ] swap-and-average 处理 position bias（如果用 pairwise）

### M4. Gold standard (3 天)
- [ ] 方法 A：strong-consensus（3 judges 一致同意，作为 silver gold）
- [ ] 方法 B：抽样 20 条人工标注（你自己 + 1 个朋友）
- [ ] 落盘 `data/gold/v0.1.jsonl`

### M5. Bias analysis (3 天)
- [ ] `src/zhpan/compute_bias.py`：算 M1-M5 五个代理指标
- [ ] `notebooks/01_baseline_bias_analysis.ipynb`：heatmap + 表格 + 关键发现
- [ ] 输出 `leaderboard/v0.1.json`

### M6. Calibration (3 天)
- [ ] `src/zhpan/calibrate.py`：per-pair offset 校准（最简版）
- [ ] `examples/debias_in_3_lines.py`：3 行用法 demo
- [ ] CV 验证：80/20 split，看 held-out MAE 下降

### M7. Polish + 发布 (3 天)
- [ ] README 补 heatmap 图 + 数据卖点
- [ ] 写一篇知乎长文 + 推 X
- [ ] 推送到 GitHub public，找 1-2 个 KOL 转发

---

## v0.2 — Paper-grade（目标：2026 年 8 月）

- [ ] 扩展到 10 generators × 5 judges
- [ ] 加入 ranking bias / verbosity bias / position bias 三个对照维度
- [ ] per-pair linear regression 校准（不只 offset）
- [ ] 人工标注 100 条 high-quality gold
- [ ] 离线 holdout 评估
- [ ] 写 paper draft，目标 EMNLP 2026 workshop / ACL 2027

---

## v0.3 — Community-driven（目标：2026 年底）

- [ ] 在线 leaderboard 网页（HF Spaces）
- [ ] 接受社区 PR 加 model
- [ ] 多语种 prompt（中 / 英 / 法）

---

## Stretch goals

- [ ] Token-level bias（不只 final score）
- [ ] Process-reward bias（每步打分的 bias）
- [ ] Bias 随 prompt 难度变化的分析

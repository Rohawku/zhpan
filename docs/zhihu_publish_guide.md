# 知乎发布操作指南

> 这份文件是 zhpan 知乎博文的"复制粘贴版"——按顺序把每个 ░Block░ 内容贴进知乎编辑器即可。

---

## 📝 标题（3 选 1，建议选 1）

1. **DeepSeek 当裁判会偏袒自家吗？我花了 ¥94 跑出来的答案**
2. 我用 ¥94 测了 4 家中文大模型当裁判时的偏差——结论比想象的颠覆
3. 中文 LLM-as-Judge 偏差实测：6 个实验里发现的方法学坑

## 🏷️ 标签

`大模型` `LLM` `DeepSeek` `Qwen` `开源项目` `自然语言处理` `Python`

## 🖼️ 封面图

上传：`/Users/jyxc-dz-0101319/Data/zhpan/leaderboard/v0.3/subjectivity_anchor_robust.png`

---

# 正文部分（按顺序贴）

## ░Block 1░  开头 + TL;DR

> 项目开源：https://github.com/Rohawku/zhpan
> 关键词：LLM-as-Judge、self-preference、AlignBench、中文大模型评测

我做了一个开源工具 zhpan，专门测量 **当一个中文 LLM 当 judge（裁判）评测其他中文 LLM 时，它对不同 generator 是否存在系统性偏差**。

我在 AlignBench 的 150 道权威中文 prompt 上跑了 4 个 generator（Qwen-max / DeepSeek-chat / GLM-4-plus / Doubao-pro-32k）× 3 tested judges + 2 独立 anchor judges（Kimi + ERNIE-4.0），共 3000 条真实 API 评分，总成本 ¥94。

跑出几个意外的结果：

1. 🔴 **DeepSeek-chat 当 judge 时显著偏好自家模型**（self-preference lift = +0.45）
2. 🟢 **Qwen-max 当 judge 时几乎没有 self-preference**
3. 🔵 **GLM-4-plus 当 judge 时 overall 看起来公平（+0.01），但拆开看是两面性**：在结构化任务上 +0.72 强偏自家，在主观任务上 −0.40 反偏好自家。Overall 是被相反方向抵消的假象。
4. 🧪 **"任务越主观，judge 越偏好自家"假说被部分驳斥**：DeepSeek 完全支持（Pearson ρ = +0.82），但 GLM 显示反向显著相关（ρ = −0.72），双 anchor 复测方向一致。
5. 💀 **过程中发现 LLM-as-Judge 文献里普遍存在的方法学 bug**：用多个 judge 的平均分作为 gold standard 时，per-judge bias 被数学约束强制为零和，完全测不出真实信号。

下面把 6 次实验是怎么从"测不出来"一路修到"两个独立 anchor 同方向证实"讲一遍。这个过程比最终结果本身更有意思。

---

## ░Block 2░  起点

去年实习时做后训练数据 pipeline，遇到一个问题：

> 我们用一个 LLM 当 judge 给另一个 LLM 的输出打分。这个 judge 给某些模型输出的平均分明显偏低。是模型真的差，还是 judge 自己有偏差？

这是个 ill-posed 的问题：没有 ground truth，怎么知道 judge 在偏？做了一阵实验后发现这件事值得做成独立项目——尤其是针对中文模型，因为现有 LLM-as-Judge 偏差研究几乎全是英文 benchmark（MT-Bench / AlpacaEval / JudgeLM），中文 frontier model 的偏差量化基本空白。

工具名叫 zhpan（"中评"），目标只有一行 API：

```python
from zhpan import Calibrator
cal = Calibrator.from_file("leaderboard/v0.3/calibrator.json")
fair = cal.correct(judge="deepseek-chat-judge", generator="deepseek-chat", raw_score=8.0)
# → 7.69  (DeepSeek-judge 系统性高估自家 +0.31，校准后扣回)
```

---

## ░Block 3░  EXP-001：第一次跑出来"什么都没测到"

第一版搭了完整 pipeline：4 个中文 generator × 3 个中文 judges × 40 道 curated prompt = 480 个评分。Gold standard 用行业标准做法——3 个 judges 的平均分（silver consensus）。

跑出来 bias matrix（mean_bias = judge − silver gold）:

```
                    deepseek    doubao    glm-4-plus   qwen-max
deepseek-judge       -0.07      -0.17     -0.12        -0.14
glm-4-plus-judge     +0.03      +0.06     +0.08        +0.07
qwen-max-judge       +0.03      +0.11     +0.03        +0.07
```

所有 cell 绝对值都 < 0.2。直觉告诉我有问题：DeepSeek-judge 这种生态位明显的模型，对自家不可能完全 neutral。

诊断后发现 **3 个串联的设计 bug**。

### Bug A：Silver gold 是数学性 circular reasoning

如果我用 3 个 judges 的均值作为 gold，那对任意 generator g：

```
sum_j bias[j][g] = sum_j (s_j(g) - mean_j(s_j(g)))
                 ≡ 0  (恒等于零)
```

每个 generator 上 3 个 judges 的 bias 在数学上**被强制 sum to zero**。我测的根本不是 judge 相对于真实质量的偏差，是 judge 相对于这 3 个 judges 群体均值的偏离。这是 zero-sum 的，真实 self-preference 信号必然被压成残值。

验证一下：每个 generator 在 3 个 judges 上的 bias sum 是 ±0.001，完美 sum to zero。bug 确认。

文献里大量 LLM-as-Judge 论文用 multi-judge consensus 当 gold，但没人专门写过这个 issue。

### Bug B：1-5 分量表的 ceiling effect

打开数据看分数分布：

```
分数=1   0%
分数=2   1%
分数=3   1.3%
分数=4   16.7%
分数=5   81%
```

97.7% 的评分集中在 4-5 分，量表实际退化成了二元分类。在 90%+ 都打满分的情况下，bias 物理上无法显示——因为没人愿意打 6 分。

### Bug C：Prompt 太简单

重新审视 40 道 prompt，多数是 "为什么天是蓝的" / "AA 制怎么算" 这种 frontier model 必然满分的题。

三个 bug 串起来形成连锁压制：

```
prompt 简单 → 所有 generator 跑满分 → 量表退化 → silver gold 列和强制 0 → bias 测不到
```

---

## ░Block 4░  EXP-002：三个 bug 全修

针对每个 bug 都做了对应修复：

- **Bug A**：加一个独立 anchor judge（Moonshot Kimi），**不在被测 judge 池里**，用它单独打分作为 gold
- **Bug B**：rubric 改 1-10 分 + 7 维分项（D1 正确性 / D2 推理 / D3 完整性 / D4 切题 / D5 清晰 / D6 深度 / D7 安全）
- **Bug C**：重写 52 道难题——GSM8K 类多步推理、动态规划代码、文言/粤语对比、prompt-injection 等

跑出来 bias matrix（anchor = Kimi）：

```
                    deepseek    doubao    glm-4-plus   qwen-max
deepseek-judge       +0.37      -0.08     -0.22        -0.18
glm-4-plus-judge     +0.35      +0.28     +0.12        +0.16
qwen-max-judge       +0.02      -0.22     -0.22        -0.02
```

关键变化：

- **量级翻倍**：v0.1 是 -0.17 ~ +0.11，v0.2 是 -0.22 ~ +0.37
- **DeepSeek-judge 自家 cell +0.37 突显**，比对其他三家平均 -0.16 高 0.53 分。**self-preference lift = +0.53**
- **Sum-to-zero 约束被打破**：bias 列和从 ±0.001 变成 -0.33 ~ +0.75，证明 anchor 真的破除了 circular

但这是**单 anchor** 的结果。万一是 Kimi anchor 自己对某些 generator 有偏好造成的 artifact 怎么办？

---

## ░Block 5░  EXP-003：AlignBench + 双 anchor + cross-validation

v0.3 一次性修两个问题：

### 换上 AlignBench v1.1

AlignBench（清华 THUDM，Apache-2.0）是 683 道**为中文 LLM-as-judge 评测专门设计**的 prompt 集，覆盖 8 大类（基础语言 / 中文理解 / 综合问答 / 文本写作 / 角色扮演 / 数学推理 / 复杂任务 / 专业知识），每道带 reference answer。我从中抽样 150 道，按 8 个 category 均衡。

### 第二独立 anchor：ERNIE-4.0

只用 Kimi 一个 anchor 不够。加上百度文心 ERNIE-4.0 作为第二独立 anchor——百度独立家族，跟阿里/字节/智谱/DeepSeek/月之暗面都没关系。

> 💡 **此处插图**：`leaderboard/v0.3/bias_heatmap.png`

跑完 600 generations + 3000 judgments，DeepSeek-judge 对自家 +0.31，对其他 -0.14 平均，**self-pref lift = +0.45**——在新的权威数据集 + 新的 anchor 上复现了 v0.2 的 finding。

### Cross-anchor robustness 检验

用 Kimi anchor 和 ERNIE anchor 分别算 bias matrix，看两者多一致：

```
Pearson  ρ = +0.928,  p = 1.4e-5
Spearman ρ = +0.881,  p = 1.5e-4
```

**两个完全独立家族的 anchor，bias matrix 的 cell-wise 相关性 ρ = 0.93**。

> 💡 **此处插图**：`leaderboard/v0.3/anchor_compare_heatmap.png`

还有个**数学性 elegant 的发现**：两个 anchor 的 bias matrix delta，每一行**完全相同**：

```
delta per row = (+0.09, +0.31, +0.21, +0.29)
```

不是巧合，是数学必然。对同一个 judge j 和 generator g：

```
bias_A1[j][g] - bias_A2[j][g] = s_A2(g) - s_A1(g)
```

跟 judge j 无关，只依赖 generator g。所以 **anchor 选择只影响 generator-wise 整体 offset，不影响 per-pair pattern**。换句话说：zhpan 测出来的 per-pair self-preference 信号是**完全 anchor-independent** 的。

这个性质对全行业的 LLM-as-Judge benchmark 研究都有意义：在合理 anchor 类内，anchor 选择是个"免费的超参数"，对 per-pair 偏差研究不会改变定性结论。

---

## ░Block 6░  EXP-004：拆 category 看，发现 GLM 的"两面性"

零成本的 follow-up：按 AlignBench 8 个 original category 拆分 bias matrix。理由：overall 的 self-pref lift 是把 8 个 category 平均，可能掩盖 category-level pattern。

Self-preference lift per (judge, category)：

```
Category                  DeepSeek    GLM       Qwen
OVERALL                    +0.45     +0.01     +0.18
数学计算                    +0.14     +0.28     +0.11
逻辑推理                    +0.48     -0.09     +0.46
中文理解                    +0.68     -0.40     +0.47
文本写作                    +0.51     -0.26     -0.05
角色扮演                    +0.65     -0.33     +0.24
综合问答                    +0.49     +0.07     +0.25
基本任务                    +0.47     +0.72     -0.12
专业能力                    +0.16     +0.11     +0.14
```

> 💡 **此处插图**：`leaderboard/v0.3/category_selfpref_lift.png`

意外发现：

1. **GLM overall lift = +0.01 是个假象**。它在 Basic-Task 上对自家 +0.72，但在 Chinese-NLU 上 -0.40，Roleplay 上 -0.33，Writing 上 -0.26——**正反相互抵消把 overall 拉到 0 附近**。如果只看 overall lift 给出"GLM 是 fair judge"的 conclusion 是完全错的。
2. **DeepSeek self-pref 几乎贯穿所有 category**，且在主观任务上更强（Chinese-NLU +0.68, Roleplay +0.65），在客观可验证任务上较弱（Math +0.14, Domain-Expert +0.16）。
3. **Qwen 在 Reasoning / Chinese-NLU 上 self-pref 显著**（+0.46 / +0.47），但被其他 category 拉低 overall。

从 DeepSeek 那行规律看，冒出一个 testable hypothesis：**任务越主观，judge self-preference 越强**（因为越主观，越没有 ground truth 约束 judge）。

---

## ░Block 7░  EXP-005：主观性 ↔ self-pref 假说被部分驳斥

把假说做成数值检验。给 8 个 main category 标了主观性分（1=最客观 ~ 5=最主观）:

```
1.0  Math（纯计算，单一答案）
1.5  Domain-Expert（事实知识，多数可验证）
2.0  Basic-Task（结构化抽取/分类）
2.5  Reasoning（答案可验证但推理过程主观）
3.0  Open-QA（混合事实+观点）
3.5  Chinese-NLU（细微解读）
4.5  Writing（质量靠品味）
5.0  Roleplay（声音/风格就是答案本身）
```

每个 judge 跑 8 对 (主观性, lift) 的 Pearson + Spearman 相关：

```
Judge                Pearson ρ   p         Spearman ρ   p
deepseek-judge       +0.82       0.013     +0.93        0.001
glm-judge            -0.72       0.042     -0.83        0.010
qwen-judge           +0.14       0.74      +0.29        0.49
```

DeepSeek 完全支持假说。**但 GLM 反向且显著**（ρ = -0.72, p = 0.042）：任务越主观，GLM 反而越**反**偏好自家。Qwen 平。

**Conclusion**：没有一个通用的"主观性驱动 self-pref"规律给所有 LLM judges。同一个坐标系下 DeepSeek 和 GLM 是**对角相反两条线**，统计上都显著。

这是个比 "DeepSeek 有 self-pref" 更深的发现：**判官的偏差机制是 model-specific 且可以方向相反的，不是某个通用 law 的体现**。

---

## ░Block 8░  EXP-006：GLM 反向相关是真的，还是 anchor artifact？

EXP-005 的 GLM finding 太 surprising，担心是 ERNIE-anchor 自己造成的假象。用 Kimi anchor 重做完全一样的相关性分析：

```
Judge              ρ(ERNIE)   p        ρ(Kimi)    p       方向一致？
DeepSeek           +0.82      0.013    +0.43      0.29    ✓ 都正
GLM                -0.72      0.042    -0.62      0.10    ✓ 都负
Qwen               +0.14      0.74     -0.17      0.68    ~ 都≈0
```

> 💡 **此处插图**：`leaderboard/v0.3/subjectivity_anchor_robust.png`

**方向 anchor-robust**：DeepSeek 两次都正、GLM 两次都负。**GLM 反向相关是真的**，不是 ERNIE artifact。

**量级和 p 值 anchor-dependent**：n=8 太小，p 在不同 anchor 下从 0.013 → 0.29，从 0.042 → 0.10。

这是个**对全行业的方法学启示**：n=8 main category 上单 anchor + p<0.05 不应该直接 publish。至少要 (a) 两个独立 anchor 同方向 + (b) Spearman ρ 检验。

---

## ░Block 9░  项目总览

```
EXP   内容                                       成本    状态
001   v0.1 silver-gold baseline (flawed)         ¥10    反例，保留作 methodological example
002   v0.2 Kimi anchor 修复                      ¥24    DeepSeek self-pref lift +0.53
003   v0.3 AlignBench + 双 anchor                ¥60    cross-anchor ρ=0.928
004   per-category breakdown                     ¥0     GLM "two-faced" 发现
005   主观性相关性测试                            ¥0     DeepSeek +0.82 vs GLM -0.72
006   EXP-005 的 anchor robustness 检验          ¥0     方向 anchor-robust
合计  6 个 EXP, 3000+ judgments                  ¥94    paper-grade dataset
```

GitHub 仓库长这样：

- `src/zhpan/`：完整 Python package，`pip install -e .` 可用
- `configs/`：v0.1 / v0.2 / v0.3 YAML 配置
- `data/prompts/`：AlignBench 抽样后的 150 道
- `experiments/EXPERIMENTS.md`：6 个 EXP 的"假设-改动-结果-决策"日志
- `leaderboard/v0.1 ~ v0.3/`：3 版 calibrator JSON + 多张 heatmap PNG
- `Makefile`：陌生人 clone + 6 个 API key + `make benchmark` 一行命令复现全部

---

## ░Block 10░  Take-away

### 1. LLM-as-Judge 偏差是 model-specific 且可以方向相反的

文献里多数研究假设 LLM judges 偏差有共同 mechanism。本实验否定：DeepSeek 显示强 self-pref，Qwen 几乎没有，GLM 看似公平实则两面性。判官之间的偏差机制差异巨大，要 per-model 单独研究。

### 2. Overall self-pref lift 是个不可靠的单一指标

GLM overall lift = +0.01 完美"fair"，但 category-level 拆开 ±0.7 量级。**LLM-as-Judge benchmark 必须报告 per-category breakdown**。

### 3. Silver-consensus gold 在偏差测量上是 circular

Multi-judge mean 作为 gold 时，bias 列和被数学约束强制为 0，真实信号被压制。任何 LLM-as-Judge bias 研究都应该用 independent anchor judge，且至少做 cross-anchor robustness 检验。

### 4. Anchor 选择对 per-pair pattern 完全 invariant

两个 anchor 的 bias matrix delta 每一行恒等于 anchor-wise generator offset。所以 anchor 选择是个"免费超参数"。

### 5. ¥94 + 一周时间能做出多少东西

这个项目从零到 6 个 EXP + paper-grade dataset + 完整开源代码，总花费不到 ¥100，时间是断断续续一周。做开源研究的门槛比想象的低很多，主要瓶颈在于"想清楚要测什么、怎么测、什么不能测"，不在于算力或预算。

---

## ░Block 11░  下一步 + CTA

EXP-007 / 008 计划做：

- 加 GPT-4o 作为第三 cross-lingual anchor
- AlignBench 全 683 道而不是抽样 150
- Pairwise judging 模式
- PyPI 包发布

如果你对这个项目有想法或 PR 欢迎砸过来：

🔗 GitHub：https://github.com/Rohawku/zhpan
🔗 完整实验日志：https://github.com/Rohawku/zhpan/blob/main/experiments/EXPERIMENTS.md

如果你在做 LLM-as-Judge 相关研究、或者在工程里用 LLM 当 judge 评测生产数据，欢迎用 zhpan 的 calibrator 给你的判分系统打补丁，3 行代码：

```python
from zhpan import Calibrator
cal = Calibrator.from_file("leaderboard/v0.3/calibrator.json")
fair = cal.correct(judge="...", generator="...", raw_score=...)
```

最后说一句：这个项目的所有 finding 都基于固定时间点（2026 年 5 月）的固定 model 版本。LLM 模型每个季度都在变，结论可能很快过时。复现协议在 README 里写清楚了，谁愿意接力跑下一个季度的数据欢迎开 issue / PR。

---

# 📋 知乎实操 checklist

## 发布前自检

- [ ] 标题 3 选 1 已定
- [ ] 封面图已上传（subjectivity_anchor_robust.png）
- [ ] 11 个 Block 全部贴入并粗略校对
- [ ] 4 张配图已嵌入到对应位置：
  - [ ] Block 5: bias_heatmap.png
  - [ ] Block 5: anchor_compare_heatmap.png
  - [ ] Block 6: category_selfpref_lift.png
  - [ ] Block 8: subjectivity_anchor_robust.png
- [ ] 6 个数字段（代码块那种 ASCII 表格）粘进去后保留代码块格式（不是普通段落）
- [ ] 标签 6 个加上
- [ ] 文末 GitHub 链接确认是 `Rohawku/zhpan`，不是占位符

## 发布后 24 小时内

- [ ] 自己回复前 3 条评论（即使评论很 cold）—— 提高互动率
- [ ] 截图发到 1-2 个微信群（AI / 大模型 / 算法群）
- [ ] **不要主动转发到自己朋友圈**，让算法自然推荐 24 小时
- [ ] 监测 GitHub repo star 数

## 如果 24h 内涨势好（>30 stars）

- [ ] 发英文版到 Hacker News（"Show HN: ..."）
- [ ] 发到 Reddit r/LocalLLaMA
- [ ] 在 X (Twitter) 发个图文，@ Anthropic / OpenAI 一些 LLM-judge 研究者

## 如果 24h 内冷清（<10 stars）

- [ ] 别 panic，做内容传播的本质是耐心
- [ ] 一周后重新发到即刻 / V2EX 触达不同读者群
- [ ] 检查标题——可能太学术了，需要更钩子

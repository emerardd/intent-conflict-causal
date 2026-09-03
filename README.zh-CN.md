# 行动前的用户授权冲突

*[English README](README.md) · [文档索引](docs/README.md)*

语言模型内部是否表征了**用户是否授权它即将执行的这个具体动作**？如果表征了，
这个表征对它最终做出的决策是否**因果承重**？

本仓库是围绕这个问题的五阶段预注册可解释性研究，对象是小型开源权重对话模型。
它的主要结果是一个否定结果，也正是它存在的理由：

> **可读出 ≠ 因果承重。**
> 授权状态可以从单个 residual stream 位置线性解码，留出集 AUROC 为 1.00；
> 但冻结该方向并干预到未授权试次上，决策只移动了统计上可检测、行为上可忽略的
> 幅度，且一次决策翻转都没有产生。

下文每一条结论都限制在实验真正支持的范围内。这种狭窄是刻意的，各阶段报告中
都写明了边界。

## 实验设计

只操纵一个变量：

> 在候选动作文本完全相同的情况下，用户是否授权**这个具体动作**。

每条提示同时包含一项许可和一项禁止。两个条件只交换「许可/禁止」与「候选动作/
干扰动作」的绑定，因此两边出现相同的动作内容和相同的授权词汇。模型在生成
单 token 决策 `A`（执行）或 `B`（阻止）之前接受测量。

设计针对的证据链：

1. **行为正对照**：未授权条件的 `log p(B) − log p(A)` 高于授权条件。
2. **表征泛化**：训练场景上拟合的「未授权减授权」方向，在完全留出的场景上
   保持方向一致。
3. **因果检验**：把授权条件的行动前激活替换到未授权条件，应降低阻止倾向；
   反向替换应提高阻止倾向。
4. **特异性控制**：候选动作在许可句与禁止句中的位置被配平；随机等范数替换
   不得复现主效应。

本项目**不**训练基础模型，**不**把 probe AUC 单独当作机制证据，也不研究自然
发生的作弊、动作风险、奖励作弊或事后合理化。

## 各阶段结果

| 阶段 | 问题 | 主要结论 |
| --- | --- | --- |
| **1** | 单模型上效应是否存在？ | 行为准确率 0.993；第 16 层留出集 AUROC 1.000；训练/测试方向余弦 0.977；按场景聚类的置换检验 `p = 0.001996`；双向替换的符号对齐效应 **0.637**，95% CI `[0.480, 0.766]`，随机方向均值 0.029。 |
| **2** | 能否跨精度、跨模型家族复现？ | 三次正式运行测试 AUROC 均为 1.000，12 个训练场景的精确 `2^12` 符号枚举均得 `p = 1/4096`：Qwen3.5-4B NF4 效应 0.700 `[0.645, 0.762]`；Qwen3.5-4B BF16 效应 1.268 `[1.205, 1.332]`；Ministral-3-3B BF16 效应 4.158 `[3.947, 4.365]`。每次运行 160 次同授权状态措辞替换，决策翻转 0%。 |
| **3** | 可读出的方向是否因果承重？ | **否。** 加入输出映射控制后，第 16 层图例前方向的跨场景 AUROC 为 1.00（换未训练措辞后 0.898），但 480 次冻结干预没有翻转任何首选输出；主效应 **−0.00651**，95% CI `[−0.01953, 0.00651]`，与随机方向无法区分。 |
| **4** | 是否只是「一维太小」？ | **也不是。** 图例前位置即使替换整条 residual 向量也没有可靠效应，而**答案位置**的同类替换确实能移动概率偏好。位置比维数更关键。本阶段是有界的探索性诊断，复用旧场景。 |
| **5** | 位置依赖能否在新场景确认？ | 12 个全新场景、48 条基线、576 次干预。`answer` 本地方向减等范数随机轴 **0.3672** `[0.3021, 0.4297]`；`answer − pre_mapping` 交互 0.3229 `[0.2604, 0.3802]`；`pre_mapping` 整向量 −0.0026 `[−0.0339, 0.0234]`，完全落入预设 ±0.1 等效界。四项预声明检查全部通过。 |
| **5b** | 效应是方向特异，还是仅由范数造成？ | 1,344 次干预，两根真实方向与每位置 10 根随机方向严格**等范数**（0.59587675，取自 Stage 3 训练激活并冻结）。答案方向减随机轴 **0.1990** `[0.1805, 0.2190]`；减同范数图例前方向 0.1771 `[0.1563, 0.1979]`；位置 × 方向身份交互 0.1927 `[0.1693, 0.2161]`。12/12 场景同向；独立 raw-artifact 复算通过。 |

### 最终结论的边界

Stage 5b 是**答案位置与答案方向联合特异的受控概率因果证据**。它不是行为控制
证据，也不是「模型知道自己在违反用户意图」的证据：

- 原始效应只恢复约 **2.3%** 的自然 margin；
- 1,344 次干预中**没有任何 top-1 翻转**；
- 任务仍把「未授权」与「规范上应当阻止」绑定，二者没有被分离；
- 没有自然发生的 prompt injection，也没有真实工具行为。

完整的证据演化、反对证据、被撤回的主张、限制与下一步，见
[docs/project-progress-complete-report.zh-CN.md](docs/project-progress-complete-report.zh-CN.md)。

## 仓库结构

```
configs/     实验配置，每次运行一个 JSON（溯源的基本单位）
docs/        预注册、结果报告、复现说明、运行日志
docs/figures/  报告中引用的图
scripts/     独立的 CPU 审计脚本，不导入分析代码即可复算已存证据
src/intent_conflict/   库代码：数据、模型、实验、各阶段分析
tests/       单元测试（41 个，纯 CPU，不需要模型权重）
```

`results/` **纳入版本控制**：prompts、baselines、interventions、activations、
冻结方向、环境记录、源码快照和 manifest 哈希都随仓库分发，因此
`scripts/audit_stage*.py` 在新克隆上直接可跑。各产物含义见
[docs/output-layout.md](docs/output-layout.md)。可再生的图目录（`plots/`）
与模型权重（`models/`）不纳入。

## 环境准备

需要 Python 3.11+。干预实验需要 CUDA GPU；测试和审计脚本不需要。

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

用纯 CPU 测试套件验证安装：

```bash
pytest -q
```

## 运行实验

Stage 1 与 Stage 2 通过 CLI 运行，配置文件是溯源的基本单位：

```bash
intent-conflict run        --config configs/smoke.json
intent-conflict run        --config configs/qwen_formal.json
intent-conflict robustness --config configs/qwen_formal.json
intent-conflict reanalyze  --config configs/qwen_formal.json
```

Stage 2 的跨运行汇总：

```bash
python -m intent_conflict.stage2_synthesis --root .
```

Stage 3–5b 是分阶段（phase）驱动的模块，从仓库根目录运行：

```bash
python -m intent_conflict.stage3_remap --phase screen
python -m intent_conflict.stage4  --precision nf4
python -m intent_conflict.stage5  --phase run
python -m intent_conflict.stage5b --phase run
```

每个阶段都有对应的复现说明，写明确切命令、固定的模型 revision 和适用的环境
边界：[Stage 3](docs/stage3-reproduction.md) ·
[Stage 4](docs/stage4-reproduction.md) ·
[Stage 5](docs/stage5-reproduction.md) ·
[Stage 5b](docs/stage5b-reproduction.md)。

### 继续门槛

行为筛选只有同时满足以下条件才进入正式 patching：

- 授权与未授权条件的决策准确率均至少 80%；
- 未授权减授权的平均行为 margin 为正；
- 至少 4 个筛选场景中有 3 个呈预测方向；
- 不出现对两种条件一律 `BLOCK` 或一律 `PROCEED` 的退化策略。

Stage 3 的 v1 三因子筛选未通过该门槛，按预设在阈值处停止，而没有被调参调到
显著。这次失败保留在记录中，见 [docs/stage3-run-log.md](docs/stage3-run-log.md)。

## 复算已存证据

`scripts/audit_stage*.py` 从存档的原始产物重新推导统计量，**且不导入分析
代码**，因此审计失败意味着真实的不一致，而不是共享的同一个 bug。它们只需要
Python 3.11+ 和 NumPy，不需要 GPU 和模型权重：

```bash
python scripts/audit_stage3.py
python scripts/audit_stage5b.py
```

这些脚本读取的原始产物随仓库分发在 `results/` 下，因此新克隆无需 GPU
即可核验每一个主要数字。

> **注意：**复算已存证据是比重跑推理更弱的主张，两者都不能验证某个潜在方向的
> **语义**解释。审计防的是分析代码与所报数字之间的实现漂移，而不是二者共有的
> 概念性错误。

## 模型

| 模型 | 用于 | 精度 |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | Stage 1–5b（主模型） | NF4 与 BF16 |
| `Ministral-3-3B-Instruct-2512` | Stage 2 跨家族复现 | BF16 |
| `google/gemma-2-2b-it` | 仅筛选 | BF16 |

Stage 4、5、5b 在配置中固定 Qwen 的 revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，并通过
`snapshot_download(..., local_files_only=True)` 解析，因此 revision 不匹配时
运行会直接失败，而不会静默回退到别的版本。解析到的 revision 记录在每次运行的
manifest 中。

本仓库不分发权重。请自行从 Hub 获取，或把各配置的 `model_name` 指向你自己的
缓存 —— Ministral 的配置预期本地目录 `models/ministral3-3b-instruct-2512`。

## 许可证

[MIT](LICENSE)。

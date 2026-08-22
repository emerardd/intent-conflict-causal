# Pre-action user-authorization conflict

这是一个用于 MATS 研究任务的最小因果机制实验。它只操纵一个变量：

> 在候选动作文本完全相同的情况下，用户是否授权这个具体动作。

每条提示同时包含一项许可和一项禁止。两个条件只交换“许可/禁止”与候选动作、干扰动作的绑定，因此两边都出现相同的动作内容和授权词汇。模型在生成单 token 决策 `A`（执行）或 `B`（阻止）之前接受测量。

主要证据链：

1. 行为正对照：未授权条件的 `log p(B)-log p(A)` 高于授权条件。
2. 表征泛化：训练场景的未授权减授权方向在完全留出的场景中保持方向一致。
3. 因果检验：将授权条件的行动前激活替换到未授权条件，应该降低阻止倾向；反向替换应该提高阻止倾向。
4. 特异性控制：候选动作在许可/禁止句中的位置被配平；随机等范数替换不得复现主效应。

本项目不训练基础模型，不把 probe AUC 单独当作机制证据，也不在第一阶段研究自然作弊、动作风险、奖励作弊或事后合理化。

## 运行

复用相邻 `prompt-injection-role-correction/.venv`：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = (Resolve-Path -LiteralPath .\src).Path
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli run --config .\configs\smoke.json
```

Qwen3.5-4B 行为筛选与正式实验：

```powershell
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli run --config .\configs\qwen_behavior_screen.json
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli run --config .\configs\qwen_formal.json
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli robustness --config .\configs\qwen_formal.json
```

Stage 1 正式运行已完成。主结果为：行为准确率 0.993；验证选出的第 16 层在最终场景留出集上 AUROC 1.000；训练/测试方向余弦 0.977；按场景聚类的置换检验 `p=0.001996`；双向激活替换的符号对齐效应为 0.637，按 4 个测试场景聚类 bootstrap 的 95% CI 为 `[0.480, 0.766]`，随机方向均值为 0.029。

事后探索性控制中，第 4 层相同替换效应仅为 0.023；第 16 层同授权状态、不同措辞间的替换平均绝对变化为 0.121，32 次均未翻转决定。详细解释、统计修正、最强反证和局限见 [`docs/final-report.zh-CN.md`](docs/final-report.zh-CN.md)。

## Stage 2 跨精度、跨家族复现

Stage 2 使用 40 个场景和全局唯一的候选/干扰代码词，固定 12/8/20 场景拆分。三次正式运行均完成：

- Qwen3.5-4B NF4：选中层 16，测试 AUROC 1.000，因果效应 0.700，95% CI `[0.645, 0.762]`。
- Qwen3.5-4B BF16：选中层 16，测试 AUROC 1.000，因果效应 1.268，95% CI `[1.205, 1.332]`。
- Ministral-3-3B-Instruct-2512 BF16：选中层 15，测试 AUROC 1.000，因果效应 4.158，95% CI `[3.947, 4.365]`。

三个运行的训练/测试方向余弦均大于 0.996，12 个训练场景的精确 `2^12` 场景符号枚举均得到 `p=1/4096`。错误早期层远弱于主效应；每个运行 160 次同授权状态措辞替换均为 0% 决策翻转。完整结论和边界见 [`docs/stage2-report.zh-CN.md`](docs/stage2-report.zh-CN.md)，机器可读审计见 [`results/stage2_comparison.json`](results/stage2_comparison.json)。

复现实验与生成汇总：

```powershell
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli run --config .\configs\qwen_stage2_4bit_v2_formal.json
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli run --config .\configs\qwen_stage2_bf16_formal.json
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.cli run --config .\configs\ministral3_bf16_formal.json
..\prompt-injection-role-correction\.venv\Scripts\python.exe -m intent_conflict.stage2_synthesis --root .
```

## 继续门槛

行为筛选只有同时满足以下条件才进入正式 patching：

- 授权与未授权决策准确率均至少 80%；
- 未授权减授权的平均行为 margin 为正；
- 至少 4 个筛选场景中有 3 个呈预测方向；
- 不出现对两种条件一律 `BLOCK` 或一律 `PROCEED` 的退化策略。

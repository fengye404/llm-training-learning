# Java 后端开发者的大模型训练学习路径（可执行版）

## 0. 学习目标（先明确）

12-16 周后，你应达到：
- 能说清预训练、SFT、偏好优化（DPO/GRPO）、RLHF 的关系
- 能独立完成一个小模型的 SFT 训练与评测
- 能完成 DPO 或 GRPO 的偏好对齐实验
- 能搭建最小可用 RLHF 流程（数据 -> 训练 -> 评测 -> 复盘）
- 能解释关键工程权衡：LoRA/QLoRA、显存、吞吐、收敛、过拟合

## 1. 阶段化路线

### Phase 1（第 1-2 周）：训练全景与最小必需基础

目标：从 Java 栈平滑切到训练栈。

你要掌握：
- 训练全流程地图：预训练 -> SFT -> 偏好优化 -> RLHF
- Python 工程化：venv/poetry、logging、argparse、notebook 到脚本迁移
- PyTorch 基础：tensor、autograd、optimizer、dataloader
- Transformer 训练最小闭环：tokenizer -> forward -> loss -> backward

产出：
- 一份 `notes/week1-2-foundation.md`
- 一个可运行的 PyTorch 最小训练脚本（哪怕是 toy 数据）

### Phase 2（第 3-6 周）：SFT 主线

目标：跑通可复现 SFT，并掌握常见坑。

你要掌握：
- 指令数据格式（instruction/input/output 或 chat template）
- TRL `SFTTrainer` 的核心参数（batch、lr、max_length、packing、LoRA）
- 评测基础：loss/perplexity + 基于任务样例的人工评估

实践建议：
- 从 0.5B-3B 的开源模型起步（先 LoRA/QLoRA）
- 做 2-3 轮对照实验：
  - 不同学习率
  - 不同数据清洗强度
  - 全量 SFT vs LoRA

产出：
- `projects/project-01-sft/train.py` + `config.yaml`
- `projects/project-01-sft/experiment_log.md`

### Phase 3（第 7-10 周）：偏好学习（DPO / GRPO）

目标：理解“从 SFT 到对齐”的关键差异。

你要掌握：
- 偏好数据结构（chosen/rejected）
- DPO 的直观含义：不用显式 reward model 直接做偏好优化
- GRPO 的使用场景：group 相对优势、在线生成训练样本

实践建议：
- 先做 DPO（实现简单、稳定）
- 再做 GRPO 小实验（小模型 + 小数据，关注训练稳定性）

产出：
- `projects/project-02-preference-alignment/dpo_train.py`
- `projects/project-02-preference-alignment/grpo_train.py`
- 一份对比报告：SFT vs DPO/GRPO 在你任务上的收益与副作用

### Phase 4（第 11-14 周）：RLHF 流程化与工程能力

目标：具备“能上线前验证”的完整思维。

你要掌握：
- Reward Modeling 基础
- 训练系统优化：gradient checkpointing、mixed precision、DeepSpeed ZeRO
- 评测与回归：离线 benchmark + 人工抽检 + 失败样例归档

实践建议：
- 用 OpenRLHF 或 LLaMA Factory 组装最小流水线
- 把训练过程产品化：
  - 参数配置版本化
  - 实验指标表格化
  - 关键日志结构化（便于回放）

产出：
- `projects/project-03-rlhf-pipeline/pipeline.md`
- `projects/project-03-rlhf-pipeline/eval_report.md`

### Phase 5（第 15-16 周）：强化与求职/转型输出

目标：把学习成果沉淀成可展示资产。

你要完成：
- 一个可复现仓库（README + 一键启动 + 实验记录）
- 一篇技术总结（数据、算法、工程、效果、踩坑）
- 一页“训练系统设计图”（可用于面试/内部分享）

## 2. 每周固定模板（建议）

- 2 小时：阅读官方文档/论文并记笔记
- 3 小时：跑实验（必须落地到脚本）
- 1 小时：复盘（记录失败原因、下周改进项）

复盘模板：
- 本周目标
- 实际结果
- 偏差原因
- 下周行动（最多 3 条）

## 3. 从 Java 背景迁移时的重点提醒

- 训练不是“写完就对”，而是“实验驱动”
- 先追求可复现，再追求最优指标
- 把你擅长的后端能力迁移过来：
  - 配置管理
  - 日志与监控
  - 任务编排
  - 资源成本控制

## 4. 工具选择（先简后繁）

优先级建议：
1. Hugging Face Transformers + TRL（学习主线）
2. LLaMA Factory（快速实验与多算法切换）
3. OpenRLHF（做更完整 RLHF/Agentic RL 流程）
4. DeepSpeed（规模化训练优化）

## 5. 验收标准（是否学会）

满足以下 4 条即可认为阶段成功：
- 能在新数据集上独立改造训练脚本
- 能解释至少 5 个关键超参数对结果的影响
- 能复现一次“有改进”的对照实验
- 能对失败实验给出可验证的原因假设

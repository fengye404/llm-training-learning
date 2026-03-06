# 大模型训练学习总规划（Java 后端转型版）

## 1. 目标与边界

- 时间：16 周（每周 6-8 小时）
- 目标：能独立完成 SFT + DPO（或 GRPO）+ 基础评测闭环
- 约束：优先小模型、低成本实验，先追求可复现再追求最优指标

## 2. 分阶段路线（16 周）

### Phase 1（第 1-2 周）基础打底

- 训练全景：预训练、SFT、偏好学习、RLHF 的关系
- Python/PyTorch 最小必需：dataloader、autograd、optimizer、训练循环
- 产出：
  - `notes/week1-2-foundation.md`
  - 一个 toy 训练脚本（可运行）

### Phase 2（第 3-6 周）SFT 主线

- 数据格式：instruction/chat template、清洗与切分
- 训练：Transformers + TRL `SFTTrainer` + LoRA/QLoRA
- 评测：loss、困惑度、固定样例人工评测
- 产出：
  - `projects/project-01-sft/train.py`
  - `projects/project-01-sft/config.yaml`
  - `projects/project-01-sft/experiment_log.md`

### Phase 3（第 7-10 周）偏好学习

- DPO 核心机制与数据结构（chosen/rejected）
- GRPO 适用场景与稳定性问题
- 产出：
  - `projects/project-02-preference-alignment/dpo_train.py`
  - `projects/project-02-preference-alignment/grpo_train.py`
  - 对比报告（SFT vs DPO/GRPO）

### Phase 4（第 11-14 周）RLHF 工程化

- Reward Modeling 基础
- 训练优化：mixed precision、gradient checkpointing、ZeRO
- 评测回归：离线指标 + 人工抽检 + 失败样例归档
- 产出：
  - `projects/project-03-rlhf-pipeline/pipeline.md`
  - `projects/project-03-rlhf-pipeline/eval_report.md`

### Phase 5（第 15-16 周）成果沉淀

- 整理可复现仓库：README、环境、运行说明、实验记录
- 形成可展示材料：技术总结 + 架构图 + 关键踩坑

## 3. 每周执行模板（固定）

- 2 小时：阅读与笔记（只看当前阶段相关文档）
- 3-4 小时：跑实验（至少 1 组可复现）
- 1 小时：复盘（失败原因、下周动作）

复盘模板：

1. 本周目标
2. 实际结果
3. 偏差原因
4. 下周 3 条动作

## 4. Java 后端经验迁移清单

- 配置管理：所有训练参数 YAML 化并版本化
- 可观测性：日志结构化、关键指标表格化
- 任务编排：脚本可重复执行，避免 notebook-only 流程
- 成本意识：先小模型验证，再放大规模

## 5. 验收标准（达到即算阶段成功）

- 能独立改造训练脚本适配新数据
- 能解释 5 个关键超参数影响（如 lr、batch、max_length、lora_r、epoch）
- 能完成至少 1 组有效对照实验
- 能对失败实验提出可验证假设并复现实验

## 6. 本周起步动作（立即执行）

1. 先完成 Python + PyTorch 最小训练脚本并记录日志
2. 选一个 0.5B-3B 开源模型，跑通 LoRA SFT baseline
3. 固化评测样例集（20-50 条）作为后续对齐回归基线

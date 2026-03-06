# LLM Training Learning Workspace

这个目录用于系统学习大模型训练全景，覆盖：
- 预训练与后训练的整体认知
- SFT（Supervised Fine-Tuning）
- 偏好学习（DPO / GRPO）
- RLHF 工程化（Reward Model + Policy Optimization + 评测）

## 目录结构

- `LEARNING_PLAN.md`：总规划（Java 后端转训练可执行版）
- `roadmap.md`：主学习路径（按阶段/周拆解）
- `resources/official-links.md`：官方资料索引
- `notes/`：你的学习笔记
- `projects/README.md`：每章示例代码索引与运行方式
- `projects/project-00-foundation/`：Phase 1 训练基础代码
- `projects/project-01-sft/`：SFT 实战项目
- `projects/project-02-preference-alignment/`：DPO/GRPO 实战项目
- `projects/project-03-rlhf-pipeline/`：端到端 RLHF 小型流水线项目
- `projects/project-04-capstone/`：学习成果沉淀与报告生成

## 建议学习节奏

- 每周 6-8 小时（工作日碎片 + 周末集中）
- 每个阶段都要有可运行产物（脚本、配置、实验记录）
- 优先“跑通一遍”再做性能优化

## Java 背景阅读入口

- `notes/python-for-java-backend.md`：Python 与 Java 对照速查
- `projects/README.md`：每章示例代码索引与一键运行命令

# 把优质课程教学思想融入当前学习路径（含链接）

这份文档不是“再给你一堆课”，而是把主流优质课程的常见教学方法抽出来，
直接映射到你当前仓库的 5 章内容，保证你学起来不抽象。

## 1. 先说结论：我们采用的 5 条教学原则

## 原则 1：分层学习（先能解释，再能动手，再做深水）

为什么：
- DeepLearning.AI 的 `Generative AI for Everyone` 明确面向零基础，强调“先理解 AI 能做什么/不能做什么”。
- Hugging Face LLM Course 明确要求更强 Python 基础，更适合第二层进入。
- Stanford CS336 明确是高强度深水区，适合作为第三层。

你在仓库里的落地：
- 第 0 层（认知）：`notes/00-零基础入门.md`
- 第 1 层（最小动手）：第 1~2 章
- 第 2 层（偏好与流水线）：第 3~4 章
- 第 3 层（工程复盘）：第 5 章

## 原则 2：先跑通一个最小闭环，再补理论

为什么：
- fast.ai 的核心方法是“先给可运行例子，再逐步深入”。
- 这对有工程经验但 AI 零基础的人最有效。

你在仓库里的落地：
- 每章都是可运行最小脚本
- 每章 README 都有“运行后你看什么现象”
- 先看现象（loss/奖励变化），再回头看公式

## 原则 3：短反馈循环（小步快跑）

为什么：
- Andrew Ng 的 Prompt Engineering 课程强调“迭代改进”，不是一次写完。
- LLM 学习中最怕“看很多概念但没有快速反馈”。

你在仓库里的落地：
- 每章脚本都能几秒到几分钟看到输出
- 建议每次只改一个超参数再运行
- 用第 5 章报告文件记录变化

## 原则 4：评测先行（没有评测就没有训练）

为什么：
- OpenAI 的 Evaluation best practices 明确指出：生成式系统有随机性，传统“固定断言”不够，需要系统化 eval。

你在仓库里的落地：
- 第 2 章看分类映射是否符合预期
- 第 3 章看 margin/概率是否朝“偏好正确”方向变化
- 第 4 章看期望奖励是否上升
- 第 5 章沉淀成标准报告

## 原则 5：工程化视角（你是 Java 后端，这一点是优势）

为什么：
- 现代课程都在强调“从 demo 到可维护系统”：配置、日志、评测、回归。

你在仓库里的落地：
- 配置文件（如 `config.yaml`）
- 可重复脚本执行
- 指标和报告沉淀

## 2. 外部优质内容：按你当前阶段推荐

## A. 零基础认知层（先看）

1. DeepLearning.AI - Generative AI for Everyone（Andrew Ng）
- 链接：https://www.deeplearning.ai/courses/generative-ai-for-everyone/
- 适合原因：明确写了“无需编程和 AI 先验”。

2. DeepLearning.AI - ChatGPT Prompt Engineering for Developers
- 链接：https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/
- 适合原因：时长短、反馈快、强调迭代。

## B. 开发实战层（与你当前仓库最匹配）

1. Generative AI with Large Language Models（DeepLearning.AI x AWS, Coursera）
- 链接：https://www.coursera.org/learn/generative-ai-with-llms
- 适合原因：覆盖 LLM 生命周期（数据、训练、评测、部署）。
- 注意：官方标注为中级，建议在你跑完本仓库第 1~2 章后再学。

2. Hugging Face LLM Course
- 链接：https://huggingface.co/learn/llm-course/en/chapter1/1
- 适合原因：从 Transformers 到微调到 Hub 分享，链路完整。
- 注意：官方写明需要较好 Python 基础。

## C. 深水进阶层（以后再上）

1. Stanford CS336: Language Modeling from Scratch
- 链接：https://cs336.stanford.edu/
- 适合原因：系统训练“从数据到训练到评测”的底层能力。
- 注意：课程明确要求 Python 熟练 + 深度学习/系统优化经验。

2. Andrej Karpathy - nanoGPT
- 链接：https://github.com/karpathy/nanoGPT
- 适合原因：代码简洁，适合理解“训练循环最小骨架”。
- 注意：仓库 README 已提示项目较旧，更偏学习原理。

## D. 对齐与评测工程层（和你第 3~4 章最相关）

1. Hugging Face TRL 文档
- 链接：https://huggingface.co/docs/trl/en/index
- 价值：SFT / DPO / GRPO / Reward 等训练器一站式文档。

2. OpenAI Evaluation best practices
- 链接：https://developers.openai.com/api/docs/guides/evaluation-best-practices
- 价值：把“评测驱动”落到可执行实践。

3. LLaMA Factory
- 链接：https://github.com/hiyouga/LlamaFactory
- 价值：快速切换多种微调/对齐实验。

4. OpenRLHF
- 链接：https://github.com/OpenRLHF/OpenRLHF
- 价值：更完整 RLHF/Agentic RL 工程能力。

## 3. 你的“最稳妥学习顺序”（避免再次看不懂）

第一阶段（本周）
1. 先只跑本仓库第 1 章，确认 loss 下降。
2. 再跑第 2 章，只看输入和输出映射。
3. 每次只改 1 个参数（如 epochs 或 learning_rate）。

第二阶段（下周）
1. 补 `Generative AI for Everyone` 的认知框架。
2. 跑第 3~4 章，观察偏好概率和奖励趋势。
3. 用第 5 章报告沉淀结果。

第三阶段（第 3-4 周）
1. 进入 Hugging Face LLM Course（配合你的脚本一起看）。
2. 再尝试 Coursera 的 LLM 课程。

## 4. 如何判断你是否“真的学会了”

- 你能不用术语，给同事解释第 1 章为什么 loss 会下降。
- 你能改一条第 2 章样本，并预测输出会怎么变。
- 你能解释第 3 章里 margin 变大代表什么。
- 你能解释第 4 章里为什么奖励上升不等于万无一失。
- 你能每周产出一份第 5 章报告。

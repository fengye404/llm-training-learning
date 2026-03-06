# 第 2 章代码：SFT 最小示例

## 这个脚本教你什么

- SFT 的核心：把输入指令映射到期望输出。
- 交叉熵损失和梯度更新的基本机制。

## 运行方式

```bash
python3 projects/project-01-sft/train.py
```

## Java 对照理解

- `DATASET` 可以类比训练数据表中的样本行。
- `weights` 是参数矩阵，类似一个内存中的 `double[][]`。
- `predict(...)` 类似推理接口里的预测逻辑。

## 进阶方向

理解本文件后，再迁移到 `Transformers + TRL SFTTrainer` 的真实训练流程。

# 第 1 章代码：训练基础

## 这个脚本教你什么

- 训练循环从输入到参数更新的完整过程。
- 损失与梯度如何驱动模型收敛。

## 运行方式

```bash
python3 projects/project-00-foundation/toy_autograd_train.py
```

## Java 对照理解

- `LinearModel` 可以类比为一个包含 `w`、`b` 字段的 Java 类。
- `train(...)` 类似你在服务层里按批次更新状态的循环。
- `loss` 就是你熟悉的可观测指标，可打日志、可看趋势。

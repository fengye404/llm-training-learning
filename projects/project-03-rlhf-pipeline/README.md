# 第 4 章代码：RLHF 流水线最小示例

## 这个脚本教你什么

- 策略生成、奖励打分、策略更新的闭环。
- 奖励函数设计如何影响最终策略。

## 运行方式

```bash
python3 projects/project-03-rlhf-pipeline/rlhf_pipeline_demo.py
```

## Java 对照理解

- `reward_model(...)` 可以类比评分服务。
- `logits` 可类比每个 prompt 对应的可变模型状态。
- epoch 循环可类比离线训练任务的重复执行。

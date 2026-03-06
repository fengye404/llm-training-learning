# 第 3 章代码：偏好对齐（DPO/GRPO）

## 文件说明

- `dpo_train.py`：DPO 边际优化最小示例。
- `grpo_train.py`：GRPO 组内相对优势最小示例。

## 运行方式

```bash
python3 projects/project-02-preference-alignment/dpo_train.py
python3 projects/project-02-preference-alignment/grpo_train.py
```

## Java 对照理解

- `scores: dict[str, float]` 可类比 `Map<String, Double>`。
- DPO 的 `margin` 可类比两个候选方案的排序分差。
- GRPO 的 `advantage` 可类比 `(当前得分 - 组均值)`。

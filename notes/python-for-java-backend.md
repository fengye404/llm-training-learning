# Java 后端开发者的 Python 速查

这个仓库的训练示例使用 Python。你可以把这份文档当作 Java -> Python 的对照桥梁。

## 1. 语法对照

- Python `dict` 约等于 Java `Map<K, V>`。
- Python `list` 约等于 Java `List<T>`。
- `@dataclass` 约等于一个简洁的 Java DTO/POJO。
- `for item in list` 约等于 Java 增强 for 循环。
- `if __name__ == "__main__":` 约等于 Java `public static void main` 入口。
- 类型标注（如 `list[str]`）类似 Java 泛型，但 Python 运行时仍是动态类型。

## 2. 运行习惯

- 不需要编译，直接 `python3 文件.py` 运行。
- 缩进是语法（重要性类似 Java 大括号）。
- 函数通常更小、更偏函数式，建议分解为可测试小函数。

## 3. 按章节看代码（重点）

### 第 1 章 `toy_autograd_train.py`

- 重点看 `train(...)`：这就是梯度下降训练循环。
- `model.w -= lr * grad_w` 是参数更新，思路和你在服务层做状态迭代一致。

### 第 2 章 `project-01-sft/train.py`

- 重点看 `softmax(...)`、交叉熵损失、权重更新。
- 这是 SFT 的最小概念版，便于理解机制。

### 第 3 章 `dpo_train.py` / `grpo_train.py`

- DPO：核心是拉大 `chosen_score - rejected_score`。
- GRPO：核心是 `advantage = reward - group_avg_reward`。

### 第 4 章 `rlhf_pipeline_demo.py`

- 看懂三步闭环：生成 -> 打分 -> 更新。
- `reward_model(...)` 在示例里是规则函数，生产环境会替换成训练出来的奖励模型。

### 第 5 章 `build_learning_report.py`

- 演示如何把实验指标整理成可分享的 Markdown 报告。
- 可以类比为“实验结果聚合服务”。

## 4. 推荐学习方式

1. 先运行一个脚本。
2. 再看该章节 README。
3. 把关键代码映射到你的 Java 心智模型。
4. 改一个超参数，观察输出变化。

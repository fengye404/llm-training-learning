#!/usr/bin/env python3
"""第 2 章：SFT 最小示例（行内超详细解释版）。

你可以把这个脚本理解成：
“先把文本变成数字，再用监督学习训练一个分类器，
最后把分类结果映射为固定回答模板”。

注意：
- 这是教学版，不是工业级训练代码。
- 目标是让零基础也能看懂 SFT 的核心机制。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


# ============================
# 第 0 部分：超参数默认值
# ============================
# learning_rate: 每次参数更新的步长
# epochs: 完整遍历训练集的轮数
DEFAULT_CONFIG = {
    "learning_rate": 0.2,
    "epochs": 120,
}


# =======================================
# 第 1 部分：读取配置（极简 YAML 解析器）
# =======================================
def load_simple_yaml(path: str) -> dict[str, float | int]:
    """读取形如 key: value 的极简配置文件。

    设计目的：
    - 让你先理解训练逻辑，不引入额外第三方库。
    - 你可以在不改代码的情况下调参。
    """

    cfg: dict[str, float | int] = {}

    # 逐行读取配置
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 空行和注释行直接跳过
            if not line or line.startswith("#"):
                continue

            # 解析 key/value
            key, value = [x.strip() for x in line.split(":", 1)]

            # 简单判断数字类型
            # 含小数点 -> float，否则 int
            if "." in value:
                cfg[key] = float(value)
            else:
                cfg[key] = int(value)

    return cfg


# =======================================
# 第 2 部分：文本预处理（分词）
# =======================================
def tokenize(text: str) -> list[str]:
    """把一条文本切分为词列表。

    这里约定训练文本使用“空格分词”，例如：
    "数据库 查询 很慢"

    这样做的理由：
    - 零基础更容易理解。
    - 不引入复杂中文分词依赖。
    """

    text = text.strip().lower()
    if not text:
        return []

    # 按一个或多个空白字符切分
    return [token for token in re.split(r"\s+", text) if token]


# =======================================
# 第 3 部分：样本结构定义
# =======================================
@dataclass
class Example:
    """监督学习样本。

    instruction: 输入文本
    label: 正确类别编号
    """

    instruction: str
    label: int


# =======================================
# 第 4 部分：标签到回答模板
# =======================================
# 真实大模型会自由生成文本；这里为了教学，先使用模板映射。
RESPONSES = [
    "建议先做 SQL 索引优化，并检查分页策略。",
    "建议增加缓存并监控命中率，再做下一步调优。",
    "建议先看日志定位，再复现问题并补回归测试。",
]


# =======================================
# 第 5 部分：教学数据集
# =======================================
# 标签含义：
# 0 -> 数据库/SQL 性能类
# 1 -> 缓存与性能调优类
# 2 -> 问题排查与修复类
DATASET = [
    Example("数据库 慢 查询 超时", 0),
    Example("大表 扫描 导致 延迟", 0),
    Example("缓存 命中率 低 延迟 高", 1),
    Example("需要 redis 缓存 优化", 1),
    Example("线上 bug 如何 排查", 2),
    Example("怎么 复现 问题 并 修复", 2),
]


# =======================================
# 第 6 部分：构建词表（token -> index）
# =======================================
def build_vocab(examples: list[Example]) -> dict[str, int]:
    """扫描训练集，为每个词分配一个唯一下标。"""

    vocab: dict[str, int] = {}

    for ex in examples:
        for token in tokenize(ex.instruction):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


# =======================================
# 第 7 部分：文本向量化（词袋模型）
# =======================================
def featurize(text: str, vocab: dict[str, int]) -> list[float]:
    """把文本转成向量。

    向量长度 = 词表大小。
    某词出现一次，对应位置 +1。
    """

    # 先创建全 0 向量
    x = [0.0] * len(vocab)

    # 扫描每个词，累计词频
    for token in tokenize(text):
        idx = vocab.get(token)
        if idx is not None:
            x[idx] += 1.0

    return x


# =======================================
# 第 8 部分：softmax（分数 -> 概率）
# =======================================
def softmax(logits: list[float]) -> list[float]:
    """把任意实数分数转换为概率分布。"""

    # 为了数值稳定，先减去最大值
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


# =======================================
# 第 9 部分：训练函数（核心）
# =======================================
def train(lr: float, epochs: int) -> tuple[list[list[float]], dict[str, int]]:
    """训练分类器权重矩阵。

    返回：
    - weights: 二维矩阵，weights[类别][词索引]
    - vocab: 训练得到的词表
    """

    # STEP 1) 准备词表
    vocab = build_vocab(DATASET)

    # STEP 2) 确定类别数量
    num_classes = len(RESPONSES)

    # STEP 3) 初始化参数矩阵（全 0）
    weights = [[0.0 for _ in range(len(vocab))] for _ in range(num_classes)]

    # STEP 4) epoch 循环
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        # STEP 5) 样本循环
        for ex in DATASET:
            # 5.1 文本 -> 向量
            x = featurize(ex.instruction, vocab)

            # 5.2 线性层：每个类别一个打分
            logits = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) for w_row in weights]

            # 5.3 打分 -> 概率
            probs = softmax(logits)

            # 5.4 交叉熵损失（只取正确类别概率）
            total_loss += -math.log(max(probs[ex.label], 1e-9))

            # 5.5 反向更新（softmax + CE 的常见梯度形式）
            for c in range(num_classes):
                grad = probs[c] - (1.0 if c == ex.label else 0.0)
                for i in range(len(vocab)):
                    weights[c][i] -= lr * grad * x[i]

        # STEP 6) 打印训练过程
        if epoch == 1 or epoch % 30 == 0 or epoch == epochs:
            avg_loss = total_loss / len(DATASET)
            print(f"epoch={epoch:03d} 平均loss={avg_loss:.4f}")

    return weights, vocab


# =======================================
# 第 10 部分：推理函数
# =======================================
def predict(text: str, weights: list[list[float]], vocab: dict[str, int]) -> tuple[int, float]:
    """给定输入文本，返回：
    - best_class: 最可能类别
    - confidence: 该类别概率
    """

    x = featurize(text, vocab)
    logits = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) for w_row in weights]
    probs = softmax(logits)

    best_class = max(range(len(probs)), key=lambda i: probs[i])
    return best_class, probs[best_class]


# =======================================
# 第 11 部分：主函数
# =======================================
def main() -> None:
    """程序入口。

    流程：
    1) 读配置
    2) 训练
    3) 用测试问题做预测
    """

    # 读取配置（默认值 + 外部配置覆盖）
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(load_simple_yaml("projects/project-01-sft/config.yaml"))

    # 训练
    weights, vocab = train(lr=float(cfg["learning_rate"]), epochs=int(cfg["epochs"]))

    # 测试输入（不参与训练）
    tests = [
        "数据库 查询 很慢",
        "缓存 命中率 低",
        "线上 问题 怎么 排查",
    ]

    print("\n预测结果:")
    for text in tests:
        idx, confidence = predict(text, weights, vocab)
        print(
            f"输入={text}\n"
            f"  -> 类别={idx}, 置信度={confidence:.3f}\n"
            f"  -> 响应={RESPONSES[idx]}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""第 2 章：SFT 最小示例（超详细注释版）。

这不是工业级训练代码，而是“概念教学版”。
它演示的是：
- 如何把文本问题转换成数字特征
- 如何通过监督学习让模型学会类别映射
- 如何根据类别返回对应回答模板

你可以先把它当成“文本分类 + 模板回答”的过程。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


# 默认配置：学习率和训练轮数
DEFAULT_CONFIG = {
    "learning_rate": 0.2,
    "epochs": 120,
}


def load_simple_yaml(path: str) -> dict[str, float | int]:
    """读取一个极简 YAML 配置文件。

    这里只支持这种最简单格式：
    key: value

    目的只是降低依赖，不引入额外库。
    """

    cfg: dict[str, float | int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = [x.strip() for x in line.split(":", 1)]
            if "." in value:
                cfg[key] = float(value)
            else:
                cfg[key] = int(value)
    return cfg


def tokenize(text: str) -> list[str]:
    """把句子切成词。

    为了让零基础更容易理解，这里约定：
    - 训练文本使用“空格分词”，例如："数据库 慢 查询"
    - tokenize 直接按空白切分
    """

    text = text.strip().lower()
    if not text:
        return []
    return [t for t in re.split(r"\s+", text) if t]


@dataclass
class Example:
    """一条监督学习样本：输入文本 + 正确类别。"""

    instruction: str
    label: int


# 类别到回复模板映射（真实大模型里会生成更自由文本）
RESPONSES = [
    "建议先做 SQL 索引优化，并检查分页策略。",
    "建议增加缓存并监控命中率，再做下一步调优。",
    "建议先看日志定位，再复现问题并补回归测试。",
]


# 训练集（教学版）
# 约定：每条输入都用空格分词，便于理解词袋特征
DATASET = [
    Example("数据库 慢 查询 超时", 0),
    Example("大表 扫描 导致 延迟", 0),
    Example("缓存 命中率 低 延迟 高", 1),
    Example("需要 redis 缓存 优化", 1),
    Example("线上 bug 如何 排查", 2),
    Example("怎么 复现 问题 并 修复", 2),
]


def build_vocab(examples: list[Example]) -> dict[str, int]:
    """建立词表：每个词对应一个下标。

    例如：
    {"数据库": 0, "慢": 1, ...}
    """

    vocab: dict[str, int] = {}
    for ex in examples:
        for token in tokenize(ex.instruction):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def featurize(text: str, vocab: dict[str, int]) -> list[float]:
    """把文本转成词袋向量（Bag-of-Words）。

    向量长度 = 词表大小。
    某个词出现一次，就在对应位置 +1。
    """

    x = [0.0] * len(vocab)
    for token in tokenize(text):
        idx = vocab.get(token)
        if idx is not None:
            x[idx] += 1.0
    return x


def softmax(logits: list[float]) -> list[float]:
    """把任意实数分数转换为概率分布。"""

    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def train(lr: float, epochs: int) -> tuple[list[list[float]], dict[str, int]]:
    """训练参数矩阵。

    返回：
    - weights: [类别][词索引] 的权重矩阵
    - vocab: 词表
    """

    vocab = build_vocab(DATASET)
    num_classes = len(RESPONSES)

    # 权重矩阵初始化为 0
    weights = [[0.0 for _ in range(len(vocab))] for _ in range(num_classes)]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        # 遍历每条样本
        for ex in DATASET:
            # 文本 -> 特征向量
            x = featurize(ex.instruction, vocab)

            # 线性打分：每个类别一个分数
            logits = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) for w_row in weights]

            # 分数 -> 概率
            probs = softmax(logits)

            # 交叉熵损失（只看正确类别概率）
            total_loss += -math.log(max(probs[ex.label], 1e-9))

            # 梯度更新
            for c in range(num_classes):
                # 这是 softmax + cross entropy 的常见梯度形式
                grad = probs[c] - (1.0 if c == ex.label else 0.0)

                for i in range(len(vocab)):
                    weights[c][i] -= lr * grad * x[i]

        if epoch == 1 or epoch % 30 == 0 or epoch == epochs:
            avg_loss = total_loss / len(DATASET)
            print(f"epoch={epoch:03d} 平均loss={avg_loss:.4f}")

    return weights, vocab


def predict(text: str, weights: list[list[float]], vocab: dict[str, int]) -> tuple[int, float]:
    """给定输入，返回预测类别和对应置信度。"""

    x = featurize(text, vocab)
    logits = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) for w_row in weights]
    probs = softmax(logits)
    best = max(range(len(probs)), key=lambda i: probs[i])
    return best, probs[best]


def main() -> None:
    """程序入口。"""

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(load_simple_yaml("projects/project-01-sft/config.yaml"))

    weights, vocab = train(lr=float(cfg["learning_rate"]), epochs=int(cfg["epochs"]))

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

#!/usr/bin/env python3
"""第 2 章：SFT 风格最小分类示例（instruction -> response 模板）。

不依赖外部深度学习库，先帮助你建立概念，再迁移到 TRL。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


DEFAULT_CONFIG = {
    "learning_rate": 0.2,
    "epochs": 120,
}


def load_simple_yaml(path: str) -> dict[str, float | int]:
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
    return re.findall(r"[a-zA-Z]+", text.lower())


@dataclass
class Example:
    instruction: str
    label: int


RESPONSES = [
    "Use SQL index and pagination to improve query performance.",
    "Add cache with TTL and monitor hit ratio before tuning.",
    "Start from logs, reproduce issue, and add regression tests.",
]


DATASET = [
    Example("Database query is slow in prod", 0),
    Example("Large table scan causes timeout", 0),
    Example("Cache miss rate is high", 1),
    Example("Response is slow, maybe need redis", 1),
    Example("Need debug unexpected behavior", 2),
    Example("How to locate bug and avoid rollback", 2),
]


def build_vocab(examples: list[Example]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for ex in examples:
        for t in tokenize(ex.instruction):
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def featurize(text: str, vocab: dict[str, int]) -> list[float]:
    x = [0.0] * len(vocab)
    for t in tokenize(text):
        idx = vocab.get(t)
        if idx is not None:
            x[idx] += 1.0
    return x


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def train(lr: float, epochs: int) -> tuple[list[list[float]], dict[str, int]]:
    vocab = build_vocab(DATASET)
    num_classes = len(RESPONSES)

    # 权重矩阵：weights[类别][词项]
    weights = [[0.0 for _ in range(len(vocab))] for _ in range(num_classes)]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for ex in DATASET:
            x = featurize(ex.instruction, vocab)
            logits = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) for w_row in weights]
            probs = softmax(logits)
            total_loss += -math.log(max(probs[ex.label], 1e-9))

            for c in range(num_classes):
                grad = probs[c] - (1.0 if c == ex.label else 0.0)
                for i in range(len(vocab)):
                    weights[c][i] -= lr * grad * x[i]

        if epoch == 1 or epoch % 30 == 0 or epoch == epochs:
            avg_loss = total_loss / len(DATASET)
            print(f"epoch={epoch:03d} 平均loss={avg_loss:.4f}")

    return weights, vocab


def predict(text: str, weights: list[list[float]], vocab: dict[str, int]) -> tuple[int, float]:
    x = featurize(text, vocab)
    logits = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) for w_row in weights]
    probs = softmax(logits)
    best = max(range(len(probs)), key=lambda i: probs[i])
    return best, probs[best]


def main() -> None:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(load_simple_yaml("projects/project-01-sft/config.yaml"))

    weights, vocab = train(lr=float(cfg["learning_rate"]), epochs=int(cfg["epochs"]))

    tests = [
        "API is slow due to database",
        "Cache hit is low and latency high",
        "Need method to debug issue quickly",
    ]

    print("\n预测结果:")
    for t in tests:
        idx, p = predict(t, weights, vocab)
        print(f"输入={t}\n  -> 类别={idx}, 置信度={p:.3f}\n  -> 响应={RESPONSES[idx]}")


if __name__ == "__main__":
    main()

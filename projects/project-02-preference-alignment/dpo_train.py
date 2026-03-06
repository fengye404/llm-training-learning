#!/usr/bin/env python3
"""第 3 章：DPO 最小示例（超详细注释版）。

你只要记住一句话：
DPO 在做的事情，就是不断拉大
“好答案分数 - 差答案分数”的差距。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PairSample:
    """一条偏好样本。

    - prompt: 用户问题
    - chosen: 更好的回答
    - rejected: 更差的回答
    """

    prompt: str
    chosen: str
    rejected: str


SAMPLES = [
    PairSample("怎么优化数据库查询？", "加索引并分析执行计划。", "先重启服务再说。"),
    PairSample("线上故障怎么排查？", "先看日志指标，再稳定复现。", "先等用户继续反馈。"),
    PairSample("怎么降低接口延迟？", "缓存热点并做瓶颈分析。", "只把超时时间调大。"),
]


def sigmoid(x: float) -> float:
    """sigmoid 函数，把任意实数压到 0~1。"""

    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    # beta 可以理解为“把 margin 放大的强度”
    beta = 1.0
    lr = 0.3
    epochs = 80

    # 用字典存每个回答当前分数
    # 可类比 Java 的 Map<String, Double>
    scores: dict[str, float] = {}
    for sample in SAMPLES:
        scores.setdefault(sample.chosen, 0.0)
        scores.setdefault(sample.rejected, 0.0)

    for epoch in range(1, epochs + 1):
        loss_sum = 0.0

        for sample in SAMPLES:
            chosen_score = scores[sample.chosen]
            rejected_score = scores[sample.rejected]

            # margin 越大越好
            margin = chosen_score - rejected_score

            # p 越接近 1，表示越确信 chosen 更好
            p = sigmoid(beta * margin)

            # DPO 风格损失（教学版）
            loss = -math.log(max(p, 1e-9))
            loss_sum += loss

            # 对 margin 的梯度
            g = -(1.0 - p) * beta

            # 更新逻辑：
            # chosen 分数上调，rejected 分数下调（方向由梯度决定）
            scores[sample.chosen] -= lr * g
            scores[sample.rejected] += lr * g

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} 平均loss={loss_sum / len(SAMPLES):.4f}")

    print("\n最终偏好边际（chosen - rejected）:")
    for sample in SAMPLES:
        margin = scores[sample.chosen] - scores[sample.rejected]
        print(f"问题={sample.prompt}\n  margin={margin:.4f}")


if __name__ == "__main__":
    main()

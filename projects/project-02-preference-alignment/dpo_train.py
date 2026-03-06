#!/usr/bin/env python3
"""第 3 章：DPO 最小示例（行内超详细解释版）。

本脚本的唯一目标：
让你看懂“偏好学习”到底如何改变模型选择。

一句话：
不断增大 chosen_score - rejected_score。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# =======================================
# 第 1 部分：样本结构
# =======================================
@dataclass
class PairSample:
    """一条偏好样本：同一问题下的好坏回答对。"""

    prompt: str
    chosen: str
    rejected: str


# =======================================
# 第 2 部分：偏好数据
# =======================================
# 每条数据都在表达：chosen 比 rejected 更优
SAMPLES = [
    PairSample("怎么优化数据库查询？", "加索引并分析执行计划。", "先重启服务再说。"),
    PairSample("线上故障怎么排查？", "先看日志指标，再稳定复现。", "先等用户继续反馈。"),
    PairSample("怎么降低接口延迟？", "缓存热点并做瓶颈分析。", "只把超时时间调大。"),
]


# =======================================
# 第 3 部分：sigmoid
# =======================================
def sigmoid(x: float) -> float:
    """把实数映射到 (0,1) 区间。"""
    return 1.0 / (1.0 + math.exp(-x))


# =======================================
# 第 4 部分：主训练流程
# =======================================
def main() -> None:
    # beta: margin 放大系数
    beta = 1.0

    # 学习率和训练轮数
    lr = 0.3
    epochs = 80

    # 用字典维护每个回答的“当前分数”
    # 初始都为 0，表示模型还没有偏好
    scores: dict[str, float] = {}
    for sample in SAMPLES:
        scores.setdefault(sample.chosen, 0.0)
        scores.setdefault(sample.rejected, 0.0)

    # epoch 循环
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0

        # 样本循环
        for sample in SAMPLES:
            # STEP 1) 取当前好/坏回答分数
            chosen_score = scores[sample.chosen]
            rejected_score = scores[sample.rejected]

            # STEP 2) 计算 margin（越大越好）
            margin = chosen_score - rejected_score

            # STEP 3) 分差 -> 概率
            # p 越接近 1，表示越确信 chosen 更好
            p = sigmoid(beta * margin)

            # STEP 4) 教学版 DPO 损失
            # p 越大，loss 越小
            loss = -math.log(max(p, 1e-9))
            loss_sum += loss

            # STEP 5) 对 margin 求导后的梯度项
            g = -(1.0 - p) * beta

            # STEP 6) 更新分数
            # chosen 分数向上推，rejected 分数向下压
            scores[sample.chosen] -= lr * g
            scores[sample.rejected] += lr * g

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} 平均loss={loss_sum / len(SAMPLES):.4f}")

    # 训练结束，打印每条样本最终 margin
    print("\n最终偏好边际（chosen - rejected）:")
    for sample in SAMPLES:
        margin = scores[sample.chosen] - scores[sample.rejected]
        print(f"问题={sample.prompt}\n  margin={margin:.4f}")


if __name__ == "__main__":
    main()

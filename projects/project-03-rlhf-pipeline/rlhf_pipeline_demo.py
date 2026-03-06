#!/usr/bin/env python3
"""第 4 章：RLHF 流水线最小示例（行内超详细解释版）。

教学目标：
- 看懂“生成 -> 打分 -> 更新”三步闭环。
- 看懂为什么高奖励策略会被强化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# =======================================
# 第 1 部分：问题结构
# =======================================
@dataclass
class PromptCase:
    """一个问题及其候选回答集合。"""

    prompt: str
    candidates: list[str]


# =======================================
# 第 2 部分：教学样本
# =======================================
CASES = [
    PromptCase("慢 SQL 怎么处理", ["补索引并分析执行计划", "先把超时调大"]),
    PromptCase("接口稳定性怎么提升", ["补监控并加重试", "故障时直接重启"]),
    PromptCase("线上 bug 怎么定位", ["复现并补回归测试", "直接随机打补丁"]),
]


# =======================================
# 第 3 部分：奖励模型（规则版）
# =======================================
def reward_model(text: str) -> float:
    """教学版奖励函数。

    真实 RLHF 里奖励模型通常是训练得到的神经网络。
    这里先用关键词规则模拟，便于零基础理解。
    """

    good_keywords = ["索引", "监控", "回归", "重试", "执行计划", "复现"]

    # STEP 1) 关键词命中加分
    score = 0.0
    for key in good_keywords:
        if key in text:
            score += 0.5

    # STEP 2) 稍微偏好信息量更多的回答（长度截断）
    score += min(len(text), 30) / 60.0

    return score


# =======================================
# 第 4 部分：softmax
# =======================================
def softmax(logits: list[float]) -> list[float]:
    """把 logits 转成概率分布。"""

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


# =======================================
# 第 5 部分：主训练流程
# =======================================
def main() -> None:
    lr = 0.25
    epochs = 40

    # 每个问题下，每个候选都有一个可学习 logit
    logits = {
        case.prompt: [0.0 for _ in case.candidates]
        for case in CASES
    }

    # epoch 循环
    for epoch in range(1, epochs + 1):
        epoch_reward = 0.0

        # 遍历每个问题
        for case in CASES:
            # STEP 1) 候选回答打分
            scores = [reward_model(candidate) for candidate in case.candidates]

            # STEP 2) 根据当前 logits 计算策略概率
            probs = softmax(logits[case.prompt])

            # STEP 3) 计算当前策略的期望奖励
            expected_reward = sum(p * r for p, r in zip(probs, scores))
            epoch_reward += expected_reward

            # STEP 4) 根据 advantage 更新参数
            # advantage = 当前候选分数 - 当前策略基线
            for i in range(len(case.candidates)):
                baseline = expected_reward
                advantage = scores[i] - baseline
                logits[case.prompt][i] += lr * advantage

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            avg_reward = epoch_reward / len(CASES)
            print(f"epoch={epoch:03d} 平均期望奖励={avg_reward:.4f}")

    # 训练完成，打印每个问题最终最优策略
    print("\n每个问题的最终优选策略:")
    for case in CASES:
        probs = softmax(logits[case.prompt])
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        print(
            f"问题={case.prompt}\n"
            f"  最优策略={case.candidates[best_idx]}\n"
            f"  概率={probs[best_idx]:.3f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""第 3 章：GRPO 组内相对优势最小示例（超详细注释版）。

核心直觉：
同一个问题下，多个候选回答做“组内比较”。
高于组平均奖励的回答会被强化。
"""

from __future__ import annotations

import math


GROUPS = [
    {
        "prompt": "慢接口怎么优化",
        "candidates": ["加缓存并补索引", "只增大超时", "关闭日志"],
        "rewards": [0.9, 0.2, 0.1],
    },
    {
        "prompt": "线上故障怎么处理",
        "candidates": ["复现并补测试", "不查根因先热修", "先观察不处理"],
        "rewards": [0.95, 0.4, 0.05],
    },
]


def softmax(logits: list[float]) -> list[float]:
    """把 logits 变成概率分布。"""

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def main() -> None:
    lr = 0.4
    epochs = 60

    # 每个问题都有一组候选回答，对应一组可学习 logits
    logits_by_prompt = {
        group["prompt"]: [0.0 for _ in group["candidates"]]
        for group in GROUPS
    }

    for epoch in range(1, epochs + 1):
        total_obj = 0.0

        for group in GROUPS:
            prompt = group["prompt"]
            logits = logits_by_prompt[prompt]

            probs = softmax(logits)
            rewards = group["rewards"]

            # 组平均奖励
            avg_reward = sum(rewards) / len(rewards)

            # 优势值：高于平均为正，低于平均为负
            advantages = [r - avg_reward for r in rewards]

            # 目标函数（教学版）
            # obj = sum(log pi(a_i|s) * A_i)
            obj = 0.0
            for p, adv in zip(probs, advantages):
                obj += math.log(max(p, 1e-9)) * adv
            total_obj += obj

            # 参数更新（教学版近似）
            for i in range(len(logits)):
                expected_adv = sum(
                    probs[j] * advantages[j]
                    for j in range(len(logits))
                )
                grad = advantages[i] - expected_adv
                logits[i] += lr * grad

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} 目标值={total_obj:.4f}")

    print("\n最终候选回答概率:")
    for group in GROUPS:
        probs = softmax(logits_by_prompt[group["prompt"]])
        print(f"问题={group['prompt']}")
        for candidate, p in zip(group["candidates"], probs):
            print(f"  {candidate:20s} 概率={p:.3f}")


if __name__ == "__main__":
    main()

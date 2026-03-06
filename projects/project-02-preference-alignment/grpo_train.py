#!/usr/bin/env python3
"""第 3 章：GRPO 风格组内相对优势示例。

这是教学版近似实现，重点是理解机制。
"""

from __future__ import annotations

import math


GROUPS = [
    {
        "prompt": "Optimize slow API",
        "candidates": ["Add cache and index", "Increase timeout", "Disable logs"],
        "rewards": [0.9, 0.2, 0.1],
    },
    {
        "prompt": "Handle prod bug",
        "candidates": ["Reproduce + add test", "Hotfix without root cause", "Wait and observe"],
        "rewards": [0.95, 0.4, 0.05],
    },
]


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def main() -> None:
    lr = 0.4
    epochs = 60

    # 每个 prompt 对应一组候选响应的 logits
    logits_by_prompt = {
        g["prompt"]: [0.0 for _ in g["candidates"]] for g in GROUPS
    }

    for epoch in range(1, epochs + 1):
        total_obj = 0.0

        for g in GROUPS:
            prompt = g["prompt"]
            logits = logits_by_prompt[prompt]
            probs = softmax(logits)
            rewards = g["rewards"]
            avg_reward = sum(rewards) / len(rewards)
            advantages = [r - avg_reward for r in rewards]

            # 目标函数: sum(log pi(a_i|s) * A_i)
            obj = 0.0
            for p, a in zip(probs, advantages):
                obj += math.log(max(p, 1e-9)) * a
            total_obj += obj

            # softmax 策略梯度的教学版更新
            for i in range(len(logits)):
                expected_adv = sum(probs[j] * advantages[j] for j in range(len(logits)))
                grad = advantages[i] - expected_adv
                logits[i] += lr * grad

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} 目标值={total_obj:.4f}")

    print("\n最终候选响应概率:")
    for g in GROUPS:
        probs = softmax(logits_by_prompt[g["prompt"]])
        print(f"prompt={g['prompt']}")
        for c, p in zip(g["candidates"], probs):
            print(f"  {c:30s} 概率={p:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 3: GRPO-style group relative advantage demo.

This is an educational approximation to explain the mechanism.
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

    # Each prompt has logits for its candidate responses.
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

            # Objective: sum(log pi(a_i|s) * A_i)
            obj = 0.0
            for p, a in zip(probs, advantages):
                obj += math.log(max(p, 1e-9)) * a
            total_obj += obj

            # Gradient for logits in softmax policy gradient style.
            for i in range(len(logits)):
                expected_adv = sum(probs[j] * advantages[j] for j in range(len(logits)))
                grad = advantages[i] - expected_adv
                logits[i] += lr * grad

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} objective={total_obj:.4f}")

    print("\nFinal candidate probabilities:")
    for g in GROUPS:
        probs = softmax(logits_by_prompt[g["prompt"]])
        print(f"prompt={g['prompt']}")
        for c, p in zip(g["candidates"], probs):
            print(f"  {c:30s} prob={p:.3f}")


if __name__ == "__main__":
    main()

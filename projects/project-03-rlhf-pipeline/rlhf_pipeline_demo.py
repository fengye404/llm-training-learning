#!/usr/bin/env python3
"""Phase 4: minimal RLHF pipeline demo.

Pipeline:
1) policy generate candidates
2) reward model score candidates
3) policy update toward higher reward
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PromptCase:
    prompt: str
    candidates: list[str]


CASES = [
    PromptCase("How to handle slow DB query?", ["add index and inspect plan", "increase timeout"]),
    PromptCase("How to improve API reliability?", ["add metrics and retries", "restart on failure"]),
    PromptCase("How to debug production bug?", ["reproduce and add regression test", "deploy random patch"]),
]


def reward_model(text: str) -> float:
    good_keywords = ["index", "metrics", "regression", "retries", "inspect", "reproduce"]
    score = 0.0
    for k in good_keywords:
        if k in text:
            score += 0.5
    score += min(len(text), 60) / 120.0
    return score


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def main() -> None:
    lr = 0.25
    epochs = 40

    logits = {case.prompt: [0.0 for _ in case.candidates] for case in CASES}

    for epoch in range(1, epochs + 1):
        epoch_reward = 0.0

        for case in CASES:
            scores = [reward_model(c) for c in case.candidates]
            probs = softmax(logits[case.prompt])
            expected_reward = sum(p * r for p, r in zip(probs, scores))
            epoch_reward += expected_reward

            for i in range(len(case.candidates)):
                baseline = expected_reward
                advantage = scores[i] - baseline
                logits[case.prompt][i] += lr * advantage

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            avg_reward = epoch_reward / len(CASES)
            print(f"epoch={epoch:03d} avg_expected_reward={avg_reward:.4f}")

    print("\nFinal selected strategy per prompt:")
    for case in CASES:
        probs = softmax(logits[case.prompt])
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        print(f"prompt={case.prompt}\n  best={case.candidates[best_idx]}\n  prob={probs[best_idx]:.3f}")


if __name__ == "__main__":
    main()

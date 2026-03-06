#!/usr/bin/env python3
"""Phase 3: DPO objective demo without heavy dependencies.

DPO key idea:
Increase score(chosen) - score(rejected).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PairSample:
    prompt: str
    chosen: str
    rejected: str


SAMPLES = [
    PairSample("How to optimize DB query?", "Use index and inspect execution plan.", "Just restart service."),
    PairSample("How to debug production issue?", "Check logs, metrics, and reproduce.", "Ignore until users complain."),
    PairSample("How to reduce latency?", "Cache hot keys and profile bottlenecks.", "Increase timeout only."),
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    beta = 1.0
    lr = 0.3
    epochs = 80

    # Policy score table, similar to a Java Map<String, Double>.
    scores: dict[str, float] = {}
    for s in SAMPLES:
        scores.setdefault(s.chosen, 0.0)
        scores.setdefault(s.rejected, 0.0)

    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        for s in SAMPLES:
            margin = scores[s.chosen] - scores[s.rejected]
            p = sigmoid(beta * margin)
            loss = -math.log(max(p, 1e-9))
            loss_sum += loss

            # dloss/dmargin = -(1 - sigmoid(beta*margin)) * beta
            g = -(1.0 - p) * beta
            scores[s.chosen] -= lr * g
            scores[s.rejected] += lr * g

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} avg_loss={loss_sum / len(SAMPLES):.4f}")

    print("\nFinal preference margins (chosen - rejected):")
    for s in SAMPLES:
        margin = scores[s.chosen] - scores[s.rejected]
        print(f"prompt={s.prompt}\n  margin={margin:.4f}")


if __name__ == "__main__":
    main()

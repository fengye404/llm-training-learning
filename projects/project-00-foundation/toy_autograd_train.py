#!/usr/bin/env python3
"""Phase 1: tiny training loop for a linear model.

Java mapping:
- variables w/b ~= fields in a POJO
- for epoch loop ~= scheduled batch job iterations
- gradient descent step ~= manual state update
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class LinearModel:
    w: float
    b: float

    def predict(self, x: float) -> float:
        return self.w * x + self.b


def make_dataset(n: int = 80) -> list[tuple[float, float]]:
    random.seed(42)
    data: list[tuple[float, float]] = []
    for _ in range(n):
        x = random.uniform(-5.0, 5.0)
        noise = random.uniform(-0.1, 0.1)
        y = 2.0 * x + 1.0 + noise
        data.append((x, y))
    return data


def train(model: LinearModel, data: list[tuple[float, float]], lr: float, epochs: int) -> None:
    n = float(len(data))
    for epoch in range(1, epochs + 1):
        grad_w = 0.0
        grad_b = 0.0
        loss = 0.0

        for x, y in data:
            pred = model.predict(x)
            err = pred - y
            loss += err * err
            grad_w += 2.0 * err * x
            grad_b += 2.0 * err

        grad_w /= n
        grad_b /= n
        loss /= n

        model.w -= lr * grad_w
        model.b -= lr * grad_b

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={loss:.6f} w={model.w:.4f} b={model.b:.4f}")


def main() -> None:
    data = make_dataset()
    model = LinearModel(w=0.0, b=0.0)
    train(model, data, lr=0.03, epochs=120)

    test_x = 3.0
    test_pred = model.predict(test_x)
    print(f"test: x={test_x}, pred={test_pred:.4f}, expected~=7.0")


if __name__ == "__main__":
    main()

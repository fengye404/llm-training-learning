#!/usr/bin/env python3
"""第 1 章：线性模型最小训练循环（超详细注释版）。

你可以把这个脚本当成“训练机制演示器”。
目标不是做复杂模型，而是看懂最核心的 4 步：
1) 预测
2) 算误差
3) 算梯度
4) 更新参数

Java 后端类比：
- 模型参数（w,b）≈ 一个会被循环更新的状态对象字段
- 每个 epoch ≈ 一轮离线批任务
- loss ≈ 监控指标（越低越好）
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class LinearModel:
    """最简单的一元线性模型: y = w * x + b。

    - w: 斜率
    - b: 截距
    """

    w: float
    b: float

    def predict(self, x: float) -> float:
        """给定输入 x，返回预测值。"""
        return self.w * x + self.b


def make_dataset(n: int = 80) -> list[tuple[float, float]]:
    """构造一份“带轻微噪声”的训练数据。

    我们故意按真实规律生成数据：
    y = 2*x + 1 + noise

    这样训练完后，理想参数会接近：w=2, b=1。
    """

    random.seed(42)  # 固定随机种子，保证你每次运行结果一致，便于学习
    data: list[tuple[float, float]] = []

    for _ in range(n):
        x = random.uniform(-5.0, 5.0)
        noise = random.uniform(-0.1, 0.1)
        y = 2.0 * x + 1.0 + noise
        data.append((x, y))

    return data


def train(model: LinearModel, data: list[tuple[float, float]], lr: float, epochs: int) -> None:
    """训练主循环。

    参数说明：
    - model: 需要学习参数的模型
    - data: 训练样本
    - lr: 学习率（每次更新步子大小）
    - epochs: 训练轮数

    数学核心（均方误差 MSE）：
    loss = average((pred - y)^2)
    """

    n = float(len(data))

    for epoch in range(1, epochs + 1):
        # 每一轮开始时，先清空累积值
        grad_w = 0.0
        grad_b = 0.0
        loss = 0.0

        # 遍历所有样本，累计 loss 和梯度
        for x, y in data:
            pred = model.predict(x)   # 第一步：预测
            err = pred - y            # 第二步：误差（预测值 - 真值）

            # 累积平方误差
            loss += err * err

            # 第三步：累计梯度（对 MSE 求导后的结果）
            grad_w += 2.0 * err * x
            grad_b += 2.0 * err

        # 求平均，防止梯度规模和样本数强绑定
        grad_w /= n
        grad_b /= n
        loss /= n

        # 第四步：梯度下降更新参数
        # 新参数 = 旧参数 - 学习率 * 梯度
        model.w -= lr * grad_w
        model.b -= lr * grad_b

        # 打印训练过程，观察是否在收敛
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={loss:.6f} w={model.w:.4f} b={model.b:.4f}")


def main() -> None:
    """程序入口。"""

    data = make_dataset()

    # 参数从 0 开始，模拟“模型一开始什么都不会”
    model = LinearModel(w=0.0, b=0.0)

    # 开始训练
    train(model, data, lr=0.03, epochs=120)

    # 训练后做一个小测试
    test_x = 3.0
    test_pred = model.predict(test_x)
    print(f"测试: x={test_x}, 预测值={test_pred:.4f}, 期望值约为7.0")


if __name__ == "__main__":
    main()

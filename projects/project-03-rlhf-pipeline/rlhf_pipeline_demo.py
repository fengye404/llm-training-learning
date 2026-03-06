#!/usr/bin/env python3
"""第 4 章：RLHF 流水线最小示例（超详细注释版）。

把流程想成一个循环：
1) 策略模型先给多个候选回答
2) 奖励模型给候选回答打分
3) 策略模型根据分数调整自己

这份代码是教学简化版，不追求工业级精确实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PromptCase:
    """一个问题及其候选回答集合。"""

    prompt: str
    candidates: list[str]


CASES = [
    PromptCase("慢 SQL 怎么处理", ["补索引并分析执行计划", "先把超时调大"]),
    PromptCase("接口稳定性怎么提升", ["补监控并加重试", "故障时直接重启"]),
    PromptCase("线上 bug 怎么定位", ["复现并补回归测试", "直接随机打补丁"]),
]


def reward_model(text: str) -> float:
    """一个教学版奖励函数。

    真正 RLHF 里，reward model 往往是训练出来的模型。
    这里用关键词规则代替，降低理解难度。
    """

    good_keywords = ["索引", "监控", "回归", "重试", "执行计划", "复现"]

    score = 0.0
    for key in good_keywords:
        if key in text:
            score += 0.5

    # 稍微偏好信息量更充分的回答（长度上限截断）
    score += min(len(text), 30) / 60.0

    return score


def softmax(logits: list[float]) -> list[float]:
    """把 logits 变成候选概率分布。"""

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def main() -> None:
    lr = 0.25
    epochs = 40

    # 每个问题下每个候选回答都有一个可学习 logit
    logits = {
        case.prompt: [0.0 for _ in case.candidates]
        for case in CASES
    }

    for epoch in range(1, epochs + 1):
        epoch_reward = 0.0

        for case in CASES:
            # 第一步：对当前候选打奖励分
            scores = [reward_model(candidate) for candidate in case.candidates]

            # 第二步：根据当前 logits 得到策略概率
            probs = softmax(logits[case.prompt])

            # 计算当前策略的期望奖励
            expected_reward = sum(p * r for p, r in zip(probs, scores))
            epoch_reward += expected_reward

            # 第三步：根据“高于基线/低于基线”更新 logits
            for i in range(len(case.candidates)):
                baseline = expected_reward
                advantage = scores[i] - baseline
                logits[case.prompt][i] += lr * advantage

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            avg_reward = epoch_reward / len(CASES)
            print(f"epoch={epoch:03d} 平均期望奖励={avg_reward:.4f}")

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

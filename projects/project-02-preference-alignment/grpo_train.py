#!/usr/bin/env python3
"""第 3 章：GRPO 最小示例（行内超详细解释版）。

核心直觉：
同一个问题有多个候选回答。
我们比较“相对组平均奖励”的优势（advantage），
然后让高优势回答概率逐步上升。
"""

from __future__ import annotations

import math


# =======================================
# 第 1 部分：训练数据
# =======================================
# 每个问题是一组候选回答 + 奖励分
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


# =======================================
# 第 2 部分：softmax
# =======================================
def softmax(logits: list[float]) -> list[float]:
    """将 logits 转为概率分布。"""

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


# =======================================
# 第 3 部分：主训练流程
# =======================================
def main() -> None:
    lr = 0.4
    epochs = 60

    # 为每个 prompt 初始化一组可学习 logits（全 0）
    logits_by_prompt = {
        group["prompt"]: [0.0 for _ in group["candidates"]]
        for group in GROUPS
    }

    for epoch in range(1, epochs + 1):
        total_obj = 0.0

        # 遍历每个问题组
        for group in GROUPS:
            prompt = group["prompt"]
            logits = logits_by_prompt[prompt]

            # STEP 1) 当前策略概率
            probs = softmax(logits)

            # STEP 2) 读取奖励并计算组均值
            rewards = group["rewards"]
            avg_reward = sum(rewards) / len(rewards)

            # STEP 3) 计算优势值
            # advantage > 0 表示优于组均值
            advantages = [r - avg_reward for r in rewards]

            # STEP 4) 计算教学版目标函数
            # obj = sum(log(pi_i) * advantage_i)
            obj = 0.0
            for p, adv in zip(probs, advantages):
                obj += math.log(max(p, 1e-9)) * adv
            total_obj += obj

            # STEP 5) 参数更新（教学版近似）
            for i in range(len(logits)):
                expected_adv = sum(
                    probs[j] * advantages[j]
                    for j in range(len(logits))
                )
                grad = advantages[i] - expected_adv
                logits[i] += lr * grad

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} 目标值={total_obj:.4f}")

    # 打印最终候选概率
    print("\n最终候选回答概率:")
    for group in GROUPS:
        probs = softmax(logits_by_prompt[group["prompt"]])
        print(f"问题={group['prompt']}")
        for candidate, p in zip(group["candidates"], probs):
            print(f"  {candidate:20s} 概率={p:.3f}")


if __name__ == "__main__":
    main()

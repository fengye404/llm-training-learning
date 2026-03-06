#!/usr/bin/env python3
"""第 5 章：根据实验快照生成学习报告（Markdown）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Experiment:
    phase: str
    metric_name: str
    before: float
    after: float
    note: str


EXPERIMENTS = [
    Experiment("第 1 章", "toy_loss", 4.8, 0.02, "线性模型完成收敛"),
    Experiment("第 2 章", "sft_avg_loss", 1.2, 0.15, "指令到输出映射已学到"),
    Experiment("第 3 章", "dpo_margin", 0.0, 2.4, "模型更偏好 chosen 响应"),
    Experiment("第 4 章", "expected_reward", 0.55, 1.35, "策略向高奖励动作收敛"),
]


def to_markdown(experiments: list[Experiment]) -> str:
    lines = []
    lines.append("# 学习报告")
    lines.append("")
    lines.append(f"生成日期: {date.today().isoformat()}")
    lines.append("")
    lines.append("| 章节 | 指标 | 变更前 | 变更后 | 差值 | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for e in experiments:
        delta = e.after - e.before
        lines.append(
            f"| {e.phase} | {e.metric_name} | {e.before:.3f} | {e.after:.3f} | {delta:+.3f} | {e.note} |"
        )

    lines.append("")
    lines.append("## 下一步行动")
    lines.append("1. 把示例中的模拟数据替换成真实训练日志。")
    lines.append("2. 增加一条失败实验并补充根因分析。")
    lines.append("3. 固化一组 benchmark 提示词用于回归检查。")
    return "\n".join(lines)


def main() -> None:
    md = to_markdown(EXPERIMENTS)
    out = "projects/project-04-capstone/learning_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已写入: {out}")


if __name__ == "__main__":
    main()

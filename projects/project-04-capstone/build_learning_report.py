#!/usr/bin/env python3
"""第 5 章：生成学习报告（超详细注释版）。

目标：
把前面章节的关键指标汇总成一份 Markdown 报告，
便于你复盘、分享、面试展示。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Experiment:
    """一条实验记录。

    - phase: 章节
    - metric_name: 指标名
    - before: 优化前
    - after: 优化后
    - note: 备注
    """

    phase: str
    metric_name: str
    before: float
    after: float
    note: str


# 教学示例数据。
# 你后续可以替换成自己真实跑出来的数值。
EXPERIMENTS = [
    Experiment("第 1 章", "toy_loss", 4.8, 0.02, "线性模型完成收敛"),
    Experiment("第 2 章", "sft_avg_loss", 1.2, 0.15, "指令到输出映射已学到"),
    Experiment("第 3 章", "dpo_margin", 0.0, 2.4, "模型更偏好 chosen 响应"),
    Experiment("第 4 章", "expected_reward", 0.55, 1.35, "策略向高奖励动作收敛"),
]


def to_markdown(experiments: list[Experiment]) -> str:
    """把实验列表拼成 Markdown 文本。"""

    lines = []
    lines.append("# 学习报告")
    lines.append("")
    lines.append(f"生成日期: {date.today().isoformat()}")
    lines.append("")
    lines.append("| 章节 | 指标 | 变更前 | 变更后 | 差值 | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for exp in experiments:
        delta = exp.after - exp.before
        lines.append(
            f"| {exp.phase} | {exp.metric_name} | {exp.before:.3f} | {exp.after:.3f} | {delta:+.3f} | {exp.note} |"
        )

    lines.append("")
    lines.append("## 下一步行动")
    lines.append("1. 把示例中的模拟数据替换成真实训练日志。")
    lines.append("2. 增加一条失败实验并补充根因分析。")
    lines.append("3. 固化一组 benchmark 提示词用于回归检查。")

    return "\n".join(lines)


def main() -> None:
    """程序入口：生成 Markdown 并写入文件。"""

    markdown_text = to_markdown(EXPERIMENTS)
    out_path = "projects/project-04-capstone/learning_report.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()

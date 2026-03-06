#!/usr/bin/env python3
"""第 5 章：生成学习报告（行内超详细解释版）。

目的：
- 把实验结果整理成 Markdown 报告。
- 形成可复盘、可分享、可追踪的学习产物。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


# =======================================
# 第 1 部分：实验记录数据结构
# =======================================
@dataclass
class Experiment:
    """一条实验记录。"""

    phase: str
    metric_name: str
    before: float
    after: float
    note: str


# =======================================
# 第 2 部分：教学示例数据
# =======================================
# 你后续可以把这里替换成真实训练结果。
EXPERIMENTS = [
    Experiment("第 1 章", "toy_loss", 4.8, 0.02, "线性模型完成收敛"),
    Experiment("第 2 章", "sft_avg_loss", 1.2, 0.15, "指令到输出映射已学到"),
    Experiment("第 3 章", "dpo_margin", 0.0, 2.4, "模型更偏好 chosen 响应"),
    Experiment("第 4 章", "expected_reward", 0.55, 1.35, "策略向高奖励动作收敛"),
]


# =======================================
# 第 3 部分：生成 Markdown 文本
# =======================================
def to_markdown(experiments: list[Experiment]) -> str:
    """把实验列表拼接为 Markdown 报告字符串。"""

    lines = []

    # STEP 1) 标题和日期
    lines.append("# 学习报告")
    lines.append("")
    lines.append(f"生成日期: {date.today().isoformat()}")
    lines.append("")

    # STEP 2) 表头
    lines.append("| 章节 | 指标 | 变更前 | 变更后 | 差值 | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")

    # STEP 3) 表体
    for exp in experiments:
        delta = exp.after - exp.before
        lines.append(
            f"| {exp.phase} | {exp.metric_name} | {exp.before:.3f} | {exp.after:.3f} | {delta:+.3f} | {exp.note} |"
        )

    # STEP 4) 下一步行动
    lines.append("")
    lines.append("## 下一步行动")
    lines.append("1. 把示例中的模拟数据替换成真实训练日志。")
    lines.append("2. 增加一条失败实验并补充根因分析。")
    lines.append("3. 固化一组 benchmark 提示词用于回归检查。")

    return "\n".join(lines)


# =======================================
# 第 4 部分：主函数
# =======================================
def main() -> None:
    """生成报告并写入文件。"""

    # STEP 1) 生成 Markdown 文本
    markdown_text = to_markdown(EXPERIMENTS)

    # STEP 2) 指定输出路径
    out_path = "projects/project-04-capstone/learning_report.md"

    # STEP 3) 写文件
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    # STEP 4) 输出提示
    print(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()

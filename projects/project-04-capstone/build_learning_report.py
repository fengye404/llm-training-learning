#!/usr/bin/env python3
"""Phase 5: build a markdown learning report from experiment snapshots."""

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
    Experiment("Phase 1", "toy_loss", 4.8, 0.02, "linear model converged"),
    Experiment("Phase 2", "sft_avg_loss", 1.2, 0.15, "instruction mapping learned"),
    Experiment("Phase 3", "dpo_margin", 0.0, 2.4, "chosen responses preferred"),
    Experiment("Phase 4", "expected_reward", 0.55, 1.35, "policy moved to high-reward actions"),
]


def to_markdown(experiments: list[Experiment]) -> str:
    lines = []
    lines.append("# Learning Report")
    lines.append("")
    lines.append(f"Generated on: {date.today().isoformat()}")
    lines.append("")
    lines.append("| Phase | Metric | Before | After | Delta | Note |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for e in experiments:
        delta = e.after - e.before
        lines.append(
            f"| {e.phase} | {e.metric_name} | {e.before:.3f} | {e.after:.3f} | {delta:+.3f} | {e.note} |"
        )

    lines.append("")
    lines.append("## Next actions")
    lines.append("1. Replace mock numbers with real logs from your scripts.")
    lines.append("2. Add one failed experiment and root-cause analysis.")
    lines.append("3. Keep one benchmark prompt set for regression check.")
    return "\n".join(lines)


def main() -> None:
    md = to_markdown(EXPERIMENTS)
    out = "projects/project-04-capstone/learning_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"report written: {out}")


if __name__ == "__main__":
    main()

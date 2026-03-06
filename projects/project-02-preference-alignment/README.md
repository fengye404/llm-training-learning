# Phase 3 Code: Preference Alignment

## Files

- `dpo_train.py`: simple DPO margin optimization demo.
- `grpo_train.py`: group-relative advantage demo.

## Run

```bash
python3 projects/project-02-preference-alignment/dpo_train.py
python3 projects/project-02-preference-alignment/grpo_train.py
```

## Java mapping

- `scores: dict[str, float]` ~= `Map<String, Double>`.
- `margin` in DPO is like comparing two candidates' ranking scores.
- GRPO `advantage` is like `(currentScore - groupAverageScore)` in ranking systems.

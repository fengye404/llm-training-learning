# Phase 4 Code: RLHF Pipeline Mini Demo

## What this script teaches

- Policy generation, reward scoring, policy update feedback loop.
- Why reward design quality affects policy behavior.

## Run

```bash
python3 projects/project-03-rlhf-pipeline/rlhf_pipeline_demo.py
```

## Java mapping

- `reward_model(...)` is like a scoring service.
- `logits` is like per-prompt mutable model state.
- Epoch loop is like repeated offline training jobs.

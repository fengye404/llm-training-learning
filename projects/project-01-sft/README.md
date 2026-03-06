# Phase 2 Code: SFT Mini Demo

## What this script teaches

- Supervised fine-tuning core idea: map instruction -> preferred output.
- Cross-entropy loss and gradient update.

## Run

```bash
python3 projects/project-01-sft/train.py
```

## Java mapping

- `DATASET` is like your training table rows.
- `weights` is like an in-memory parameter matrix (similar to a 2D `double[][]`).
- `predict(...)` is like model inference endpoint logic.

## Real-world extension

After understanding this file, move to `Transformers + TRL SFTTrainer` with real tokenizers and base models.

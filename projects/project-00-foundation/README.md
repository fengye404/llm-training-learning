# Phase 1 Code: Foundation

## What this script teaches

- How a training loop works end-to-end.
- How loss and gradients change model parameters.

## Run

```bash
python3 projects/project-00-foundation/toy_autograd_train.py
```

## Java mapping

- `LinearModel` is like a simple Java class with fields `w` and `b`.
- `train(...)` is like your service-layer loop that updates state every batch.
- `loss` is a metric similar to what you would record in logs/monitoring.

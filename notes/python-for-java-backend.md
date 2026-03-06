# Python for Java Backend (Quick Mapping)

This repository uses Python for model-training learning examples.
Use this note as a Java-to-Python bridge.

## 1. Syntax mapping

- `dict` in Python ~= `Map<K, V>` in Java.
- `list` in Python ~= `List<T>` in Java.
- `@dataclass` ~= Java DTO/POJO with generated constructor/fields.
- `for item in list` ~= Java enhanced for-loop.
- `if __name__ == "__main__":` ~= Java `public static void main` entry.
- Type hints like `list[str]` are similar to Java generics, but runtime is dynamic.

## 2. Runtime habits

- No compile step: run directly with `python3 file.py`.
- Indentation is syntax (similar importance to braces in Java).
- Functions are first-class and lightweight; prefer small pure functions.

## 3. Chapter-by-chapter reading focus

### Phase 1 `toy_autograd_train.py`

- Focus on `train(...)`: this is your gradient-descent batch loop.
- `model.w -= lr * grad_w` is parameter update (same concept as state mutation in service loop).

### Phase 2 `project-01-sft/train.py`

- Focus on `softmax(...)`, cross-entropy loss, and weight update loop.
- This is a tiny supervised classifier standing in for real SFT.

### Phase 3 `dpo_train.py` / `grpo_train.py`

- DPO: optimize `chosen_score - rejected_score` margin.
- GRPO: use `advantage = reward - group_avg_reward` to update policy.

### Phase 4 `rlhf_pipeline_demo.py`

- Understand the 3-step loop: generate -> score -> update.
- `reward_model(...)` is intentionally simple; in production this is a trained model/service.

### Phase 5 `build_learning_report.py`

- Shows how to convert experiment metrics into markdown output.
- Treat as report aggregation service.

## 4. Suggested way to learn

1. Run one script.
2. Read its chapter README.
3. Compare code blocks to your Java mental model.
4. Modify one hyperparameter and observe output change.

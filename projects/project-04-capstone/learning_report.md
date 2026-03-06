# Learning Report

Generated on: 2026-03-07

| Phase | Metric | Before | After | Delta | Note |
|---|---:|---:|---:|---:|---|
| Phase 1 | toy_loss | 4.800 | 0.020 | -4.780 | linear model converged |
| Phase 2 | sft_avg_loss | 1.200 | 0.150 | -1.050 | instruction mapping learned |
| Phase 3 | dpo_margin | 0.000 | 2.400 | +2.400 | chosen responses preferred |
| Phase 4 | expected_reward | 0.550 | 1.350 | +0.800 | policy moved to high-reward actions |

## Next actions
1. Replace mock numbers with real logs from your scripts.
2. Add one failed experiment and root-cause analysis.
3. Keep one benchmark prompt set for regression check.
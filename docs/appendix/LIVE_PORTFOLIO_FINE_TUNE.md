# Live Portfolio Fine Tune

## Purpose

This document explains the historical 12-candidate portfolio fine-tune layer.
Phase1 used the rounded `stable_compact_current` output from this layer.
Phase2 then added a separate current/capacity portfolio blend in
`tune_portfolio_blend_current_capacity.py`.

## Role In The Pipeline

`live_portfolio_fine_tune_3_7_13.py` converted each retained model pair into a
legal long-only portfolio by tuning:

| Parameter | Meaning |
| --- | --- |
| general/short weight | Ensemble weight between the candidate's two XGB profiles |
| `top_k` | Number of selected stocks |
| internal cap | Concentration cap before final normalization |
| risk filter | Whether to remove recent halt/illiquidity names |
| weighting | Equal or rank-weighted portfolio |

It produced the Phase1 source candidate and the two Phase2 source candidates:

| Candidate | Portfolio Role |
| --- | --- |
| `stable_compact_current` | 31-stock rank-weighted portfolio with higher recent mean and higher sensitivity |
| `stable_compact_capacity` | 33-stock equal-weighted portfolio with stronger robustness in recent anchors |

## Commands

Prerequisite: strict live-style model caches are available.

```bash
conda run -n ml26s python live_walk_forward_validation.py \
  --num-windows 13 \
  --window-length 5 \
  --out-dir experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing
```

Run the retained-candidate portfolio fine-tune:

```bash
conda run -n ml26s python live_portfolio_fine_tune_3_7_13.py \
  --out-dir experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13 \
  --source-run experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing \
  --progress-every 50000
```

Run the robustness audit:

```bash
conda run -n ml26s python audit_portfolio_param_robustness.py \
  --out-dir experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/robustness_audit \
  --skip-existing
```

## Current Phase2 Decision

The final phase2 file is not a single 12-candidate output. It is a
portfolio-level blend:

```text
0.45 * stable_compact_current + 0.55 * stable_compact_capacity
top_k = 44
```

Generate it with:

```bash
conda run -n ml26s python tune_portfolio_blend_current_capacity.py \
  --write-best-submission \
  --submission-out submissions/submission2.csv \
  --scores-out submissions/submission2.scores.csv
```

Then validate:

```bash
conda run -n ml26s python validate_submission.py submissions/submission2.csv
```

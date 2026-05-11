# Research Tuning Workflow

## Goal

The research workflow finds candidate feature families, XGB shapes, and
portfolio rules without using future data. It is slower than the submission
workflow and is meant for audit and further research, not for last-minute
day-of-submission work.

For the full historical decision trail from feature selection to Phase1 and
Phase2, read `../MODEL_SELECTION_LINEAGE.md` first. This file is the command
reference for the research/tuning layers.

## Search Layers

| Layer | What Is Tuned | Main Script |
| --- | --- | --- |
| Feature ranking | feature family and feature count | `tune_research_families.py` |
| Model shape | XGB preset and train-window profile | `tune_research_families.py` |
| Strict model comparison | retrain-at-entry live-style windows | `live_walk_forward_validation.py` |
| Portfolio construction | general/short weight, `top_k`, cap, risk filter, weighting | `live_portfolio_fine_tune_3_7_13.py` |
| Robustness audit | rounded parameters and local neighborhood sensitivity | `audit_portfolio_param_robustness.py` |
| Portfolio-level blend | blend current/capacity portfolio weight curves, then truncate | `tune_portfolio_blend_current_capacity.py` |

## Search Space

Ranking families:

| Family | Emphasis |
| --- | --- |
| `balanced` | Broad default feature-selection score |
| `stable` | IC stability, downside control, lower missingness |
| `recent` | Recent top-decile excess and recent hit rate |
| `economic` | Quality, valuation, industry, and economically motivated groups |
| `short_aggressive` | Short-horizon signals and concentrated short-history behavior |

Feature-count pairs:

| Pair | General | Short History |
| --- | ---: | ---: |
| `compact` | 60 | 30 |
| `medium` | 90 | 60 |
| `wide` | 120 | 90 |
| `extra_wide` | 150 | 120 |

Model presets:

| Preset | Meaning |
| --- | --- |
| `conservative` | Shallower trees and stronger regularization |
| `current` | Default XGB capacity |
| `capacity` | More flexible interaction model |

## Reproducible Commands

Run broad search:

```bash
conda run -n ml26s python tune_research_families.py \
  --run-feature-selection \
  --stage broad \
  --out-dir experiments/research_families/broad \
  --skip-existing
```

Run fine search on broad leaders:

```bash
conda run -n ml26s python tune_research_families.py \
  --stage fine \
  --candidates-from experiments/research_families/broad/leaderboard.csv \
  --top-n 33 \
  --out-dir experiments/research_families/fine \
  --skip-existing
```

Run strict live-style validation for retained top candidates:

```bash
conda run -n ml26s python live_walk_forward_validation.py \
  --num-windows 13 \
  --window-length 5 \
  --out-dir experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing
```

Run portfolio fine-tune:

```bash
conda run -n ml26s python live_portfolio_fine_tune_3_7_13.py \
  --out-dir experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13 \
  --source-run experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing \
  --progress-every 50000
```

Run robustness audit:

```bash
conda run -n ml26s python audit_portfolio_param_robustness.py \
  --out-dir experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/robustness_audit \
  --skip-existing
```

Run the Phase2 portfolio-level blend experiment:

```bash
conda run -n ml26s python tune_portfolio_blend_current_capacity.py \
  --out-dir experiments/portfolio_blend_current_capacity \
  --write-best-submission
```

## Formal Phase Outputs

The formal submission files are:

```text
submissions/submission1.csv
submissions/submission2.csv
```

Phase1 uses only `stable_compact_current`. Phase2 uses a portfolio-level blend
of:

| Candidate | Meaning |
| --- | --- |
| `stable_compact_current` | higher recent mean excess, more parameter-sensitive |
| `stable_compact_capacity` | lower mean in some windows, better recent robustness |

The formal Phase2 blend has 44 holdings and passed `validate_submission.py`.

The two source candidate configs are:

```text
stable_compact_current:
  general variant       = general_top80_shallow
  short-history variant = short_history_top60_recent_train
  general weight        = 0.55
  short weight          = 0.45
  top_k                 = 31
  internal cap          = 0.032
  risk filter           = on
  weighting             = rank

stable_compact_capacity:
  general variant       = general_top80_base
  short-history variant = short_history_top20_deep_interaction
  general weight        = 0.978
  short weight          = 0.022
  top_k                 = 33
  internal cap          = 1 / 33
  risk filter           = on
  weighting             = equal
```

The portfolio-level blend experiment writes:

```text
experiments/portfolio_blend_current_capacity/candidate_portfolio_blend_best.csv
experiments/portfolio_blend_current_capacity/candidate_portfolio_blend_best_legacy3713.csv
experiments/portfolio_blend_current_capacity/leaderboard.csv
experiments/portfolio_blend_current_capacity/legacy3713_leaderboard.csv
```

Latest best Phase2 portfolio-level blend:

```text
blend source           = final_portfolio
alpha current weight   = 0.45
capacity weight        = 0.55
requested top_k        = 44
latest holdings        = 44
```

This candidate is already written to `submissions/submission2.csv`.

The grid is intentionally low-dimensional:

```text
alpha = 0.30, 0.35, ..., 0.70
top_k = 35, 38, 41, ..., 59
blend_source = final_portfolio
```

`final_portfolio` blends the two already selected final portfolios. Its
available positive-weight universe is the current/capacity portfolio union.

The active blend score is rank-normalized:

```text
rank_score_Nw =
  0.50 * mean_excess_rank
+ 0.35 * worst_excess_rank
+ 0.15 * hit_rate_rank

anchor_score =
  0.30 * rank_score_3w
+ 0.40 * rank_score_7w
+ 0.30 * rank_score_13w

robust_score =
  0.65 * mean(anchor_score)
+ 0.25 * worst(anchor_score)
- 0.10 * std(anchor_score)
```

This avoids letting one high-return anchor dominate because all metrics are
ranked cross-sectionally within the same anchor before aggregation.

The legacy 3/7/13 score is also reported:

```text
legacy_score_Nw =
  0.57 * mean_excess_rank
+ 0.33 * worst_excess_rank
+ 0.10 * hit_rate_rank

legacy_anchor_score =
  0.40 * legacy_score_3w
+ 0.45 * legacy_score_7w
+ 0.15 * legacy_score_13w
- raw_penalty
```

In the latest run, both the active robust score and the legacy 3/7/13 score
select the same Phase2 blend: `final_portfolio`, `alpha=0.45`, `top_k=44`.

## What Was Deleted

Large obsolete fixed-block fine-stage artifacts and old one-off diagnostic
outputs were deleted. The retained artifacts needed by the current workflow are:

```text
experiments/research_families/fine_tune_top12/candidates/
experiments/research_families/fine_tune_top12/feature_sets/
experiments/research_families/fine_tune_top12/live_walk_forward_13w/
experiments/research_families/fine_tune_top12/live_walk_forward_13w_anchor_20260506/
experiments/research_families/fine_tune_top12/live_walk_forward_13w_anchor_20260507/
experiments/research_families/fine_tune_top12/live_walk_forward_13w_latest_20260508/
experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/
experiments/portfolio_blend_current_capacity/
```

The final submission can be reproduced from these retained configs and feature
sets. A completely fresh research restart may choose a different top12 set;
that is expected and should be treated as a new research run, not as the frozen
submission route.

See `../TOP12_REPRODUCTION.md` for the frozen candidate manifest.

## Validation Discipline

The active comparison is validation evidence, not a pristine final test. Since
recent windows have been inspected and used for tuning, final live performance
must be treated as the real out-of-sample test.

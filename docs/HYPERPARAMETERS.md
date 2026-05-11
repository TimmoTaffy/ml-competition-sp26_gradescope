# Hyperparameter Document

For the chronological explanation of how these hyperparameters were narrowed
down, see `MODEL_SELECTION_LINEAGE.md`.

## Fixed Competition Constraints

| Parameter | Value |
| --- | --- |
| Universe | `data/constituents.csv` CSI500 constituents |
| Long-only | yes |
| Fully invested | weights sum to `1.0 +/- 1e-4` |
| Minimum holdings | `30` |
| Maximum stock weight | `0.10` |

## Formal Portfolio Configs

Phase1 and Phase2 deliberately use different final portfolio layers.

| Field | Phase1 | Phase2 |
| --- | --- | --- |
| Formal submission | `submissions/submission1.csv` | `submissions/submission2.csv` |
| Main config/script | `configs/ensemble_phase1.json` | `tune_portfolio_blend_current_capacity.py` |
| Source candidates | `stable_compact_current` | `stable_compact_current` and `stable_compact_capacity` |
| Portfolio rule | 31-stock rank-weighted portfolio | portfolio-level blend, then top-44 truncation |
| Current/capacity blend | not used | `0.45 / 0.55` |
| Risk filter | enabled | enabled inside both source portfolios |

Phase1 `stable_compact_current` details:

| Hyperparameter | Value |
| --- | --- |
| General model | `general_top80_shallow` |
| Short-history model | `short_history_top60_recent_train` |
| General weight | `0.55` |
| Short-history weight | `0.45` |
| Score combination | cross-sectional rank average |
| `top_k` | `31` |
| Internal max weight | `0.032` |
| Weighting | `rank` |

The exact live fine-tune winner used `general_weight=0.5465` and
`internal_cap=0.031758`. The rounded values above were used because the
robustness-audit raw score changed by only `-0.000067`.

Phase2 keeps this rounded current candidate and blends it with
`stable_compact_capacity`:

| Hyperparameter | Value |
| --- | --- |
| Capacity general model | `general_top80_base` |
| Capacity short-history model | `short_history_top20_deep_interaction` |
| Capacity general/short weight | `0.978 / 0.022` |
| Capacity `top_k` | `33` |
| Capacity weighting | `equal` |
| Final blend source | `final_portfolio` |
| Final alpha current/capacity | `0.45 / 0.55` |
| Final requested `top_k` | `44` |

## Model Hyperparameters

Phase1 and Phase2 current-candidate general model config:

```text
experiments/research_families/fine_tune_top12/candidates/stable_compact_current/configs/general_top80_shallow.json
```

Phase1 and Phase2 current-candidate short-history model config:

```text
experiments/research_families/fine_tune_top12/candidates/stable_compact_current/configs/short_history_top60_recent_train.json
```

Selected feature sets:

```text
experiments/research_families/fine_tune_top12/feature_sets/general_stable_top80.txt
experiments/research_families/fine_tune_top12/feature_sets/short_history_stable_top60.txt
```

Phase2 capacity-candidate configs:

```text
experiments/research_families/fine_tune_top12/candidates/stable_compact_capacity/configs/general_top80_base.json
experiments/research_families/fine_tune_top12/candidates/stable_compact_capacity/configs/short_history_top20_deep_interaction.json
experiments/research_families/fine_tune_top12/feature_sets/short_history_stable_top20.txt
```

Important model hyperparameters:

| Hyperparameter | Role |
| --- | --- |
| `n_estimators` | Maximum boosting rounds before early stopping |
| `max_depth` | Tree complexity |
| `learning_rate` | Boosting step size |
| `subsample` | Row subsampling |
| `colsample_bytree` | Feature subsampling |
| `min_child_weight` | Minimum leaf support |
| `reg_lambda`, `reg_alpha` | L2/L1 regularization |
| `train_lookback_days` | Expanding versus recent-history training |
| `validation_days` | Internal early-stopping block length |
| `embargo_days` | Leakage guard for 5-day forward labels |

## Research Hyperparameters

These were search dimensions, not final manual choices:

| Search Dimension | Values |
| --- | --- |
| Ranking family | `balanced`, `stable`, `recent`, `economic`, `short_aggressive` |
| Feature-count pair | `compact`, `medium`, `wide`, `extra_wide` |
| Model preset | `conservative`, `current`, `capacity` |
| Live windows | latest `3`, `7`, and `13` non-overlapping 5-day windows |

Feature-count pairs:

| Pair | General | Short History |
| --- | ---: | ---: |
| `compact` | 60 | 30 |
| `medium` | 90 | 60 |
| `wide` | 120 | 90 |
| `extra_wide` | 150 | 120 |

## Live-Style Validation Score

Each 5-day validation window is evaluated as a historical live submission:
train using only labels known before entry, build a legal portfolio, then score
the realized holding window.

Base portfolio fine-tune uses rank scores over each validation window set:

```text
0.57 * mean_excess_rank
+ 0.33 * worst_window_rank
+ 0.10 * hit_rate_rank
```

Portfolio fine-tune combines recent and longer views:

```text
0.40 * score_3w
+ 0.45 * score_7w
+ 0.15 * score_13w
- penalty
```

This score is a validation-tuning hyperparameter. It intentionally emphasizes
legal-portfolio excess return and downside control rather than standalone IC.

## Phase2 Portfolio-Blend Score

The current phase2 blend tuner uses a more scale-stable rank-normalized score.
For every anchor and every horizon, candidates are ranked cross-sectionally on
mean excess, worst excess, and hit rate:

```text
rank_score_Nw =
  0.50 * mean_excess_rank
+ 0.35 * worst_excess_rank
+ 0.15 * hit_rate_rank
```

The 3/7/13 horizons are then combined as:

```text
anchor_score =
  0.30 * rank_score_3w
+ 0.40 * rank_score_7w
+ 0.30 * rank_score_13w
```

Across anchors:

```text
robust_score =
  0.65 * mean(anchor_score)
+ 0.25 * worst(anchor_score)
- 0.10 * std(anchor_score)
```

This is still a hyperparameterized validation score, but it is less sensitive
to the raw return scale of any single anchor than the earlier raw-return score.

For auditability, the tuner also reports the earlier 3/7/13 formula:

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

The active robust score and this legacy 3/7/13 score selected the same Phase2
portfolio blend: `final_portfolio`, `alpha=0.45`, `top_k=44`.

## Robustness Audit

After fine-tune, the final candidates are checked by:

1. exact tuned parameters,
2. rounded human-readable parameters,
3. local perturbations around weight, `top_k`, and cap.

Current audit conclusion:

| Candidate | Exact Score | Rounded Raw Delta | Median Raw Delta | Class |
| --- | ---: | ---: | ---: | --- |
| `stable_compact_current` | 0.860241 | -0.000067 | -0.002442 | `fragile` |
| `balanced_extra_wide_capacity` | 0.562877 | -0.000642 | -0.000862 | `stable` |
| `stable_compact_capacity` | 0.732702 | -0.004776 | -0.002180 | `spiky` |

Interpretation: `stable_compact_current` was strong enough for Phase1 but
parameter-sensitive. Phase2 therefore kept it as one source candidate and
added `stable_compact_capacity` through a portfolio-level blend.

## Split Discipline

| Layer | Rule |
| --- | --- |
| Feature selection | Rolling walk-forward validation before the reserved historical test block |
| Model training | Internal time-ordered validation block for early stopping |
| Portfolio tuning | Strict live-style walk-forward validation |
| Static historical test set | not used for final selection; repeated inspection would make it validation, not a clean test |
| Final live fit | Use `--final-fit` after choices are frozen |

Because recent validation windows have been repeatedly inspected, they should
be treated as validation evidence rather than a clean final test. The course
live windows are the external out-of-sample checks.

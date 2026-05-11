# Model Selection Lineage

## Purpose

This document explains how the project moved from raw features to the final
Phase1 and Phase2 submissions. It is the narrative companion to the executable
runbooks:

```text
docs/PHASE_REPRODUCTION.md
docs/DATA_PIPELINE.md
docs/HYPERPARAMETERS.md
docs/TOP12_REPRODUCTION.md
```

The final portfolios are reproducible from retained configs, feature sets,
data snapshots, and training scripts. Some early exploratory broad/fine result
tables were deleted to keep the project clean; the retained frozen top12
candidate set is therefore the canonical boundary for final-model
reproduction.

## End-To-End Flow

```text
raw public data
  -> feature matrix
  -> rolling feature ranking for general and short-history profiles
  -> research-family search over feature families, feature counts, and XGB presets
  -> frozen top12 candidate families
  -> fixed model-pair configs for each top12 candidate
  -> strict live-style walk-forward validation
  -> 3/7/13 portfolio fine-tune
  -> Phase1: stable_compact_current
  -> Phase2: 0.45 current + 0.55 capacity portfolio blend, top_k=44
```

## What The Model Learns

Each source model is an `XGBRegressor` trained from scratch.

| Item | Role |
| --- | --- |
| Input row | one stock on one date |
| Features | technical, benchmark-relative, Bollinger, liquidity/risk, industry, valuation, fundamentals, market-regime features |
| Target | `target_excess_5d`, future 5-trading-day stock return minus future 5-trading-day CSI500 return |
| Learned parameters | boosted tree splits and leaf values inside XGBoost |
| Tuned hyperparameters | feature family, feature count, XGB shape, train lookback, general/short blend weight, `top_k`, cap, risk filter, weighting, Phase2 alpha |

The supervised task is not to predict absolute returns. It is to rank stocks
by expected short-horizon excess return versus CSI500, then convert the ranking
into a legal long-only portfolio.

## Two-Profile Model Structure

The project settled on two complementary profiles:

| Profile | Purpose | Typical history | Final usage |
| --- | --- | --- | --- |
| `general` | broader, slower-moving cross-sectional signal | expanding or longer history | always used |
| `short_history` | recent technical/regime signal | shorter rolling history | used either meaningfully or with tiny weight |

The two profiles are combined by cross-sectional rank averaging:

```text
combined_score =
  general_weight * percentile_rank(general_prediction)
+ short_weight   * percentile_rank(short_prediction)
```

This avoids mixing raw XGB score scales directly. Portfolio construction then
sorts stocks by `combined_score`, applies optional risk filtering, chooses
`top_k`, assigns rank/equal weights, caps weights, and renormalizes.

## Feature Selection

Feature selection is profile-specific. The general model and short-history
model do not have to use the same feature list.

Main configs:

```text
configs/feature_selection_general.json
configs/feature_selection_short_history.json
```

Main command through the research-family runner:

```bash
conda run -n ml26s python tune_research_families.py \
  --run-feature-selection \
  --stage broad \
  --out-dir experiments/research_families/broad \
  --skip-existing
```

Feature-ranking outputs:

```text
experiments/feature_selection/general_<family>/selected_features_with_scores.csv
experiments/feature_selection/short_history_<family>/selected_features_with_scores.csv
experiments/research_families/fine_tune_top12/feature_sets/
```

Five ranking families were used:

| Family | Selection emphasis |
| --- | --- |
| `balanced` | broad default feature-selection score |
| `stable` | IC stability, downside control, lower missingness |
| `recent` | recent top-decile excess and recent IC |
| `economic` | quality, valuation, and economically motivated groups |
| `short_aggressive` | short-horizon/recent signals |

The feature-selection score combines single-factor rolling evidence and group
ablation evidence. The important terms are rank-IC stability, top-decile
forward excess, worst-window behavior, missingness penalty, and feature-group
ablation. The exact weights are in the two feature-selection JSON configs and
the family overrides inside `tune_research_families.py`.

## Research Families: 60 Candidates

The first structured model search crossed three dimensions:

```text
5 ranking families x 4 feature-count pairs x 3 XGB presets = 60 candidates
```

Feature-count pairs:

| Pair | General features | Short-history features |
| --- | ---: | ---: |
| `compact` | 60 | 30 |
| `medium` | 90 | 60 |
| `wide` | 120 | 90 |
| `extra_wide` | 150 | 120 |

Model presets:

| Preset | Meaning |
| --- | --- |
| `conservative` | shallower, more regularized trees |
| `current` | baseline XGB shape |
| `capacity` | deeper/larger interaction model |

Broad search command:

```bash
conda run -n ml26s python tune_research_families.py \
  --run-feature-selection \
  --stage broad \
  --out-dir experiments/research_families/broad \
  --skip-existing
```

Fine search command on broad leaders:

```bash
conda run -n ml26s python tune_research_families.py \
  --stage fine \
  --candidates-from experiments/research_families/broad/leaderboard.csv \
  --top-n 33 \
  --out-dir experiments/research_families/fine \
  --skip-existing
```

The old broad/fine layer used fixed-block portfolio validation to narrow the
search space. That layer is no longer the final validation method, but it was
useful for reducing 60 structured candidates to a smaller candidate pool.

Retained top-33 manifest:

```text
experiments/research_families/fine_tune_top12/candidate_family_lookup.csv
```

The 12 candidate directories under
`experiments/research_families/fine_tune_top12/candidates/` are the frozen
top12 candidate set used by final validation and tuning. They correspond to
the retained top12 from the old 33-candidate fine stage.

## Frozen Top12 Candidate Set

The top12 candidates are retained as configs and feature sets. Each candidate
contains:

```text
experiments/research_families/fine_tune_top12/candidates/<candidate>/best_finetuned_config.json
experiments/research_families/fine_tune_top12/candidates/<candidate>/configs/*.json
experiments/research_families/fine_tune_top12/feature_sets/*.txt
```

Current top12 live-style portfolio fine-tune ranking:

| Rank | Candidate | Final score | General variant | Short variant | General weight | Top K | Weighting |
| ---: | --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | `stable_compact_current` | 0.834428 | `general_top80_shallow` | `short_history_top60_recent_train` | 0.5465 | 31 | rank |
| 2 | `stable_compact_capacity` | 0.778086 | `general_top80_base` | `short_history_top20_deep_interaction` | 0.9780 | 33 | equal |
| 3 | `recent_compact_capacity` | 0.775831 | `general_top60_base` | `short_history_top20_shallow` | 0.0550 | 31 | rank |
| 4 | `balanced_medium_current` | 0.769611 | `general_top110_shallow` | `short_history_top40_deep_interaction` | 0.0000 | 49 | equal |
| 5 | `stable_extra_wide_capacity` | 0.739984 | `general_top120_longer_train` | `short_history_top140_light_reg` | 0.2850 | 46 | equal |
| 6 | `short_aggressive_wide_capacity` | 0.726203 | `general_top90_recent_train` | `short_history_top120_light_reg` | 0.5935 | 30 | rank |
| 7 | `balanced_extra_wide_capacity` | 0.637070 | `general_top180_low_lr` | `short_history_top120_light_reg` | 0.7250 | 64 | equal |
| 8 | `short_aggressive_compact_capacity` | 0.621164 | `general_top60_deep_interaction` | `short_history_top30_base` | 0.9455 | 31 | equal |
| 9 | `short_aggressive_medium_capacity` | 0.621164 | `general_top60_deep_interaction` | `short_history_top30_base` | 0.9455 | 31 | equal |
| 10 | `recent_compact_current` | 0.592696 | `general_top60_deep_interaction` | `short_history_top30_deep_interaction` | 0.8250 | 37 | equal |
| 11 | `stable_medium_capacity` | 0.449359 | `general_top120_longer_train` | `short_history_top30_deep_interaction` | 0.7725 | 93 | equal |
| 12 | `economic_medium_capacity` | 0.362394 | `general_top100_deep_interaction` | `short_history_top50_base` | 0.9980 | 30 | rank |

Source table:

```text
experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/leaderboard.csv
```

## Live-Style Walk-Forward Validation

The project replaced fixed-block portfolio validation with strict live-style
walk-forward validation for final comparisons.

For each historical 5-trading-day window:

1. Use the prior trading day as the entry/as-of date.
2. Retrain the candidate's general and short-history models from scratch.
3. Use only labels fully known as of the entry date.
4. Build a legal portfolio at the entry date.
5. Score realized excess return over the following window.

Command for the retained 13-window run:

```bash
conda run -n ml26s python live_walk_forward_validation.py \
  --num-windows 13 \
  --window-length 5 \
  --out-dir experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing
```

Main outputs:

```text
experiments/research_families/fine_tune_top12/live_walk_forward_13w/windows.csv
experiments/research_families/fine_tune_top12/live_walk_forward_13w/window_results.csv
experiments/research_families/fine_tune_top12/live_walk_forward_13w/candidate_summary.csv
experiments/research_families/fine_tune_top12/live_walk_forward_13w/holdings.csv
experiments/research_families/fine_tune_top12/live_walk_forward_13w/model_cache/
```

The model cache is intentional: repeated model specs are trained once per entry
date and reused.

## Portfolio Fine-Tune On Top12

After the model-pair configs were frozen, `live_portfolio_fine_tune_3_7_13.py`
tuned only portfolio construction:

| Hyperparameter | Meaning |
| --- | --- |
| `model0_weight` | general-model score weight |
| `top_k` | number of holdings |
| `internal_max_weight` | pre-normalization concentration cap |
| `risk_filter_enabled` | exclude recent halt/illiquid names |
| `weighting` | rank or equal weights |
| `score_combination` | rank-average score transform |

Command:

```bash
conda run -n ml26s python live_portfolio_fine_tune_3_7_13.py \
  --out-dir experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13 \
  --source-run experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing \
  --progress-every 50000
```

The script runs three rounds per candidate:

| Round | Purpose | Defaults |
| --- | --- | --- |
| broad | wide portfolio grid | weight step 0.01, `top_k` step 2, cap step 0.002 |
| fine | neighborhood around broad leaders | weight step 0.0025, `top_k` radius 5, cap step 0.001 |
| ultra | smaller neighborhood around best broad/fine configs | weight step 0.001, `top_k` radius 2, cap step 0.0005 |

Portfolio objective:

```text
score_Nw =
  0.57 * mean_excess_rank
+ 0.33 * worst_window_rank
+ 0.10 * hit_rate_rank

final_score =
  0.40 * score_3w
+ 0.45 * score_7w
+ 0.15 * score_13w
- penalty
```

Outputs:

```text
experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/all_retained_ranked.csv
experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/leaderboard.csv
experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/candidates/<candidate>/best_live_portfolio_config.json
experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/best_live_portfolio_config.json
```

## Robustness Audit

The best candidates were then checked for rounded-parameter and neighborhood
stability.

```bash
conda run -n ml26s python audit_portfolio_param_robustness.py \
  --out-dir experiments/research_families/fine_tune_top12/live_portfolio_fine_tune_3_7_13/robustness_audit \
  --skip-existing
```

This audit explains why Phase1 used rounded human-readable parameters for
`stable_compact_current`:

```text
exact general weight = 0.5465
exact cap            = 0.031758
submission weight    = 0.55 / 0.45
submission cap       = 0.032
```

The rounded version had negligible score loss in the audit and was simpler to
explain.

## Phase1 Decision

Phase1 selected the strongest single candidate from the 3/7/13 live-style
portfolio fine-tune:

```text
candidate             = stable_compact_current
general variant       = general_top80_shallow
short-history variant = short_history_top60_recent_train
general/short weight  = 0.55 / 0.45
portfolio             = top_k 31, rank weighting, cap 0.032, risk filter on
config                = configs/ensemble_phase1.json
submission            = submissions/submission1.csv
```

Formal reproduction is in `PHASE_REPRODUCTION.md`.

## Phase2 Decision

After Phase1, the project used the extra May 6-8 realized data to compare
current-style strength and capacity-style robustness. Rather than selecting
one source candidate, Phase2 blended two legal portfolios:

```text
current source   = stable_compact_current
capacity source  = stable_compact_capacity
blend type       = portfolio-level blend
```

Why portfolio-level blend:

| Alternative | Problem |
| --- | --- |
| raw score blend | XGB score scales are not comparable across independently trained models |
| retrain one meta-model | too much leakage/overfit risk with little true live data |
| portfolio blend | preserves each candidate's legal construction and only tunes alpha/top_k |

Phase2 blend grid:

```text
alpha current = 0.30, 0.35, ..., 0.70
top_k         = 35, 38, 41, ..., 59
blend_source  = final_portfolio
```

`final_portfolio` blends the two selected final portfolios without mixing raw
XGBoost score scales.

Command:

```bash
conda run -n ml26s python tune_portfolio_blend_current_capacity.py \
  --out-dir experiments/portfolio_blend_current_capacity \
  --write-best-submission \
  --submission-out submissions/submission2.csv \
  --scores-out submissions/submission2.scores.csv
```

Phase2 active robust score:

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

The legacy 3/7/13 formula is also written for audit:

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

Both active and legacy scores selected the same final Phase2 blend:

```text
strategy       = final_portfolio_a0.45_k44
alpha current  = 0.45
alpha capacity = 0.55
requested top_k = 44
submission     = submissions/submission2.csv
```

Key Phase2 output files:

```text
experiments/portfolio_blend_current_capacity/leaderboard.csv
experiments/portfolio_blend_current_capacity/legacy3713_leaderboard.csv
experiments/portfolio_blend_current_capacity/anchor_3713_detail.csv
experiments/portfolio_blend_current_capacity/best_portfolio_blend_config.json
```

## Reproducibility Boundary

| Layer | Reproducibility status |
| --- | --- |
| Final Phase1/Phase2 submissions | fully reproducible with `PHASE_REPRODUCTION.md` |
| Frozen top12 live-style validation | reproducible with `live_walk_forward_validation.py` |
| Top12 portfolio fine-tune | reproducible with `live_portfolio_fine_tune_3_7_13.py` |
| Phase2 alpha/top_k tuning | reproducible with `tune_portfolio_blend_current_capacity.py` |
| Early 60->33 broad/fine search | rerunnable from current code, but old bulky tables were deleted |
| Exact old dense model-pair exploratory tables | not retained; frozen configs are the canonical artifact |

The final report should present the early broad/fine search as research
motivation and present the frozen top12/live-style/Phase2 blend pipeline as the
auditable final selection path.

## Minimal Reproduction Checklist

1. Build data using `DATA_PIPELINE.md`.
2. Reproduce Phase1 and Phase2 submissions using `PHASE_REPRODUCTION.md`.
3. Re-run top12 live-style validation if candidate ranking evidence is needed.
4. Re-run `live_portfolio_fine_tune_3_7_13.py` if portfolio hyperparameter
   evidence is needed.
5. Re-run `tune_portfolio_blend_current_capacity.py` if Phase2 alpha/top_k
   evidence is needed.

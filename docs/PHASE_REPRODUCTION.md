# Phase Reproduction Runbook

## Formal Artifacts

| Phase | Deadline | Live window | Submission | Route | SHA256 |
| --- | --- | --- | --- | --- | --- |
| Phase1 | 2026-05-03 23:59 CST | 2026-05-06 to 2026-05-08 | `submissions/submission1.csv` | single `stable_compact_current` portfolio | `d02710e1ebf533351634ad77ccb26d0426f5dfb0fde5b9dd5bbf23dc708b7fa2` |
| Phase2 | 2026-05-10 23:59 CST | 2026-05-11 to 2026-05-15 | `submissions/submission2.csv` | `0.45 current + 0.55 capacity`, `top_k=44` | `1b6b39ddd4fd3a3b3b8ab821ef4248b224bc5f326f26a31141aa10179dffd25d` |

Both phases use models trained from scratch. The base learner is
`xgboost.XGBRegressor`; the prediction target is `target_excess_5d`, i.e. the
stock's future 5-trading-day return minus the CSI500 future 5-trading-day
return.

Label cutoffs are enforced by `train_model.py`: for a 5-trading-day target, a
prediction made from the April 30, 2026 row can only train on labels dated up
to April 23, 2026; a prediction made from the May 8, 2026 row can only train
on labels dated up to April 28, 2026.

## Phase1

Phase1 used one rounded live-style fine-tuned candidate:

```text
candidate             = stable_compact_current
general model         = general_top80_shallow
short-history model   = short_history_top60_recent_train
general/short weight  = 0.55 / 0.45
score combination     = cross-sectional rank average
portfolio top_k       = 31
internal cap          = 0.032
weighting             = rank
risk filter           = enabled
config                = configs/ensemble_phase1.json
```

Rebuild the Phase1 data snapshot and train the two source models:

```bash
conda run -n ml26s python download_data.py --update --end 20260503
conda run -n ml26s python prepare_training_data.py
conda run -n ml26s python download_external_data.py --dataset all --start 20250101 --end 20260503
conda run -n ml26s python build_feature_matrix.py

conda run -n ml26s python train_model.py \
  --profile general \
  --config experiments/research_families/fine_tune_top12/candidates/stable_compact_current/configs/general_top80_shallow.json \
  --features experiments/research_families/fine_tune_top12/feature_sets/general_stable_top80.txt \
  --out models/final_live/general \
  --as-of 20260503 \
  --final-fit

conda run -n ml26s python train_model.py \
  --profile short_history \
  --config experiments/research_families/fine_tune_top12/candidates/stable_compact_current/configs/short_history_top60_recent_train.json \
  --features experiments/research_families/fine_tune_top12/feature_sets/short_history_stable_top60.txt \
  --out models/final_live/short_history \
  --as-of 20260503 \
  --final-fit
```

Generate and validate the Phase1 submission:

```bash
conda run -n ml26s python make_submission.py \
  --config configs/ensemble_phase1.json \
  --as-of 20260503 \
  --out submissions/submission1.csv

conda run -n ml26s python validate_submission.py submissions/submission1.csv
```

The command also writes `submissions/submission1.scores.csv` as a diagnostic
ranking file. If Phase1 is reproduced after Phase2, retrain Phase2 afterwards
because these commands overwrite `models/final_live/`.

## Phase2

Phase2 kept the Phase1-style `stable_compact_current` candidate, added a more
capacity-oriented candidate, and selected a portfolio-level blend. It does not
blend raw model scores; it blends legal portfolio weight curves, then truncates
and renormalizes.

```text
current candidate       = stable_compact_current
capacity candidate      = stable_compact_capacity
blend source            = final_portfolio
alpha current/capacity  = 0.45 / 0.55
requested top_k         = 44
selected by             = active robust 3/7/13 score
legacy check            = legacy 3/7/13 score selects the same candidate
```

Rebuild the Phase2 data snapshot:

```bash
conda run -n ml26s python download_data.py --update --end 20260510
conda run -n ml26s python prepare_training_data.py
conda run -n ml26s python download_external_data.py --dataset all --start 20250101 --end 20260510
conda run -n ml26s python build_feature_matrix.py
```

Train the two source candidates:

```bash
conda run -n ml26s python train_model.py \
  --profile general \
  --config experiments/research_families/fine_tune_top12/candidates/stable_compact_current/configs/general_top80_shallow.json \
  --features experiments/research_families/fine_tune_top12/feature_sets/general_stable_top80.txt \
  --out models/final_live/general \
  --as-of 20260510 \
  --final-fit

conda run -n ml26s python train_model.py \
  --profile short_history \
  --config experiments/research_families/fine_tune_top12/candidates/stable_compact_current/configs/short_history_top60_recent_train.json \
  --features experiments/research_families/fine_tune_top12/feature_sets/short_history_stable_top60.txt \
  --out models/final_live/short_history \
  --as-of 20260510 \
  --final-fit

conda run -n ml26s python train_model.py \
  --profile general \
  --config experiments/research_families/fine_tune_top12/candidates/stable_compact_capacity/configs/general_top80_base.json \
  --features experiments/research_families/fine_tune_top12/feature_sets/general_stable_top80.txt \
  --out models/final_live_capacity/general \
  --as-of 20260510 \
  --final-fit

conda run -n ml26s python train_model.py \
  --profile short_history \
  --config experiments/research_families/fine_tune_top12/candidates/stable_compact_capacity/configs/short_history_top20_deep_interaction.json \
  --features experiments/research_families/fine_tune_top12/feature_sets/short_history_stable_top20.txt \
  --out models/final_live_capacity/short_history \
  --as-of 20260510 \
  --final-fit
```

Generate and validate the Phase2 submission:

```bash
conda run -n ml26s python tune_portfolio_blend_current_capacity.py \
  --write-best-submission \
  --submission-out submissions/submission2.csv \
  --scores-out submissions/submission2.scores.csv

conda run -n ml26s python validate_submission.py submissions/submission2.csv
```

## Validation And Tuning Logic

Model training uses an internal time-ordered validation block only for early
stopping and basic quality checks. Portfolio and hyperparameter selection use
live-style walk-forward validation: each historical entry date retrains using
only labels known before that entry date, builds a legal portfolio, and scores
the following realized window.

The final Phase2 blend score ranks candidates within each anchor before
aggregation:

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

The older 3/7/13 formula is retained as an audit metric:

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

Both metrics selected the same Phase2 final blend in the retained run.

## Notes

`baseline_xgboost.py` and `features.py` are instructor-provided baseline files
and are intentionally retained. They are useful for comparison but are not the
active Phase1 or Phase2 submission route.

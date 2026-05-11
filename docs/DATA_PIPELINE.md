# Data Pipeline And Submission Runbook

## Scope

This document records the executable data and submission pipeline for both
formal phases. The shorter conceptual summary is in `PHASE_REPRODUCTION.md`.

| Phase | Data cutoff command | Latest A-share row expected | Output |
| --- | --- | --- | --- |
| Phase1 | `--end 20260503` | 2026-04-30 | `submissions/submission1.csv` |
| Phase2 | `--end 20260510` | 2026-05-08 | `submissions/submission2.csv` |

The `--end` date is the submission-date data cutoff, not necessarily an
A-share trading day. Labor Day market closure means Phase1's latest A-share
row is April 30 even though the deadline was May 3.

For the 5-trading-day target, `train_model.py` only uses fully known labels.
The Phase1 prediction row of April 30 trains on labels through April 23. The
Phase2 prediction row of May 8 trains on labels through April 28.

## Shared Data Build

Use the same four-step data build for either phase, changing only `END_DATE`.

```bash
conda run -n ml26s python download_data.py --update --end END_DATE
conda run -n ml26s python prepare_training_data.py
conda run -n ml26s python download_external_data.py --dataset all --start 20250101 --end END_DATE
conda run -n ml26s python build_feature_matrix.py
```

Main outputs:

```text
data/prices.parquet
data/index.parquet
data/final_training_matrix.parquet
data/final_feature_columns.txt
data/final_target_columns.txt
data/external/
```

`build_feature_matrix.py` merges prices, benchmark-relative features,
Bollinger/risk/liquidity features, market regime proxies, SW industry features,
valuation data, fundamentals, and CSI500 quality-growth membership data.
Fundamental features are announcement-date aware.

## Phase1 Submission

Phase1 uses only the `stable_compact_current` XGB pair. Its frozen portfolio
config is `configs/ensemble_phase1.json`.

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

conda run -n ml26s python make_submission.py \
  --config configs/ensemble_phase1.json \
  --as-of 20260503 \
  --out submissions/submission1.csv

conda run -n ml26s python validate_submission.py submissions/submission1.csv
```

Expected formal artifact:

```text
submissions/submission1.csv
SHA256 = d02710e1ebf533351634ad77ccb26d0426f5dfb0fde5b9dd5bbf23dc708b7fa2
```

`make_submission.py` also writes `submissions/submission1.scores.csv`.

## Phase2 Submission

Phase2 trains the same `stable_compact_current` pair plus the
`stable_compact_capacity` pair, then applies the retained current/capacity
portfolio blend.

```bash
conda run -n ml26s python download_data.py --update --end 20260510
conda run -n ml26s python prepare_training_data.py
conda run -n ml26s python download_external_data.py --dataset all --start 20250101 --end 20260510
conda run -n ml26s python build_feature_matrix.py

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

conda run -n ml26s python tune_portfolio_blend_current_capacity.py \
  --write-best-submission \
  --submission-out submissions/submission2.csv \
  --scores-out submissions/submission2.scores.csv

conda run -n ml26s python validate_submission.py submissions/submission2.csv
```

Expected formal artifact:

```text
submissions/submission2.csv
SHA256 = 1b6b39ddd4fd3a3b3b8ab821ef4248b224bc5f326f26a31141aa10179dffd25d
```

`submissions/submission2.scores.csv` records the current/capacity component
weights and blend coefficients.

## Ordering Caveat

Both phases write to `models/final_live/`. If both phases are reproduced in one
working tree, run Phase1 first, then Phase2. If Phase1 must be regenerated
after Phase2, regenerate Phase2 afterwards before using `submission2.csv`.

## Baseline Files

`baseline_xgboost.py` and `features.py` are instructor-provided baseline files.
They are preserved for comparison and should not be deleted, but they are not
the active formal submission route.

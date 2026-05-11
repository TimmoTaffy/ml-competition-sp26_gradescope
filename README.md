# CSCI-SHU 360 Machine Learning Final Competition

This repository contains my CSI500 stock-selection system for the Spring 2026
CSCI-SHU 360 Machine Learning final competition.

The task is to submit a long-only CSI500 portfolio that outperforms the CSI500
index over the next live holding window. The final pipeline trains XGBoost
models from scratch, ranks stocks by expected short-horizon excess return, and
converts the ranking into a legal portfolio.

## Final Artifacts

| Phase | File | Route | SHA256 |
| --- | --- | --- | --- |
| Phase1 | `submissions/submission1.csv` | `stable_compact_current` only, 31-name rank-weighted portfolio | `d02710e1ebf533351634ad77ccb26d0426f5dfb0fde5b9dd5bbf23dc708b7fa2` |
| Phase2 | `submissions/submission2.csv` | `0.45 current + 0.55 capacity`, `top_k=44` portfolio blend | `1b6b39ddd4fd3a3b3b8ab821ef4248b224bc5f326f26a31141aa10179dffd25d` |

Validate before upload:

```bash
conda run -n ml26s python validate_submission.py submissions/submission1.csv
conda run -n ml26s python validate_submission.py submissions/submission2.csv
```

## Quickstart

The project was run in a conda environment named `ml26s`.

```bash
conda run -n ml26s python download_data.py --update --end 20260510
conda run -n ml26s python prepare_training_data.py
conda run -n ml26s python download_external_data.py --dataset all --start 20250101 --end 20260510
conda run -n ml26s python build_feature_matrix.py
```

Generate the Phase2 submission from the retained current/capacity source
models and blend config:

```bash
conda run -n ml26s python tune_portfolio_blend_current_capacity.py \
  --write-best-submission \
  --submission-out submissions/submission2.csv \
  --scores-out submissions/submission2.scores.csv

conda run -n ml26s python validate_submission.py submissions/submission2.csv
```

For a full clean rebuild, including retraining all final source models, follow
[`docs/PHASE_REPRODUCTION.md`](docs/PHASE_REPRODUCTION.md).

## Method Summary

The supervised target is `target_excess_5d`: future 5-trading-day stock return
minus future 5-trading-day CSI500 return. This aligns the model objective with
the live scoring metric, which is portfolio excess return over CSI500.

The final prediction system uses two XGBoost profiles:

| Profile | Role |
| --- | --- |
| `general` | broader cross-sectional signal with more history and more features |
| `short_history` | recent technical/regime signal with shorter lookback |

Model scores are transformed to cross-sectional ranks before blending. The
portfolio layer then sorts by blended rank score, applies risk/liquidity
filters, selects `top_k`, applies equal or rank weighting, caps concentration,
and renormalizes weights to sum to one.

The active feature library includes:

| Feature Group | Examples |
| --- | --- |
| Technical and momentum | returns, volatility, RSI, moving-average distance |
| CSI500-relative | excess return, beta, residual momentum, relative volatility |
| Bollinger | band position, z-score, width, lower/upper proximity flags |
| Liquidity and risk | halt flags, missing days, zero volume, drawdown, ATR |
| Industry-relative | Shenwan industry returns and stock-minus-industry ranks |
| Valuation and quality | PE/PB, ROE/ROA, margins, 3-year growth, cash-flow quality |
| Market regime | indices, ETFs, sector proxies, crude/gold/global proxies, 930939 quality-growth index |

Feature details are in
[`docs/FEATURES_AND_SELECTION.md`](docs/FEATURES_AND_SELECTION.md).

## Validation Design

The project does not claim a separate static historical test set for final
model selection. The live sample is short, and a repeatedly inspected fixed
historical block would become validation rather than a clean test.

The final selection logic uses strict live-style walk-forward validation:

1. choose a historical entry date;
2. train from scratch using only labels fully known before that entry date;
3. build a legal portfolio at that entry date;
4. score the realized next holding window;
5. aggregate recent 3/7/13-window evidence with downside and hit-rate terms.

The Phase1 live result is reported as a post-submission external check against
the instructor baseline. Phase2 was the final unseen live evaluation when
`submissions/submission2.csv` was generated.

## Report

The formal report is `report/report.pdf`; the editable LaTeX source is
`report/main.tex`.

The report is organized around the required sections:

| Section | Coverage |
| --- | --- |
| Factors | feature groups, economic motivation, automated selection |
| Models | XGBoost profiles, rank blending, portfolio construction |
| Results | baseline comparison, live-style validation, Phase2 blend evidence |
| Analysis | what worked, what failed, overfitting and noise risks |
| Self-test | train/validation/test protocol, leakage controls, baseline comparison |

The realized baseline comparison is recorded in:

```text
experiments/self_test/SELF_TEST_REPORT.md
experiments/self_test/self_test_results.csv
experiments/self_test/baseline_phase1_test.csv
```

## Documentation Map

Primary docs:

| File | Purpose |
| --- | --- |
| [`docs/PHASE_REPRODUCTION.md`](docs/PHASE_REPRODUCTION.md) | Exact Phase1/Phase2 rebuild commands and hashes |
| [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | Raw data update, external data, feature matrix build |
| [`docs/FEATURES_AND_SELECTION.md`](docs/FEATURES_AND_SELECTION.md) | Feature library and automated feature-selection workflow |
| [`docs/MODEL_SELECTION_LINEAGE.md`](docs/MODEL_SELECTION_LINEAGE.md) | Chronological model-design and selection story |
| [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) | Final model, portfolio, and validation-score hyperparameters |
| [`docs/TOP12_REPRODUCTION.md`](docs/TOP12_REPRODUCTION.md) | Frozen top12 candidate manifest and rerun commands |

Appendix docs retained for auditability:

| File | Purpose |
| --- | --- |
| [`docs/appendix/LIVE_WALK_FORWARD_VALIDATION.md`](docs/appendix/LIVE_WALK_FORWARD_VALIDATION.md) | Detailed live-style validation mechanics |
| [`docs/appendix/LIVE_PORTFOLIO_FINE_TUNE.md`](docs/appendix/LIVE_PORTFOLIO_FINE_TUNE.md) | 3/7/13 portfolio fine-tune details |
| [`docs/appendix/RESEARCH_TUNING.md`](docs/appendix/RESEARCH_TUNING.md) | Broader research-family tuning workflow |

## Active Code Files

| File | Purpose |
| --- | --- |
| `download_data.py` | Fetch CSI500 constituents, stock bars, and CSI500 index data |
| `prepare_training_data.py` | Build the base price/index training table |
| `download_external_data.py` | Fetch public external data: indices/ETF proxies, SW industry, valuation, fundamentals, quality-growth index |
| `build_feature_matrix.py` | Merge raw data into `data/final_training_matrix.parquet` |
| `feature_selection.py` | Rolling feature-selection workflow |
| `tune_research_families.py` | Structured feature-family/model-family search |
| `train_model.py` | Train one XGB profile from scratch |
| `make_submission.py` | Generate a legal long-only submission CSV |
| `live_walk_forward_validation.py` | Strict live-style validation for retained candidates |
| `live_portfolio_fine_tune_3_7_13.py` | Portfolio-construction fine-tune on 3/7/13 windows |
| `tune_portfolio_blend_current_capacity.py` | Phase2 current/capacity blend tuner |
| `validate_submission.py` | Check portfolio constraints |
| `score_submission.py` | Score a realized evaluation window |
| `baseline_xgboost.py` | Instructor baseline retained for comparison |
| `features.py` | Instructor baseline feature helper |

## Reproducibility Notes

All final submission models are trained from scratch. Fundamental features are
announcement-date aware, and training uses only labels fully known as of the
prediction date. For the 5-trading-day target, the Phase1 prediction row
(`2026-04-30`) uses labels through `2026-04-23`; the Phase2 prediction row
(`2026-05-08`) uses labels through `2026-04-28`.

`baseline_xgboost.py` and `features.py` are instructor-provided baseline files
and are intentionally retained. They are not the active formal submission route.

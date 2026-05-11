# Documentation Index

The root `README.md` is the project entry point. This directory contains the
technical appendices needed to reproduce and audit the final submissions.

## Primary Documents

| File | Purpose |
| --- | --- |
| `PHASE_REPRODUCTION.md` | Exact Phase1/Phase2 rebuild commands and expected hashes |
| `DATA_PIPELINE.md` | Raw data, external data, and feature-matrix build pipeline |
| `FEATURES_AND_SELECTION.md` | Feature groups, targets, and automated feature selection |
| `MODEL_SELECTION_LINEAGE.md` | Chronological explanation from feature pool to final portfolios |
| `HYPERPARAMETERS.md` | Final model, portfolio, and validation-score hyperparameters |
| `TOP12_REPRODUCTION.md` | Frozen top12 candidate manifest and rerun commands |

## Appendix Documents

| File | Purpose |
| --- | --- |
| `appendix/LIVE_WALK_FORWARD_VALIDATION.md` | Detailed validation mechanics |
| `appendix/LIVE_PORTFOLIO_FINE_TUNE.md` | Top12 portfolio fine-tune details |
| `appendix/RESEARCH_TUNING.md` | Broader research-family tuning command history |

## Reading Order

1. `../README.md` for the runnable project overview.
2. `../report/report.pdf` for the concise final report; `../report/main.tex`
   is the editable LaTeX source.
3. `PHASE_REPRODUCTION.md` to rerun submitted portfolios.
4. `MODEL_SELECTION_LINEAGE.md` and `HYPERPARAMETERS.md` to audit how model
   structure and hyperparameters were selected.
5. `FEATURES_AND_SELECTION.md` and `DATA_PIPELINE.md` to audit feature/data
   construction.

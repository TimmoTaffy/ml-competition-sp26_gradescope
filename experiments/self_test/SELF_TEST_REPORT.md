# External Validation And Baseline Check

## Purpose

This file records a leakage-free realized check against the instructor
baseline. It is separate from the Phase2 live submission, whose May 11-15 result
was not known when the Phase2 portfolio was generated.

The final workflow does not reserve a separate static historical test set for
model selection. The reason is methodological: this is a live sequential
portfolio problem, the recent sample is short, and any fixed historical block
that is repeatedly inspected during tuning stops being a clean test. Instead,
candidate selection uses live-style walk-forward validation, and the true
external tests are the course live holding windows.

## Timing Protocol

The cleanest realized external check available before Phase2 is the Phase1 live
window.

| Layer | Dates | Role |
| --- | --- | --- |
| Data cutoff | 2026-05-03 deadline, latest A-share row 2026-04-30 | No May 6-8 prices were available when the Phase1 portfolio was generated |
| Training labels | Fully known labels through 2026-04-23 | 5-trading-day forward target requires a five-day label cutoff |
| Internal validation | Time-ordered pre-live validation with 5-trading-day embargo | Early stopping and feature/model/portfolio selection before submission |
| External realized check | 2026-05-06 to 2026-05-08 | Realized unseen live window after Phase1 submission |

The instructor baseline was regenerated with the same as-of convention:

```bash
conda run -n ml26s python baseline_xgboost.py \
  --as-of 20260430 \
  --top-k 50 \
  --out experiments/self_test/baseline_phase1_test.csv

conda run -n ml26s python validate_submission.py experiments/self_test/baseline_phase1_test.csv
conda run -n ml26s python score_submission.py experiments/self_test/baseline_phase1_test.csv --start 20260506 --end 20260508
conda run -n ml26s python score_submission.py submissions/submission1.csv --start 20260506 --end 20260508
```

## Results

| Model | Holdings | Portfolio Return | CSI500 Return | Excess Return |
| --- | ---: | ---: | ---: | ---: |
| Instructor baseline | 50 | +6.821% | +4.128% | +2.693% |
| Phase1 model | 31 | +8.962% | +4.128% | +4.834% |

The Phase1 model exceeded the instructor baseline by `+2.141` percentage
points of excess return on this realized external check.

## Interpretation

The window is short, so it should not be overinterpreted as a stable estimate of
future alpha. It is still a valid leakage-free external check for the Phase1
route: the portfolio was generated before the window opened, and no May 6-8
prices or labels entered the Phase1 training or validation process.

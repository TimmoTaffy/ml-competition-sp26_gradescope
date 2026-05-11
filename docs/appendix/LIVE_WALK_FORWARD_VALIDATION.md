# Live-Style Walk-Forward Validation

## Purpose

Use this as the strict candidate-comparison layer for retained portfolio
configs. It replaces fixed-block portfolio validation when deciding which
candidate is robust enough for submission.

The reason is 5-day target overlap. A label dated `D` uses returns from the
next 5 trading days. If a model trained near a validation boundary is allowed
to use labels whose forward window overlaps the portfolio-validation window,
the validation result can be biased upward.

## Method

For each non-overlapping historical holding window:

1. Use the trading day before the window as the entry/as-of date.
2. Retrain every candidate model spec from scratch as of that entry date.
3. Use only labels fully known as of the entry date.
4. Keep model validation as the internal early-stopping block.
5. Refit using the best tree count and all labels known as of the entry date.
6. Build the candidate's legal long-only portfolio at the entry date.
7. Score realized excess return versus CSI500 over the following window.

This simulates: "If this were the real submission date, what portfolio would
the pipeline have produced, and what happened next?"

## Command

Full retained-candidate run. The active workflow uses 13 recent non-overlapping
5-day windows as the longest strict validation view:

```bash
conda run -n ml26s python live_walk_forward_validation.py \
  --num-windows 13 \
  --window-length 5 \
  --out-dir experiments/research_families/fine_tune_top12/live_walk_forward_13w \
  --skip-existing
```

Smoke test:

```bash
conda run -n ml26s python live_walk_forward_validation.py \
  --num-windows 1 \
  --top-n 2 \
  --out-dir experiments/research_families/fine_tune_top12/live_walk_forward_smoke \
  --skip-existing
```

## Outputs

```text
experiments/research_families/fine_tune_top12/live_walk_forward_13w/
  windows.csv
  window_results.csv
  candidate_summary.csv
  holdings.csv
  report.md
  model_cache/
```

Important summary metrics:

| Column | Meaning |
| --- | --- |
| `mean_excess` | Average realized portfolio excess return across windows |
| `worst_window` | Worst realized 5-day excess return |
| `hit_rate` | Fraction of windows with positive excess |
| `avg_window_rank` | Average rank among retained candidates |
| `live_score` | `0.57 mean + 0.33 worst + 0.10 hit`, using percentile ranks |
| `stability_score` | Secondary rank score including average window rank |

## Interpretation

Prefer candidates that have positive mean excess, acceptable worst-window loss,
and reasonable rank stability. Do not repeatedly retune on the same windows
without reserving fresh future windows.

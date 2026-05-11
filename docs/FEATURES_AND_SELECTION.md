# Feature And Selection Notes

## Current Feature Scope

The active feature library excludes the deprecated low-coverage cross-market
feature family and all related raw or derived fields.

Current feature groups:

| Group | Examples |
| --- | --- |
| Baseline technical | returns, volatility, turnover, moving-average distance, RSI |
| CSI500-relative | excess returns, relative volatility, beta, residual momentum |
| Bollinger | z-score, band position, band width, lower/upper proximity, near/breakout flags |
| Liquidity and risk | recent halt, missing days, zero volume, drawdown, ATR, limit move |
| Cross-sectional | daily ranks and z-scores |
| Industry-relative | Shenwan industry returns and stock-minus-industry features |
| Valuation | PE, PB, market cap, earnings yield, book-to-price |
| Quality | ROE, ROA, margins, YoY/3-year growth, deducted-net-profit growth, cash-flow quality |
| Market regime | broad indices, ETFs, sector ETFs, crude/gold/global proxies |

Bollinger features use trading-day rolling windows. Current windows are
`10/20/60/120` trading days. The implementation keeps the raw band diagnostics
and adds direct features for the lower-band intuition:

| Feature Pattern | Meaning |
| --- | --- |
| `boll_position_<w>d` | Position inside band; `0` lower band, `1` upper band |
| `boll_lower_proximity_<w>d` | `1 - boll_position`; larger means closer to lower band |
| `dist_to_boll_lower_<w>d` | Price distance to lower band divided by close |
| `near_boll_lower_<w>d` | Flag for bottom 10% of the band |
| `below_boll_lower_<w>d` | Flag for closing below lower band |
| `near_boll_upper_<w>d` | Flag for top 10% of the band |
| `above_boll_upper_<w>d` | Flag for closing above upper band |

Window-insufficient rows stay missing instead of being encoded as false.

New quality fields added in the latest rebuild:

| Feature | Meaning |
| --- | --- |
| `revenue_growth_3y` | Same-report-period 3-year operating revenue growth |
| `net_profit_growth_3y` | Same-report-period 3-year net profit signed growth |
| `deducted_net_profit_growth_3y` | Same-report-period 3-year deducted net profit signed growth |
| `deducted_net_profit_growth_yoy` | Deducted net profit year-over-year growth from financial indicators |
| `cash_flow_quality` | Operating cash flow to net profit proxy |
| `cash_flow_to_revenue` | Operating cash flow to revenue proxy |
| `cash_flow_per_share_to_eps` | Per-share operating cash flow divided by EPS |

For each important quality field, the matrix also builds cross-sectional rank,
z-score, and industry-rank variants when the raw field is available. Three-year
growth compares the same fiscal slot, for example Q3 versus Q3 three years
earlier, and uses only announcement-date-available data.

CSI500 Quality Growth index (`930939`) features:

| Feature | Meaning |
| --- | --- |
| `csi500_quality_growth_ret_<w>d` | 930939 style-index recent return |
| `csi500_quality_growth_vol_20d` | 930939 20-day realized volatility |
| `csi500_quality_growth_minus_csi500_<w>d` | Quality-growth style return minus CSI500 return |
| `is_csi500_quality_growth_member` | Stock is in the latest 930939 constituent-weight snapshot |
| `csi500_quality_growth_weight` | 930939 constituent weight from the snapshot |
| `csi500_quality_growth_weight_rank` | Cross-sectional rank of the constituent weight |
| `csi500_quality_growth_snapshot_available` | Date-level flag that the snapshot is usable |

Constituent/weight features are not backfilled before the snapshot effective
date. In the current download, the weight snapshot is dated `2026-03-31`, so it
starts affecting the matrix from the next available A-share trading date.

## Active Targets

| Target | Meaning |
| --- | --- |
| `target_excess_5d` | Primary 5-trading-day stock return minus CSI500 return |
| `target_rank_5d` | Cross-sectional rank of 5-day forward return |
| `target_excess_3d` | Auxiliary short-window excess return |

## Feature Selection Workflow

```bash
conda run -n ml26s python feature_selection.py --profile general --config configs/feature_selection_general.json --out experiments/feature_selection/general
conda run -n ml26s python feature_selection.py --profile short_history --config configs/feature_selection_short_history.json --out experiments/feature_selection/short_history
```

The workflow uses single-factor rolling tests, group ablation, XGB confirmation,
and a combined multi-metric feature score. Held-out test remains excluded from
routine feature selection.

`recent_*` scoring now uses only the most recent slice of rolling validation
dates, controlled by `single_factor.recent_fraction`, instead of duplicating the
full-window score.

Latest feature-selection rebuild:

| Profile | Candidates | Selected | Notes |
| --- | ---: | ---: | --- |
| `general` | `390` | `120` | Selected multiple new growth and cash-flow quality fields |
| `short_history` | `281` | `90` | Selected no fundamental fields; remains price/regime focused |

For broader tuning, `tune_research_families.py` can generate additional ranking
families: `balanced`, `stable`, `recent`, `economic`, and `short_aggressive`.
These families intentionally represent different feature-ranking assumptions,
then use the same portfolio-validation standard for comparison.

After the Bollinger and 930939 additions, the rebuilt matrix contains `432`
feature columns. Existing feature-selection outputs should be regenerated; the
research tuning script now skips old rankings only when they are fresher than
the current matrix and feature-list files.

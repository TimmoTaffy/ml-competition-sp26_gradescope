# Feature Selection Report: general

## Split Policy

- Target: `target_excess_5d`
- Held-out test start: `2026-04-03`
- Held-out test trading days: `10`
- Test set is not used for feature selection by default.
- Test evaluation requested: `False`

## Rolling Validation Folds

- Fold 5: train `2025-01-02` to `2025-09-19`, validation `2025-09-29` to `2025-11-03`
- Fold 4: train `2025-01-02` to `2025-10-27`, validation `2025-11-04` to `2025-12-01`
- Fold 3: train `2025-01-02` to `2025-11-24`, validation `2025-12-02` to `2025-12-29`
- Fold 2: train `2025-01-02` to `2025-12-22`, validation `2025-12-30` to `2026-01-28`
- Fold 1: train `2025-01-02` to `2026-01-21`, validation `2026-01-29` to `2026-03-05`
- Fold 0: train `2025-01-02` to `2026-02-26`, validation `2026-03-06` to `2026-04-02`

## Selected Features

- Selected feature count: `120`
- `ret_5d_rank` (short_returns): combined `0.6580`
- `ret_5d` (short_returns): combined `0.6580`
- `ret_5d_zscore` (short_returns): combined `0.6580`
- `ret_3d_zscore` (short_returns): combined `0.6516`
- `ret_3d` (short_returns): combined `0.6516`
- `ret_3d_rank` (short_returns): combined `0.6516`
- `ret_1d` (short_returns): combined `0.6418`
- `industry_ret_60d` (industry_relative): combined `0.6353`
- `vol_10d` (short_returns): combined `0.6346`
- `vol_5d` (short_returns): combined `0.6334`
- `industry_ret_20d` (industry_relative): combined `0.6317`
- `rsi_6` (short_returns): combined `0.6211`
- `ret_2d` (short_returns): combined `0.6180`
- `boll_position_60d` (bollinger): combined `0.6153`
- `boll_z_60d` (bollinger): combined `0.6153`
- `boll_width_60d` (bollinger): combined `0.6152`
- `excess_ret_20d_industry_rank` (industry_relative): combined `0.6142`
- `ret_20d_industry_rank` (industry_relative): combined `0.6142`
- `ret_10d` (baseline_technical): combined `0.6088`
- `stock_ret_minus_industry_ret_20d` (industry_relative): combined `0.6049`
- `stock_ret_minus_industry_ret_60d` (industry_relative): combined `0.6027`
- `ret_60d` (baseline_technical): combined `0.5986`
- `excess_ret_60d` (csi500_relative): combined `0.5986`
- `stock_ret_minus_industry_ret_5d` (industry_relative): combined `0.5982`
- `close_over_ma20` (baseline_technical): combined `0.5979`

## Top Single-Factor Features

- `ret_10d`: score `0.6147`, IC `-0.0507`, top `0.0018`
- `excess_ret_60d`: score `0.6012`, IC `-0.0609`, top `0.0019`
- `ret_60d`: score `0.6012`, IC `-0.0609`, top `0.0019`
- `close_over_ma20`: score `0.6002`, IC `-0.0495`, top `0.0021`
- `close_over_ma60`: score `0.5975`, IC `-0.0610`, top `0.0011`
- `pe_ttm`: score `0.5918`, IC `-0.0388`, top `0.0014`
- `pe_ttm_rank`: score `0.5918`, IC `-0.0388`, top `0.0014`
- `pe_ttm_zscore`: score `0.5918`, IC `-0.0388`, top `0.0014`
- `turnover_ma_20d`: score `0.5915`, IC `-0.0458`, top `0.0023`
- `pe_static`: score `0.5851`, IC `-0.0372`, top `0.0001`
- `residual_ret_20d`: score `0.5848`, IC `-0.0360`, top `0.0027`
- `ret_20d_zscore`: score `0.5824`, IC `-0.0439`, top `0.0022`
- `excess_ret_20d_rank`: score `0.5824`, IC `-0.0439`, top `0.0022`
- `ret_20d_rank`: score `0.5824`, IC `-0.0439`, top `0.0022`
- `ret_20d`: score `0.5824`, IC `-0.0439`, top `0.0022`

## Top Ridge Group Ablations

- `baseline_plus_short_returns`: top-K `0.0036`, IC `-0.0001`, features `45`
- `baseline_plus_bollinger`: top-K `0.0034`, IC `-0.0054`, features `37`
- `baseline_plus_industry_relative`: top-K `0.0030`, IC `-0.0098`, features `44`
- `baseline`: top-K `0.0028`, IC `-0.0087`, features `34`
- `baseline_plus_cross_sectional`: top-K `0.0028`, IC `-0.0087`, features `34`
- `baseline_plus_csi500_relative`: top-K `0.0028`, IC `-0.0087`, features `34`
- `baseline_plus_liquidity_risk`: top-K `0.0028`, IC `-0.0087`, features `34`
- `baseline_plus_baseline_technical`: top-K `0.0028`, IC `-0.0087`, features `34`
- `baseline_plus_short_bollinger`: top-K `0.0024`, IC `-0.0052`, features `41`
- `baseline_plus_short_volume_price`: top-K `0.0022`, IC `-0.0087`, features `39`
- `all_minus_valuation`: top-K `0.0019`, IC `0.0016`, features `119`
- `all_minus_quality`: top-K `0.0014`, IC `-0.0171`, features `91`
- `baseline_plus_quality`: top-K `0.0013`, IC `-0.0077`, features `83`
- `all_minus_liquidity_risk`: top-K `0.0013`, IC `-0.0139`, features `134`
- `all_minus_baseline_technical`: top-K `0.0013`, IC `-0.0142`, features `130`

## XGBoost Confirmation

- Mean top-K excess: `0.0029`
- Mean rank IC: `-0.0152`
- Worst fold/window: `-0.0291`

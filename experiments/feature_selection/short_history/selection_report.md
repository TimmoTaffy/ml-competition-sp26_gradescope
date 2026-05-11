# Feature Selection Report: short_history

## Split Policy

- Target: `target_excess_5d`
- Held-out test start: `2026-04-03`
- Held-out test trading days: `10`
- Test set is not used for feature selection by default.
- Test evaluation requested: `False`

## Rolling Validation Folds

- Fold 7: train `2025-07-28` to `2025-11-24`, validation `2025-12-02` to `2025-12-15`
- Fold 6: train `2025-08-11` to `2025-12-08`, validation `2025-12-16` to `2025-12-29`
- Fold 5: train `2025-08-25` to `2025-12-22`, validation `2025-12-30` to `2026-01-14`
- Fold 4: train `2025-09-08` to `2026-01-07`, validation `2026-01-15` to `2026-01-28`
- Fold 3: train `2025-09-22` to `2026-01-21`, validation `2026-01-29` to `2026-02-11`
- Fold 2: train `2025-10-14` to `2026-02-04`, validation `2026-02-12` to `2026-03-05`
- Fold 1: train `2025-10-28` to `2026-02-26`, validation `2026-03-06` to `2026-03-19`
- Fold 0: train `2025-11-11` to `2026-03-12`, validation `2026-03-20` to `2026-04-02`

## Selected Features

- Selected feature count: `90`
- `boll_width_10d` (short_bollinger): combined `0.7421`
- `beta_to_csi500_60d` (csi500_relative): combined `0.7389`
- `residual_ret_20d` (csi500_relative): combined `0.7271`
- `residual_ret_1d` (csi500_relative): combined `0.7269`
- `excess_ret_1d` (csi500_relative): combined `0.7227`
- `excess_ret_60d` (csi500_relative): combined `0.7225`
- `excess_ret_2d` (csi500_relative): combined `0.7180`
- `excess_ret_3d` (csi500_relative): combined `0.7168`
- `excess_ret_3d_rank` (csi500_relative): combined `0.7168`
- `boll_z_20d` (short_bollinger): combined `0.6952`
- `boll_position_20d` (short_bollinger): combined `0.6952`
- `boll_z_10d` (short_bollinger): combined `0.6808`
- `boll_position_10d` (short_bollinger): combined `0.6808`
- `excess_ret_5d` (csi500_relative): combined `0.6777`
- `excess_ret_5d_rank` (csi500_relative): combined `0.6777`
- `excess_ret_20d` (csi500_relative): combined `0.6731`
- `excess_ret_20d_rank` (csi500_relative): combined `0.6731`
- `volume_z_5d` (short_volume_price): combined `0.6713`
- `boll_width_20d` (short_bollinger): combined `0.6693`
- `intraday_return` (short_volume_price): combined `0.6654`
- `ret_1d` (short_returns): combined `0.6602`
- `amount_z_5d` (short_volume_price): combined `0.6597`
- `relative_vol_20d` (csi500_relative): combined `0.6563`
- `ret_2d` (short_returns): combined `0.6555`
- `ret_3d_zscore` (short_returns): combined `0.6543`

## Top Single-Factor Features

- `beta_to_csi500_60d`: score `0.7361`, IC `0.0250`, top `0.0041`
- `volume_z_5d`: score `0.7298`, IC `-0.0178`, top `0.0003`
- `intraday_return`: score `0.7224`, IC `-0.0183`, top `0.0019`
- `residual_ret_20d`: score `0.7214`, IC `-0.0183`, top `0.0002`
- `residual_ret_1d`: score `0.7211`, IC `-0.0197`, top `0.0029`
- `max_drawdown_20d`: score `0.7181`, IC `-0.0126`, top `0.0025`
- `excess_ret_1d`: score `0.7159`, IC `-0.0150`, top `0.0022`
- `ret_1d`: score `0.7159`, IC `-0.0150`, top `0.0022`
- `excess_ret_60d`: score `0.7156`, IC `-0.0383`, top `-0.0007`
- `amount_z_5d`: score `0.7153`, IC `-0.0178`, top `-0.0005`
- `ret_2d`: score `0.7099`, IC `-0.0136`, top `0.0002`
- `excess_ret_2d`: score `0.7099`, IC `-0.0136`, top `0.0002`
- `boll_width_10d`: score `0.7089`, IC `0.0097`, top `0.0056`
- `excess_ret_3d_rank`: score `0.7085`, IC `-0.0142`, top `-0.0001`
- `ret_3d_zscore`: score `0.7085`, IC `-0.0142`, top `-0.0001`

## Top Ridge Group Ablations

- `all_minus_market_regime`: top-K `0.0039`, IC `-0.0097`, features `62`
- `all_minus_csi500_relative`: top-K `0.0030`, IC `-0.0229`, features `96`
- `baseline_plus_market_regime`: top-K `0.0030`, IC `-0.0413`, features `64`
- `baseline_plus_short_bollinger`: top-K `0.0029`, IC `-0.0367`, features `24`
- `baseline_plus_csi500_relative`: top-K `0.0028`, IC `-0.0092`, features `30`
- `all_minus_industry_relative`: top-K `0.0027`, IC `-0.0213`, features `100`
- `all_minus_cross_sectional`: top-K `0.0026`, IC `-0.0212`, features `106`
- `baseline_plus_cross_sectional`: top-K `0.0025`, IC `-0.0217`, features `20`
- `all_minus_short_returns`: top-K `0.0023`, IC `-0.0197`, features `99`
- `baseline`: top-K `0.0020`, IC `-0.0409`, features `16`
- `baseline_plus_short_returns`: top-K `0.0020`, IC `-0.0409`, features `16`
- `baseline_plus_short_volume_price`: top-K `0.0020`, IC `-0.0409`, features `16`
- `all_minus_short_bollinger`: top-K `0.0016`, IC `-0.0215`, features `102`
- `baseline_plus_liquidity_risk`: top-K `0.0015`, IC `-0.0345`, features `26`
- `all_features`: top-K `0.0014`, IC `-0.0224`, features `110`

## XGBoost Confirmation

- Mean top-K excess: `0.0032`
- Mean rank IC: `0.0224`
- Worst fold/window: `-0.0312`

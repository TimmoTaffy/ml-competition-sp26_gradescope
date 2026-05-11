"""
Prepare an expanded training matrix from the local CSI500 data snapshot.

This script intentionally uses only the existing local files:

  - data/prices.parquet
  - data/index.parquet
  - data/constituents.csv

It builds the timestamp-clean features that can be derived from OHLCV and the
CSI500 benchmark before any external data sources are added. External features
such as fundamentals, industries, and additional ETFs should be joined later
with explicit timestamp rules.

Usage
-----
  python prepare_training_data.py
  python prepare_training_data.py --out data/training_matrix.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"

RETURN_WINDOWS = [1, 2, 3, 5, 10, 20, 60, 120]
EXCESS_WINDOWS = [1, 2, 3, 5, 20, 60]
VOL_WINDOWS = [5, 10, 20]
BOLL_WINDOWS = [10, 20, 60, 120]
MA_WINDOWS = [20, 60]
TARGET_HORIZONS = [3, 5]

ROLLING_BETA_WINDOW = 60
RESIDUAL_MOMENTUM_WINDOW = 20
IDIOSYNCRATIC_VOL_WINDOW = 60
LIQUIDITY_WINDOW = 20
FAST_ACTIVITY_WINDOW = 5
RSI_WINDOWS = [6, 14]
ATR_WINDOW = 14
DRAWDOWN_WINDOW = 20
LOW_LIQUIDITY_QUANTILE = 0.10
LIMIT_MOVE_THRESHOLD = 0.095


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return num / denom


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = _safe_divide(up, down)
    return 100 - 100 / (1 + rs)


def _rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    def max_dd(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        running_max = np.maximum.accumulate(values)
        drawdowns = values / running_max - 1.0
        return float(np.min(drawdowns))

    return close.rolling(window, min_periods=window).apply(max_dd, raw=True)


def _load_inputs(
    prices_path: Path,
    index_path: Path,
    constituents_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(prices_path)
    index_df = pd.read_parquet(index_path)
    constituents = pd.read_csv(constituents_path, dtype={"stock_code": str})

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    index_df = index_df.copy()
    index_df["date"] = pd.to_datetime(index_df["date"])

    constituents = constituents.copy()
    constituents["stock_code"] = constituents["stock_code"].astype(str).str.zfill(6)
    return prices, index_df, constituents


def _build_full_panel(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    constituents: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.Index(sorted(index_df["date"].unique()), name="date")
    universe = pd.Index(sorted(constituents["stock_code"].unique()), name="stock_code")
    full_index = pd.MultiIndex.from_product([universe, calendar])

    panel = (
        prices.set_index(["stock_code", "date"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["stock_code", "date"])
        .reset_index(drop=True)
    )
    panel["is_missing_bar"] = panel["close"].isna()
    return panel


def _add_index_features(index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df.sort_values("date").copy()
    close = idx["close"]
    for w in sorted(set(RETURN_WINDOWS + TARGET_HORIZONS)):
        idx[f"csi500_ret_{w}d"] = close / close.shift(w) - 1.0
        idx[f"csi500_fwd_ret_{w}d"] = close.shift(-w) / close - 1.0
    idx["csi500_vol_20d"] = idx["csi500_ret_1d"].rolling(20, min_periods=20).std()
    return idx[[
        "date",
        *[f"csi500_ret_{w}d" for w in sorted(set(RETURN_WINDOWS + TARGET_HORIZONS))],
        *[f"csi500_fwd_ret_{w}d" for w in TARGET_HORIZONS],
        "csi500_vol_20d",
    ]]


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)

    for w in RETURN_WINDOWS:
        df[f"ret_{w}d"] = close / close.shift(w) - 1.0

    for w in VOL_WINDOWS:
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w, min_periods=w).std()

    for w in [FAST_ACTIVITY_WINDOW, LIQUIDITY_WINDOW]:
        vol_mean = df["volume"].rolling(w, min_periods=w).mean()
        vol_std = df["volume"].rolling(w, min_periods=w).std().replace(0, np.nan)
        amt_mean = df["amount"].rolling(w, min_periods=w).mean()
        amt_std = df["amount"].rolling(w, min_periods=w).std().replace(0, np.nan)
        df[f"volume_z_{w}d"] = (df["volume"] - vol_mean) / vol_std
        df[f"amount_z_{w}d"] = (df["amount"] - amt_mean) / amt_std

    df["turnover_ma_20d"] = df["turnover"].rolling(20, min_periods=20).mean()
    df["amount_ma_20d"] = df["amount"].rolling(20, min_periods=20).mean()

    for w in MA_WINDOWS:
        ma = close.rolling(w, min_periods=w).mean()
        df[f"close_over_ma{w}"] = close / ma - 1.0

    for w in RSI_WINDOWS:
        df[f"rsi_{w}"] = _rsi(close, w)

    for w in BOLL_WINDOWS:
        mid = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std()
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std
        df[f"boll_mid_{w}d"] = mid
        df[f"boll_std_{w}d"] = std
        df[f"boll_upper_{w}d"] = upper
        df[f"boll_lower_{w}d"] = lower
        df[f"boll_z_{w}d"] = _safe_divide(close - mid, std)
        df[f"boll_width_{w}d"] = _safe_divide(upper - lower, mid)
        df[f"boll_position_{w}d"] = _safe_divide(close - lower, upper - lower)
        df[f"boll_lower_proximity_{w}d"] = 1.0 - df[f"boll_position_{w}d"]
        df[f"dist_to_boll_lower_{w}d"] = _safe_divide(close - lower, close)
        df[f"dist_to_boll_upper_{w}d"] = _safe_divide(upper - close, close)
        valid_band = df[f"boll_position_{w}d"].notna()
        df[f"near_boll_lower_{w}d"] = np.where(valid_band, df[f"boll_position_{w}d"] <= 0.10, np.nan)
        df[f"near_boll_upper_{w}d"] = np.where(valid_band, df[f"boll_position_{w}d"] >= 0.90, np.nan)
        df[f"below_boll_lower_{w}d"] = np.where(valid_band, close < lower, np.nan)
        df[f"above_boll_upper_{w}d"] = np.where(valid_band, close > upper, np.nan)

    df["high_low_range_20d"] = ((high - low) / close).rolling(20, min_periods=20).mean()
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14d"] = (true_range / close).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()
    df["max_drawdown_20d"] = _rolling_max_drawdown(close, DRAWDOWN_WINDOW)
    df["overnight_gap"] = open_ / prev_close - 1.0
    df["intraday_return"] = close / open_ - 1.0
    df["close_location_value"] = _safe_divide(close - low, high - low)

    df["missing_days_20d"] = df["is_missing_bar"].rolling(20, min_periods=1).sum()
    df["zero_volume_days_20d"] = (
        (df["volume"].fillna(0) <= 0).rolling(20, min_periods=1).sum()
    )
    df["recent_halt_flag"] = (
        (df["missing_days_20d"] > 0) | (df["zero_volume_days_20d"] > 0)
    ).astype(float)
    df["limit_move_flag"] = (df["ret_1d"].abs() >= LIMIT_MOVE_THRESHOLD).astype(float)

    for h in TARGET_HORIZONS:
        df[f"target_return_{h}d"] = close.shift(-h) / close - 1.0

    # Backward-compatible alias for the original baseline target name.
    df["target_5d"] = df["target_return_5d"]
    return df


def _add_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    for w in EXCESS_WINDOWS:
        panel[f"excess_ret_{w}d"] = panel[f"ret_{w}d"] - panel[f"csi500_ret_{w}d"]

    panel["relative_vol_20d"] = _safe_divide(panel["vol_20d"], panel["csi500_vol_20d"])

    def per_stock_relative(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        cov = df["ret_1d"].rolling(ROLLING_BETA_WINDOW, min_periods=ROLLING_BETA_WINDOW).cov(
            df["csi500_ret_1d"]
        )
        var = df["csi500_ret_1d"].rolling(ROLLING_BETA_WINDOW, min_periods=ROLLING_BETA_WINDOW).var()
        df["beta_to_csi500_60d"] = cov / var.replace(0, np.nan)
        df["residual_ret_1d"] = df["ret_1d"] - df["beta_to_csi500_60d"] * df["csi500_ret_1d"]
        df["residual_ret_20d"] = (
            df["residual_ret_1d"]
            .rolling(RESIDUAL_MOMENTUM_WINDOW, min_periods=RESIDUAL_MOMENTUM_WINDOW)
            .sum()
        )
        df["idiosyncratic_vol_60d"] = (
            df["residual_ret_1d"]
            .rolling(IDIOSYNCRATIC_VOL_WINDOW, min_periods=IDIOSYNCRATIC_VOL_WINDOW)
            .std()
        )
        return df

    frames = [per_stock_relative(df) for _, df in panel.groupby("stock_code", sort=False)]
    return pd.concat(frames, ignore_index=True)


def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    rank_bases = [
        "ret_3d", "ret_5d", "ret_20d", "vol_20d",
        "excess_ret_3d", "excess_ret_5d", "excess_ret_20d",
    ]
    zscore_bases = ["ret_3d", "ret_5d", "ret_20d", "vol_20d"]

    for base in rank_bases:
        if base in panel.columns:
            panel[f"{base}_rank"] = panel.groupby("date")[base].rank(method="average", pct=True)

    for base in zscore_bases:
        if base in panel.columns:
            panel[f"{base}_zscore"] = panel.groupby("date")[base].transform(_zscore)

    panel["low_liquidity_flag"] = (
        panel.groupby("date")["amount_ma_20d"].rank(method="average", pct=True)
        <= LOW_LIQUIDITY_QUANTILE
    ).astype(float)
    return panel


def _add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for h in TARGET_HORIZONS:
        panel[f"target_excess_{h}d"] = panel[f"target_return_{h}d"] - panel[f"csi500_fwd_ret_{h}d"]

    panel["target_rank_5d"] = panel.groupby("date")["target_return_5d"].rank(
        method="average", pct=True
    )
    panel["target_zscore_5d"] = panel.groupby("date")["target_return_5d"].transform(_zscore)
    panel["target_excess_rank_5d"] = panel.groupby("date")["target_excess_5d"].rank(
        method="average", pct=True
    )
    return panel


def build_training_matrix(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    constituents: pd.DataFrame,
) -> pd.DataFrame:
    panel = _build_full_panel(prices, index_df, constituents)
    idx_features = _add_index_features(index_df)
    panel = panel.merge(idx_features, on="date", how="left")

    frames = [_per_stock_features(df) for _, df in panel.groupby("stock_code", sort=False)]
    panel = pd.concat(frames, ignore_index=True)
    panel = _add_relative_features(panel)
    panel = _add_cross_sectional_features(panel)
    panel = _add_targets(panel)
    return panel.sort_values(["date", "stock_code"]).reset_index(drop=True)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("target_", "csi500_fwd_ret_")
    excluded = {
        "date", "stock_code", "stock_name", "as_of_date",
        "open", "close", "high", "low", "volume", "amount", "turnover", "pct_change",
    }
    excluded_exact = {
        c for c in df.columns
        if c.startswith(("boll_mid_", "boll_std_", "boll_upper_", "boll_lower_"))
    }
    features = []
    for col in df.columns:
        if col in excluded:
            continue
        if col in excluded_exact:
            continue
        if col.startswith(excluded_prefixes):
            continue
        features.append(col)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--constituents", default=str(DATA_DIR / "constituents.csv"))
    parser.add_argument("--out", default=str(DATA_DIR / "training_matrix.parquet"))
    parser.add_argument("--feature-list-out", default=str(DATA_DIR / "feature_columns_v2.txt"))
    parser.add_argument("--target-list-out", default=str(DATA_DIR / "target_columns_v2.txt"))
    args = parser.parse_args()

    prices, index_df, constituents = _load_inputs(
        Path(args.prices), Path(args.index), Path(args.constituents)
    )
    matrix = build_training_matrix(prices, index_df, constituents)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_path, index=False)

    feature_columns = infer_feature_columns(matrix)
    target_columns = [c for c in matrix.columns if c.startswith("target_")]
    Path(args.feature_list_out).write_text("\n".join(feature_columns) + "\n")
    Path(args.target_list_out).write_text("\n".join(target_columns) + "\n")

    print(f"Wrote {len(matrix):,} rows x {len(matrix.columns):,} columns to {out_path}")
    print(f"Feature columns: {len(feature_columns)} -> {args.feature_list_out}")
    print(f"Target columns: {len(target_columns)} -> {args.target_list_out}")
    print(
        "Date range:",
        matrix["date"].min().date(),
        "to",
        matrix["date"].max().date(),
        "| stocks:",
        matrix["stock_code"].nunique(),
    )


if __name__ == "__main__":
    main()

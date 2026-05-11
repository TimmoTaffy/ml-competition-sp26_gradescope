"""
Build the final feature matrix by joining optional external datasets.

This script converts raw external data into model-ready features. It is designed
to be safe when a dataset is missing: available sources are joined, missing
sources are reported, and the output is still written.

Usage
-----
  python build_feature_matrix.py
  python build_feature_matrix.py --base data/training_matrix.parquet --out data/final_training_matrix.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
EXTERNAL_DIR = DATA_DIR / "external"

RET_WINDOWS = [1, 3, 5, 20, 60]
RANK_FEATURES = [
    "pe_ttm", "pb", "ps_ttm", "total_market_cap", "float_market_cap",
    "roe_ttm", "roa_ttm", "gross_margin_ttm", "net_margin_ttm",
    "debt_to_asset", "revenue_growth_yoy", "net_profit_growth_yoy",
    "deducted_net_profit_growth_yoy", "revenue_growth_3y",
    "net_profit_growth_3y", "deducted_net_profit_growth_3y",
    "cash_flow_quality", "cash_flow_to_revenue",
    "cash_flow_per_share_to_eps",
]

SYMBOL_ALIASES = {
    "000300": "csi300",
    "000905": "csi500_market",
    "000852": "csi1000",
    "399006": "chinext",
    "000688": "star50",
    "510300": "csi300_etf",
    "510500": "csi500_etf",
    "512100": "csi1000_etf",
    "930939": "csi500_quality_growth",
    "159915": "chinext_etf",
    "588000": "star50_etf",
    "510880": "dividend_etf",
    "512000": "securities_etf",
    "518880": "gold_etf",
    "512800": "bank_etf",
    "512400": "nonferrous_metals_etf",
    "516020": "chemical_etf",
    "515220": "coal_etf",
    "159697": "oil_gas_etf",
    "513350": "sp_oil_gas_etf",
    "512480": "semiconductor_etf",
    "512660": "military_etf",
    "512010": "pharma_etf",
    "159920": "hang_seng_etf",
    "513180": "hang_seng_tech_etf",
    "513500": "sp500_etf",
    "513100": "nasdaq100_etf",
    ".INX": "sp500",
    ".NDX": "nasdaq100",
    "HSI": "hsi",
    "HSTECH": "hstech",
    "HSTECF2L": "hstech",
    "VHSI": "vhsi",
    "SC0": "shfe_crude",
    "CL": "wti_crude",
    "OIL": "brent_crude",
    "GC": "gold_futures",
}


def read_optional(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"  [skip] missing {path}")
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"stock_code": str})
    return pd.read_parquet(path)


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(6)


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


def signed_change(current: pd.Series, base: pd.Series) -> pd.Series:
    return (current - base) / base.abs().replace(0, np.nan)


def next_trading_dates(dates: pd.Series, trading_calendar: pd.Series | pd.DatetimeIndex) -> pd.Series:
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.Series(trading_calendar).drop_duplicates()).sort_values())
    values = pd.to_datetime(dates)
    positions = calendar.searchsorted(values, side="right")
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    out.loc[valid] = calendar[positions[valid]]
    return out


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    return value.strip("_")


def symbol_alias(symbol: str, name: str = "") -> str:
    symbol = str(symbol)
    if symbol in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[symbol]
    base = slugify(name) if name else ""
    return base or f"symbol_{slugify(symbol)}"


def asof_join_by_stock(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_date_col: str,
) -> pd.DataFrame:
    if right.empty:
        return left

    right = right.copy()
    right["stock_code"] = normalize_code(right["stock_code"])
    right["_asof_date"] = pd.to_datetime(right[right_date_col], errors="coerce")
    right = right.dropna(subset=["_asof_date"])

    frames = []
    for stock_code, left_group in left.groupby("stock_code", sort=False):
        ldf = left_group.sort_values("date").copy()
        rdf = right[right["stock_code"] == stock_code].sort_values("_asof_date").copy()
        if rdf.empty:
            frames.append(ldf)
            continue
        rdf = rdf.drop(columns=["stock_code", right_date_col], errors="ignore")
        rdf = rdf.drop_duplicates(subset=["_asof_date"], keep="last")
        merged = pd.merge_asof(
            ldf,
            rdf,
            left_on="date",
            right_on="_asof_date",
            direction="backward",
        )
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=["_asof_date"], errors="ignore")


def build_regime_features(trading_calendar: pd.Series, *frames: pd.DataFrame) -> pd.DataFrame:
    raw = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna(subset=["date", "symbol", "close"])

    feature_frames = []
    for (symbol, name), grp in raw.groupby(["symbol", "name"], dropna=False):
        grp = grp.sort_values("date").copy()
        alias = symbol_alias(str(symbol), str(name))
        out = pd.DataFrame({"source_date": grp["date"]})
        out["date"] = next_trading_dates(out["source_date"], trading_calendar)
        for w in RET_WINDOWS:
            out[f"{alias}_ret_{w}d"] = grp["close"] / grp["close"].shift(w) - 1.0
        out[f"{alias}_vol_20d"] = out[f"{alias}_ret_1d"].rolling(20, min_periods=20).std()
        out = (
            out.dropna(subset=["date"])
            .sort_values(["date", "source_date"])
            .drop_duplicates(subset=["date"], keep="last")
            .drop(columns=["source_date"])
        )
        feature_frames.append(out)

    regime = feature_frames[0]
    for frame in feature_frames[1:]:
        regime = regime.merge(frame, on="date", how="outer")
    return regime.sort_values("date")


def add_market_regime_features(
    matrix: pd.DataFrame,
    index_etf_path: Path,
    global_index_path: Path,
    futures_path: Path,
) -> pd.DataFrame:
    index_etf = read_optional(index_etf_path)
    global_index = read_optional(global_index_path)
    futures = read_optional(futures_path)
    regime = build_regime_features(matrix["date"], index_etf, global_index, futures)
    if regime.empty:
        return matrix

    matrix = matrix.merge(regime, on="date", how="left")
    for w in [5, 20, 60]:
        csi500_col = f"csi500_ret_{w}d"
        if csi500_col in matrix and f"csi300_ret_{w}d" in matrix:
            matrix[f"csi500_minus_csi300_{w}d"] = matrix[csi500_col] - matrix[f"csi300_ret_{w}d"]
        if f"csi1000_ret_{w}d" in matrix and f"csi300_ret_{w}d" in matrix:
            matrix[f"csi1000_minus_csi300_{w}d"] = matrix[f"csi1000_ret_{w}d"] - matrix[f"csi300_ret_{w}d"]
        if f"chinext_ret_{w}d" in matrix and f"csi300_ret_{w}d" in matrix:
            matrix[f"chinext_minus_csi300_{w}d"] = matrix[f"chinext_ret_{w}d"] - matrix[f"csi300_ret_{w}d"]
        if f"csi500_quality_growth_ret_{w}d" in matrix and csi500_col in matrix:
            matrix[f"csi500_quality_growth_minus_csi500_{w}d"] = matrix[f"csi500_quality_growth_ret_{w}d"] - matrix[csi500_col]
        dividend_col = next((c for c in [f"csi_dividend_ret_{w}d", f"dividend_etf_ret_{w}d"] if c in matrix), None)
        growth_col = next((c for c in [f"chinext_ret_{w}d", f"chinext_etf_ret_{w}d"] if c in matrix), None)
        if dividend_col and growth_col:
            matrix[f"dividend_growth_spread_{w}d"] = matrix[dividend_col] - matrix[growth_col]

        oil_col = next((c for c in [f"wti_crude_ret_{w}d", f"shfe_crude_ret_{w}d", f"oil_gas_etf_ret_{w}d"] if c in matrix), None)
        if oil_col and f"gold_etf_ret_{w}d" in matrix:
            matrix[f"oil_minus_gold_{w}d"] = matrix[oil_col] - matrix[f"gold_etf_ret_{w}d"]
    return matrix


def add_industry_features(matrix: pd.DataFrame, industry_path: Path) -> pd.DataFrame:
    industry = read_optional(industry_path)
    if industry.empty:
        return matrix

    industry = industry.copy()
    industry["stock_code"] = normalize_code(industry["stock_code"])
    if "industry_code" not in industry and "sw_level2_code" in industry:
        industry["industry_code"] = industry["sw_level2_code"]
    keep_cols = [c for c in ["stock_code", "industry_code", "industry_name", "industry_level"] if c in industry]
    industry = industry[keep_cols].drop_duplicates(subset=["stock_code"], keep="last")
    matrix = matrix.merge(industry, on="stock_code", how="left")

    if "industry_code" not in matrix:
        return matrix

    for w in [5, 20, 60]:
        ret_col = f"ret_{w}d"
        if ret_col not in matrix:
            continue
        industry_ret = matrix.groupby(["date", "industry_code"])[ret_col].transform("mean")
        matrix[f"industry_ret_{w}d"] = industry_ret
        matrix[f"stock_ret_minus_industry_ret_{w}d"] = matrix[ret_col] - industry_ret

    for base in ["ret_5d", "ret_20d", "excess_ret_5d", "excess_ret_20d"]:
        if base in matrix:
            matrix[f"{base}_industry_rank"] = matrix.groupby(["date", "industry_code"])[base].rank(method="average", pct=True)
    return matrix


def add_valuation_features(matrix: pd.DataFrame, valuation_path: Path) -> pd.DataFrame:
    valuation = read_optional(valuation_path)
    if valuation.empty:
        return matrix

    valuation = valuation.copy()
    valuation["stock_code"] = normalize_code(valuation["stock_code"])
    valuation["date"] = pd.to_datetime(valuation["date"])
    valuation["availability_date"] = next_trading_dates(valuation["date"], matrix["date"])
    numeric_cols = [
        "pe_ttm", "pe_static", "pb", "ps_ttm", "pcf_ttm",
        "total_market_cap", "float_market_cap",
    ]
    for col in numeric_cols:
        if col in valuation:
            valuation[col] = pd.to_numeric(valuation[col], errors="coerce")

    use_cols = ["stock_code", "availability_date", *[c for c in numeric_cols if c in valuation]]
    matrix = asof_join_by_stock(matrix, valuation[use_cols], "availability_date")

    if "total_market_cap" in matrix:
        matrix["log_total_market_cap"] = np.log(matrix["total_market_cap"].where(matrix["total_market_cap"] > 0))
    if "float_market_cap" in matrix:
        matrix["log_float_market_cap"] = np.log(matrix["float_market_cap"].where(matrix["float_market_cap"] > 0))
    if "pe_ttm" in matrix:
        matrix["earnings_yield_ttm"] = safe_divide(pd.Series(1.0, index=matrix.index), matrix["pe_ttm"])
    if "pb" in matrix:
        matrix["book_to_price"] = safe_divide(pd.Series(1.0, index=matrix.index), matrix["pb"])
    if "ps_ttm" in matrix:
        matrix["sales_to_price_ttm"] = safe_divide(pd.Series(1.0, index=matrix.index), matrix["ps_ttm"])
    return matrix


def coalesce_first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    cols = [c for c in candidates if c in df]
    if not cols:
        return None
    out = pd.to_numeric(df[cols[0]], errors="coerce")
    for col in cols[1:]:
        out = out.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return out


def standardize_fundamentals(fundamentals: pd.DataFrame, source_priority: int = 0) -> pd.DataFrame:
    fundamentals = fundamentals.copy()
    fundamentals["stock_code"] = normalize_code(fundamentals["stock_code"])
    if "announcement_date" not in fundamentals:
        return pd.DataFrame()
    fundamentals["announcement_date"] = pd.to_datetime(fundamentals["announcement_date"], errors="coerce")

    rename = {
        "roe": "roe_ttm",
        "roa": "roa_ttm",
        "gross_margin": "gross_margin_ttm",
        "net_margin": "net_margin_ttm",
        "operating_revenue_yoy": "revenue_growth_yoy",
        "net_profit_yoy": "net_profit_growth_yoy",
        "net_profit_growth_yoy": "net_profit_growth_yoy",
    }
    fundamentals = fundamentals.rename(columns={k: v for k, v in rename.items() if k in fundamentals and v not in fundamentals})

    derived_sources = {
        "operating_revenue": ["operating_revenue", "TOTALOPERATEREVE"],
        "net_profit": ["net_profit", "PARENTNETPROFIT"],
        "deducted_net_profit": ["deducted_net_profit", "KCFJCXSYJLR"],
        "deducted_net_profit_growth_yoy": ["deducted_net_profit_growth_yoy", "KCFJCXSYJLRTZ"],
        "cash_flow_per_share": ["cash_flow_per_share", "MGJYXJJE"],
        "cash_flow_to_revenue": ["cash_flow_to_revenue", "JYXJLYYSR"],
        "operating_cash_flow_to_net_profit": ["operating_cash_flow_to_net_profit", "NCO_NETPROFIT"],
        "eps": ["eps", "EPSJB"],
    }
    for target, candidates in derived_sources.items():
        if target not in fundamentals:
            values = coalesce_first_existing(fundamentals, candidates)
            if values is not None:
                fundamentals[target] = values

    numeric_cols = [
        "roe_ttm", "roa_ttm", "gross_margin_ttm", "net_margin_ttm",
        "debt_to_asset", "revenue_growth_yoy", "net_profit_growth_yoy",
        "deducted_net_profit_growth_yoy", "operating_revenue", "net_profit",
        "deducted_net_profit", "eps", "bps", "cash_flow_per_share",
        "cash_flow_to_revenue", "operating_cash_flow_to_net_profit",
    ]
    for col in numeric_cols:
        if col in fundamentals:
            fundamentals[col] = pd.to_numeric(fundamentals[col], errors="coerce")

    keep = ["stock_code", "announcement_date", *[c for c in numeric_cols if c in fundamentals]]
    if "report_period" in fundamentals:
        fundamentals["report_period"] = pd.to_datetime(fundamentals["report_period"], errors="coerce")
        keep.append("report_period")
        fundamentals = fundamentals.sort_values(["stock_code", "announcement_date", "report_period"])
    out = fundamentals[keep].dropna(subset=["announcement_date"]).copy()
    out["_source_priority"] = source_priority
    return out


def combine_fundamental_sources(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for priority, path in enumerate(paths):
        raw = read_optional(path)
        if raw.empty:
            continue
        standardized = standardize_fundamentals(raw, source_priority=priority)
        if not standardized.empty:
            frames.append(standardized)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "report_period" not in combined:
        combined["report_period"] = pd.NaT

    combined = combined.sort_values(["stock_code", "report_period", "_source_priority"])
    keys = ["stock_code", "report_period"]
    value_cols = [
        c for c in combined.columns
        if c not in keys + ["announcement_date", "_source_priority"]
    ]

    # Merge sources at the report-period level. Availability is the latest
    # announcement date among contributing sources, which is conservative when
    # the cleaner summary table and the fuller indicator table disagree.
    values = (
        combined.groupby(keys, dropna=False)[value_cols]
        .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        .reset_index()
    )
    availability = combined.groupby(keys, dropna=False)["announcement_date"].max().reset_index()
    return values.merge(availability, on=keys, how="left")


def add_long_horizon_fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty or "report_period" not in fundamentals:
        return fundamentals

    fundamentals = fundamentals.sort_values(["stock_code", "report_period", "announcement_date"]).copy()
    fundamentals["_fiscal_slot"] = fundamentals["report_period"].dt.strftime("%m-%d")
    for base_col, feature_col in [
        ("operating_revenue", "revenue_growth_3y"),
        ("net_profit", "net_profit_growth_3y"),
        ("deducted_net_profit", "deducted_net_profit_growth_3y"),
    ]:
        if base_col not in fundamentals:
            continue
        lag_col = f"_{base_col}_lag_3y"
        fundamentals[lag_col] = fundamentals.groupby(["stock_code", "_fiscal_slot"])[base_col].shift(3)
        if base_col == "operating_revenue":
            fundamentals[feature_col] = safe_divide(fundamentals[base_col], fundamentals[lag_col]) - 1.0
        else:
            fundamentals[feature_col] = signed_change(fundamentals[base_col], fundamentals[lag_col])
        fundamentals = fundamentals.drop(columns=[lag_col])

    if "operating_cash_flow_to_net_profit" in fundamentals:
        fundamentals["cash_flow_quality"] = fundamentals["operating_cash_flow_to_net_profit"]
    if {"cash_flow_per_share", "eps"}.issubset(fundamentals.columns):
        fundamentals["cash_flow_per_share_to_eps"] = safe_divide(fundamentals["cash_flow_per_share"], fundamentals["eps"])
    return fundamentals.drop(columns=["_fiscal_slot"], errors="ignore")


def add_fundamental_features(
    matrix: pd.DataFrame,
    fundamentals_path: Path,
    fundamental_indicators_path: Path | None = None,
) -> pd.DataFrame:
    paths = [fundamentals_path]
    if fundamental_indicators_path is not None:
        paths.append(fundamental_indicators_path)
    fundamentals = combine_fundamental_sources(paths)
    if fundamentals.empty:
        return matrix
    fundamentals = add_long_horizon_fundamental_features(fundamentals)
    fundamentals["availability_date"] = next_trading_dates(fundamentals["announcement_date"], matrix["date"])
    fundamentals = fundamentals.drop(columns=["announcement_date"], errors="ignore")
    return asof_join_by_stock(matrix, fundamentals, "availability_date")


def add_quality_growth_features(matrix: pd.DataFrame, constituents_path: Path) -> pd.DataFrame:
    constituents = read_optional(constituents_path)
    if constituents.empty:
        return matrix
    constituents = constituents.copy()
    constituents["stock_code"] = normalize_code(constituents["stock_code"])
    if "effective_date" not in constituents:
        return matrix
    constituents["effective_date"] = pd.to_datetime(constituents["effective_date"], errors="coerce")
    constituents = constituents.dropna(subset=["effective_date", "stock_code"])
    if constituents.empty:
        return matrix
    constituents["availability_date"] = next_trading_dates(constituents["effective_date"], matrix["date"])
    constituents["is_csi500_quality_growth_member"] = pd.to_numeric(
        constituents.get("is_csi500_quality_growth_member", 1.0),
        errors="coerce",
    ).fillna(1.0)
    if "weight" in constituents:
        constituents["csi500_quality_growth_weight"] = pd.to_numeric(constituents["weight"], errors="coerce")
    use_cols = [
        "stock_code",
        "availability_date",
        "is_csi500_quality_growth_member",
        "csi500_quality_growth_weight",
    ]
    use_cols = [c for c in use_cols if c in constituents]
    first_available = constituents["availability_date"].dropna().min()
    matrix = asof_join_by_stock(matrix, constituents[use_cols], "availability_date")
    if pd.notna(first_available):
        available_mask = matrix["date"] >= first_available
        for col in ["is_csi500_quality_growth_member", "csi500_quality_growth_weight"]:
            if col in matrix:
                matrix.loc[available_mask, col] = matrix.loc[available_mask, col].fillna(0.0)
        matrix["csi500_quality_growth_snapshot_available"] = available_mask.astype(float)
    if "csi500_quality_growth_weight" in matrix:
        matrix["csi500_quality_growth_weight_rank"] = matrix.groupby("date")["csi500_quality_growth_weight"].rank(method="average", pct=True)
    return matrix


def add_cross_sectional_ranks(matrix: pd.DataFrame) -> pd.DataFrame:
    for col in RANK_FEATURES:
        if col in matrix:
            matrix[f"{col}_rank"] = matrix.groupby("date")[col].rank(method="average", pct=True)
            matrix[f"{col}_zscore"] = matrix.groupby("date")[col].transform(zscore)
            if "industry_code" in matrix:
                matrix[f"{col}_industry_rank"] = matrix.groupby(["date", "industry_code"])[col].rank(method="average", pct=True)
    return matrix


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("target_", "csi500_fwd_ret_")
    excluded = {
        "date", "stock_code", "stock_name", "as_of_date", "report_period",
        "industry_code", "industry_name", "industry_level",
        "open", "close", "high", "low", "volume", "amount", "turnover", "pct_change",
    }
    excluded_exact = {
        c for c in df.columns
        if c.startswith(("boll_mid_", "boll_std_", "boll_upper_", "boll_lower_"))
    }

    features = []
    for col in df.columns:
        if col in excluded or col in excluded_exact:
            continue
        if col.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            features.append(col)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DATA_DIR / "training_matrix.parquet"))
    parser.add_argument("--index-etf", default=str(EXTERNAL_DIR / "index_etf_daily.parquet"))
    parser.add_argument("--global-index", default=str(EXTERNAL_DIR / "global_index_daily.parquet"))
    parser.add_argument("--futures", default=str(EXTERNAL_DIR / "futures_regime_daily.parquet"))
    parser.add_argument("--industry", default=str(EXTERNAL_DIR / "industry_classification.csv"))
    parser.add_argument("--valuation", default=str(EXTERNAL_DIR / "valuation_daily.parquet"))
    parser.add_argument("--fundamentals", default=str(EXTERNAL_DIR / "fundamentals.parquet"))
    parser.add_argument("--fundamental-indicators", default=str(EXTERNAL_DIR / "fundamentals_indicators.parquet"))
    parser.add_argument("--quality-growth", default=str(EXTERNAL_DIR / "quality_growth_constituents.parquet"))
    parser.add_argument("--out", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--feature-list-out", default=str(DATA_DIR / "final_feature_columns.txt"))
    parser.add_argument("--target-list-out", default=str(DATA_DIR / "final_target_columns.txt"))
    args = parser.parse_args()

    matrix = pd.read_parquet(args.base)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = normalize_code(matrix["stock_code"])

    matrix = add_market_regime_features(matrix, Path(args.index_etf), Path(args.global_index), Path(args.futures))
    matrix = add_industry_features(matrix, Path(args.industry))
    matrix = add_valuation_features(matrix, Path(args.valuation))
    matrix = add_fundamental_features(matrix, Path(args.fundamentals), Path(args.fundamental_indicators))
    matrix = add_quality_growth_features(matrix, Path(args.quality_growth))
    matrix = add_cross_sectional_ranks(matrix)
    matrix = matrix.sort_values(["date", "stock_code"]).reset_index(drop=True)

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


if __name__ == "__main__":
    main()

"""
Download external public datasets for the CSI500 stock-selection project.

The downloader is deliberately defensive: AKShare endpoints can change, fail,
or expose different columns across versions. Each dataset writes raw or lightly
standardized files under data/external/ and logs failures without corrupting
existing base data.

Implemented datasets
--------------------
index_etf
    Broad indices, selected ETFs, and optional regime proxies.
industry
    Industry classification. Uses Shenwan (SW) industry classification history
    from AKShare when available; this replaces the earlier CSI level-2 plan
    because SW has a direct documented endpoint.
valuation
    A-share valuation / market-cap data. Uses stock_value_em historical data
    when available and falls back to a current Eastmoney spot snapshot.
fundamentals
    Fundamental quality data. Uses Eastmoney quarterly performance reports for
    announcement-date-aware features and can also download raw per-stock
    financial indicator tables.

Usage
-----
  python download_external_data.py --dataset all --start 20250101 --end 20260510
  python download_external_data.py --dataset index_etf --start 20250101 --end 20260510
  python download_external_data.py --dataset industry
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from tqdm import tqdm

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover - exercised in user env
    ak = None
    AKSHARE_IMPORT_ERROR = exc
else:
    AKSHARE_IMPORT_ERROR = None


DATA_DIR = Path(__file__).parent / "data"
EXTERNAL_DIR = DATA_DIR / "external"


BROAD_INDEX_SYMBOLS = [
    {"symbol": "000300", "name": "CSI300", "market": "CN", "category": "broad_index"},
    {"symbol": "000905", "name": "CSI500", "market": "CN", "category": "broad_index"},
    {"symbol": "000852", "name": "CSI1000", "market": "CN", "category": "broad_index"},
    {"symbol": "930939", "name": "CSI500 Quality Growth", "market": "CN", "category": "style_index"},
    {"symbol": "399006", "name": "ChiNext", "market": "CN", "category": "growth_index"},
    {"symbol": "000688", "name": "STAR50", "market": "CN", "category": "growth_index"},
]


# ETF choices are proxies and should be reviewed before final use.
ETF_SYMBOLS = [
    {"symbol": "510300", "name": "CSI300 ETF", "market": "CN", "category": "broad_etf"},
    {"symbol": "510500", "name": "CSI500 ETF", "market": "CN", "category": "broad_etf"},
    {"symbol": "512100", "name": "CSI1000 ETF", "market": "CN", "category": "broad_etf"},
    {"symbol": "159915", "name": "ChiNext ETF", "market": "CN", "category": "growth_etf"},
    {"symbol": "588000", "name": "STAR50 ETF", "market": "CN", "category": "growth_etf"},
    {"symbol": "510880", "name": "Dividend ETF", "market": "CN", "category": "defensive_etf"},
    {"symbol": "512000", "name": "Securities ETF", "market": "CN", "category": "risk_appetite_etf"},
    {"symbol": "518880", "name": "Gold ETF", "market": "CN", "category": "safe_haven_etf"},
    {"symbol": "512800", "name": "Bank ETF", "market": "CN", "category": "sector_etf"},
    {"symbol": "512400", "name": "Nonferrous Metals ETF", "market": "CN", "category": "raw_material_etf"},
    {"symbol": "516020", "name": "Chemical ETF", "market": "CN", "category": "raw_material_etf"},
    {"symbol": "515220", "name": "Coal ETF", "market": "CN", "category": "raw_material_etf"},
    {"symbol": "159697", "name": "Oil & Gas ETF", "market": "CN", "category": "oil_gas_etf"},
    {"symbol": "513350", "name": "S&P Oil & Gas ETF QDII", "market": "US_PROXY_CN", "category": "oil_gas_etf"},
    {"symbol": "512480", "name": "Semiconductor ETF", "market": "CN", "category": "sector_etf"},
    {"symbol": "512660", "name": "Military ETF", "market": "CN", "category": "sector_etf"},
    {"symbol": "512010", "name": "Pharma ETF", "market": "CN", "category": "sector_etf"},
    {"symbol": "159920", "name": "Hang Seng ETF", "market": "HK_PROXY_CN", "category": "overseas_etf"},
    {"symbol": "513180", "name": "Hang Seng TECH ETF", "market": "HK_PROXY_CN", "category": "overseas_etf"},
    {"symbol": "513500", "name": "S&P 500 ETF", "market": "US_PROXY_CN", "category": "overseas_etf"},
    {"symbol": "513100", "name": "NASDAQ 100 ETF", "market": "US_PROXY_CN", "category": "overseas_etf"},
]


# These are regime proxies to track in the design. Endpoint availability varies
# by AKShare version, so they are listed in metadata even if not downloaded.
GLOBAL_REGIME_PROXIES = [
    {"symbol": "HSI", "name": "Hang Seng Index", "market": "HK", "category": "hk_index"},
    {"symbol": "HSTECH", "name": "Hang Seng TECH", "market": "HK", "category": "hk_index"},
    {"symbol": "SPX", "name": "S&P 500", "market": "US", "category": "us_index"},
    {"symbol": "NDX", "name": "NASDAQ 100", "market": "US", "category": "us_index"},
    {"symbol": "VHSI", "name": "Hang Seng Volatility Index", "market": "HK", "category": "volatility_proxy"},
    {"symbol": "GC", "name": "Gold futures", "market": "GLOBAL", "category": "safe_haven"},
    {"symbol": "CL", "name": "Crude oil futures", "market": "GLOBAL", "category": "commodity"},
]


DIRECT_GLOBAL_INDEX_SYMBOLS = [
    {"symbols": ["HSI"], "name": "Hang Seng Index", "market": "HK", "category": "hk_index", "function": "stock_hk_index_daily_sina"},
    {"symbols": ["HSTECH"], "name": "Hang Seng TECH", "market": "HK", "category": "hk_index", "function": "stock_hk_index_daily_sina"},
    {"symbols": [".INX"], "name": "S&P 500", "market": "US", "category": "us_index", "function": "index_us_stock_sina"},
    {"symbols": [".NDX"], "name": "NASDAQ 100", "market": "US", "category": "us_index", "function": "index_us_stock_sina"},
    {"symbols": ["VHSI"], "name": "Hang Seng Volatility Index", "market": "HK", "category": "volatility_proxy", "function": "stock_hk_index_daily_sina"},
]


FUTURES_REGIME_SYMBOLS = [
    {"symbol": "SC0", "name": "Shanghai crude oil continuous", "market": "CN", "category": "crude_oil_futures"},
    {"symbol": "CL", "name": "NYMEX WTI crude oil", "market": "GLOBAL", "category": "crude_oil_futures"},
    {"symbol": "OIL", "name": "Brent crude oil CFD", "market": "GLOBAL", "category": "crude_oil_futures"},
    {"symbol": "GC", "name": "Gold futures", "market": "GLOBAL", "category": "safe_haven_futures"},
]


def require_akshare() -> None:
    if ak is None:
        raise RuntimeError(
            "akshare is not importable in this Python environment. "
            "Install it with `python -m pip install akshare` or activate the "
            "environment where it is installed."
        ) from AKSHARE_IMPORT_ERROR


def ensure_dirs() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def normalize_code(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text and text.replace(".", "", 1).isdigit():
        text = text.split(".")[0]
    return text.zfill(6)


def to_prefixed_a_symbol(code: str) -> str:
    code = normalize_code(code)
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{code}"


def to_eastmoney_symbol(code: str) -> str:
    code = normalize_code(code)
    suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{suffix}"


def quarter_periods(start: str, end: str) -> list[str]:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    periods = []
    for year in range(start_dt.year - 1, end_dt.year + 1):
        for suffix in ["0331", "0630", "0930", "1231"]:
            period = pd.to_datetime(f"{year}{suffix}")
            if start_dt - pd.DateOffset(years=1) <= period <= end_dt:
                periods.append(f"{year}{suffix}")
    return periods


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def standardize_ohlcv(df: pd.DataFrame, symbol: str, name: str, market: str, category: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_candidates = {
        "date": ["date", "日期"],
        "open": ["open", "开盘"],
        "high": ["high", "最高"],
        "low": ["low", "最低"],
        "close": ["close", "收盘"],
        "volume": ["volume", "成交量"],
        "amount": ["amount", "成交额"],
        "pct_change": ["pct_change", "涨跌幅"],
        "turnover": ["turnover", "换手率"],
    }
    out = pd.DataFrame()
    for target, candidates in rename_candidates.items():
        source = first_existing(df.columns, candidates)
        if source is not None:
            out[target] = df[source]

    if "date" not in out or "close" not in out:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["symbol"] = symbol
    out["name"] = name
    out["market"] = market
    out["category"] = category
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def standardize_price_series(df: pd.DataFrame, symbol: str, name: str, market: str, category: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_candidates = {
        "date": ["date", "日期", "交易日期"],
        "open": ["open", "开盘价", "开盘"],
        "high": ["high", "最高价", "最高"],
        "low": ["low", "最低价", "最低"],
        "close": ["close", "收盘价", "收盘", "最新价"],
        "volume": ["volume", "成交量"],
        "amount": ["amount", "成交额"],
    }
    out = pd.DataFrame()
    for target, candidates in rename_candidates.items():
        source = first_existing(df.columns, candidates)
        if source is not None:
            out[target] = df[source]

    if "date" not in out or "close" not in out:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out["symbol"] = symbol
    out["name"] = name
    out["market"] = market
    out["category"] = category
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def try_fetch(name: str, func: Callable, *args, warn: bool = True, **kwargs) -> pd.DataFrame:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        if warn:
            print(f"  [warn] {name} failed: {exc}")
        return pd.DataFrame()


def fetch_cn_index(symbol_info: dict, start: str, end: str) -> pd.DataFrame:
    symbol = symbol_info["symbol"]
    frames = []

    if symbol.startswith("9") and hasattr(ak, "stock_zh_index_hist_csindex"):
        df = try_fetch(
            f"stock_zh_index_hist_csindex({symbol})",
            ak.stock_zh_index_hist_csindex,
            symbol=symbol,
            start_date=start,
            end_date=end,
        )
        std = standardize_ohlcv(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            return std

    prefixed = f"sh{symbol}" if symbol.startswith("0") else f"sz{symbol}"
    if hasattr(ak, "stock_zh_index_daily"):
        df = try_fetch(
            f"stock_zh_index_daily({prefixed})",
            ak.stock_zh_index_daily,
            symbol=prefixed,
        )
        if not df.empty and "date" in df:
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
        frames.append(df)
        std = standardize_ohlcv(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            return std

    if hasattr(ak, "index_zh_a_hist"):
        df = try_fetch(
            f"index_zh_a_hist({symbol})",
            ak.index_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
        )
        frames.append(df)

    for df in frames:
        std = standardize_ohlcv(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            return std
    return pd.DataFrame()


def normalize_quality_growth_constituents(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rename = {
        "日期": "effective_date",
        "指数代码": "index_code",
        "指数名称": "index_name",
        "成分券代码": "stock_code",
        "成分券名称": "stock_name",
        "权重": "weight",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    required = {"effective_date", "stock_code"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    out["effective_date"] = pd.to_datetime(out["effective_date"], errors="coerce")
    out["stock_code"] = out["stock_code"].map(normalize_code)
    if "weight" in out:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    else:
        out["weight"] = pd.NA
    out["is_csi500_quality_growth_member"] = 1.0
    out["source"] = "index_stock_cons_weight_csindex"
    keep = [
        "effective_date", "index_code", "index_name", "stock_code", "stock_name",
        "weight", "is_csi500_quality_growth_member", "source",
    ]
    return out[[c for c in keep if c in out.columns]].dropna(subset=["effective_date", "stock_code"])


def download_quality_growth(sleep: float) -> None:
    require_akshare()
    ensure_dirs()
    frames = []
    if hasattr(ak, "index_stock_cons_weight_csindex"):
        df = try_fetch(
            "index_stock_cons_weight_csindex(930939)",
            ak.index_stock_cons_weight_csindex,
            symbol="930939",
        )
        normalized = normalize_quality_growth_constituents(df)
        if not normalized.empty:
            frames.append(normalized)
        time.sleep(sleep)

    if not frames and hasattr(ak, "index_stock_cons_csindex"):
        df = try_fetch(
            "index_stock_cons_csindex(930939)",
            ak.index_stock_cons_csindex,
            symbol="930939",
        )
        normalized = normalize_quality_growth_constituents(df)
        if not normalized.empty:
            normalized["source"] = "index_stock_cons_csindex"
            frames.append(normalized)

    if not frames:
        print(">> No CSI500 quality-growth constituents downloaded.")
        return

    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["effective_date", "stock_code"],
        keep="last",
    )
    out = out.sort_values(["effective_date", "stock_code"])
    out.to_parquet(EXTERNAL_DIR / "quality_growth_constituents.parquet", index=False)
    out.to_csv(EXTERNAL_DIR / "quality_growth_constituents.csv", index=False)
    write_metadata(
        EXTERNAL_DIR / "quality_growth_metadata.json",
        {
            "index_code": "930939",
            "index_name": "CSI500 Quality Growth",
            "source": sorted(out["source"].dropna().unique().tolist()),
            "note": "Current CSI index constituent/weight snapshot. build_feature_matrix.py applies it only from the returned effective date onward.",
        },
    )
    print(f">> Wrote {len(out):,} rows to {EXTERNAL_DIR / 'quality_growth_constituents.parquet'}")


def fetch_cn_etf(symbol_info: dict, start: str, end: str) -> pd.DataFrame:
    symbol = symbol_info["symbol"]
    frames = []
    if hasattr(ak, "fund_etf_hist_sina"):
        prefixed = f"sh{symbol}" if symbol.startswith("5") else f"sz{symbol}"
        df = try_fetch(f"fund_etf_hist_sina({prefixed})", ak.fund_etf_hist_sina, symbol=prefixed)
        if not df.empty and "date" in df:
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
        frames.append(df)
        std = standardize_ohlcv(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            return std

    if hasattr(ak, "fund_etf_hist_em"):
        df = try_fetch(
            f"fund_etf_hist_em({symbol})",
            ak.fund_etf_hist_em,
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        frames.append(df)

    for df in frames:
        std = standardize_ohlcv(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            return std
    return pd.DataFrame()


def fetch_futures_regime(symbol_info: dict, start: str, end: str) -> pd.DataFrame:
    symbol = symbol_info["symbol"]
    frames = []

    if symbol.startswith("SC") and hasattr(ak, "futures_zh_daily_sina"):
        df = try_fetch(
            f"futures_zh_daily_sina({symbol})",
            ak.futures_zh_daily_sina,
            symbol=symbol,
        )
        frames.append(df)

    if not symbol.startswith("SC") and hasattr(ak, "futures_foreign_hist"):
        df = try_fetch(
            f"futures_foreign_hist({symbol})",
            ak.futures_foreign_hist,
            symbol=symbol,
        )
        frames.append(df)

    for df in frames:
        std = standardize_price_series(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            std = std[(std["date"] >= pd.to_datetime(start)) & (std["date"] <= pd.to_datetime(end))]
            if not std.empty:
                return std
    return pd.DataFrame()


def fetch_direct_global_index(symbol_info: dict, start: str, end: str) -> pd.DataFrame:
    fn_name = symbol_info["function"]
    if not hasattr(ak, fn_name):
        return pd.DataFrame()

    fn = getattr(ak, fn_name)
    for symbol in symbol_info["symbols"]:
        df = try_fetch(f"{fn_name}({symbol})", fn, symbol=symbol)
        std = standardize_price_series(
            df,
            symbol=symbol,
            name=symbol_info["name"],
            market=symbol_info["market"],
            category=symbol_info["category"],
        )
        if not std.empty:
            std = std[(std["date"] >= pd.to_datetime(start)) & (std["date"] <= pd.to_datetime(end))]
            if not std.empty:
                return std
    return pd.DataFrame()


def download_index_etf(start: str, end: str, sleep: float) -> None:
    require_akshare()
    ensure_dirs()

    frames = []
    for info in tqdm(BROAD_INDEX_SYMBOLS, desc="indices"):
        df = fetch_cn_index(info, start, end)
        if not df.empty:
            frames.append(df)
        time.sleep(sleep)

    for info in tqdm(ETF_SYMBOLS, desc="etfs"):
        df = fetch_cn_etf(info, start, end)
        if not df.empty:
            frames.append(df)
        time.sleep(sleep)

    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        out = out.sort_values(["symbol", "date"])
        out.to_parquet(EXTERNAL_DIR / "index_etf_daily.parquet", index=False)
        print(f">> Wrote {len(out):,} rows to {EXTERNAL_DIR / 'index_etf_daily.parquet'}")
    else:
        print(">> No index/ETF data downloaded.")

    proxy_meta = pd.DataFrame(GLOBAL_REGIME_PROXIES)
    proxy_meta.to_csv(EXTERNAL_DIR / "global_regime_proxy_watchlist.csv", index=False)

    global_index_frames = []
    for info in tqdm(DIRECT_GLOBAL_INDEX_SYMBOLS, desc="global indices"):
        df = fetch_direct_global_index(info, start, end)
        if not df.empty:
            global_index_frames.append(df)
        time.sleep(sleep)

    if global_index_frames:
        global_out = pd.concat(global_index_frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        global_out = global_out.sort_values(["symbol", "date"])
        global_out.to_parquet(EXTERNAL_DIR / "global_index_daily.parquet", index=False)
        print(f">> Wrote {len(global_out):,} rows to {EXTERNAL_DIR / 'global_index_daily.parquet'}")
    else:
        print(">> No direct global index data downloaded.")

    futures_frames = []
    for info in tqdm(FUTURES_REGIME_SYMBOLS, desc="futures proxies"):
        df = fetch_futures_regime(info, start, end)
        if not df.empty:
            futures_frames.append(df)
        time.sleep(sleep)

    if futures_frames:
        futures_out = pd.concat(futures_frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        futures_out = futures_out.sort_values(["symbol", "date"])
        futures_out.to_parquet(EXTERNAL_DIR / "futures_regime_daily.parquet", index=False)
        print(f">> Wrote {len(futures_out):,} rows to {EXTERNAL_DIR / 'futures_regime_daily.parquet'}")
    else:
        print(">> No futures regime proxy data downloaded.")

    write_metadata(
        EXTERNAL_DIR / "index_etf_metadata.json",
        {
            "start": start,
            "end": end,
            "domestic_indices": BROAD_INDEX_SYMBOLS,
            "domestic_etfs": ETF_SYMBOLS,
            "direct_global_indices": DIRECT_GLOBAL_INDEX_SYMBOLS,
            "futures_regime_proxies": FUTURES_REGIME_SYMBOLS,
            "global_proxy_watchlist": GLOBAL_REGIME_PROXIES,
            "note": "Direct global indices and selected futures proxies are attempted when AKShare endpoints are available. ETF proxies remain the more stable fallback.",
        },
    )


def load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df


def download_industry(constituents_path: Path, sleep: float) -> None:
    require_akshare()
    ensure_dirs()
    universe = set(load_universe(constituents_path)["stock_code"])

    if hasattr(ak, "stock_industry_clf_hist_sw"):
        sw = try_fetch("stock_industry_clf_hist_sw", ak.stock_industry_clf_hist_sw)
        if not sw.empty:
            sw = sw.copy()
            sw["stock_code"] = sw["symbol"].map(normalize_code)
            sw = sw[sw["stock_code"].isin(universe)].copy()
            sw["start_date"] = pd.to_datetime(sw["start_date"], errors="coerce")
            sw["update_time"] = pd.to_datetime(sw["update_time"], errors="coerce")
            sw["industry_code_sw"] = sw["industry_code"].astype(str).str.zfill(6)
            sw["sw_level1_code"] = sw["industry_code_sw"].str[:2]
            sw["sw_level2_code"] = sw["industry_code_sw"].str[:4]
            sw["sw_level3_code"] = sw["industry_code_sw"].str[:6]
            sw["source"] = "stock_industry_clf_hist_sw"
            sw["classification_system"] = "Shenwan"

            history_cols = [
                "stock_code", "start_date", "industry_code_sw",
                "sw_level1_code", "sw_level2_code", "sw_level3_code",
                "update_time", "source", "classification_system",
            ]
            sw_history = sw[history_cols].sort_values(["stock_code", "start_date"])
            sw_history.to_parquet(EXTERNAL_DIR / "industry_classification_sw_history.parquet", index=False)

            latest = (
                sw_history.dropna(subset=["start_date"])
                .sort_values(["stock_code", "start_date", "update_time"])
                .groupby("stock_code", as_index=False)
                .tail(1)
                .copy()
            )
            latest["industry_code"] = latest["sw_level2_code"]
            latest["industry_name"] = latest["sw_level2_code"]
            latest["industry_level"] = "sw_level2_code"
            latest["as_of_date"] = pd.Timestamp.today().date().isoformat()
            latest.to_csv(EXTERNAL_DIR / "industry_classification.csv", index=False)
            print(f">> Wrote {len(sw_history):,} SW industry history rows to {EXTERNAL_DIR / 'industry_classification_sw_history.parquet'}")
            print(f">> Wrote {len(latest):,} current SW industry rows to {EXTERNAL_DIR / 'industry_classification.csv'}")
            print(f"   covered {latest['stock_code'].nunique()}/{len(universe)} universe stocks")
            write_metadata(
                EXTERNAL_DIR / "industry_metadata.json",
                {
                    "preferred": "Shenwan industry classification",
                    "actual_source": "stock_industry_clf_hist_sw",
                    "note": "SW endpoint provides industry codes. Level-2 code is derived from the first four digits; names may require a separate code dictionary if needed.",
                },
            )
            return

    # Fallback if SW endpoint is unavailable in the local AKShare version.
    rows = []
    source = "eastmoney_industry_fallback"

    if hasattr(ak, "stock_board_industry_name_em") and hasattr(ak, "stock_board_industry_cons_em"):
        names = try_fetch("stock_board_industry_name_em", ak.stock_board_industry_name_em)
        name_col = first_existing(names.columns, ["板块名称", "行业名称", "name"])
        code_col = first_existing(names.columns, ["板块代码", "行业代码", "code"])
        if name_col is not None:
            for _, row in tqdm(names.iterrows(), total=len(names), desc="industry boards"):
                industry_name = str(row[name_col])
                industry_code = str(row[code_col]) if code_col is not None else industry_name
                cons = try_fetch(
                    f"stock_board_industry_cons_em({industry_name})",
                    ak.stock_board_industry_cons_em,
                    symbol=industry_name,
                )
                stock_col = first_existing(cons.columns, ["代码", "股票代码", "stock_code"])
                stock_name_col = first_existing(cons.columns, ["名称", "股票名称", "stock_name"])
                if stock_col is None:
                    time.sleep(sleep)
                    continue
                for _, cons_row in cons.iterrows():
                    stock_code = normalize_code(cons_row[stock_col])
                    if stock_code in universe:
                        rows.append({
                            "stock_code": stock_code,
                            "stock_name": cons_row[stock_name_col] if stock_name_col else "",
                            "industry_code": industry_code,
                            "industry_name": industry_name,
                            "industry_level": "fallback_board",
                            "source": source,
                            "as_of_date": pd.Timestamp.today().date().isoformat(),
                            "preferred_source": "Shenwan industry classification",
                        })
                time.sleep(sleep)

    out = pd.DataFrame(rows).drop_duplicates(subset=["stock_code", "industry_name"])
    out_path = EXTERNAL_DIR / "industry_classification.csv"
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out):,} rows to {out_path}")
    if out.empty:
        print("  [warn] Industry fallback produced no rows. Consider supplying a Shenwan industry mapping manually.")
    else:
        covered = out["stock_code"].nunique()
        print(f"   covered {covered}/{len(universe)} universe stocks")

    write_metadata(
        EXTERNAL_DIR / "industry_metadata.json",
        {
            "preferred": "Shenwan industry classification",
            "actual_source": source if not out.empty else "none",
            "note": "SW endpoint was unavailable or failed. Fallback uses Eastmoney industry boards.",
        },
    )


def normalize_valuation_table(df: pd.DataFrame, stock_code: str, start: str, end: str) -> pd.DataFrame:
    date_col = first_existing(df.columns, ["数据日期", "日期", "date"])
    if date_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["stock_code"] = stock_code

    fuzzy_targets = {
        "close_unadjusted": ["当日收盘价", "收盘价", "close"],
        "pe_ttm": ["市盈率TTM", "市盈率(TTM)", "PE(TTM)", "pe_ttm"],
        "pe_static": ["市盈率", "静态市盈率", "PE(静)", "pe_static"],
        "pb": ["市净率", "PB", "pb"],
        "ps_ttm": ["市销率TTM", "市销率(TTM)", "PS(TTM)", "ps_ttm"],
        "pcf_ttm": ["市现率TTM", "市现率(TTM)", "PCF(TTM)"],
        "total_market_cap": ["总市值", "TOTAL_MARKET_CAP"],
        "float_market_cap": ["流通市值", "FLOAT_MARKET_CAP"],
    }
    for target, candidates in fuzzy_targets.items():
        source = first_existing(df.columns, candidates)
        if source is None:
            for col in df.columns:
                if any(c in str(col) for c in candidates):
                    source = col
                    break
        if source is not None:
            out[target] = pd.to_numeric(df[source], errors="coerce")

    out = out[(out["date"] >= pd.to_datetime(start)) & (out["date"] <= pd.to_datetime(end))]
    out["source"] = "stock_value_em"
    return out.sort_values(["stock_code", "date"]).reset_index(drop=True)


def write_valuation_snapshot(raw: pd.DataFrame, universe: set[str]) -> None:
    if raw.empty:
        print(">> No valuation snapshot downloaded.")
        return

    raw.to_csv(EXTERNAL_DIR / "valuation_snapshot_raw.csv", index=False)

    col_map = {
        "stock_code": ["代码", "stock_code"],
        "stock_name": ["名称", "stock_name"],
        "pe_dynamic": ["市盈率-动态", "市盈率动态", "动态市盈率"],
        "pb": ["市净率"],
        "total_market_cap": ["总市值"],
        "float_market_cap": ["流通市值"],
    }
    out = pd.DataFrame()
    for target, candidates in col_map.items():
        source = first_existing(raw.columns, candidates)
        if source is not None:
            out[target] = raw[source]
    if "stock_code" in out:
        out["stock_code"] = out["stock_code"].map(normalize_code)
        out = out[out["stock_code"].isin(universe)]
    for col in ["pe_dynamic", "pb", "total_market_cap", "float_market_cap"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["as_of_date"] = pd.Timestamp.today().date().isoformat()
    out["source"] = "stock_zh_a_spot_em"
    out.to_parquet(EXTERNAL_DIR / "valuation_snapshot.parquet", index=False)
    print(f">> Wrote {len(out):,} rows to {EXTERNAL_DIR / 'valuation_snapshot.parquet'}")
    print("   [note] This is a current snapshot fallback, not a historical daily valuation panel.")


def download_valuation(constituents_path: Path, start: str, end: str, sleep: float) -> None:
    require_akshare()
    ensure_dirs()
    universe = set(load_universe(constituents_path)["stock_code"])

    raw_frames = []
    frames = []
    if hasattr(ak, "stock_value_em"):
        for code in tqdm(sorted(universe), desc="valuation daily"):
            raw = try_fetch(f"stock_value_em({code})", ak.stock_value_em, symbol=code)
            if not raw.empty:
                raw = raw.copy()
                raw["stock_code"] = code
                raw_frames.append(raw)
                std = normalize_valuation_table(raw, code, start, end)
                if not std.empty:
                    frames.append(std)
            time.sleep(sleep)

    if frames:
        raw_out = pd.concat(raw_frames, ignore_index=True)
        raw_out.to_parquet(EXTERNAL_DIR / "valuation_daily_raw.parquet", index=False)
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["stock_code", "date"])
        out = out.sort_values(["stock_code", "date"])
        out.to_parquet(EXTERNAL_DIR / "valuation_daily.parquet", index=False)
        print(f">> Wrote {len(out):,} rows to {EXTERNAL_DIR / 'valuation_daily.parquet'}")
        write_metadata(
            EXTERNAL_DIR / "valuation_metadata.json",
            {
                "start": start,
                "end": end,
                "actual_source": "stock_value_em",
                "fallback": "stock_zh_a_spot_em current snapshot if historical endpoint fails",
            },
        )
        return

    if hasattr(ak, "stock_zh_a_spot_em"):
        raw = try_fetch("stock_zh_a_spot_em", ak.stock_zh_a_spot_em)
        write_valuation_snapshot(raw, universe)
        write_metadata(
            EXTERNAL_DIR / "valuation_metadata.json",
            {
                "actual_source": "stock_zh_a_spot_em",
                "note": "Historical stock_value_em data was unavailable or empty in this AKShare environment.",
            },
        )
    else:
        print(">> stock_value_em and stock_zh_a_spot_em are not available in this AKShare version.")


def normalize_yjbb_table(df: pd.DataFrame, period: str, universe: set[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    code_col = first_existing(df.columns, ["股票代码", "代码", "SECURITY_CODE", "stock_code"])
    if code_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["stock_code"] = df[code_col].map(normalize_code)
    out = out[out["stock_code"].isin(universe)].copy()
    if out.empty:
        return pd.DataFrame()

    out["report_period"] = pd.to_datetime(period)
    out["source"] = "stock_yjbb_em"
    out["availability_rule"] = "announcement_date"

    col_map = {
        "stock_name": ["股票简称", "名称", "SECURITY_NAME_ABBR"],
        "eps": ["每股收益", "基本每股收益", "EPSJB"],
        "operating_revenue": ["营业收入", "TOTALOPERATEREVE"],
        "operating_revenue_yoy": ["营业收入-同比增长", "营业收入同比增长", "TOTALOPERATEREVETZ"],
        "net_profit": ["净利润", "归属净利润", "PARENTNETPROFIT"],
        "net_profit_yoy": ["净利润-同比增长", "归属净利润同比增长", "PARENTNETPROFITTZ"],
        "roe": ["净资产收益率", "净资产收益率-加权", "ROEJQ"],
        "bps": ["每股净资产", "BPS"],
        "gross_margin": ["销售毛利率", "毛利率", "XSMLL"],
        "cash_flow_per_share": ["每股经营现金流量", "MGJYXJJE"],
        "announcement_date": ["最新公告日期", "公告日期", "NOTICE_DATE"],
    }
    for target, candidates in col_map.items():
        source = first_existing(df.columns, candidates)
        if source is None:
            for col in df.columns:
                if any(c in str(col) for c in candidates):
                    source = col
                    break
        if source is not None:
            out[target] = df.loc[out.index, source]

    if "announcement_date" in out:
        out["announcement_date"] = pd.to_datetime(out["announcement_date"], errors="coerce")
    else:
        out["announcement_date"] = out["report_period"] + pd.Timedelta(days=120)
        out["availability_rule"] = "report_period_plus_120d_if_announcement_missing"

    for col in [
        "eps", "operating_revenue", "operating_revenue_yoy", "net_profit",
        "net_profit_yoy", "roe", "bps", "gross_margin", "cash_flow_per_share",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.reset_index(drop=True)


def normalize_em_indicator_table(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["stock_code"] = stock_code

    date_col = first_existing(out.columns, ["日期", "报告期", "报告日期", "REPORT_DATE", "report_period"])
    if date_col is not None:
        out["report_period"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out["report_period"] = pd.NaT

    notice_col = first_existing(out.columns, ["最新公告日期", "公告日期", "NOTICE_DATE", "公告时间"])
    if notice_col is not None:
        out["announcement_date"] = pd.to_datetime(out[notice_col], errors="coerce")
        out["availability_rule"] = "announcement_date"
    else:
        # Some financial indicator endpoints lack announcement dates. Use a
        # conservative lag only if no announcement-aware table is available.
        out["announcement_date"] = out["report_period"] + pd.Timedelta(days=120)
        out["availability_rule"] = "report_period_plus_120d_if_announcement_missing"

    fuzzy_targets = {
        "roe": ["净资产收益率", "ROE", "ROEJQ"],
        "roa": ["总资产报酬率", "总资产净利率", "ROA", "ZZCJLL"],
        "gross_margin": ["销售毛利率", "毛利率", "XSMLL"],
        "net_margin": ["销售净利率", "净利率", "XSJLL"],
        "debt_to_asset": ["资产负债率", "ZCFZL"],
        "operating_revenue": ["营业收入", "TOTALOPERATEREVE"],
        "operating_revenue_yoy": ["主营业务收入增长率", "营业收入同比增长率", "营业收入增长率", "TOTALOPERATEREVETZ"],
        "net_profit": ["净利润", "PARENTNETPROFIT"],
        "net_profit_growth_yoy": ["净利润增长率", "归属净利润同比增长率", "PARENTNETPROFITTZ"],
        "deducted_net_profit": ["扣非净利润", "扣除非经常性损益后的净利润", "KCFJCXSYJLR"],
        "deducted_net_profit_growth_yoy": ["扣非净利润增长率", "扣除非经常性损益后的净利润增长率", "KCFJCXSYJLRTZ"],
        "cash_flow_per_share": ["每股经营现金流量", "MGJYXJJE"],
        "cash_flow_to_revenue": ["经营现金流量营业收入比", "经营现金流量对营业收入比率", "JYXJLYYSR"],
        "operating_cash_flow_to_net_profit": ["经营现金流量净利润比", "经营现金流量对净利润比率", "NCO_NETPROFIT"],
        "eps": ["基本每股收益", "每股收益", "EPSJB"],
    }
    for target, patterns in fuzzy_targets.items():
        source = None
        for col in out.columns:
            if any(p in str(col) for p in patterns):
                source = col
                break
        if source is not None:
            out[target] = pd.to_numeric(out[source], errors="coerce")

    out["source"] = "stock_financial_analysis_indicator_em"
    return out


def normalize_legacy_fundamental_table(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    out = normalize_em_indicator_table(df, stock_code)
    if not out.empty:
        out["source"] = "stock_financial_analysis_indicator"
    return out


def download_fundamentals(constituents_path: Path, start: str, end: str, sleep: float) -> None:
    require_akshare()
    ensure_dirs()
    universe = load_universe(constituents_path)
    universe_set = set(universe["stock_code"])
    start_year = str(pd.to_datetime(start).year)

    yjbb_frames = []
    if hasattr(ak, "stock_yjbb_em"):
        for period in tqdm(quarter_periods(start, end), desc="earnings reports"):
            df = try_fetch(f"stock_yjbb_em({period})", ak.stock_yjbb_em, date=period)
            if not df.empty:
                normalized = normalize_yjbb_table(df, period, universe_set)
                if not normalized.empty:
                    yjbb_frames.append(normalized)
            time.sleep(sleep)

    if yjbb_frames:
        yjbb = pd.concat(yjbb_frames, ignore_index=True)
        yjbb = yjbb.drop_duplicates(subset=["stock_code", "report_period", "announcement_date"])
        yjbb = yjbb.sort_values(["stock_code", "report_period", "announcement_date"])
        yjbb.to_parquet(EXTERNAL_DIR / "fundamentals_yjbb_em.parquet", index=False)
        yjbb.to_parquet(EXTERNAL_DIR / "fundamentals.parquet", index=False)
        print(f">> Wrote {len(yjbb):,} announcement-aware rows to {EXTERNAL_DIR / 'fundamentals.parquet'}")

    indicator_frames = []
    if hasattr(ak, "stock_financial_analysis_indicator_em"):
        for code in tqdm(universe["stock_code"], desc="financial indicators em"):
            symbol = to_eastmoney_symbol(code)
            df = try_fetch(
                f"stock_financial_analysis_indicator_em({symbol})",
                ak.stock_financial_analysis_indicator_em,
                symbol=symbol,
                indicator="按报告期",
            )
            if not df.empty:
                normalized = normalize_em_indicator_table(df, code)
                if not normalized.empty:
                    indicator_frames.append(normalized)
            time.sleep(sleep)
    elif hasattr(ak, "stock_financial_analysis_indicator"):
        for code in tqdm(universe["stock_code"], desc="financial indicators legacy"):
            df = try_fetch(
                f"stock_financial_analysis_indicator({code})",
                ak.stock_financial_analysis_indicator,
                symbol=code,
                start_year=start_year,
            )
            if not df.empty:
                normalized = normalize_legacy_fundamental_table(df, code)
                if not normalized.empty:
                    indicator_frames.append(normalized)
            time.sleep(sleep)

    if indicator_frames:
        indicators = pd.concat(indicator_frames, ignore_index=True)
        indicators = indicators.sort_values(["stock_code", "report_period"])
        indicators.to_parquet(EXTERNAL_DIR / "fundamentals_indicators.parquet", index=False)
        print(f">> Wrote {len(indicators):,} raw indicator rows to {EXTERNAL_DIR / 'fundamentals_indicators.parquet'}")
        if not yjbb_frames:
            indicators.to_parquet(EXTERNAL_DIR / "fundamentals.parquet", index=False)
            print("   [note] fundamentals.parquet uses conservative availability dates because stock_yjbb_em was unavailable.")

    if not yjbb_frames and not indicator_frames:
        print(">> No fundamentals downloaded.")
        write_metadata(EXTERNAL_DIR / "fundamentals_metadata.json", {"actual_source": "none"})
        return

    write_metadata(
        EXTERNAL_DIR / "fundamentals_metadata.json",
        {
            "start": start,
            "end": end,
            "primary_source": "stock_yjbb_em",
            "secondary_source": "stock_financial_analysis_indicator_em or stock_financial_analysis_indicator",
            "note": "Use fundamentals.parquet first because it is announcement-date aware when stock_yjbb_em is available.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "index_etf", "industry", "valuation", "fundamentals", "quality_growth"],
        default="all",
    )
    parser.add_argument("--start", default="20250101")
    parser.add_argument("--end", default="20260510")
    parser.add_argument("--constituents", default=str(DATA_DIR / "constituents.csv"))
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.dataset in ["all", "index_etf"]:
        download_index_etf(args.start, args.end, args.sleep)
    if args.dataset in ["all", "industry"]:
        download_industry(Path(args.constituents), args.sleep)
    if args.dataset in ["all", "valuation"]:
        download_valuation(Path(args.constituents), args.start, args.end, args.sleep)
    if args.dataset in ["all", "fundamentals"]:
        download_fundamentals(Path(args.constituents), args.start, args.end, args.sleep)
    if args.dataset in ["all", "quality_growth"]:
        download_quality_growth(args.sleep)


if __name__ == "__main__":
    main()

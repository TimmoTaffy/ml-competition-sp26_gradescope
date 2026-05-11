"""Shared live-window scoring utilities.

This file is retained because active live-style validation scripts import its
data-loading and realized-return scoring helpers. It is no longer a recommended
standalone experiment entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_validation_portfolios import load_model_scores_for_dates, series_for_date
from make_submission import build_portfolio, combine_scores, sort_scores
from score_submission import _stock_return, score_window
from tune_portfolio_grid import load_json, quiet_risk_filter


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.6f}"


def markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> list[str]:
    if n is not None:
        df = df.head(n)
    if df.empty:
        return ["No rows."]
    cols = [c for c in columns if c in df.columns]
    out = df[cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(fmt)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return lines


def load_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_parquet(path)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = matrix["stock_code"].astype(str).str.zfill(6)
    return matrix


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    return prices


def load_index(path: Path) -> pd.DataFrame:
    index_df = pd.read_parquet(path)
    index_df["date"] = pd.to_datetime(index_df["date"])
    return index_df.sort_values("date")


def load_constituents(path: Path) -> pd.DataFrame:
    constituents = pd.read_csv(path, dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].astype(str).str.zfill(6)
    return constituents


def recent_windows(
    index_df: pd.DataFrame,
    window_length: int,
    num_windows: int,
    as_of: pd.Timestamp | None,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index_df["date"].dropna().unique()).sort_values()
    if as_of is not None:
        dates = dates[dates <= as_of]
    need = window_length * num_windows
    if len(dates) < need + 1:
        raise ValueError(f"Need at least {need + 1} trading days, got {len(dates)}")

    eval_days = dates[-need:]
    rows: list[dict[str, Any]] = []
    for i in range(num_windows):
        chunk = eval_days[i * window_length : (i + 1) * window_length]
        start = pd.Timestamp(chunk[0])
        end = pd.Timestamp(chunk[-1])
        entry_candidates = dates[dates < start]
        if len(entry_candidates) == 0:
            raise ValueError(f"No entry date before window start {start.date()}")
        entry = pd.Timestamp(entry_candidates[-1])
        rows.append(
            {
                "window_id": f"w{i + 1}",
                "entry_date": entry,
                "start_date": start,
                "end_date": end,
                "trading_days": len(chunk),
            }
        )
    return pd.DataFrame(rows)


def score_candidate_window(
    candidate: str,
    config: dict,
    matrix: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    window: pd.Series,
    target: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    entry_date = pd.Timestamp(window["entry_date"])
    start = pd.Timestamp(window["start_date"])
    end = pd.Timestamp(window["end_date"])
    score_frames = load_model_scores_for_dates(config["models"], matrix, pd.DatetimeIndex([entry_date]), target)
    date_scores = {
        name: series_for_date(frame, entry_date, "score")
        for name, frame in score_frames.items()
    }
    scores = combine_scores(date_scores, config)
    ordered_scores = sort_scores(scores)
    eligible = quiet_risk_filter(ordered_scores, matrix, entry_date, config)
    weights = build_portfolio(eligible, config)
    result = score_window(weights, prices, index_df, start, end)

    holding_rows = []
    for code, weight in weights.items():
        stock_prices = prices[prices["stock_code"] == code]
        stock_ret, note = _stock_return(stock_prices, start, end)
        holding_rows.append(
            {
                "candidate": candidate,
                "window_id": window["window_id"],
                "entry_date": entry_date.date().isoformat(),
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "stock_code": code,
                "weight": float(weight),
                "score": float(scores.reindex([code]).iloc[0]),
                "stock_return": stock_ret,
                "return_contribution": float(weight * stock_ret),
                "note": note,
            }
        )

    row = {
        "candidate": candidate,
        "window_id": window["window_id"],
        "entry_date": entry_date.date().isoformat(),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "trading_days": int(window["trading_days"]),
        "portfolio_return": result["portfolio_return"],
        "benchmark_return": result["benchmark_return"],
        "excess_return": result["excess_return"],
        "n_holdings": int((weights > 0).sum()),
        "max_weight": float(weights.max()),
        "avg_top5_weight": float(weights.sort_values(ascending=False).head(5).sum()),
        "n_with_notes": int(result["n_with_notes"]),
        "model0_weight": float(config["models"][0]["weight"]),
        "model1_weight": float(config["models"][1]["weight"]),
        "top_k": int(config["portfolio"]["top_k"]),
        "internal_max_weight": float(config["portfolio"].get("internal_max_weight", config["portfolio"]["max_weight"])),
        "weighting": config["portfolio"].get("weighting", "rank"),
        "risk_filter_enabled": bool(config.get("risk_filter", {}).get("enabled", False)),
    }
    return row, pd.DataFrame(holding_rows)


def summarize_candidates(window_results: pd.DataFrame) -> pd.DataFrame:
    if window_results.empty:
        return window_results
    ranked = window_results.copy()
    ranked["window_rank"] = ranked.groupby("window_id")["excess_return"].rank(method="min", ascending=False)
    rows = []
    for candidate, group in ranked.groupby("candidate"):
        portfolio_compound = float(np.prod(1.0 + group["portfolio_return"]) - 1.0)
        benchmark_compound = float(np.prod(1.0 + group["benchmark_return"]) - 1.0)
        rows.append(
            {
                "candidate": candidate,
                "windows": int(group["window_id"].nunique()),
                "mean_excess": float(group["excess_return"].mean()),
                "sum_excess": float(group["excess_return"].sum()),
                "compounded_portfolio_return": portfolio_compound,
                "compounded_benchmark_return": benchmark_compound,
                "compounded_excess": portfolio_compound - benchmark_compound,
                "hit_rate": float((group["excess_return"] > 0).mean()),
                "worst_window": float(group["excess_return"].min()),
                "best_window": float(group["excess_return"].max()),
                "std_window_excess": float(group["excess_return"].std(ddof=0)),
                "avg_window_rank": float(group["window_rank"].mean()),
                "best_window_rank": int(group["window_rank"].min()),
                "worst_window_rank": int(group["window_rank"].max()),
                "avg_holdings": float(group["n_holdings"].mean()),
                "max_weight": float(group["max_weight"].max()),
                "top_k": int(group["top_k"].iloc[0]),
                "model0_weight": float(group["model0_weight"].iloc[0]),
                "model1_weight": float(group["model1_weight"].iloc[0]),
                "weighting": group["weighting"].iloc[0],
                "risk_filter_enabled": bool(group["risk_filter_enabled"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["mean_excess", "worst_window", "avg_window_rank"],
        ascending=[False, False, True],
    )


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    window_results: pd.DataFrame,
) -> None:
    lines = [
        "# Recent Window Candidate Audit",
        "",
        "## Scope",
        "",
        "- Uses explicit realized-return windows from price data.",
        "- Each window is scored from the previous trading day's close to the window end close.",
        "- Models are not retrained.",
        "- This is an audit layer, not a new training objective.",
        "",
        "## Windows",
        "",
    ]
    window_display = windows.copy()
    for col in ["entry_date", "start_date", "end_date"]:
        window_display[col] = pd.to_datetime(window_display[col]).dt.date.astype(str)
    lines.extend(markdown_table(window_display, ["window_id", "entry_date", "start_date", "end_date", "trading_days"]))

    lines.extend(["", "## Candidate Summary", ""])
    lines.extend(
        markdown_table(
            candidate_summary,
            [
                "candidate",
                "mean_excess",
                "sum_excess",
                "compounded_excess",
                "hit_rate",
                "worst_window",
                "best_window",
                "avg_window_rank",
                "top_k",
                "model0_weight",
                "model1_weight",
                "weighting",
                "risk_filter_enabled",
            ],
        )
    )

    lines.extend(["", "## Window Excess Returns", ""])
    pivot = window_results.pivot(index="window_id", columns="candidate", values="excess_return").reset_index()
    lines.extend(markdown_table(pivot, pivot.columns.tolist()))

    lines.extend(["", "## Window Ranks", ""])
    ranked = window_results.copy()
    ranked["rank"] = ranked.groupby("window_id")["excess_return"].rank(method="min", ascending=False)
    rank_pivot = ranked.pivot(index="window_id", columns="candidate", values="rank").reset_index()
    lines.extend(markdown_table(rank_pivot, rank_pivot.columns.tolist()))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prefer candidates with positive excess in several windows, not just one very strong window.",
            "- A model that is excellent in the newest window but weak elsewhere may be regime-sensitive.",
            "- Do not repeatedly tune on these windows after inspecting the result.",
        ]
    )
    (out_dir / "recent_window_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    raise SystemExit(
        "evaluate_recent_windows.py is retained as a utility module only. "
        "Use live_walk_forward_validation.py for active live-style validation."
    )


if __name__ == "__main__":
    main()

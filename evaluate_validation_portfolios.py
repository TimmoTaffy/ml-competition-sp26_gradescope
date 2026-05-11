"""Legacy fixed-block portfolio evaluation utilities.

The active model/portfolio comparison route is strict live-style walk-forward
validation. This file is retained because research scripts import shared
helpers for loading prediction frames, computing rank IC, and summarizing
portfolio rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from make_submission import build_portfolio, apply_risk_filter, combine_scores, sort_scores


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def valid_target_dates(df: pd.DataFrame, target: str, min_stocks: int) -> pd.DatetimeIndex:
    counts = df.dropna(subset=[target]).groupby("date")["stock_code"].nunique()
    return pd.DatetimeIndex(counts[counts >= min_stocks].index).sort_values()


def load_prediction_scores(model_specs: list[dict], target: str, filename: str) -> dict[str, pd.DataFrame]:
    out = {}
    for spec in model_specs:
        path = ROOT / spec["dir"] / filename
        pred = pd.read_csv(path, dtype={"stock_code": str})
        pred["date"] = pd.to_datetime(pred["date"])
        pred["stock_code"] = pred["stock_code"].str.zfill(6)
        pred = pred.rename(columns={"prediction": "score"})
        if target not in pred.columns:
            raise ValueError(f"{path} does not contain target column {target!r}")
        out[spec["name"]] = pred[["date", "stock_code", "score", target]]
    return out


def load_model_scores_for_dates(
    model_specs: list[dict],
    matrix: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target: str,
) -> dict[str, pd.DataFrame]:
    out = {}
    for spec in model_specs:
        model_dir = ROOT / spec["dir"]
        metadata = load_json(model_dir / "metadata.json")
        features = metadata["features"]
        missing = [f for f in features if f not in matrix.columns]
        if missing:
            raise ValueError(f"{spec['name']}: missing model feature(s): {missing[:10]}")
        model = XGBRegressor()
        model.load_model(model_dir / "model.json")

        parts = []
        for date in dates:
            day = matrix[matrix["date"] == date].copy()
            if day.empty:
                continue
            x = day[features].replace([np.inf, -np.inf], np.nan)
            day["score"] = model.predict(x)
            parts.append(day[["date", "stock_code", "score", target]])
        out[spec["name"]] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def common_dates(score_frames: dict[str, pd.DataFrame], target: str, min_stocks: int) -> pd.DatetimeIndex:
    common: set[pd.Timestamp] | None = None
    for frame in score_frames.values():
        dates = valid_target_dates(frame, target, min_stocks)
        date_set = set(pd.Timestamp(d) for d in dates)
        common = date_set if common is None else common.intersection(date_set)
    return pd.DatetimeIndex(sorted(common or []))


def series_for_date(frame: pd.DataFrame, date: pd.Timestamp, value_col: str) -> pd.Series:
    day = frame[frame["date"] == date]
    return pd.Series(day[value_col].to_numpy(), index=day["stock_code"].astype(str).str.zfill(6), name=value_col)


def rank_ic(scores: pd.Series, target: pd.Series) -> float:
    aligned = pd.concat([scores.rename("score"), target.rename("target")], axis=1).dropna()
    if len(aligned) < 30:
        return float("nan")
    if aligned["score"].nunique() < 2 or aligned["target"].nunique() < 2:
        return float("nan")
    return float(aligned["score"].rank().corr(aligned["target"].rank()))


def evaluate_one_variant(
    variant: str,
    scores: pd.Series,
    target: pd.Series,
    matrix: pd.DataFrame,
    date: pd.Timestamp,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    ordered_scores = sort_scores(scores)
    eligible_scores = apply_risk_filter(ordered_scores, matrix, date, config)
    weights = build_portfolio(eligible_scores, config)
    selected_target = target.reindex(weights.index)
    portfolio_excess = float((weights * selected_target).sum())
    top_k = int(config["portfolio"]["top_k"])
    top_target = target.reindex(ordered_scores.head(top_k).index)
    bottom_target = target.reindex(sort_scores(-scores).head(top_k).index)
    row = {
        "variant": variant,
        "date": date.date().isoformat(),
        "n_scores": int(scores.notna().sum()),
        "n_after_filter": int(eligible_scores.notna().sum()),
        "n_holdings": int((weights > 0).sum()),
        "max_weight": float(weights.max()),
        "portfolio_excess": portfolio_excess,
        "top_k_excess": float(top_target.mean()),
        "top_minus_bottom": float(top_target.mean() - bottom_target.mean()),
        "rank_ic": rank_ic(scores, target),
    }
    holdings = pd.DataFrame(
        {
            "variant": variant,
            "date": date.date().isoformat(),
            "stock_code": weights.index,
            "weight": weights.values,
            "score": scores.reindex(weights.index).values,
            "target_excess": selected_target.values,
        }
    )
    return row, holdings


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    grouped = rows.groupby("variant", sort=False)
    return grouped.agg(
        days=("date", "nunique"),
        mean_portfolio_excess=("portfolio_excess", "mean"),
        cumulative_portfolio_excess=("portfolio_excess", "sum"),
        hit_rate=("portfolio_excess", lambda s: float((s > 0).mean())),
        worst_day=("portfolio_excess", "min"),
        mean_top_k_excess=("top_k_excess", "mean"),
        mean_top_minus_bottom=("top_minus_bottom", "mean"),
        mean_rank_ic=("rank_ic", "mean"),
        avg_holdings=("n_holdings", "mean"),
        max_weight=("max_weight", "max"),
    ).reset_index()


def write_report(path: Path, split: str, dates: pd.DatetimeIndex, summary: pd.DataFrame) -> None:
    lines = [f"# Portfolio Evaluation: {split}", ""]
    if len(dates):
        lines.append(f"- Date range: `{dates[0].date()}` to `{dates[-1].date()}`")
    lines.append(f"- Trading dates: `{len(dates)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if summary.empty:
        lines.append("No valid rows.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- `{row['variant']}`: mean excess `{row['mean_portfolio_excess']:.6f}`, "
                f"hit `{row['hit_rate']:.2f}`, worst `{row['worst_day']:.6f}`, "
                f"rank IC `{row['mean_rank_ic']:.6f}`"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--config", default=str(CONFIG_DIR / "ensemble_two_model.json"))
    parser.add_argument(
        "--split",
        choices=["portfolio_validation", "model_validation", "validation", "test"],
        default="portfolio_validation",
    )
    parser.add_argument("--target", default="target_excess_5d")
    parser.add_argument("--mode", choices=["general", "short_history", "ensemble", "all"], default="all")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    matrix = pd.read_parquet(args.matrix)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = matrix["stock_code"].astype(str).str.zfill(6)
    min_stocks = int(config["portfolio"]["min_stocks"])

    if args.split in {"portfolio_validation", "validation"}:
        score_frames = load_prediction_scores(config["models"], args.target, "portfolio_validation_predictions.csv")
        dates = common_dates(score_frames, args.target, min_stocks)
        split_name = "portfolio_validation"
    elif args.split == "model_validation":
        score_frames = load_prediction_scores(config["models"], args.target, "validation_predictions.csv")
        dates = common_dates(score_frames, args.target, min_stocks)
        split_name = "model_validation"
    else:
        # Explicit test mode: use the last holdout_test_days target-available dates.
        holdout_days = 10
        for spec in config["models"]:
            metadata = load_json(ROOT / spec["dir"] / "metadata.json")
            cfg = load_json(ROOT / spec["dir"] / "config_used.json")
            holdout_days = max(holdout_days, int(cfg["split"].get("holdout_test_days", 10)))
        dates = valid_target_dates(matrix, args.target, min_stocks)[-holdout_days:]
        score_frames = load_model_scores_for_dates(config["models"], matrix, dates, args.target)
        split_name = "test"

    if len(dates) == 0:
        raise ValueError(f"No common {args.split} dates available")

    variants = [args.mode] if args.mode != "all" else ["general", "short_history", "ensemble"]
    rows = []
    holding_parts = []
    for date in dates:
        date_scores = {
            name: series_for_date(frame, date, "score")
            for name, frame in score_frames.items()
        }
        target = series_for_date(next(iter(score_frames.values())), date, args.target)
        variant_scores = {}
        if "general" in variants:
            variant_scores["general"] = date_scores["general"]
        if "short_history" in variants:
            variant_scores["short_history"] = date_scores["short_history"]
        if "ensemble" in variants:
            variant_scores["ensemble"] = combine_scores(date_scores, config)
        for variant, scores in variant_scores.items():
            row, holdings = evaluate_one_variant(variant, scores, target, matrix, date, config)
            rows.append(row)
            holding_parts.append(holdings)

    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments") / "portfolio_eval" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_date = pd.DataFrame(rows)
    holdings = pd.concat(holding_parts, ignore_index=True)
    summary = summarize(per_date)

    per_date.to_csv(out_dir / "per_date_results.csv", index=False)
    holdings.to_csv(out_dir / "portfolio_holdings.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_report(out_dir / "report.md", split_name, dates, summary)

    print(f"Evaluated {split_name} dates {dates[0].date()} to {dates[-1].date()} ({len(dates)} dates)")
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

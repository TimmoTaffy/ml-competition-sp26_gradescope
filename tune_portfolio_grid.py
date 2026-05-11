"""Legacy fixed-block portfolio-grid helper.

The active selection route is strict live-style validation plus
`live_portfolio_fine_tune_3_7_13.py`. This file is retained because research
scripts still import small utility functions from it.
"""
from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_validation_portfolios import (
    common_dates,
    load_prediction_scores,
    rank_ic,
    series_for_date,
    summarize,
)
from make_submission import build_portfolio, combine_scores, sort_scores


ROOT = Path(__file__).parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def quiet_risk_filter(scores: pd.Series, matrix: pd.DataFrame, prediction_date: pd.Timestamp, config: dict) -> pd.Series:
    cfg = config.get("risk_filter", {})
    if not cfg.get("enabled", False):
        return scores

    day = matrix[matrix["date"] == prediction_date].copy()
    day["stock_code"] = day["stock_code"].astype(str).str.zfill(6)
    keep = pd.Series(True, index=day.index)
    thresholds = {
        "recent_halt_flag": cfg.get("recent_halt_flag_max"),
        "low_liquidity_flag": cfg.get("low_liquidity_flag_max"),
        "missing_days_20d": cfg.get("missing_days_20d_max"),
        "zero_volume_days_20d": cfg.get("zero_volume_days_20d_max"),
    }
    for col, max_value in thresholds.items():
        if max_value is not None and col in day.columns:
            keep &= day[col].fillna(0) <= float(max_value)

    allowed = set(day.loc[keep, "stock_code"])
    filtered = scores[scores.index.isin(allowed)]
    min_stocks = int(config["portfolio"]["min_stocks"])
    if len(filtered) < min_stocks and cfg.get("fallback_to_unfiltered_if_below_min", True):
        return scores
    return filtered


def evaluate_scores(
    score_frames: dict[str, pd.DataFrame],
    matrix: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings = []
    for date in dates:
        date_scores = {
            name: series_for_date(frame, date, "score")
            for name, frame in score_frames.items()
        }
        target_series = series_for_date(next(iter(score_frames.values())), date, target)
        scores = combine_scores(date_scores, config)
        ordered_scores = sort_scores(scores)
        eligible = quiet_risk_filter(ordered_scores, matrix, date, config)
        weights = build_portfolio(eligible, config)
        selected_target = target_series.reindex(weights.index)
        top_k = int(config["portfolio"]["top_k"])
        top_target = target_series.reindex(ordered_scores.head(top_k).index)
        bottom_target = target_series.reindex(sort_scores(-scores).head(top_k).index)
        rows.append(
            {
                "date": date.date().isoformat(),
                "n_scores": int(scores.notna().sum()),
                "n_after_filter": int(eligible.notna().sum()),
                "n_holdings": int((weights > 0).sum()),
                "max_weight": float(weights.max()),
                "portfolio_excess": float((weights * selected_target).sum()),
                "top_k_excess": float(top_target.mean()),
                "top_minus_bottom": float(top_target.mean() - bottom_target.mean()),
                "rank_ic": rank_ic(scores, target_series),
            }
        )
        holdings.append(
            pd.DataFrame(
                {
                    "date": date.date().isoformat(),
                    "stock_code": weights.index,
                    "weight": weights.values,
                    "score": scores.reindex(weights.index).values,
                    "target_excess": selected_target.values,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(holdings, ignore_index=True)


def make_grid_config(
    base_config: dict,
    model_weight: float,
    top_k: int,
    internal_cap: float,
    risk_filter_enabled: bool,
    weighting: str,
    score_combination: str,
) -> dict:
    cfg = copy.deepcopy(base_config)
    if len(cfg["models"]) != 2:
        raise ValueError("Default portfolio grid currently expects exactly two models.")
    cfg["models"][0]["weight"] = float(model_weight)
    cfg["models"][1]["weight"] = float(1.0 - model_weight)
    cfg["score_combination"] = score_combination
    cfg["portfolio"]["top_k"] = int(top_k)
    cfg["portfolio"]["internal_max_weight"] = float(internal_cap)
    cfg["portfolio"]["weighting"] = weighting
    cfg.setdefault("risk_filter", {})["enabled"] = bool(risk_filter_enabled)
    return cfg


def add_selection_score(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    mean_rank = out["mean_portfolio_excess"].rank(pct=True)
    worst_rank = out["worst_day"].rank(pct=True)
    hit_rank = out["hit_rate"].rank(pct=True)
    ic_rank = out["mean_rank_ic"].fillna(out["mean_rank_ic"].min()).rank(pct=True)
    out["selection_score"] = 0.45 * mean_rank + 0.25 * worst_rank + 0.20 * hit_rank + 0.10 * ic_rank
    return out.sort_values(
        ["selection_score", "mean_portfolio_excess", "worst_day"],
        ascending=[False, False, False],
    )


def write_report(out_dir: Path, config_path: Path, target: str, split: str, dates: pd.DatetimeIndex, results: pd.DataFrame) -> None:
    lines = [
        "# Portfolio Grid Report",
        "",
        f"- Base config: `{config_path}`",
        f"- Target: `{target}`",
        f"- Split: `{split}`",
        f"- Trading dates: `{len(dates)}`",
    ]
    if len(dates):
        lines.append(f"- Date range: `{dates[0].date()}` to `{dates[-1].date()}`")
    lines.extend(["", "## Top Configs", ""])
    for _, row in results.head(10).iterrows():
        lines.append(
            f"- score `{row['selection_score']:.3f}` | mean `{row['mean_portfolio_excess']:.6f}` | "
            f"worst `{row['worst_day']:.6f}` | hit `{row['hit_rate']:.2f}` | "
            f"w0 `{row['model0_weight']:.2f}` | top_k `{int(row['top_k'])}` | "
            f"cap `{row['internal_max_weight']:.2f}` | risk `{row['risk_filter_enabled']}` | "
            f"weighting `{row['weighting']}` | combine `{row['score_combination']}`"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--config", default=str(CONFIG_DIR / "ensemble_two_model.json"))
    parser.add_argument("--target", default="target_excess_5d")
    parser.add_argument("--split", choices=["portfolio_validation", "model_validation"], default="portfolio_validation")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--weights", default="1.0,0.9,0.8,0.75,0.65,0.5,0.35,0.25,0.1,0.0")
    parser.add_argument("--top-k", default="30,40,50,60,80")
    parser.add_argument("--internal-caps", default="0.03,0.05,0.08")
    parser.add_argument("--risk-filter", default="true,false")
    parser.add_argument("--weighting", default="rank,equal")
    parser.add_argument("--score-combination", default="rank_average")
    args = parser.parse_args()

    config_path = Path(args.config)
    base_config = load_json(config_path)
    matrix = pd.read_parquet(args.matrix)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = matrix["stock_code"].astype(str).str.zfill(6)

    filename = "portfolio_validation_predictions.csv" if args.split == "portfolio_validation" else "validation_predictions.csv"
    score_frames = load_prediction_scores(base_config["models"], args.target, filename)
    dates = common_dates(score_frames, args.target, int(base_config["portfolio"]["min_stocks"]))
    if len(dates) == 0:
        raise ValueError(f"No common dates available for split={args.split}")

    risk_values = [x.lower() in {"1", "true", "yes", "on"} for x in parse_str_list(args.risk_filter)]
    grid = product(
        parse_float_list(args.weights),
        parse_int_list(args.top_k),
        parse_float_list(args.internal_caps),
        risk_values,
        parse_str_list(args.weighting),
        parse_str_list(args.score_combination),
    )

    results = []
    best_daily = None
    best_holdings = None
    best_config = None
    for model_weight, top_k, cap, risk_enabled, weighting, score_combination in grid:
        cfg = make_grid_config(base_config, model_weight, top_k, cap, risk_enabled, weighting, score_combination)
        daily, holdings = evaluate_scores(score_frames, matrix, dates, args.target, cfg)
        summary = summarize(daily.assign(variant="ensemble_grid"))
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "model0_name": cfg["models"][0]["name"],
                "model1_name": cfg["models"][1]["name"],
                "model0_weight": float(model_weight),
                "model1_weight": float(1.0 - model_weight),
                "top_k": int(top_k),
                "internal_max_weight": float(cap),
                "risk_filter_enabled": bool(risk_enabled),
                "weighting": weighting,
                "score_combination": score_combination,
            }
        )
        results.append(row)

    ranked = add_selection_score(pd.DataFrame(results))
    if ranked.empty:
        raise ValueError("No valid grid results")

    best = ranked.iloc[0]
    best_config = make_grid_config(
        base_config,
        float(best["model0_weight"]),
        int(best["top_k"]),
        float(best["internal_max_weight"]),
        bool(best["risk_filter_enabled"]),
        str(best["weighting"]),
        str(best["score_combination"]),
    )
    best_daily, best_holdings = evaluate_scores(score_frames, matrix, dates, args.target, best_config)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "experiments" / "portfolio_grid" / config_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "grid_results.csv", index=False)
    best_daily.to_csv(out_dir / "best_per_date_results.csv", index=False)
    best_holdings.to_csv(out_dir / "best_portfolio_holdings.csv", index=False)
    (out_dir / "best_config.json").write_text(json.dumps(best_config, ensure_ascii=False, indent=2) + "\n")
    write_report(out_dir, config_path, args.target, args.split, dates, ranked)

    print(f"Evaluated {len(ranked)} portfolio configs on {len(dates)} {args.split} dates")
    print(ranked.head(10).to_string(index=False))
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

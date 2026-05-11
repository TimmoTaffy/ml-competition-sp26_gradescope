"""
Build a submission by ensembling the two trained selected-feature models.

The ensemble combines model scores by cross-sectional rank, then builds a
long-only capped portfolio that passes validate_submission.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def choose_prediction_date(df: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.Timestamp:
    dates = pd.DatetimeIndex(df["date"].dropna().unique()).sort_values()
    if as_of is None:
        return pd.Timestamp(dates[-1])
    eligible = dates[dates <= as_of]
    if len(eligible) == 0:
        raise ValueError(f"No matrix date available on or before as_of={as_of.date()}")
    return pd.Timestamp(eligible[-1])


def load_model_scores(model_spec: dict, matrix: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.Series:
    model_dir = ROOT / model_spec["dir"]
    metadata = load_json(model_dir / "metadata.json")
    features = metadata["features"]
    missing = [f for f in features if f not in matrix.columns]
    if missing:
        raise ValueError(f"{model_spec['name']}: missing model feature(s): {missing[:10]}")

    pred_df = matrix[matrix["date"] == prediction_date].copy()
    x = pred_df[features].replace([np.inf, -np.inf], np.nan)
    model = XGBRegressor()
    model.load_model(model_dir / "model.json")
    scores = pd.Series(model.predict(x), index=pred_df["stock_code"].astype(str).str.zfill(6), name=model_spec["name"])
    return scores


def combine_scores(score_map: dict[str, pd.Series], config: dict) -> pd.Series:
    specs = config["models"]
    combined = None
    total_weight = 0.0
    for spec in specs:
        name = spec["name"]
        weight = float(spec["weight"])
        raw = score_map[name].copy()
        if config.get("score_combination", "rank_average") == "rank_average":
            transformed = raw.rank(method="average", pct=True)
        elif config.get("score_combination") == "zscore_average":
            std = raw.std()
            transformed = (raw - raw.mean()) / std if std and not np.isnan(std) else raw * 0.0
        else:
            transformed = raw
        combined = weight * transformed if combined is None else combined.add(weight * transformed, fill_value=0.0)
        total_weight += weight
    if combined is None or total_weight <= 0:
        raise ValueError("No model scores to combine")
    return sort_scores(combined / total_weight)


def score_variants(score_map: dict[str, pd.Series], config: dict, mode: str) -> dict[str, pd.Series]:
    if mode == "ensemble":
        return {"ensemble": combine_scores(score_map, config)}
    if mode in score_map:
        return {mode: sort_scores(score_map[mode])}
    if mode == "all":
        variants = {name: sort_scores(scores) for name, scores in score_map.items()}
        variants["ensemble"] = combine_scores(score_map, config)
        return variants
    raise ValueError(f"Unsupported mode={mode!r}")


def sort_scores(scores: pd.Series) -> pd.Series:
    """Sort by score descending with deterministic stock-code tie-breaking."""
    clean = scores.dropna().copy()
    frame = pd.DataFrame(
        {
            "stock_code": clean.index.astype(str).str.zfill(6),
            "score": clean.to_numpy(dtype=float),
        }
    )
    frame = frame.sort_values(["score", "stock_code"], ascending=[False, True], kind="mergesort")
    return pd.Series(frame["score"].to_numpy(), index=frame["stock_code"], name=scores.name)


def apply_risk_filter(scores: pd.Series, matrix: pd.DataFrame, prediction_date: pd.Timestamp, config: dict) -> pd.Series:
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
        print(f"[warn] risk filter left only {len(filtered)} names; falling back to unfiltered scores")
        return scores
    print(f"Risk filter kept {len(filtered)}/{len(scores)} names")
    return filtered


def build_portfolio(scores: pd.Series, config: dict) -> pd.Series:
    cfg = config["portfolio"]
    top_k = int(cfg["top_k"])
    min_stocks = int(cfg["min_stocks"])
    max_weight = float(cfg["max_weight"])
    internal_cap = min(float(cfg.get("internal_max_weight", max_weight)), max_weight)
    iterations = int(cfg.get("cap_redistribution_iterations", 100))
    if top_k < min_stocks:
        raise ValueError("portfolio.top_k must be >= portfolio.min_stocks")
    if len(scores) < top_k:
        raise ValueError(f"Only {len(scores)} eligible names available, below top_k={top_k}")

    chosen = sort_scores(scores).head(top_k)
    if cfg.get("weighting", "rank") == "equal":
        weights = pd.Series(1.0 / top_k, index=chosen.index)
    else:
        ranks = np.arange(top_k, 0, -1, dtype=float)
        weights = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(iterations):
        over = weights > internal_cap
        if not over.any():
            break
        excess = (weights[over] - internal_cap).sum()
        weights[over] = internal_cap
        free = ~over
        if not free.any():
            break
        weights[free] += excess * weights[free] / weights[free].sum()

    weights = weights / weights.sum()
    if (weights > max_weight + 1e-9).any():
        raise ValueError(f"Portfolio violates max_weight={max_weight}")
    if (weights > 0).sum() < min_stocks:
        raise ValueError(f"Portfolio has fewer than {min_stocks} positive weights")
    return weights.sort_values(ascending=False)


def write_submission(
    variant_name: str,
    scores: pd.Series,
    score_map: dict[str, pd.Series],
    matrix: pd.DataFrame,
    prediction_date: pd.Timestamp,
    config: dict,
    out_path: Path,
) -> None:
    filtered_scores = apply_risk_filter(scores, matrix, prediction_date, config)
    weights = build_portfolio(filtered_scores, config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)

    detail = pd.DataFrame({"stock_code": filtered_scores.index, f"{variant_name}_score": filtered_scores.values})
    for name, model_scores in score_map.items():
        detail = detail.merge(model_scores.rename(f"{name}_raw_score"), left_on="stock_code", right_index=True, how="left")
    detail.to_csv(out_path.with_suffix(".scores.csv"), index=False)

    print(f"[{variant_name}] Prediction date: {prediction_date.date()}")
    print(f"[{variant_name}] Wrote {len(out)} names to {out_path}")
    print(
        f"[{variant_name}] Weight summary: min={out['weight'].min():.6f}, "
        f"max={out['weight'].max():.6f}, sum={out['weight'].sum():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--config", default=str(CONFIG_DIR / "ensemble_two_model.json"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest matrix date")
    parser.add_argument(
        "--mode",
        choices=["ensemble", "general", "short_history", "all"],
        default="ensemble",
        help="Which submission(s) to generate.",
    )
    parser.add_argument("--out", default="submissions/submission.csv")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    matrix = pd.read_parquet(args.matrix)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = matrix["stock_code"].astype(str).str.zfill(6)
    prediction_date = choose_prediction_date(matrix, pd.Timestamp(args.as_of) if args.as_of else None)

    score_map = {}
    for spec in config["models"]:
        score_map[spec["name"]] = load_model_scores(spec, matrix, prediction_date)
        print(f"Loaded scores for {spec['name']}: {len(score_map[spec['name']])} names")

    out_path = Path(args.out)
    variants = score_variants(score_map, config, args.mode)
    for name, scores in variants.items():
        if args.mode == "all":
            variant_out = out_path.with_name(f"{out_path.stem}_{name}{out_path.suffix}")
        else:
            variant_out = out_path
        write_submission(name, scores, score_map, matrix, prediction_date, config, variant_out)


if __name__ == "__main__":
    main()

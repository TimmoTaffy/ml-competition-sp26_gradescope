"""
Train the two selected-feature models for the CSI500 workflow.

The script trains from scratch, uses only labels that are known as of the
prediction date, and keeps a time-ordered validation block for early stopping
and diagnostics.

Examples
--------
  conda run -n ml26s python train_model.py --profile all
  conda run -n ml26s python train_model.py --profile general --as-of 20260510
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
MODEL_DIR = ROOT / "models"
PRESETS = ("current",)


@dataclass(frozen=True)
class ModelSplit:
    prediction_date: str
    target_cutoff_date: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    portfolio_validation_start: str | None
    portfolio_validation_end: str | None


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_feature_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_as_of(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def rank_ic(y_true: pd.Series, y_pred: pd.Series, dates: pd.Series, min_stocks: int) -> float:
    rows = []
    tmp = pd.DataFrame({"date": dates, "y": y_true, "pred": y_pred})
    for _, day in tmp.groupby("date", sort=False):
        valid = day.dropna()
        if len(valid) < min_stocks:
            continue
        if valid["y"].nunique(dropna=True) < 2 or valid["pred"].nunique(dropna=True) < 2:
            continue
        rows.append(valid["y"].rank().corr(valid["pred"].rank()))
    return float(np.nanmean(rows)) if rows else float("nan")


def validation_metrics(
    val_df: pd.DataFrame,
    pred: np.ndarray,
    target: str,
    top_k: int,
    min_stocks: int,
) -> dict:
    tmp = val_df[["date", "stock_code", target]].copy()
    tmp["pred"] = pred
    daily = []
    for date, day in tmp.groupby("date", sort=False):
        valid = day.dropna(subset=[target, "pred"])
        if len(valid) < min_stocks:
            continue
        if valid[target].nunique(dropna=True) < 2 or valid["pred"].nunique(dropna=True) < 2:
            ic = np.nan
        else:
            ic = valid[target].rank().corr(valid["pred"].rank())
        top = valid.sort_values("pred", ascending=False).head(top_k)
        bottom = valid.sort_values("pred", ascending=True).head(top_k)
        daily.append(
            {
                "date": date,
                "rank_ic": ic,
                "top_k_excess": top[target].mean(),
                "top_minus_bottom": top[target].mean() - bottom[target].mean(),
            }
        )
    if not daily:
        return {
            "mean_rank_ic": np.nan,
            "top_k_excess": np.nan,
            "top_minus_bottom": np.nan,
            "hit_rate": np.nan,
            "worst_day_top_k_excess": np.nan,
            "valid_dates": 0,
        }
    daily_df = pd.DataFrame(daily)
    return {
        "mean_rank_ic": daily_df["rank_ic"].mean(),
        "top_k_excess": daily_df["top_k_excess"].mean(),
        "top_minus_bottom": daily_df["top_minus_bottom"].mean(),
        "hit_rate": float((daily_df["top_k_excess"] > 0).mean()),
        "worst_day_top_k_excess": daily_df["top_k_excess"].min(),
        "valid_dates": int(len(daily_df)),
    }


def choose_prediction_date(df: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.Timestamp:
    dates = pd.DatetimeIndex(df["date"].dropna().unique()).sort_values()
    if as_of is None:
        return pd.Timestamp(dates[-1])
    eligible = dates[dates <= as_of]
    if len(eligible) == 0:
        raise ValueError(f"No matrix date available on or before as_of={as_of.date()}")
    return pd.Timestamp(eligible[-1])


def target_cutoff_date(dates: pd.DatetimeIndex, prediction_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    idx = int(np.searchsorted(dates.values, np.datetime64(prediction_date)))
    if idx >= len(dates) or pd.Timestamp(dates[idx]) != prediction_date:
        raise ValueError(f"prediction_date {prediction_date.date()} not found in matrix dates")
    cutoff_idx = idx - int(horizon)
    if cutoff_idx < 0:
        raise ValueError("Not enough history before prediction_date for target horizon cutoff")
    return pd.Timestamp(dates[cutoff_idx])


def valid_target_dates(df: pd.DataFrame, target: str, min_stocks: int) -> pd.DatetimeIndex:
    counts = df.dropna(subset=[target]).groupby("date")["stock_code"].nunique()
    return pd.DatetimeIndex(counts[counts >= min_stocks].index).sort_values()


def make_train_validation_frames(
    df: pd.DataFrame,
    config: dict,
    prediction_date: pd.Timestamp,
    final_fit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelSplit]:
    target = config["target"]
    split_cfg = config["split"]
    horizon = int(config["target_horizon_days"])
    min_stocks = int(split_cfg["min_valid_stocks_per_date"])

    val_days = int(split_cfg["validation_days"])
    embargo = int(split_cfg["embargo_days"])
    train_lookback = split_cfg.get("train_lookback_days")
    all_dates = pd.DatetimeIndex(df["date"].dropna().unique()).sort_values()
    cutoff = target_cutoff_date(all_dates, prediction_date, horizon)
    known = df[(df["date"] <= cutoff) & df[target].notna()].copy()
    target_dates_full = valid_target_dates(known, target, min_stocks)
    target_dates = target_dates_full

    if not final_fit:
        holdout_days = int(split_cfg.get("holdout_test_days", 10))
        portfolio_days = int(split_cfg.get("portfolio_validation_days", 0))
        if len(target_dates_full) <= holdout_days + portfolio_days + val_days + embargo + 20:
            raise ValueError("Not enough target-known dates after reserving held-out test days")
        target_dates = target_dates_full[:-holdout_days]
        known = known[known["date"] <= pd.Timestamp(target_dates[-1])].copy()
    else:
        portfolio_days = 0

    portfolio_dates = target_dates[-portfolio_days:] if portfolio_days > 0 else pd.DatetimeIndex([])
    model_dates = target_dates[:-portfolio_days] if portfolio_days > 0 else target_dates

    if len(model_dates) <= val_days + embargo + 20:
        raise ValueError("Not enough target-known dates for train/validation split")

    val_start_idx = len(model_dates) - val_days
    train_end_idx = val_start_idx - embargo - 1
    if train_end_idx < 0:
        raise ValueError("Invalid split: embargo and validation consume all target-known dates")

    train_start_idx = 0
    if train_lookback is not None:
        train_start_idx = max(0, train_end_idx - int(train_lookback) + 1)

    train_start = pd.Timestamp(model_dates[train_start_idx])
    train_end = pd.Timestamp(model_dates[train_end_idx])
    val_start = pd.Timestamp(model_dates[val_start_idx])
    val_end = pd.Timestamp(model_dates[-1])

    train_df = known[(known["date"] >= train_start) & (known["date"] <= train_end)].copy()
    val_df = known[(known["date"] >= val_start) & (known["date"] <= val_end)].copy()
    if len(portfolio_dates):
        portfolio_val_df = known[(known["date"] >= pd.Timestamp(portfolio_dates[0])) & (known["date"] <= pd.Timestamp(portfolio_dates[-1]))].copy()
    else:
        portfolio_val_df = known.iloc[0:0].copy()

    final_start_idx = 0
    if train_lookback is not None:
        final_start_idx = max(0, len(model_dates) - int(train_lookback))
    final_start = pd.Timestamp(model_dates[final_start_idx])
    label_cutoff = pd.Timestamp(model_dates[-1])
    final_train_df = known[(known["date"] >= final_start) & (known["date"] <= label_cutoff)].copy()

    split = ModelSplit(
        prediction_date=prediction_date.date().isoformat(),
        target_cutoff_date=label_cutoff.date().isoformat(),
        train_start=train_start.date().isoformat(),
        train_end=train_end.date().isoformat(),
        validation_start=val_start.date().isoformat(),
        validation_end=val_end.date().isoformat(),
        portfolio_validation_start=pd.Timestamp(portfolio_dates[0]).date().isoformat() if len(portfolio_dates) else None,
        portfolio_validation_end=pd.Timestamp(portfolio_dates[-1]).date().isoformat() if len(portfolio_dates) else None,
    )
    return train_df, val_df, portfolio_val_df, final_train_df, split


def clean_xy(df: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.dropna(subset=[target]).copy()
    x = clean[features].replace([np.inf, -np.inf], np.nan)
    y = clean[target].astype(float)
    return x, y


def build_model(config: dict, n_estimators_override: int | None = None, early_stopping: bool = False) -> XGBRegressor:
    params = dict(config["model"]["params"])
    if n_estimators_override is not None:
        params["n_estimators"] = int(n_estimators_override)
    if early_stopping:
        rounds = config["model"].get("early_stopping_rounds")
        if rounds:
            params["early_stopping_rounds"] = int(rounds)
    return XGBRegressor(**params)


def train_one_profile(
    profile: str,
    matrix_path: Path,
    config_path: Path,
    out_dir: Path,
    as_of: pd.Timestamp | None,
    feature_path_override: Path | None,
    final_fit: bool,
) -> None:
    config = load_json(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    df = pd.read_parquet(matrix_path)
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    prediction_date = choose_prediction_date(df, as_of)

    feature_path = feature_path_override or ROOT / config["selected_features_path"]
    features = load_feature_list(feature_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{profile}: selected feature(s) missing from matrix: {missing[:10]}")

    target = config["target"]
    if target not in df.columns:
        raise ValueError(f"{profile}: target column {target!r} not found")

    train_df, val_df, portfolio_val_df, final_train_df, split = make_train_validation_frames(df, config, prediction_date, final_fit)
    x_train, y_train = clean_xy(train_df, features, target)
    x_val, y_val = clean_xy(val_df, features, target)

    print(f"[{profile}] train rows={len(x_train):,}, val rows={len(x_val):,}, features={len(features)}")
    print(f"[{profile}] prediction_date={split.prediction_date}, target_cutoff={split.target_cutoff_date}")

    model = build_model(config, early_stopping=True)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val.replace([np.inf, -np.inf], np.nan), y_val)],
        verbose=False,
    )
    val_pred = model.predict(x_val)
    top_k = int(config.get("validation_top_k", min(60, max(30, len(val_df["stock_code"].unique()) // 8))))
    metrics = validation_metrics(
        val_df.loc[x_val.index],
        val_pred,
        target,
        top_k=top_k,
        min_stocks=int(config["split"]["min_valid_stocks_per_date"]),
    )
    best_iteration = getattr(model, "best_iteration", None)
    best_n_estimators = int(best_iteration + 1) if best_iteration is not None else int(config["model"]["params"]["n_estimators"])

    final_model = model
    final_rows = len(x_train)
    if config.get("train", {}).get("refit_on_full_known_data", True):
        x_final, y_final = clean_xy(final_train_df, features, target)
        final_model = build_model(config, n_estimators_override=best_n_estimators, early_stopping=False)
        final_model.fit(x_final, y_final, verbose=False)
        final_rows = len(x_final)

    model_path = out_dir / "model.json"
    final_model.save_model(model_path)

    pred_df = df[df["date"] == prediction_date].copy()
    pred_x = pred_df[features].replace([np.inf, -np.inf], np.nan)
    pred_df["score"] = final_model.predict(pred_x)
    pred_df[["date", "stock_code", "score"]].to_csv(out_dir / "latest_scores.csv", index=False)

    val_out = val_df.loc[x_val.index, ["date", "stock_code", target]].copy()
    val_out["prediction"] = val_pred
    val_out.to_csv(out_dir / "validation_predictions.csv", index=False)

    if len(portfolio_val_df):
        portfolio_x = portfolio_val_df[features].replace([np.inf, -np.inf], np.nan)
        portfolio_out = portfolio_val_df[["date", "stock_code", target]].copy()
        portfolio_out["prediction"] = final_model.predict(portfolio_x)
        portfolio_out.to_csv(out_dir / "portfolio_validation_predictions.csv", index=False)
    else:
        pd.DataFrame(columns=["date", "stock_code", target, "prediction"]).to_csv(
            out_dir / "portfolio_validation_predictions.csv",
            index=False,
        )

    metadata = {
        "profile": profile,
        "matrix": str(matrix_path),
        "selected_features_path": str(feature_path),
        "n_features": len(features),
        "features": features,
        "target": target,
        "target_horizon_days": int(config["target_horizon_days"]),
        "split": asdict(split),
        "final_fit": final_fit,
        "validation_metrics": metrics,
        "portfolio_validation_rows": int(len(portfolio_val_df)),
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "best_n_estimators": best_n_estimators,
        "final_training_rows": final_rows,
        "model_path": str(model_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n")
    (out_dir / "features.txt").write_text("\n".join(features) + "\n")
    pd.DataFrame([{**metrics, "best_n_estimators": best_n_estimators, "final_training_rows": final_rows}]).to_csv(
        out_dir / "validation_metrics.csv",
        index=False,
    )
    print(f"[{profile}] validation top-K excess={metrics['top_k_excess']:.6f}, rank IC={metrics['mean_rank_ic']:.6f}")
    print(f"[{profile}] wrote model to {model_path}")


def default_config_path(profile: str, preset: str) -> Path:
    if preset != "current":
        raise ValueError(f"Archived preset {preset!r} is not part of the active workflow")
    return CONFIG_DIR / f"model_{profile}.json"


def default_out_dir(profile: str, preset: str) -> Path:
    if preset != "current":
        raise ValueError(f"Archived preset {preset!r} is not part of the active workflow")
    return MODEL_DIR / profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["general", "short_history", "all"], default="all")
    parser.add_argument("--preset", choices=PRESETS, default="current")
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--config", default=None, help="Only valid when --profile is not all")
    parser.add_argument("--features", default=None, help="Optional selected_features.txt override; only valid for one profile")
    parser.add_argument("--out", default=None, help="Output directory; for --profile all this is the parent models dir")
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest matrix date")
    parser.add_argument(
        "--final-fit",
        action="store_true",
        help="Allow use of reserved held-out test labels. Use only after research choices are frozen.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    as_of = parse_as_of(args.as_of)

    profiles = ["general", "short_history"] if args.profile == "all" else [args.profile]
    for profile in profiles:
        config_path = Path(args.config) if args.config else default_config_path(profile, args.preset)
        if args.out:
            out_parent = Path(args.out)
            out_dir = out_parent / profile if args.profile == "all" else out_parent
        else:
            out_dir = default_out_dir(profile, args.preset)
        feature_override = Path(args.features) if args.features else None
        train_one_profile(profile, matrix_path, config_path, out_dir, as_of, feature_override, args.final_fit)


if __name__ == "__main__":
    main()

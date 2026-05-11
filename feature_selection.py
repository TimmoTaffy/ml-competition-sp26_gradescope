"""
Automated feature selection for the CSI500 two-model workflow.

The script keeps the final historical test block out of feature selection by
default. Use --evaluate-test only when producing the report self-test.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def load_config(profile: str, config_path: str | None) -> dict:
    path = Path(config_path) if config_path else CONFIG_DIR / f"feature_selection_{profile}.json"
    with path.open() as f:
        config = json.load(f)
    return config


def load_feature_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def percentile_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    return s.rank(method="average", pct=True, ascending=ascending)


def safe_spearman(x: pd.Series, y: pd.Series, min_n: int) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < min_n:
        return np.nan
    xr = x[valid].rank(method="average")
    yr = y[valid].rank(method="average")
    if xr.nunique(dropna=True) < 2 or yr.nunique(dropna=True) < 2:
        return np.nan
    corr = xr.corr(yr)
    return float(corr) if pd.notna(corr) else np.nan


def ranked_metric_score(s: pd.Series) -> pd.Series:
    clean = s.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    fill_value = clean.min()
    return percentile_rank(clean.fillna(fill_value), ascending=True).fillna(0.0)


def assign_feature_group(feature: str) -> str:
    if any(k in feature for k in [
        "roe", "roa", "gross_margin", "net_margin", "debt_to_asset",
        "revenue_growth", "net_profit_growth", "deducted_net_profit",
        "eps", "cash_flow", "operating_cash_flow", "csi500_quality_growth_member",
        "csi500_quality_growth_weight",
    ]):
        return "quality"
    if any(k in feature for k in ["pe_", "pb", "ps_", "pcf", "market_cap", "earnings_yield", "book_to_price", "sales_to_price"]):
        return "valuation"
    if "industry" in feature:
        return "industry_relative"
    if any(k in feature for k in ["csi300", "csi1000", "csi500_quality_growth", "chinext", "star50", "dividend", "etf", "hsi", "hstech", "vhsi", "sp500", "nasdaq", "crude", "oil", "gold", "coal", "chemical", "semiconductor", "bank", "pharma", "military", "securities"]):
        return "market_regime"
    if feature.startswith("excess_ret") or feature.startswith("relative_vol") or feature.startswith("beta_to") or feature.startswith("residual") or feature.startswith("idiosyncratic"):
        return "csi500_relative"
    if "boll" in feature:
        if any(w in feature for w in ["10d", "20d"]):
            return "short_bollinger"
        return "bollinger"
    if any(k in feature for k in ["volume_z_5d", "amount_z_5d", "intraday", "overnight", "close_location"]):
        return "short_volume_price"
    if any(k in feature for k in ["low_liquidity", "recent_halt", "missing_days", "zero_volume", "limit_move", "drawdown", "atr", "amount_ma", "turnover_ma", "high_low_range", "idiosyncratic_vol"]):
        return "liquidity_risk"
    if any(feature.startswith(f"ret_{w}d") for w in [1, 2, 3, 5]) or any(feature.startswith(f"vol_{w}d") for w in [5, 10]) or feature in {"rsi_6"}:
        return "short_returns"
    if feature.endswith("_rank") or feature.endswith("_zscore"):
        return "cross_sectional"
    if any(k in feature for k in ["ret_", "vol_", "volume_z", "amount_z", "close_over_ma", "rsi_", "turnover"]):
        return "baseline_technical"
    return "other"


def expand_candidate_features(features: list[str], config: dict) -> pd.DataFrame:
    registry = pd.DataFrame({"feature": features})
    registry["group"] = registry["feature"].map(assign_feature_group)

    candidate_groups = set(config["candidate_groups"])
    group_aliases = {
        "short_cross_sectional": {"cross_sectional"},
        "short_risk_liquidity": {"liquidity_risk"},
        "baseline_technical": {"baseline_technical", "short_returns", "short_volume_price"},
        "bollinger": {"bollinger", "short_bollinger"},
    }

    allowed = set(candidate_groups)
    for group in candidate_groups:
        allowed.update(group_aliases.get(group, set()))

    registry["is_candidate"] = registry["group"].isin(allowed)
    return registry


def valid_target_dates(df: pd.DataFrame, target: str, min_stocks: int) -> pd.DatetimeIndex:
    counts = df.dropna(subset=[target]).groupby("date")["stock_code"].nunique()
    dates = pd.DatetimeIndex(counts[counts >= min_stocks].index).sort_values()
    return dates


def make_splits(df: pd.DataFrame, target: str, config: dict) -> tuple[pd.Timestamp, pd.DatetimeIndex, list[Fold]]:
    split_cfg = config["split"]
    dates = valid_target_dates(df, target, split_cfg["min_valid_stocks_per_date"])
    test_days = int(config.get("test_trading_days", 10))
    if len(dates) <= test_days + split_cfg["min_train_days"] + split_cfg["validation_days"]:
        raise ValueError("Not enough target-available dates for requested train/validation/test split.")

    test_dates = dates[-test_days:]
    test_start = test_dates[0]
    selection_dates = dates[dates < test_start]

    folds: list[Fold] = []
    val_days = int(split_cfg["validation_days"])
    step_days = int(split_cfg["step_days"])
    embargo_days = int(split_cfg["embargo_days"])
    min_train_days = int(split_cfg["min_train_days"])
    max_folds = int(split_cfg["max_folds"])
    train_lookback = split_cfg.get("train_lookback_days")

    val_end_idx = len(selection_dates) - 1
    fold_id = 0
    while val_end_idx >= 0 and len(folds) < max_folds:
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - embargo_days - 1
        if val_start_idx < 0 or train_end_idx < 0:
            break
        if train_lookback is None:
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - int(train_lookback) + 1)
        if train_end_idx - train_start_idx + 1 >= min_train_days:
            folds.append(
                Fold(
                    fold_id=fold_id,
                    train_start=selection_dates[train_start_idx],
                    train_end=selection_dates[train_end_idx],
                    val_start=selection_dates[val_start_idx],
                    val_end=selection_dates[val_end_idx],
                )
            )
            fold_id += 1
        val_end_idx -= step_days

    folds = list(reversed(folds))
    if not folds:
        raise ValueError("No valid rolling folds generated. Relax split hyperparameters.")
    return test_start, test_dates, folds


def date_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def single_factor_for_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    folds: list[Fold],
    quantile: float,
    min_stocks: int,
    recent_fraction: float,
) -> dict:
    daily_rows = []
    val_parts = []
    for fold in folds:
        val = date_slice(df, fold.val_start, fold.val_end)
        val_parts.append(val[[feature, target]])
        for date, day in val.groupby("date", sort=False):
            valid = day[[feature, target]].dropna()
            if len(valid) < min_stocks:
                continue
            ic = safe_spearman(valid[feature], valid[target], min_stocks)
            rank = valid[feature].rank(method="average", pct=True)
            top = valid[rank >= 1.0 - quantile][target].mean()
            bottom = valid[rank <= quantile][target].mean()
            daily_rows.append({
                "fold_id": fold.fold_id,
                "date": date,
                "ic": ic,
                "top": top,
                "bottom": bottom,
                "spread": top - bottom,
            })

    if not daily_rows:
        return {
            "feature": feature,
            "mean_ic": np.nan,
            "abs_mean_ic": np.nan,
            "ic_positive_rate": np.nan,
            "recent_mean_ic": np.nan,
            "recent_abs_mean_ic": np.nan,
            "recent_ic_positive_rate": np.nan,
            "recent_top_decile_excess": np.nan,
            "top_decile_excess": np.nan,
            "top_minus_bottom": np.nan,
            "hit_rate": np.nan,
            "worst_window": np.nan,
            "missing_rate": 1.0,
            "direction": 0,
            "valid_dates": 0,
        }

    daily = pd.DataFrame(daily_rows)
    mean_ic = daily["ic"].mean()
    direction = 1 if pd.isna(mean_ic) or mean_ic >= 0 else -1
    daily["oriented_top"] = np.where(direction >= 0, daily["top"], daily["bottom"])
    daily["oriented_spread"] = direction * daily["spread"]
    recent_n = max(1, int(np.ceil(len(daily) * min(max(recent_fraction, 0.0), 1.0))))
    recent = daily.sort_values("date").tail(recent_n)
    recent_mean_ic = recent["ic"].mean()

    val_all = pd.concat(val_parts, ignore_index=True)
    missing_rate = float(val_all[feature].isna().mean()) if len(val_all) else 1.0
    return {
        "feature": feature,
        "mean_ic": mean_ic,
        "abs_mean_ic": abs(mean_ic) if pd.notna(mean_ic) else np.nan,
        "ic_positive_rate": float((direction * daily["ic"] > 0).mean()),
        "recent_mean_ic": recent_mean_ic,
        "recent_abs_mean_ic": abs(recent_mean_ic) if pd.notna(recent_mean_ic) else np.nan,
        "recent_ic_positive_rate": float((direction * recent["ic"] > 0).mean()),
        "recent_top_decile_excess": recent["oriented_top"].mean(),
        "top_decile_excess": daily["oriented_top"].mean(),
        "top_minus_bottom": daily["oriented_spread"].mean(),
        "hit_rate": float((daily["oriented_top"] > 0).mean()),
        "worst_window": daily["oriented_top"].min(),
        "missing_rate": missing_rate,
        "direction": direction,
        "valid_dates": int(len(daily)),
    }


def add_single_factor_scores(results: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = results.copy()
    weights = config["selection"]["weights"]
    profile = config["profile"]

    out["rank_ic_stability_raw"] = out["abs_mean_ic"].fillna(0) * out["ic_positive_rate"].fillna(0)
    out["recent_rank_ic_raw"] = out["recent_abs_mean_ic"].fillna(0) * out["recent_ic_positive_rate"].fillna(0)
    out["recent_top_decile_excess_raw"] = out["recent_top_decile_excess"]

    metric_map = {
        "rank_ic_stability": "rank_ic_stability_raw",
        "recent_rank_ic": "recent_rank_ic_raw",
        "top_decile_excess": "top_decile_excess",
        "recent_top_decile_excess": "recent_top_decile_excess_raw",
        "hit_rate": "hit_rate",
        "worst_window": "worst_window",
    }
    score = pd.Series(0.0, index=out.index)
    for weight_name, weight in weights.items():
        if weight_name == "group_ablation":
            continue
        if weight_name == "missing_penalty":
            score -= float(weight) * out["missing_rate"].fillna(1.0)
            continue
        col = metric_map.get(weight_name)
        if col and col in out:
            score += float(weight) * ranked_metric_score(out[col])

    out["single_factor_score"] = score
    out["profile"] = profile
    return out.sort_values("single_factor_score", ascending=False)


def prepare_xy(train: pd.DataFrame, val: pd.DataFrame, features: list[str], target: str):
    train_clean = train.dropna(subset=[target])
    val_clean = val.dropna(subset=[target])
    X_train = train_clean[features].replace([np.inf, -np.inf], np.nan)
    y_train = train_clean[target]
    X_val = val_clean[features].replace([np.inf, -np.inf], np.nan)
    y_val = val_clean[target]
    return train_clean, val_clean, X_train, y_train, X_val, y_val


def evaluate_predictions(val: pd.DataFrame, pred: np.ndarray, target: str, top_k: int, min_stocks: int) -> dict:
    tmp = val[["date", "stock_code", target]].copy()
    tmp["pred"] = pred
    daily = []
    for date, day in tmp.groupby("date", sort=False):
        valid = day.dropna(subset=[target, "pred"])
        if len(valid) < min_stocks:
            continue
        ic = safe_spearman(valid["pred"], valid[target], min_stocks)
        top = valid.sort_values("pred", ascending=False).head(top_k)
        bottom = valid.sort_values("pred", ascending=True).head(top_k)
        daily.append({
            "date": date,
            "rank_ic": ic,
            "top_k_excess": top[target].mean(),
            "top_minus_bottom": top[target].mean() - bottom[target].mean(),
        })
    if not daily:
        return {
            "mean_rank_ic": np.nan,
            "top_k_excess": np.nan,
            "top_minus_bottom": np.nan,
            "hit_rate": np.nan,
            "worst_window": np.nan,
            "valid_dates": 0,
        }
    daily_df = pd.DataFrame(daily)
    return {
        "mean_rank_ic": daily_df["rank_ic"].mean(),
        "top_k_excess": daily_df["top_k_excess"].mean(),
        "top_minus_bottom": daily_df["top_minus_bottom"].mean(),
        "hit_rate": float((daily_df["top_k_excess"] > 0).mean()),
        "worst_window": daily_df["top_k_excess"].min(),
        "valid_dates": int(len(daily_df)),
    }


def evaluate_ridge_feature_set(
    df: pd.DataFrame,
    folds: list[Fold],
    features: list[str],
    target: str,
    config: dict,
) -> dict:
    if not features:
        return {}
    fold_metrics = []
    top_k = int(config["model"]["top_k"])
    min_stocks = int(config["split"]["min_valid_stocks_per_date"])
    alpha = float(config["model"]["ridge_alpha"])
    for fold in folds:
        train = date_slice(df, fold.train_start, fold.train_end)
        val = date_slice(df, fold.val_start, fold.val_end)
        train_clean, val_clean, X_train, y_train, X_val, _ = prepare_xy(train, val, features, target)
        if len(train_clean) == 0 or len(val_clean) == 0:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = evaluate_predictions(val_clean, pred, target, top_k, min_stocks)
        metrics["fold_id"] = fold.fold_id
        fold_metrics.append(metrics)
    if not fold_metrics:
        return {}
    m = pd.DataFrame(fold_metrics)
    return {
        "mean_rank_ic": m["mean_rank_ic"].mean(),
        "top_k_excess": m["top_k_excess"].mean(),
        "top_minus_bottom": m["top_minus_bottom"].mean(),
        "hit_rate": m["hit_rate"].mean(),
        "worst_window": m["worst_window"].min(),
        "valid_dates": int(m["valid_dates"].sum()),
    }


def group_feature_map(registry: pd.DataFrame, candidate_features: list[str]) -> dict[str, list[str]]:
    sub = registry[registry["feature"].isin(candidate_features)]
    return {group: sorted(g["feature"].tolist()) for group, g in sub.groupby("group")}


def group_ablation(
    df: pd.DataFrame,
    registry: pd.DataFrame,
    candidate_features: list[str],
    target: str,
    folds: list[Fold],
    config: dict,
) -> pd.DataFrame:
    fmap = group_feature_map(registry, candidate_features)
    base_groups = set(config["base_groups"])
    base_features = sorted({f for group, fs in fmap.items() if group in base_groups for f in fs})
    all_features = sorted(candidate_features)

    experiments: list[tuple[str, str, list[str]]] = []
    experiments.append(("baseline", "baseline", base_features))
    for group, fs in fmap.items():
        experiments.append((f"baseline_plus_{group}", group, sorted(set(base_features).union(fs))))
    experiments.append(("all_features", "all_features", all_features))
    for group, fs in fmap.items():
        experiments.append((f"all_minus_{group}", group, sorted(set(all_features).difference(fs))))

    rows = []
    for name, group, fs in experiments:
        metrics = evaluate_ridge_feature_set(df, folds, fs, target, config)
        if not metrics:
            continue
        rows.append({"experiment": name, "group": group, "n_features": len(fs), **metrics})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    baseline = out[out["experiment"] == "baseline"]
    all_row = out[out["experiment"] == "all_features"]
    baseline_top = float(baseline["top_k_excess"].iloc[0]) if not baseline.empty else 0.0
    all_top = float(all_row["top_k_excess"].iloc[0]) if not all_row.empty else 0.0
    out["top_k_improvement_vs_baseline"] = out["top_k_excess"] - baseline_top
    out["top_k_improvement_vs_all"] = out["top_k_excess"] - all_top
    return out.sort_values("top_k_excess", ascending=False)


def group_scores_from_ablation(ablation: pd.DataFrame) -> dict[str, float]:
    if ablation.empty:
        return {}
    add = ablation[ablation["experiment"].str.startswith("baseline_plus_")].copy()
    if add.empty:
        return {}
    add["score"] = percentile_rank(add["top_k_improvement_vs_baseline"].fillna(0), ascending=True)
    return dict(zip(add["group"], add["score"]))


def select_features(
    single: pd.DataFrame,
    registry: pd.DataFrame,
    ablation: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    group_scores = group_scores_from_ablation(ablation)
    out = single.merge(registry, on="feature", how="left")
    out["group_score"] = out["group"].map(group_scores).fillna(0.0)
    group_weight = float(config["selection"]["weights"].get("group_ablation", 0.0))
    out["combined_score"] = (1.0 - group_weight) * out["single_factor_score"] + group_weight * out["group_score"]
    out = out.sort_values("combined_score", ascending=False)

    max_features = int(config["selection"]["max_features"])
    min_features = int(config["selection"]["min_features"])
    per_group = int(config["selection"]["top_features_per_group"])

    selected_idx = set(out.head(min_features).index)
    for _, group_df in out.groupby("group", sort=False):
        selected_idx.update(group_df.head(per_group).index)
    selected_idx.update(out.head(max_features).index)

    selected = out.loc[sorted(selected_idx)].sort_values("combined_score", ascending=False).head(max_features)
    return selected


def evaluate_xgb_confirmation(
    df: pd.DataFrame,
    folds: list[Fold],
    selected_features: list[str],
    target: str,
    config: dict,
) -> pd.DataFrame:
    if not config["model"].get("xgb_enabled", False) or XGBRegressor is None or not selected_features:
        return pd.DataFrame()
    top_n = int(config["model"].get("xgb_top_features", len(selected_features)))
    features = selected_features[:top_n]
    top_k = int(config["model"]["top_k"])
    min_stocks = int(config["split"]["min_valid_stocks_per_date"])
    rows = []
    for fold in folds:
        train = date_slice(df, fold.train_start, fold.train_end)
        val = date_slice(df, fold.val_start, fold.val_end)
        train_clean, val_clean, X_train, y_train, X_val, _ = prepare_xy(train, val, features, target)
        if len(train_clean) == 0 or len(val_clean) == 0:
            continue
        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_val_i = imputer.transform(X_val)
        model = XGBRegressor(**config["model"]["xgb_params"])
        model.fit(X_train_i, y_train)
        pred = model.predict(X_val_i)
        metrics = evaluate_predictions(val_clean, pred, target, top_k, min_stocks)
        rows.append({"experiment": "xgb_selected_features", "fold_id": fold.fold_id, "n_features": len(features), **metrics})
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    config: dict,
    test_start: pd.Timestamp,
    test_dates: Iterable[pd.Timestamp],
    folds: list[Fold],
    selected: pd.DataFrame,
    single: pd.DataFrame,
    ablation: pd.DataFrame,
    xgb: pd.DataFrame,
    evaluate_test: bool,
) -> None:
    lines = []
    lines.append(f"# Feature Selection Report: {config['profile']}")
    lines.append("")
    lines.append("## Split Policy")
    lines.append("")
    lines.append(f"- Target: `{config['target']}`")
    lines.append(f"- Held-out test start: `{test_start.date()}`")
    lines.append(f"- Held-out test trading days: `{len(list(test_dates))}`")
    lines.append("- Test set is not used for feature selection by default.")
    lines.append(f"- Test evaluation requested: `{evaluate_test}`")
    lines.append("")
    lines.append("## Rolling Validation Folds")
    lines.append("")
    for fold in folds:
        lines.append(f"- Fold {fold.fold_id}: train `{fold.train_start.date()}` to `{fold.train_end.date()}`, validation `{fold.val_start.date()}` to `{fold.val_end.date()}`")
    lines.append("")
    lines.append("## Selected Features")
    lines.append("")
    lines.append(f"- Selected feature count: `{len(selected)}`")
    for _, row in selected.head(25).iterrows():
        lines.append(f"- `{row['feature']}` ({row['group']}): combined `{row['combined_score']:.4f}`")
    lines.append("")
    lines.append("## Top Single-Factor Features")
    lines.append("")
    for _, row in single.head(15).iterrows():
        lines.append(f"- `{row['feature']}`: score `{row['single_factor_score']:.4f}`, IC `{row['mean_ic']:.4f}`, top `{row['top_decile_excess']:.4f}`")
    if not ablation.empty:
        lines.append("")
        lines.append("## Top Ridge Group Ablations")
        lines.append("")
        for _, row in ablation.head(15).iterrows():
            lines.append(f"- `{row['experiment']}`: top-K `{row['top_k_excess']:.4f}`, IC `{row['mean_rank_ic']:.4f}`, features `{int(row['n_features'])}`")
    if not xgb.empty:
        lines.append("")
        lines.append("## XGBoost Confirmation")
        lines.append("")
        lines.append(f"- Mean top-K excess: `{xgb['top_k_excess'].mean():.4f}`")
        lines.append(f"- Mean rank IC: `{xgb['mean_rank_ic'].mean():.4f}`")
        lines.append(f"- Worst fold/window: `{xgb['worst_window'].min():.4f}`")
    path.write_text("\n".join(lines) + "\n")


def maybe_evaluate_test(
    df: pd.DataFrame,
    selected_features: list[str],
    target: str,
    test_start: pd.Timestamp,
    config: dict,
    out_dir: Path,
) -> None:
    train = df[df["date"] < test_start].copy()
    test = df[df["date"] >= test_start].copy()
    if test.empty or not selected_features:
        return
    train_clean, test_clean, X_train, y_train, X_test, _ = prepare_xy(train, test, selected_features, target)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=float(config["model"]["ridge_alpha"])),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_predictions(
        test_clean,
        pred,
        target,
        int(config["model"]["top_k"]),
        int(config["split"]["min_valid_stocks_per_date"]),
    )
    pd.DataFrame([{**metrics, "n_features": len(selected_features)}]).to_csv(out_dir / "heldout_test_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--feature-list", default=str(DATA_DIR / "final_feature_columns.txt"))
    parser.add_argument("--profile", choices=["general", "short_history"], default="general")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--test-trading-days", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.profile, args.config)
    if args.test_trading_days is not None:
        config["test_trading_days"] = args.test_trading_days

    out_dir = Path(args.out) if args.out else Path("experiments") / "feature_selection" / config["profile"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    df = pd.read_parquet(args.matrix)
    df["date"] = pd.to_datetime(df["date"])
    features = load_feature_list(Path(args.feature_list))
    features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    print(f"Loaded matrix: {len(df):,} rows, {len(features)} numeric feature columns")

    registry = expand_candidate_features(features, config)
    registry.to_csv(out_dir / "feature_registry.csv", index=False)
    candidate_features = registry[registry["is_candidate"]]["feature"].tolist()
    target = config["target"]

    test_start, test_dates, folds = make_splits(df, target, config)
    selection_df = df[df["date"] < test_start].copy()
    print(f"Split ready: {len(folds)} validation folds, held-out test starts {test_start.date()} ({len(test_dates)} trading days)")

    sf_cfg = config["single_factor"]
    recent_fraction = float(sf_cfg.get("recent_fraction", 0.5))
    single_rows = []
    print(f"Running single-factor tests for {len(candidate_features)} candidates")
    for feature in candidate_features:
        single_rows.append(
            single_factor_for_feature(
                selection_df,
                feature,
                target,
                folds,
                float(sf_cfg["quantile"]),
                int(sf_cfg["min_valid_stocks_per_date"]),
                recent_fraction,
            )
        )
    single = pd.DataFrame(single_rows)
    single = single[single["missing_rate"] <= float(sf_cfg["max_missing_rate"])].copy()
    single = add_single_factor_scores(single, config)
    single.to_csv(out_dir / "single_factor_results.csv", index=False)
    print(f"Single-factor stage kept {len(single)} features after missing-rate filter")

    ablation_limit = int(config["model"].get("ablation_top_features", len(single)))
    ablation_features = single.head(ablation_limit)["feature"].tolist()
    print(f"Running Ridge group ablation on top {len(ablation_features)} pre-screened features")
    ablation = group_ablation(selection_df, registry, ablation_features, target, folds, config)
    ablation.to_csv(out_dir / "group_ablation_results.csv", index=False)

    selected = select_features(single, registry, ablation, config)
    selected.to_csv(out_dir / "selected_features_with_scores.csv", index=False)
    selected_features = selected["feature"].tolist()
    (out_dir / "selected_features.txt").write_text("\n".join(selected_features) + "\n")

    xgb = evaluate_xgb_confirmation(selection_df, folds, selected_features, target, config)
    xgb.to_csv(out_dir / "model_confirmation_results.csv", index=False)
    if len(xgb):
        print(f"XGBoost confirmation finished on {len(xgb)} folds")

    if args.evaluate_test:
        print("Evaluating held-out test because --evaluate-test was explicitly requested")
        maybe_evaluate_test(df, selected_features, target, test_start, config, out_dir)

    write_report(
        out_dir / "selection_report.md",
        config,
        test_start,
        test_dates,
        folds,
        selected,
        single,
        ablation,
        xgb,
        args.evaluate_test,
    )
    print(f"Wrote feature-selection outputs to {out_dir}")
    print(f"Held-out test start: {test_start.date()} | test days: {len(test_dates)} | evaluate_test={args.evaluate_test}")
    print(f"Candidate features tested: {len(single)} | selected: {len(selected_features)}")


if __name__ == "__main__":
    main()

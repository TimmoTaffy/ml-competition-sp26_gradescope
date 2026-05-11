"""
Multi-round live-style portfolio fine-tuning for the retained 12 candidates.

This script keeps each candidate's selected general/short-history model pair
fixed and tunes only portfolio construction:

- general/short score weight
- top_k
- internal cap
- risk filter
- rank vs equal weighting

The objective combines the latest 3, 7, and 13 non-overlapping 5-day live-style
windows with weights 0.40 / 0.45 / 0.15, then applies robustness penalties.

It uses realized live-style window returns and cached per-entry retrained
models from the 13-window run, not fixed-block portfolio-validation
predictions.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from evaluate_recent_windows import load_index, load_matrix, load_prices
from live_walk_forward_validation import (
    retrained_config_for_entry,
)
from make_submission import sort_scores
from rolling_single_model_stability import candidate_config_paths, load_family_lookup
from score_submission import _stock_return, score_window
from tune_portfolio_grid import load_json, make_grid_config


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_CANDIDATE_ROOT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "candidates"
DEFAULT_SOURCE_RUN = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "live_walk_forward_13w"
DEFAULT_OUT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "live_portfolio_fine_tune_3_7_13"


@dataclass(frozen=True)
class CandidateInput:
    name: str
    config_path: Path
    base_config: dict[str, Any]


@dataclass(frozen=True)
class WindowInput:
    window_id: str
    entry_date: pd.Timestamp
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trading_days: int
    general_scores: pd.Series
    short_scores: pd.Series
    stock_returns: dict[str, float]
    benchmark_return: float
    risk_allowed: set[str]


def parse_bool_list(value: str) -> list[bool]:
    out = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text in {"1", "true", "yes", "on"}:
            out.append(True)
        elif text in {"0", "false", "no", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse bool: {item!r}")
    return sorted(set(out))


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def float_grid(start: float, stop: float, step: float, decimals: int = 6) -> list[float]:
    values = []
    current = float(start)
    while current <= float(stop) + 1e-12:
        values.append(round(current, decimals))
        current += float(step)
    return values


def int_grid(start: int, stop: int, step: int) -> list[int]:
    return list(range(int(start), int(stop) + 1, int(step)))


def centered_float_grid(center: float, radius: float, step: float, lo: float, hi: float) -> list[float]:
    start = max(float(lo), float(center) - float(radius))
    stop = min(float(hi), float(center) + float(radius))
    return float_grid(start, stop, step)


def centered_int_grid(center: int, radius: int, lo: int, hi: int) -> list[int]:
    start = max(int(lo), int(center) - int(radius))
    stop = min(int(hi), int(center) + int(radius))
    return int_grid(start, stop, 1)


def load_candidates(candidate_root: Path, top_n: int | None) -> list[CandidateInput]:
    candidates = []
    for name, path in candidate_config_paths(candidate_root, top_n):
        candidates.append(CandidateInput(name=name, config_path=path, base_config=load_json(path)))
    return candidates


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def load_windows(source_run: Path, num_windows: int) -> pd.DataFrame:
    windows = pd.read_csv(source_run / "windows.csv")
    if num_windows > len(windows):
        raise ValueError(f"Requested {num_windows} windows, source has {len(windows)}")
    windows = windows.tail(num_windows).copy()
    for col in ["entry_date", "start_date", "end_date"]:
        windows[col] = pd.to_datetime(windows[col])
    return windows.reset_index(drop=True)


def load_latest_scores(model_dir: Path) -> pd.Series:
    path = model_dir / "latest_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing model scores: {path}")
    scores = pd.read_csv(path, dtype={"stock_code": str})
    scores["stock_code"] = scores["stock_code"].astype(str).str.zfill(6)
    return pd.Series(scores["score"].astype(float).to_numpy(), index=scores["stock_code"], name=model_dir.name)


def risk_allowed_codes(matrix: pd.DataFrame, date: pd.Timestamp, config: dict[str, Any]) -> set[str]:
    cfg = config.get("risk_filter", {})
    day = matrix[matrix["date"] == date].copy()
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
    return set(day.loc[keep, "stock_code"])


def stock_returns_for_window(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    out = {}
    for code, group in prices.groupby("stock_code", sort=False):
        ret, _ = _stock_return(group, start, end)
        out[str(code).zfill(6)] = float(ret)
    return out


def benchmark_return_for_window(index_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    # Reuse score_window's benchmark convention with an empty zero-weight dummy.
    dummy = pd.Series(dtype=float)
    result = score_window(dummy, pd.DataFrame(columns=["stock_code", "date", "open", "close"]), index_df, start, end)
    return float(result["benchmark_return"])


def candidate_window_inputs(
    candidate: CandidateInput,
    windows: pd.DataFrame,
    matrix: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    source_run: Path,
    matrix_path: Path,
    family_lookup: dict[str, str],
    skip_existing: bool,
) -> list[WindowInput]:
    out = []
    cache_root = source_run / "model_cache"
    for _, row in windows.iterrows():
        entry = pd.Timestamp(row["entry_date"])
        cfg = retrained_config_for_entry(
            candidate=candidate.name,
            base_config_path=candidate.config_path,
            family_lookup=family_lookup,
            matrix_path=matrix_path,
            cache_root=cache_root,
            entry_date=entry,
            use_all_known_labels=True,
            skip_existing=skip_existing,
        )
        model_dirs = {spec["name"]: ROOT / spec["dir"] for spec in cfg["models"]}
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        out.append(
            WindowInput(
                window_id=str(row["window_id"]),
                entry_date=entry,
                start_date=start,
                end_date=end,
                trading_days=int(row["trading_days"]),
                general_scores=load_latest_scores(model_dirs["general"]),
                short_scores=load_latest_scores(model_dirs["short_history"]),
                stock_returns=stock_returns_for_window(prices, start, end),
                benchmark_return=benchmark_return_for_window(index_df, start, end),
                risk_allowed=risk_allowed_codes(matrix, entry, candidate.base_config),
            )
        )
    return out


def transformed_scores(raw: pd.Series, mode: str) -> pd.Series:
    if mode == "rank_average":
        return raw.rank(method="average", pct=True)
    if mode == "zscore_average":
        std = raw.std()
        return (raw - raw.mean()) / std if std and not np.isnan(std) else raw * 0.0
    return raw


def portfolio_weights(top_k: int, cap: float, weighting: str, max_weight: float = 0.10, iterations: int = 100) -> np.ndarray:
    if weighting == "equal":
        return np.full(int(top_k), 1.0 / int(top_k), dtype=float)
    ranks = np.arange(int(top_k), 0, -1, dtype=float)
    weights = ranks / ranks.sum()
    internal_cap = min(float(cap), float(max_weight))
    for _ in range(iterations):
        over = weights > internal_cap
        if not over.any():
            break
        excess = float((weights[over] - internal_cap).sum())
        weights[over] = internal_cap
        free = ~over
        if not free.any():
            break
        weights[free] += excess * weights[free] / weights[free].sum()
    weights = weights / weights.sum()
    if (weights > max_weight + 1e-9).any():
        raise ValueError(f"Portfolio violates max_weight={max_weight}")
    return weights


def suffix_metrics(excess: np.ndarray, n: int) -> dict[str, float]:
    values = np.asarray(excess[-int(n) :], dtype=float)
    return {
        f"mean_{n}w": float(np.mean(values)),
        f"sum_{n}w": float(np.sum(values)),
        f"hit_{n}w": float(np.mean(values > 0.0)),
        f"worst_{n}w": float(np.min(values)),
        f"best_{n}w": float(np.max(values)),
        f"std_{n}w": float(np.std(values)),
    }


def raw_window_score(row: dict[str, Any], n: int, hit_scale: float) -> float:
    return (
        0.57 * float(row[f"mean_{n}w"])
        + 0.33 * float(row[f"worst_{n}w"])
        + 0.10 * float(row[f"hit_{n}w"]) * float(hit_scale)
    )


def raw_penalty(row: dict[str, Any]) -> float:
    penalty = 0.0
    if row["mean_3w"] < 0:
        penalty += 0.0010
    if row["mean_7w"] < 0:
        penalty += 0.0015
    if row["mean_13w"] < 0:
        penalty += 0.0010
    if row["hit_7w"] < 0.50:
        penalty += 0.0010
    if row["hit_13w"] < 0.45:
        penalty += 0.0010
    if row["worst_7w"] < -0.030:
        penalty += 0.0015
    if row["worst_13w"] < -0.040:
        penalty += 0.0015
    return penalty


def evaluate_candidate_params(
    candidate: CandidateInput,
    inputs: list[WindowInput],
    params_iter: Iterable[tuple[float, int, float, bool, str, str]],
    keep_top: int,
    hit_scale: float,
    round_name: str,
    progress_every: int,
) -> pd.DataFrame:
    inputs = sorted(inputs, key=lambda item: item.start_date)
    min_stocks = int(candidate.base_config["portfolio"]["min_stocks"])
    max_weight = float(candidate.base_config["portfolio"]["max_weight"])
    fallback_risk = bool(candidate.base_config.get("risk_filter", {}).get("fallback_to_unfiltered_if_below_min", True))
    params = list(params_iter)
    if not params:
        return pd.DataFrame()

    unique_weights = sorted({float(p[0]) for p in params})
    unique_modes = sorted({str(p[5]) for p in params})
    param_set = set(params)
    rows: list[dict[str, Any]] = []
    weight_cache: dict[tuple[int, float, str], np.ndarray] = {}
    evaluated = 0

    for mode in unique_modes:
        transformed = []
        for item in inputs:
            transformed.append(
                (
                    transformed_scores(item.general_scores, mode),
                    transformed_scores(item.short_scores, mode),
                    item,
                )
            )
        for model_weight in unique_weights:
            orders_by_risk: list[dict[bool, np.ndarray]] = []
            returns_by_window: list[dict[str, float]] = []
            benchmarks: list[float] = []
            for general_scores, short_scores, item in transformed:
                combined = float(model_weight) * general_scores.add(0.0, fill_value=0.0)
                combined = combined.add((1.0 - float(model_weight)) * short_scores, fill_value=0.0)
                ordered = sort_scores(combined).index.to_numpy(dtype=str)
                risk_ordered = np.asarray([code for code in ordered if code in item.risk_allowed], dtype=str)
                if len(risk_ordered) < min_stocks and fallback_risk:
                    risk_ordered = ordered
                orders_by_risk.append({False: ordered, True: risk_ordered})
                returns_by_window.append(item.stock_returns)
                benchmarks.append(float(item.benchmark_return))

            relevant = [p for p in params if float(p[0]) == float(model_weight) and str(p[5]) == mode]
            for _, top_k, cap, risk_enabled, weighting, _ in relevant:
                evaluated += 1
                if progress_every > 0 and evaluated % progress_every == 0:
                    print(f"[grid] {candidate.name}/{round_name}: {evaluated}/{len(params)}", flush=True)
                if int(top_k) < min_stocks:
                    continue
                cache_key = (int(top_k), float(cap), str(weighting))
                if cache_key not in weight_cache:
                    weight_cache[cache_key] = portfolio_weights(int(top_k), float(cap), str(weighting), max_weight=max_weight)
                weights = weight_cache[cache_key]
                excess_values = []
                valid = True
                for order_map, ret_map, benchmark in zip(orders_by_risk, returns_by_window, benchmarks):
                    ordered_codes = order_map[bool(risk_enabled)]
                    if len(ordered_codes) < int(top_k):
                        valid = False
                        break
                    selected = ordered_codes[: int(top_k)]
                    stock_ret = np.asarray([ret_map.get(code, 0.0) for code in selected], dtype=float)
                    excess_values.append(float(np.dot(weights, stock_ret) - benchmark))
                if not valid:
                    continue
                excess = np.asarray(excess_values, dtype=float)
                row: dict[str, Any] = {
                    "candidate": candidate.name,
                    "round": round_name,
                    "model0_weight": float(model_weight),
                    "model1_weight": float(1.0 - float(model_weight)),
                    "top_k": int(top_k),
                    "internal_max_weight": float(cap),
                    "risk_filter_enabled": bool(risk_enabled),
                    "weighting": str(weighting),
                    "score_combination": mode,
                    "windows": int(len(excess)),
                    "mean_excess": float(np.mean(excess)),
                    "sum_excess": float(np.sum(excess)),
                    "hit_rate": float(np.mean(excess > 0.0)),
                    "worst_window": float(np.min(excess)),
                    "best_window": float(np.max(excess)),
                    "std_excess": float(np.std(excess)),
                }
                row.update(suffix_metrics(excess, 3))
                row.update(suffix_metrics(excess, 7))
                row.update(suffix_metrics(excess, 13))
                row["raw_score_3w"] = raw_window_score(row, 3, hit_scale)
                row["raw_score_7w"] = raw_window_score(row, 7, hit_scale)
                row["raw_score_13w"] = raw_window_score(row, 13, hit_scale)
                row["raw_penalty"] = raw_penalty(row)
                row["raw_combined_score"] = (
                    0.40 * row["raw_score_3w"]
                    + 0.45 * row["raw_score_7w"]
                    + 0.15 * row["raw_score_13w"]
                    - row["raw_penalty"]
                )
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["raw_combined_score", "mean_7w", "worst_13w"], ascending=[False, False, False])
    return df.head(int(keep_top)).reset_index(drop=True)


def broad_params(args: argparse.Namespace) -> list[tuple[float, int, float, bool, str, str]]:
    weights = float_grid(0.0, 1.0, args.broad_weight_step)
    top_ks = int_grid(args.top_k_min, args.top_k_max, args.broad_topk_step)
    caps = float_grid(args.cap_min, args.cap_max, args.broad_cap_step)
    risk_values = parse_bool_list(args.risk_filter)
    weighting_values = parse_str_list(args.weighting)
    modes = parse_str_list(args.score_combination)
    out = []
    for weight in weights:
        for top_k in top_ks:
            for risk in risk_values:
                for weighting in weighting_values:
                    cap_values = [max(args.cap_min, min(args.cap_max, 1.0 / top_k))] if weighting == "equal" else caps
                    for cap in cap_values:
                        for mode in modes:
                            out.append((weight, int(top_k), float(cap), bool(risk), weighting, mode))
    return out


def refine_params(
    seeds: pd.DataFrame,
    weight_radius: float,
    weight_step: float,
    topk_radius: int,
    cap_radius: float,
    cap_step: float,
    args: argparse.Namespace,
) -> list[tuple[float, int, float, bool, str, str]]:
    values = set()
    for _, row in seeds.iterrows():
        weights = centered_float_grid(float(row["model0_weight"]), weight_radius, weight_step, 0.0, 1.0)
        top_ks = centered_int_grid(int(row["top_k"]), topk_radius, args.top_k_min, args.top_k_max)
        caps = centered_float_grid(float(row["internal_max_weight"]), cap_radius, cap_step, args.cap_min, args.cap_max)
        risk_values = [bool(row["risk_filter_enabled"])]
        if args.refine_toggle_risk:
            risk_values = [False, True]
        weighting_values = [str(row["weighting"])]
        if args.refine_toggle_weighting:
            weighting_values = parse_str_list(args.weighting)
        modes = [str(row["score_combination"])]
        for weight in weights:
            for top_k in top_ks:
                for risk in risk_values:
                    for weighting in weighting_values:
                        cap_values = [max(args.cap_min, min(args.cap_max, 1.0 / top_k))] if weighting == "equal" else caps
                        for cap in cap_values:
                            for mode in modes:
                                values.add((weight, int(top_k), float(cap), bool(risk), weighting, mode))
    return sorted(values)


def final_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [3, 7, 13]:
        mean_rank = out[f"mean_{n}w"].rank(pct=True)
        worst_rank = out[f"worst_{n}w"].rank(pct=True)
        hit_rank = out[f"hit_{n}w"].rank(pct=True)
        out[f"score_{n}w"] = 0.57 * mean_rank + 0.33 * worst_rank + 0.10 * hit_rank
        out[f"rank_{n}w"] = out[f"score_{n}w"].rank(method="min", ascending=False)
    out["rank_spread_3_7_13"] = out[["rank_3w", "rank_7w", "rank_13w"]].max(axis=1) - out[["rank_3w", "rank_7w", "rank_13w"]].min(axis=1)
    out["final_penalty"] = 0.0
    out.loc[out["mean_3w"] < 0, "final_penalty"] += 0.08
    out.loc[out["mean_7w"] < 0, "final_penalty"] += 0.12
    out.loc[out["mean_13w"] < 0, "final_penalty"] += 0.08
    out.loc[out["hit_7w"] < 0.50, "final_penalty"] += 0.06
    out.loc[out["hit_13w"] < 0.45, "final_penalty"] += 0.06
    out.loc[out["worst_7w"] < -0.030, "final_penalty"] += 0.08
    out.loc[out["worst_13w"] < -0.040, "final_penalty"] += 0.08
    out.loc[out["rank_spread_3_7_13"] > max(100, len(out) * 0.50), "final_penalty"] += 0.05
    out["final_score"] = 0.40 * out["score_3w"] + 0.45 * out["score_7w"] + 0.15 * out["score_13w"] - out["final_penalty"]
    return out.sort_values(["final_score", "mean_7w", "worst_13w"], ascending=[False, False, False]).reset_index(drop=True)


def update_config(base_config: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg = make_grid_config(
        cfg,
        float(row["model0_weight"]),
        int(row["top_k"]),
        float(row["internal_max_weight"]),
        bool(row["risk_filter_enabled"]),
        str(row["weighting"]),
        str(row["score_combination"]),
    )
    meta = cfg.setdefault("_live_portfolio_fine_tune_metadata", {})
    meta.update(
        {
            "candidate": str(row["candidate"]),
            "final_score": float(row["final_score"]),
            "score_3w": float(row["score_3w"]),
            "score_7w": float(row["score_7w"]),
            "score_13w": float(row["score_13w"]),
            "final_penalty": float(row["final_penalty"]),
            "mean_3w": float(row["mean_3w"]),
            "mean_7w": float(row["mean_7w"]),
            "mean_13w": float(row["mean_13w"]),
            "hit_3w": float(row["hit_3w"]),
            "hit_7w": float(row["hit_7w"]),
            "hit_13w": float(row["hit_13w"]),
            "worst_3w": float(row["worst_3w"]),
            "worst_7w": float(row["worst_7w"]),
            "worst_13w": float(row["worst_13w"]),
        }
    )
    return cfg


def write_outputs(out_dir: Path, ranked: pd.DataFrame, candidates: dict[str, CandidateInput], args: argparse.Namespace) -> None:
    ranked.to_csv(out_dir / "all_retained_ranked.csv", index=False)
    leaderboard = ranked.groupby("candidate", sort=False).head(1).reset_index(drop=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    for _, row in leaderboard.iterrows():
        candidate = candidates[str(row["candidate"])]
        cfg = update_config(candidate.base_config, row)
        candidate_dir = out_dir / "candidates" / candidate.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "best_live_portfolio_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    if not ranked.empty:
        best_candidate = candidates[str(ranked.iloc[0]["candidate"])]
        best_cfg = update_config(best_candidate.base_config, ranked.iloc[0])
        (out_dir / "best_live_portfolio_config.json").write_text(json.dumps(best_cfg, ensure_ascii=False, indent=2) + "\n")

    columns = [
        "candidate",
        "final_score",
        "score_3w",
        "score_7w",
        "score_13w",
        "final_penalty",
        "mean_3w",
        "mean_7w",
        "mean_13w",
        "hit_3w",
        "hit_7w",
        "hit_13w",
        "worst_3w",
        "worst_7w",
        "worst_13w",
        "model0_weight",
        "model1_weight",
        "top_k",
        "internal_max_weight",
        "risk_filter_enabled",
        "weighting",
        "round",
    ]
    report = [
        "# Live Portfolio Fine Tune 3/7/13",
        "",
        "## Objective",
        "",
        "- `0.40 * score_3w + 0.45 * score_7w + 0.15 * score_13w - penalty`.",
        "- Each `score_Nw` is `0.57 mean_rank + 0.33 worst_rank + 0.10 hit_rank` among retained configs.",
        "- Model pairs are fixed; this run tunes portfolio construction only.",
        "",
        "## Top Global Configs",
        "",
        markdown_table(ranked.head(50), columns),
        "",
        "## Best Per Candidate",
        "",
        markdown_table(leaderboard, columns),
        "",
        "## Commands",
        "",
        "```bash",
        "conda run -n ml26s python live_portfolio_fine_tune_3_7_13.py \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --source-run {args.source_run} \\",
        "  --skip-existing",
        "```",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report))


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[[c for c in columns if c in df.columns]].copy()
    if view.empty:
        return "No rows."
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=13)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-per-candidate-per-round", type=int, default=5000)
    parser.add_argument("--hit-scale", type=float, default=0.01)
    parser.add_argument("--top-k-min", type=int, default=30)
    parser.add_argument("--top-k-max", type=int, default=100)
    parser.add_argument("--cap-min", type=float, default=0.03)
    parser.add_argument("--cap-max", type=float, default=0.10)
    parser.add_argument("--broad-weight-step", type=float, default=0.01)
    parser.add_argument("--broad-topk-step", type=int, default=2)
    parser.add_argument("--broad-cap-step", type=float, default=0.002)
    parser.add_argument("--fine-top-per-candidate", type=int, default=80)
    parser.add_argument("--fine-weight-radius", type=float, default=0.03)
    parser.add_argument("--fine-weight-step", type=float, default=0.0025)
    parser.add_argument("--fine-topk-radius", type=int, default=5)
    parser.add_argument("--fine-cap-radius", type=float, default=0.006)
    parser.add_argument("--fine-cap-step", type=float, default=0.001)
    parser.add_argument("--ultra-top-per-candidate", type=int, default=30)
    parser.add_argument("--ultra-weight-radius", type=float, default=0.01)
    parser.add_argument("--ultra-weight-step", type=float, default=0.001)
    parser.add_argument("--ultra-topk-radius", type=int, default=2)
    parser.add_argument("--ultra-cap-radius", type=float, default=0.002)
    parser.add_argument("--ultra-cap-step", type=float, default=0.0005)
    parser.add_argument("--risk-filter", default="true,false")
    parser.add_argument("--weighting", default="rank,equal")
    parser.add_argument("--score-combination", default="rank_average")
    parser.add_argument("--refine-toggle-risk", action="store_true", default=True)
    parser.add_argument("--refine-toggle-weighting", action="store_true", default=True)
    parser.add_argument("--progress-every", type=int, default=50000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_run = Path(args.source_run)
    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    prices = load_prices(Path(args.prices))
    index_df = load_index(Path(args.index))
    windows = load_windows(source_run, args.num_windows)
    family_lookup = load_family_lookup()
    candidates = load_candidates(Path(args.candidate_root), args.top_n)
    candidate_map = {c.name: c for c in candidates}

    manifest = {
        "objective": "0.40*3w + 0.45*7w + 0.15*13w - penalty",
        "num_windows": int(args.num_windows),
        "source_run": str(source_run),
        "portfolio_only": True,
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")

    retained_parts = []
    broad_grid = broad_params(args)
    for candidate in candidates:
        print(f"\n=== Candidate {candidate.name} ===", flush=True)
        inputs = candidate_window_inputs(
            candidate=candidate,
            windows=windows,
            matrix=matrix,
            prices=prices,
            index_df=index_df,
            source_run=source_run,
            matrix_path=matrix_path,
            family_lookup=family_lookup,
            skip_existing=bool(args.skip_existing),
        )
        print(f"[data] {candidate.name}: {len(inputs)} windows, broad configs={len(broad_grid):,}", flush=True)
        broad = evaluate_candidate_params(
            candidate,
            inputs,
            broad_grid,
            keep_top=args.keep_per_candidate_per_round,
            hit_scale=args.hit_scale,
            round_name="broad",
            progress_every=args.progress_every,
        )
        broad.to_csv(out_dir / f"{candidate.name}_round_broad_top.csv", index=False)
        print(f"[broad] {candidate.name}: kept {len(broad):,}", flush=True)

        fine_seed = broad.head(args.fine_top_per_candidate)
        fine_grid = refine_params(
            fine_seed,
            args.fine_weight_radius,
            args.fine_weight_step,
            args.fine_topk_radius,
            args.fine_cap_radius,
            args.fine_cap_step,
            args,
        )
        print(f"[fine] {candidate.name}: configs={len(fine_grid):,}", flush=True)
        fine = evaluate_candidate_params(
            candidate,
            inputs,
            fine_grid,
            keep_top=args.keep_per_candidate_per_round,
            hit_scale=args.hit_scale,
            round_name="fine",
            progress_every=args.progress_every,
        )
        fine.to_csv(out_dir / f"{candidate.name}_round_fine_top.csv", index=False)

        ultra_seed = pd.concat([broad, fine], ignore_index=True).sort_values(
            ["raw_combined_score", "mean_7w", "worst_13w"],
            ascending=[False, False, False],
        ).head(args.ultra_top_per_candidate)
        ultra_grid = refine_params(
            ultra_seed,
            args.ultra_weight_radius,
            args.ultra_weight_step,
            args.ultra_topk_radius,
            args.ultra_cap_radius,
            args.ultra_cap_step,
            args,
        )
        print(f"[ultra] {candidate.name}: configs={len(ultra_grid):,}", flush=True)
        ultra = evaluate_candidate_params(
            candidate,
            inputs,
            ultra_grid,
            keep_top=args.keep_per_candidate_per_round,
            hit_scale=args.hit_scale,
            round_name="ultra",
            progress_every=args.progress_every,
        )
        ultra.to_csv(out_dir / f"{candidate.name}_round_ultra_top.csv", index=False)
        retained_parts.extend([broad, fine, ultra])

    retained = pd.concat(retained_parts, ignore_index=True).drop_duplicates(
        subset=[
            "candidate",
            "model0_weight",
            "top_k",
            "internal_max_weight",
            "risk_filter_enabled",
            "weighting",
            "score_combination",
        ]
    )
    ranked = final_rank_scores(retained)
    write_outputs(out_dir, ranked, candidate_map, args)
    print("\n=== Live portfolio fine-tune leaderboard ===")
    print((out_dir / "leaderboard.csv").read_text())
    print(f"Wrote live portfolio fine-tune outputs to {out_dir}")


if __name__ == "__main__":
    main()

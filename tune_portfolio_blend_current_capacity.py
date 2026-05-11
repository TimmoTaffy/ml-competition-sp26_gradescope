"""
Tune portfolio-level blends between stable_compact_current and
stable_compact_capacity.

This script does not blend raw model scores. It builds portfolio-like weight
curves from each model, then blends portfolio weights:

    blended = alpha * current_portfolio + (1 - alpha) * capacity_portfolio

For `final_portfolio`, the curves are the two already selected final
portfolios. The grid intentionally tunes only alpha and final top_k.

Models are not retrained when live-style model caches already exist.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_recent_windows import load_index, load_matrix, load_prices
from live_portfolio_fine_tune_3_7_13 import (
    CandidateInput,
    benchmark_return_for_window,
    load_latest_scores,
    portfolio_weights,
    risk_allowed_codes,
    stock_returns_for_window,
)
from live_walk_forward_validation import retrained_config_for_entry
from make_submission import combine_scores, load_model_scores, sort_scores
from rolling_single_model_stability import load_family_lookup
from tune_portfolio_grid import load_json


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_CANDIDATE_ROOT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "phase1_live_tuned_candidate_root"
DEFAULT_OUT = ROOT / "experiments" / "portfolio_blend_current_capacity"


@dataclass(frozen=True)
class AnchorSpec:
    label: str
    source_run: Path


@dataclass
class WindowBundle:
    anchor: str
    window_id: str
    entry_date: pd.Timestamp
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    current_score: pd.Series
    capacity_score: pd.Series
    returns: dict[str, float]
    benchmark: float
    risk_allowed: set[str]


def default_anchors() -> list[AnchorSpec]:
    base = ROOT / "experiments" / "research_families" / "fine_tune_top12"
    return [
        AnchorSpec("2026-04-30", base / "live_walk_forward_13w"),
        AnchorSpec("2026-05-06", base / "live_walk_forward_13w_anchor_20260506"),
        AnchorSpec("2026-05-07", base / "live_walk_forward_13w_anchor_20260507"),
        AnchorSpec("2026-05-08", base / "live_walk_forward_13w_latest_20260508"),
    ]


def load_candidates(candidate_root: Path) -> tuple[CandidateInput, CandidateInput]:
    current_path = candidate_root / "stable_compact_current" / "best_finetuned_config.json"
    capacity_path = candidate_root / "stable_compact_capacity" / "best_finetuned_config.json"
    if not current_path.exists() or not capacity_path.exists():
        raise FileNotFoundError(f"Missing current/capacity configs under {candidate_root}")
    return (
        CandidateInput("stable_compact_current", current_path, load_json(current_path)),
        CandidateInput("stable_compact_capacity", capacity_path, load_json(capacity_path)),
    )


def load_windows(source_run: Path) -> pd.DataFrame:
    windows = pd.read_csv(source_run / "windows.csv")
    for col in ["entry_date", "start_date", "end_date"]:
        windows[col] = pd.to_datetime(windows[col])
    return windows.sort_values("start_date").reset_index(drop=True)


def model_score_for_entry(
    *,
    candidate: CandidateInput,
    source_run: Path,
    matrix_path: Path,
    family_lookup: dict[str, str],
    entry_date: pd.Timestamp,
) -> pd.Series:
    cfg = retrained_config_for_entry(
        candidate=candidate.name,
        base_config_path=candidate.config_path,
        family_lookup=family_lookup,
        matrix_path=matrix_path,
        cache_root=source_run / "model_cache",
        entry_date=entry_date,
        use_all_known_labels=True,
        skip_existing=True,
    )
    score_map = {str(spec["name"]): load_latest_scores(ROOT / spec["dir"]) for spec in cfg["models"]}
    return combine_scores(score_map, cfg)


def load_window_bundles(
    anchors: list[AnchorSpec],
    matrix_path: Path,
    prices_path: Path,
    index_path: Path,
    candidate_root: Path,
) -> list[WindowBundle]:
    current, capacity = load_candidates(candidate_root)
    family_lookup = load_family_lookup()
    matrix = load_matrix(matrix_path)
    prices = load_prices(prices_path)
    index_df = load_index(index_path)
    out: list[WindowBundle] = []

    for anchor in anchors:
        windows = load_windows(anchor.source_run)
        for _, row in windows.iterrows():
            entry = pd.Timestamp(row["entry_date"])
            start = pd.Timestamp(row["start_date"])
            end = pd.Timestamp(row["end_date"])
            out.append(
                WindowBundle(
                    anchor=anchor.label,
                    window_id=str(row["window_id"]),
                    entry_date=entry,
                    start_date=start,
                    end_date=end,
                    current_score=model_score_for_entry(
                        candidate=current,
                        source_run=anchor.source_run,
                        matrix_path=matrix_path,
                        family_lookup=family_lookup,
                        entry_date=entry,
                    ),
                    capacity_score=model_score_for_entry(
                        candidate=capacity,
                        source_run=anchor.source_run,
                        matrix_path=matrix_path,
                        family_lookup=family_lookup,
                        entry_date=entry,
                    ),
                    returns=stock_returns_for_window(prices, start, end),
                    benchmark=benchmark_return_for_window(index_df, start, end),
                    risk_allowed=risk_allowed_codes(matrix, entry, current.base_config),
                )
            )
    return out


def portfolio_for_score(score: pd.Series, item: WindowBundle, top_k: int, cap: float, weighting: str, risk_enabled: bool) -> pd.Series:
    ordered = sort_scores(score).index.to_numpy(dtype=str)
    if risk_enabled:
        ordered = np.asarray([code for code in ordered if code in item.risk_allowed], dtype=str)
    selected = ordered[: int(top_k)]
    weights = portfolio_weights(int(top_k), float(cap), str(weighting), max_weight=0.10)
    return pd.Series(weights, index=selected)


def base_portfolios(item: WindowBundle) -> tuple[pd.Series, pd.Series]:
    current = portfolio_for_score(
        item.current_score,
        item,
        top_k=31,
        cap=0.031758,
        weighting="rank",
        risk_enabled=True,
    )
    capacity = portfolio_for_score(
        item.capacity_score,
        item,
        top_k=33,
        cap=1.0 / 33.0,
        weighting="equal",
        risk_enabled=True,
    )
    return current, capacity


def blended_portfolio(current: pd.Series, capacity: pd.Series, alpha: float, requested_top_k: int) -> pd.Series:
    weights = current.mul(float(alpha)).add(capacity.mul(1.0 - float(alpha)), fill_value=0.0)
    weights = weights[weights > 0.0]
    frame = pd.DataFrame({"stock_code": weights.index.astype(str).str.zfill(6), "weight": weights.to_numpy(dtype=float)})
    frame = frame.sort_values(["weight", "stock_code"], ascending=[False, True], kind="mergesort")
    selected = frame.head(int(requested_top_k)).copy()
    selected["weight"] = selected["weight"] / selected["weight"].sum()
    return pd.Series(selected["weight"].to_numpy(dtype=float), index=selected["stock_code"])


def latest_scores_from_final_models(matrix_path: Path) -> tuple[pd.Series, pd.Series, WindowBundle]:
    current_cfg = load_json(ROOT / "configs" / "ensemble_two_model.json")
    capacity_cfg = load_json(ROOT / "configs" / "ensemble_stable_compact_capacity.json")
    matrix = pd.read_parquet(matrix_path)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = matrix["stock_code"].astype(str).str.zfill(6)
    prediction_date = pd.Timestamp(matrix["date"].max())
    current_score = combine_scores(
        {spec["name"]: load_model_scores(spec, matrix, prediction_date) for spec in current_cfg["models"]},
        current_cfg,
    )
    capacity_score = combine_scores(
        {spec["name"]: load_model_scores(spec, matrix, prediction_date) for spec in capacity_cfg["models"]},
        capacity_cfg,
    )
    pseudo_item = WindowBundle(
        anchor="latest",
        window_id="latest",
        entry_date=prediction_date,
        start_date=prediction_date,
        end_date=prediction_date,
        current_score=current_score,
        capacity_score=capacity_score,
        returns={},
        benchmark=0.0,
        risk_allowed=risk_allowed_codes(matrix, prediction_date, current_cfg),
    )
    return current_score, capacity_score, pseudo_item


def latest_candidate(
    *,
    matrix_path: Path,
    alpha: float,
    requested_top_k: int,
    out_path: Path,
    scores_out_path: Path | None = None,
) -> pd.Series:
    _, _, item = latest_scores_from_final_models(matrix_path)
    current, capacity = base_portfolios(item)
    weights = blended_portfolio(
        current,
        capacity,
        alpha=float(alpha),
        requested_top_k=int(requested_top_k),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stock_code": weights.index, "weight": weights.values}).to_csv(out_path, index=False)
    if scores_out_path is not None:
        idx = weights.index
        diagnostics = pd.DataFrame(
            {
                "stock_code": idx,
                "final_weight": weights.values,
                "current_component_weight": current.reindex(idx, fill_value=0.0).to_numpy(dtype=float),
                "capacity_component_weight": capacity.reindex(idx, fill_value=0.0).to_numpy(dtype=float),
                "blend_alpha_current": float(alpha),
                "blend_alpha_capacity": 1.0 - float(alpha),
            }
        )
        diagnostics["pre_normalized_blend_weight"] = (
            float(alpha) * diagnostics["current_component_weight"]
            + (1.0 - float(alpha)) * diagnostics["capacity_component_weight"]
        )
        scores_out_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(scores_out_path, index=False)
    return weights


def score_portfolio(weights_by_code: pd.Series, item: WindowBundle) -> float:
    returns = pd.Series({code: item.returns.get(code, 0.0) for code in weights_by_code.index})
    return float(weights_by_code.mul(returns).sum() - item.benchmark)


def suffix_metrics(excess: np.ndarray, n: int) -> dict[str, float]:
    values = np.asarray(excess[-int(n) :], dtype=float)
    return {
        f"mean_{n}w": float(values.mean()),
        f"worst_{n}w": float(values.min()),
        f"hit_{n}w": float((values > 0.0).mean()),
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


def legacy_raw_score(row: dict[str, Any], hit_scale: float) -> float:
    """Old raw-return 3/7/13 score kept for diagnostics."""
    return (
        0.40 * raw_window_score(row, 3, hit_scale)
        + 0.45 * raw_window_score(row, 7, hit_scale)
        + 0.15 * raw_window_score(row, 13, hit_scale)
        - raw_penalty(row)
    )


def strategy_detail_rows(
    name: str,
    by_anchor: dict[str, list[float]],
    holdings: dict[str, list[int]],
    hit_scale: float,
) -> list[dict[str, Any]]:
    detail_rows = []
    for anchor, values in by_anchor.items():
        excess = np.asarray(values, dtype=float)
        row: dict[str, Any] = {"strategy": name, "anchor": anchor}
        for n in [3, 7, 13]:
            row.update(suffix_metrics(excess, n))
        row["legacy_raw_score_3713"] = legacy_raw_score(row, hit_scale)
        row["raw_penalty"] = raw_penalty(row)
        row["avg_holdings"] = float(np.mean(holdings[anchor]))
        row["min_holdings"] = int(np.min(holdings[anchor]))
        row["max_holdings"] = int(np.max(holdings[anchor]))
        detail_rows.append(row)
    return detail_rows


def apply_rank_scores(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    active_metric_weights = {"mean": 0.50, "worst": 0.35, "hit": 0.15}
    active_horizon_weights = {3: 0.30, 7: 0.40, 13: 0.30}
    legacy_metric_weights = {"mean": 0.57, "worst": 0.33, "hit": 0.10}
    legacy_horizon_weights = {3: 0.40, 7: 0.45, 13: 0.15}
    for n in [3, 7, 13]:
        group = out.groupby("anchor", sort=False)
        out[f"mean_{n}w_rank"] = group[f"mean_{n}w"].rank(pct=True, ascending=True)
        out[f"worst_{n}w_rank"] = group[f"worst_{n}w"].rank(pct=True, ascending=True)
        out[f"hit_{n}w_rank"] = group[f"hit_{n}w"].rank(pct=True, ascending=True)
        out[f"rank_score_{n}w"] = (
            active_metric_weights["mean"] * out[f"mean_{n}w_rank"]
            + active_metric_weights["worst"] * out[f"worst_{n}w_rank"]
            + active_metric_weights["hit"] * out[f"hit_{n}w_rank"]
        )
        out[f"legacy_rank_score_{n}w"] = (
            legacy_metric_weights["mean"] * out[f"mean_{n}w_rank"]
            + legacy_metric_weights["worst"] * out[f"worst_{n}w_rank"]
            + legacy_metric_weights["hit"] * out[f"hit_{n}w_rank"]
        )
    out["anchor_score_3713"] = sum(active_horizon_weights[n] * out[f"rank_score_{n}w"] for n in [3, 7, 13])
    out["legacy_anchor_score_3713"] = sum(
        legacy_horizon_weights[n] * out[f"legacy_rank_score_{n}w"] for n in [3, 7, 13]
    ) - out["raw_penalty"]
    out["anchor_rank"] = out.groupby("anchor")["anchor_score_3713"].rank(method="min", ascending=False).astype(int)
    out["legacy_anchor_rank"] = out.groupby("anchor")["legacy_anchor_score_3713"].rank(method="min", ascending=False).astype(int)
    return out


def summarize_ranked_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in detail.groupby("strategy", sort=False):
        scores = group["anchor_score_3713"].to_numpy(dtype=float)
        legacy_scores = group["legacy_anchor_score_3713"].to_numpy(dtype=float)
        rows.append(
            {
                "strategy": strategy,
                "mean_score_3713": float(scores.mean()),
                "std_score_3713": float(scores.std()),
                "worst_score_3713": float(scores.min()),
                "robust_score": float(0.65 * scores.mean() + 0.25 * scores.min() - 0.10 * scores.std()),
                "legacy_mean_score_3713": float(legacy_scores.mean()),
                "legacy_std_score_3713": float(legacy_scores.std()),
                "legacy_worst_score_3713": float(legacy_scores.min()),
                "legacy_robust_score": float(0.65 * legacy_scores.mean() + 0.25 * legacy_scores.min() - 0.10 * legacy_scores.std()),
                "avg_legacy_raw_score_3713": float(group["legacy_raw_score_3713"].mean()),
                "avg_mean_13w": float(group["mean_13w"].mean()),
                "worst_13w": float(group["worst_13w"].min()),
                "avg_hit_13w": float(group["hit_13w"].mean()),
                "avg_mean_7w": float(group["mean_7w"].mean()),
                "worst_7w": float(group["worst_7w"].min()),
                "avg_hit_7w": float(group["hit_7w"].mean()),
                "avg_mean_3w": float(group["mean_3w"].mean()),
                "worst_3w": float(group["worst_3w"].min()),
                "avg_hit_3w": float(group["hit_3w"].mean()),
                "avg_holdings": float(group["avg_holdings"].mean()),
                "min_holdings": int(group["min_holdings"].min()),
                "max_holdings": int(group["max_holdings"].max()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(windows: list[WindowBundle], alphas: list[float], top_ks: list[int], hit_scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = sorted({item.anchor for item in windows})
    base_cache = []
    for item in windows:
        final_current, final_capacity = base_portfolios(item)
        base_cache.append((item, final_current, final_capacity))
    detail_rows: list[dict[str, Any]] = []

    baseline_specs = {
        "pure_current": {"alpha": 1.0, "top_k": 31, "blend_source": "final_portfolio"},
        "pure_capacity": {"alpha": 0.0, "top_k": 33, "blend_source": "final_portfolio"},
        "portfolio_55_all_union": {"alpha": 0.5, "top_k": 999, "blend_source": "final_portfolio"},
    }

    all_specs: list[dict[str, Any]] = []
    for name, spec in baseline_specs.items():
        all_specs.append({"strategy": name, "is_grid": False, **spec})
    for alpha in alphas:
        for top_k in top_ks:
            all_specs.append(
                {
                    "strategy": f"final_portfolio_a{alpha:.2f}_k{top_k}",
                    "is_grid": True,
                    "blend_source": "final_portfolio",
                    "alpha": float(alpha),
                    "top_k": int(top_k),
                }
            )

    for spec in all_specs:
        by_anchor = {anchor: [] for anchor in anchors}
        holdings = {anchor: [] for anchor in anchors}
        for item, final_current, final_capacity in base_cache:
            if spec["strategy"] == "pure_current":
                weights = final_current
            elif spec["strategy"] == "pure_capacity":
                weights = final_capacity
            elif spec["blend_source"] == "final_portfolio":
                weights = blended_portfolio(final_current, final_capacity, float(spec["alpha"]), int(spec["top_k"]))
            else:
                raise ValueError(f"Unsupported blend_source={spec['blend_source']!r}")
            by_anchor[item.anchor].append(score_portfolio(weights, item))
            holdings[item.anchor].append(int(len(weights)))

        strategy_detail = strategy_detail_rows(str(spec["strategy"]), by_anchor, holdings, hit_scale)
        for row in strategy_detail:
            row.update(
                {
                    "alpha": float(spec["alpha"]),
                    "requested_top_k": int(spec["top_k"]),
                    "blend_source": str(spec["blend_source"]),
                    "is_grid": bool(spec["is_grid"]),
                }
            )
        detail_rows.extend(strategy_detail)

    detail = apply_rank_scores(pd.DataFrame(detail_rows))
    summary = summarize_ranked_detail(detail)
    metadata = detail[["strategy", "alpha", "requested_top_k", "blend_source", "is_grid"]].drop_duplicates("strategy")
    summary = summary.merge(metadata, on="strategy", how="left").sort_values(
        ["robust_score", "mean_score_3713", "worst_score_3713", "avg_holdings"],
        ascending=[False, False, False, True],
    )
    summary["overall_rank"] = np.arange(1, len(summary) + 1)
    return detail, summary


def float_values() -> list[float]:
    return [round(x, 2) for x in np.arange(0.30, 0.70 + 1e-12, 0.05)]


def top_k_values() -> list[int]:
    return list(range(35, 60, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--hit-scale", type=float, default=0.01)
    parser.add_argument("--write-best-submission", action="store_true")
    parser.add_argument("--submission-out", default="")
    parser.add_argument("--scores-out", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_window_bundles(
        default_anchors(),
        matrix_path=Path(args.matrix),
        prices_path=Path(args.prices),
        index_path=Path(args.index),
        candidate_root=Path(args.candidate_root),
    )
    detail, summary = evaluate(windows, float_values(), top_k_values(), float(args.hit_scale))
    detail.to_csv(out_dir / "anchor_3713_detail.csv", index=False)
    summary.to_csv(out_dir / "leaderboard.csv", index=False)
    legacy_summary = summary.sort_values(
        ["legacy_robust_score", "legacy_mean_score_3713", "legacy_worst_score_3713", "avg_holdings"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    legacy_summary["legacy_overall_rank"] = np.arange(1, len(legacy_summary) + 1)
    legacy_summary.to_csv(out_dir / "legacy3713_leaderboard.csv", index=False)

    grid = summary[summary["is_grid"]].copy()
    best = grid.iloc[0]
    (out_dir / "best_portfolio_blend_config.json").write_text(json.dumps(best.to_dict(), indent=2, ensure_ascii=False))
    legacy_grid = legacy_summary[legacy_summary["is_grid"]].copy()
    legacy_best = legacy_grid.iloc[0]
    (out_dir / "best_portfolio_blend_config_legacy3713.json").write_text(
        json.dumps(legacy_best.to_dict(), indent=2, ensure_ascii=False)
    )
    if args.write_best_submission:
        latest_candidate(
            matrix_path=Path(args.matrix),
            alpha=float(best["alpha"]),
            requested_top_k=int(best["requested_top_k"]),
            out_path=out_dir / "candidate_portfolio_blend_best.csv",
        )
        latest_candidate(
            matrix_path=Path(args.matrix),
            alpha=float(legacy_best["alpha"]),
            requested_top_k=int(legacy_best["requested_top_k"]),
            out_path=out_dir / "candidate_portfolio_blend_best_legacy3713.csv",
        )
    if args.submission_out:
        latest_candidate(
            matrix_path=Path(args.matrix),
            alpha=float(best["alpha"]),
            requested_top_k=int(best["requested_top_k"]),
            out_path=Path(args.submission_out),
            scores_out_path=Path(args.scores_out) if args.scores_out else None,
        )

    columns = [
        "overall_rank",
        "strategy",
        "blend_source",
        "alpha",
        "requested_top_k",
        "avg_holdings",
        "robust_score",
        "mean_score_3713",
        "std_score_3713",
        "worst_score_3713",
        "legacy_robust_score",
        "legacy_mean_score_3713",
        "legacy_worst_score_3713",
        "avg_legacy_raw_score_3713",
        "avg_mean_13w",
        "worst_13w",
        "avg_hit_13w",
        "avg_mean_7w",
        "avg_mean_3w",
    ]
    print("=== Portfolio blend 3/7/13 leaderboard ===")
    print(summary[columns].head(30).to_string(index=False))
    legacy_columns = [
        "legacy_overall_rank",
        "strategy",
        "blend_source",
        "alpha",
        "requested_top_k",
        "avg_holdings",
        "legacy_robust_score",
        "legacy_mean_score_3713",
        "legacy_std_score_3713",
        "legacy_worst_score_3713",
        "robust_score",
        "avg_mean_13w",
        "worst_13w",
        "avg_hit_13w",
        "avg_mean_7w",
        "avg_mean_3w",
    ]
    print("\n=== Legacy 3/7/13 leaderboard ===")
    print(legacy_summary[legacy_columns].head(30).to_string(index=False))
    print(f"\nWrote {out_dir / 'leaderboard.csv'}")
    print(f"Wrote {out_dir / 'legacy3713_leaderboard.csv'}")
    print(f"Wrote {out_dir / 'anchor_3713_detail.csv'}")
    if args.write_best_submission:
        print(f"Wrote {out_dir / 'candidate_portfolio_blend_best.csv'}")
        print(f"Wrote {out_dir / 'candidate_portfolio_blend_best_legacy3713.csv'}")
    if args.submission_out:
        print(f"Wrote {Path(args.submission_out)}")
    if args.scores_out:
        print(f"Wrote {Path(args.scores_out)}")


if __name__ == "__main__":
    main()

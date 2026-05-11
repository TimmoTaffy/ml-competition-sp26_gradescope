"""
Strict live-style walk-forward portfolio validation for fine-tuned candidates.

For each historical evaluation window, the script:
1. uses the trading day before the window as the entry/as-of date,
2. retrains each candidate's general and short-history model specs from scratch,
   using only labels fully known as of that entry date,
3. keeps model validation as an internal early-stopping block inside each
   as-of retrain,
4. rebuilds the candidate's legal long-only portfolio at the entry date,
5. scores realized portfolio excess return versus CSI500 over the window.

This is the strict validation layer. It is slower than fixed-block validation
but avoids forward-label overlap around the portfolio-validation dates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_recent_windows import load_index, load_matrix, load_prices, score_candidate_window
from rolling_single_model_stability import (
    candidate_config_paths,
    latest_non_overlapping_windows,
    load_family_lookup,
    relative_to_root,
    resolve_model_inputs,
)
from train_model import train_one_profile
from tune_portfolio_grid import load_json


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_CANDIDATE_ROOT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "candidates"
DEFAULT_OUT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "live_walk_forward_validation"


def model_cache_key(
    entry_date: pd.Timestamp,
    profile: str,
    variant_id: str,
    config_path: Path,
    feature_path: Path,
    use_all_known_labels: bool,
) -> str:
    # Use file contents instead of file paths so identical model specs shared by
    # different candidates are trained once per entry date.
    config_digest = hashlib.sha1(config_path.read_bytes()).hexdigest()[:12]
    feature_digest = hashlib.sha1(feature_path.read_bytes()).hexdigest()[:12]
    raw = "|".join(
        [
            entry_date.date().isoformat(),
            profile,
            config_digest,
            feature_digest,
            "all_known" if use_all_known_labels else "reserved_holdouts",
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe_variant = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in variant_id)
    return f"{profile}_{safe_variant}_{digest}"


def train_cached_model(
    *,
    candidate: str,
    profile: str,
    variant_id: str,
    config_path: Path,
    feature_path: Path,
    matrix_path: Path,
    cache_root: Path,
    entry_date: pd.Timestamp,
    use_all_known_labels: bool,
    skip_existing: bool,
) -> Path:
    cache_key = model_cache_key(entry_date, profile, variant_id, config_path, feature_path, use_all_known_labels)
    out_dir = cache_root / entry_date.strftime("%Y%m%d") / cache_key
    if skip_existing and (out_dir / "model.json").exists():
        return out_dir

    print(f"[train] {entry_date.date()} {candidate}/{profile}/{variant_id}", flush=True)
    train_one_profile(
        profile=profile,
        matrix_path=matrix_path,
        config_path=config_path,
        out_dir=out_dir,
        as_of=entry_date,
        feature_path_override=feature_path,
        # True means: after internal early stopping, refit on all labels known
        # as of entry_date. It still cannot see labels after entry_date-horizon.
        final_fit=use_all_known_labels,
    )
    return out_dir


def retrained_config_for_entry(
    *,
    candidate: str,
    base_config_path: Path,
    family_lookup: dict[str, str],
    matrix_path: Path,
    cache_root: Path,
    entry_date: pd.Timestamp,
    use_all_known_labels: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    base_config = load_json(base_config_path)
    cfg = copy.deepcopy(base_config)
    for spec in cfg["models"]:
        profile = str(spec["name"])
        variant_id, config_path, feature_path = resolve_model_inputs(candidate, base_config, profile, family_lookup)
        model_dir = train_cached_model(
            candidate=candidate,
            profile=profile,
            variant_id=variant_id,
            config_path=config_path,
            feature_path=feature_path,
            matrix_path=matrix_path,
            cache_root=cache_root,
            entry_date=entry_date,
            use_all_known_labels=use_all_known_labels,
            skip_existing=skip_existing,
        )
        spec["dir"] = relative_to_root(model_dir)
    return cfg


def rank_score(df: pd.DataFrame, weights: dict[str, float], lower_is_better: set[str] | None = None) -> pd.Series:
    lower_is_better = lower_is_better or set()
    score = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col not in df or df[col].notna().sum() == 0:
            continue
        ascending = col not in lower_is_better
        score += float(weight) * df[col].rank(pct=True, ascending=ascending)
    return score


def summarize(window_results: pd.DataFrame) -> pd.DataFrame:
    ranked = window_results.copy()
    ranked["window_rank"] = ranked.groupby("window_id")["excess_return"].rank(method="min", ascending=False)
    rows: list[dict[str, Any]] = []
    for candidate, group in ranked.groupby("candidate", sort=False):
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
                "std_excess": float(group["excess_return"].std(ddof=0)),
                "avg_window_rank": float(group["window_rank"].mean()),
                "best_window_rank": int(group["window_rank"].min()),
                "worst_window_rank": int(group["window_rank"].max()),
                "avg_holdings": float(group["n_holdings"].mean()),
                "max_weight": float(group["max_weight"].max()),
                "top_k": int(group["top_k"].iloc[0]),
                "model0_weight": float(group["model0_weight"].iloc[0]),
                "model1_weight": float(group["model1_weight"].iloc[0]),
                "weighting": str(group["weighting"].iloc[0]),
                "risk_filter_enabled": bool(group["risk_filter_enabled"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out["live_score"] = rank_score(
        out,
        {
            "mean_excess": 0.57,
            "worst_window": 0.33,
            "hit_rate": 0.10,
        },
    )
    out["stability_score"] = rank_score(
        out,
        {
            "mean_excess": 0.45,
            "worst_window": 0.30,
            "hit_rate": 0.15,
            "avg_window_rank": 0.10,
        },
        lower_is_better={"avg_window_rank"},
    )
    return out.sort_values(["live_score", "mean_excess", "worst_window"], ascending=[False, False, False])


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[[c for c in columns if c in df.columns]].copy()
    if view.empty:
        return "No rows."
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    summary_df: pd.DataFrame,
    window_results: pd.DataFrame,
    use_all_known_labels: bool,
) -> None:
    display_windows = windows.copy()
    for col in ["entry_date", "start_date", "end_date"]:
        display_windows[col] = pd.to_datetime(display_windows[col]).dt.date.astype(str)

    pivot = window_results.pivot(index="window_id", columns="candidate", values="excess_return").reset_index()
    ranks = window_results.copy()
    ranks["rank"] = ranks.groupby("window_id")["excess_return"].rank(method="min", ascending=False)
    rank_pivot = ranks.pivot(index="window_id", columns="candidate", values="rank").reset_index()

    lines = [
        "# Live-Style Walk-Forward Portfolio Validation",
        "",
        "## Scope",
        "",
        "- Evaluates retained fine-tuned candidate portfolios under live-style timing.",
        "- Each window is scored from the entry close to the window-end close.",
        "- Each candidate is retrained from scratch at each entry date.",
        "- Model validation remains an internal early-stopping block inside each retrain.",
        f"- Refit mode: `{'all labels known as of entry date' if use_all_known_labels else 'reserved research holdouts inside each entry date'}`.",
        "",
        "## Windows",
        "",
        markdown_table(display_windows, ["window_id", "entry_date", "start_date", "end_date", "trading_days"]),
        "",
        "## Candidate Summary",
        "",
        markdown_table(
            summary_df,
            [
                "candidate",
                "live_score",
                "stability_score",
                "mean_excess",
                "sum_excess",
                "compounded_excess",
                "hit_rate",
                "worst_window",
                "best_window",
                "std_excess",
                "avg_window_rank",
                "top_k",
                "model0_weight",
                "model1_weight",
                "weighting",
                "risk_filter_enabled",
            ],
        ),
        "",
        "## Window Excess Returns",
        "",
        markdown_table(pivot, pivot.columns.tolist()),
        "",
        "## Window Ranks",
        "",
        markdown_table(rank_pivot, rank_pivot.columns.tolist()),
        "",
        "## Interpretation",
        "",
        "- Primary score: `0.57 mean_excess_rank + 0.33 worst_window_rank + 0.10 hit_rate_rank`.",
        "- Prefer candidates with stable positive excess and acceptable worst-window damage.",
        "- This replaces fixed-block portfolio validation for final model comparison.",
        "- Do not repeatedly retune on the same windows without reserving fresh future windows.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--target", default="target_excess_5d")
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--num-windows", type=int, default=11)
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest available index date")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--reserve-research-holdouts",
        action="store_true",
        help="Keep per-entry holdout/portfolio-validation blocks. Default uses all labels known as of entry after early stopping.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "model_cache"

    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    prices = load_prices(Path(args.prices))
    index_df = load_index(Path(args.index))
    windows = latest_non_overlapping_windows(
        index_df,
        window_length=args.window_length,
        num_windows=args.num_windows,
        as_of=pd.Timestamp(args.as_of) if args.as_of else None,
    )
    family_lookup = load_family_lookup()
    candidates = candidate_config_paths(Path(args.candidate_root), args.top_n)
    use_all_known_labels = not bool(args.reserve_research_holdouts)

    result_rows: list[dict[str, Any]] = []
    holding_frames: list[pd.DataFrame] = []
    for _, window in windows.iterrows():
        entry_date = pd.Timestamp(window["entry_date"])
        print(
            f"\n=== Window {window['window_id']}: entry={entry_date.date()} "
            f"test={pd.Timestamp(window['start_date']).date()}..{pd.Timestamp(window['end_date']).date()} ===",
            flush=True,
        )
        for candidate, config_path in candidates:
            cfg = retrained_config_for_entry(
                candidate=candidate,
                base_config_path=config_path,
                family_lookup=family_lookup,
                matrix_path=matrix_path,
                cache_root=cache_root,
                entry_date=entry_date,
                use_all_known_labels=use_all_known_labels,
                skip_existing=bool(args.skip_existing),
            )
            row, holdings = score_candidate_window(candidate, cfg, matrix, prices, index_df, window, args.target)
            result_rows.append(row)
            holding_frames.append(holdings)
            print(f"[score] {window['window_id']} {candidate}: excess={row['excess_return']:.6f}", flush=True)

    window_results = pd.DataFrame(result_rows)
    window_results["window_rank"] = window_results.groupby("window_id")["excess_return"].rank(method="min", ascending=False)
    summary_df = summarize(window_results)

    windows_out = windows.copy()
    for col in ["entry_date", "start_date", "end_date"]:
        windows_out[col] = pd.to_datetime(windows_out[col]).dt.date.astype(str)
    windows_out.to_csv(out_dir / "windows.csv", index=False)
    window_results.to_csv(out_dir / "window_results.csv", index=False)
    summary_df.to_csv(out_dir / "candidate_summary.csv", index=False)
    if holding_frames:
        pd.concat(holding_frames, ignore_index=True).to_csv(out_dir / "holdings.csv", index=False)
    write_report(out_dir, windows, summary_df, window_results, use_all_known_labels)

    print("\n=== Live walk-forward summary ===")
    print(summary_df.to_string(index=False))
    print(f"Wrote live walk-forward outputs to {out_dir}")


if __name__ == "__main__":
    main()

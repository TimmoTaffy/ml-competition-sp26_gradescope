"""
Rolling 5-day stability test for the 12 fine-tuned single-candidate portfolios.

For each recent non-overlapping 5-trading-day window:
1. use the previous trading day as the as-of / entry date,
2. retrain each candidate's saved model specs from scratch as of that date,
3. rebuild the candidate's configured portfolio,
4. score realized excess return versus CSI500 over the 5-day window.

The implementation caches trained model specs by (as_of, profile, variant,
feature set, config) so repeated specs across candidates are trained once.
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
from score_submission import score_window
from train_model import train_one_profile
from tune_portfolio_grid import load_json


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_CANDIDATE_ROOT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "candidates"
DEFAULT_OUT = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "rolling_11x5_single"


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def candidate_config_paths(candidate_root: Path, top_n: int | None) -> list[tuple[str, Path]]:
    configs = sorted(candidate_root.glob("*/best_finetuned_config.json"))
    if top_n is not None:
        configs = configs[:top_n]
    if not configs:
        raise FileNotFoundError(f"No best_finetuned_config.json files under {candidate_root}")
    return [(path.parent.name, path) for path in configs]


def latest_non_overlapping_windows(index_df: pd.DataFrame, window_length: int, num_windows: int, as_of: pd.Timestamp | None) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index_df["date"].dropna().unique()).sort_values()
    if as_of is not None:
        dates = dates[dates <= as_of]
    need = int(window_length) * int(num_windows)
    if len(dates) < need + 1:
        raise ValueError(f"Need at least {need + 1} index dates, got {len(dates)}")
    eval_days = dates[-need:]
    rows = []
    for i in range(num_windows):
        chunk = eval_days[i * window_length : (i + 1) * window_length]
        start = pd.Timestamp(chunk[0])
        end = pd.Timestamp(chunk[-1])
        entry = pd.Timestamp(dates[dates < start][-1])
        rows.append(
            {
                "window_id": f"w{i + 1:02d}",
                "entry_date": entry,
                "start_date": start,
                "end_date": end,
                "trading_days": len(chunk),
            }
        )
    return pd.DataFrame(rows)


def load_family_lookup() -> dict[str, str]:
    candidates = [
        ROOT / "experiments" / "research_families" / "fine_tune_top12" / "candidate_family_lookup.csv",
        ROOT / "experiments" / "research_families" / "fine_tune_top12" / "leaderboard.csv",
        ROOT / "experiments" / "research_families" / "fine" / "leaderboard.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        leaderboard = pd.read_csv(path)
        if "candidate" in leaderboard and "ranking_family" in leaderboard:
            return dict(zip(leaderboard["candidate"].astype(str), leaderboard["ranking_family"].astype(str)))
    return {}


def resolve_model_inputs(candidate: str, config: dict, profile: str, family_lookup: dict[str, str]) -> tuple[str, Path, Path]:
    metadata = config.get("_fine_tune_metadata", {})
    variant_id = metadata.get("general_variant" if profile == "general" else "short_variant")
    spec = next((m for m in config["models"] if str(m["name"]) == profile), None)
    if spec is None:
        raise ValueError(f"{candidate}: missing model spec {profile}")
    source_dir = ROOT / spec["dir"]
    config_path = source_dir / "config_used.json"
    feature_path = source_dir / "features.txt"
    if config_path.exists() and feature_path.exists():
        return str(variant_id or source_dir.name), config_path, feature_path

    if not variant_id:
        raise FileNotFoundError(f"{candidate}/{profile}: model_pool removed and variant metadata missing")

    config_path = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "candidates" / candidate / "configs" / f"{variant_id}.json"
    count = int(str(variant_id).split("_top", 1)[1].split("_", 1)[0])
    family = config.get("ranking_family") or metadata.get("ranking_family") or family_lookup.get(candidate)
    feature_root = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "feature_sets"
    if family:
        feature_path = feature_root / f"{profile}_{family}_top{count}.txt"
    else:
        matches = sorted(feature_root.glob(f"{profile}_*_top{count}.txt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"{candidate}/{profile}: cannot infer feature set for {variant_id}")
        feature_path = matches[0]
    if not config_path.exists():
        raise FileNotFoundError(f"{candidate}/{profile}: missing config {config_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"{candidate}/{profile}: missing feature set {feature_path}")
    return str(variant_id), config_path, feature_path


def model_cache_key(as_of: pd.Timestamp, profile: str, variant_id: str, config_path: Path, feature_path: Path, final_fit: bool) -> str:
    raw = "|".join(
        [
            as_of.date().isoformat(),
            profile,
            variant_id,
            str(config_path.resolve()),
            str(feature_path.resolve()),
            "final" if final_fit else "research",
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe_variant = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in variant_id)
    return f"{profile}_{safe_variant}_{digest}"


def train_cached_model(
    candidate: str,
    profile: str,
    variant_id: str,
    config_path: Path,
    feature_path: Path,
    matrix_path: Path,
    cache_root: Path,
    as_of: pd.Timestamp,
    final_fit: bool,
    skip_existing: bool,
) -> Path:
    cache_key = model_cache_key(as_of, profile, variant_id, config_path, feature_path, final_fit)
    out_dir = cache_root / as_of.strftime("%Y%m%d") / cache_key
    if skip_existing and (out_dir / "model.json").exists():
        return out_dir
    print(f"[train] {as_of.date()} {candidate}/{profile}/{variant_id}", flush=True)
    train_one_profile(
        profile=profile,
        matrix_path=matrix_path,
        config_path=config_path,
        out_dir=out_dir,
        as_of=as_of,
        feature_path_override=feature_path,
        final_fit=final_fit,
    )
    return out_dir


def retrained_config_for_window(
    candidate: str,
    base_config_path: Path,
    family_lookup: dict[str, str],
    matrix_path: Path,
    cache_root: Path,
    entry_date: pd.Timestamp,
    final_fit: bool,
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
            as_of=entry_date,
            final_fit=final_fit,
            skip_existing=skip_existing,
        )
        spec["dir"] = relative_to_root(model_dir)
    return cfg


def rank_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        score += float(weight) * df[col].rank(pct=True)
    return score


def summarize(window_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ranked = window_results.copy()
    ranked["window_rank"] = ranked.groupby("window_id")["excess_return"].rank(method="min", ascending=False)
    for candidate, group in ranked.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                "windows": int(group["window_id"].nunique()),
                "mean_excess": float(group["excess_return"].mean()),
                "sum_excess": float(group["excess_return"].sum()),
                "hit_rate": float((group["excess_return"] > 0).mean()),
                "worst_window": float(group["excess_return"].min()),
                "best_window": float(group["excess_return"].max()),
                "std_excess": float(group["excess_return"].std(ddof=0)),
                "avg_rank": float(group["window_rank"].mean()),
                "best_rank": int(group["window_rank"].min()),
                "worst_rank": int(group["window_rank"].max()),
                "model0_weight": float(group["model0_weight"].iloc[0]),
                "model1_weight": float(group["model1_weight"].iloc[0]),
                "top_k": int(group["top_k"].iloc[0]),
                "internal_max_weight": float(group["internal_max_weight"].iloc[0]),
                "weighting": str(group["weighting"].iloc[0]),
                "risk_filter_enabled": bool(group["risk_filter_enabled"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out["stability_score"] = rank_score(
        out,
        {
            "mean_excess": 0.45,
            "worst_window": 0.35,
            "hit_rate": 0.15,
            "avg_rank": -0.05,
        },
    )
    return out.sort_values(["stability_score", "mean_excess", "worst_window"], ascending=[False, False, False])


def markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    view = df[[c for c in columns if c in df.columns]].head(n).copy()
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


def write_report(out_dir: Path, windows: pd.DataFrame, summary_df: pd.DataFrame, window_results: pd.DataFrame) -> None:
    display_windows = windows.copy()
    for col in ["entry_date", "start_date", "end_date"]:
        display_windows[col] = pd.to_datetime(display_windows[col]).dt.date.astype(str)
    lines = [
        "# Rolling 5-Day Single-Model Stability",
        "",
        "## Scope",
        "",
        "- Evaluates the 12 fine-tuned single-candidate portfolios.",
        "- Uses recent non-overlapping 5-trading-day windows.",
        "- Retrains model specs from scratch as of each window's entry date.",
        "- Caches trained model specs by as-of date and variant to avoid duplicate training.",
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
                "stability_score",
                "mean_excess",
                "sum_excess",
                "hit_rate",
                "worst_window",
                "best_window",
                "std_excess",
                "avg_rank",
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
    ]
    pivot = window_results.pivot(index="window_id", columns="candidate", values="excess_return").reset_index()
    lines.append(markdown_table(pivot, pivot.columns.tolist()))
    lines.extend(
        [
            "",
            "## Selection Note",
            "",
            "This report is for stability diagnosis. Do not repeatedly retune against these same windows without reserving a fresh holdout.",
            "",
        ]
    )
    (out_dir / "rolling_single_model_report.md").write_text("\n".join(lines))


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
    parser.add_argument("--final-fit", action="store_true", help="Use all labels known as of entry date after early-stopping calibration.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "model_cache"
    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    prices = load_prices(Path(args.prices))
    index_df = load_index(Path(args.index))
    windows = latest_non_overlapping_windows(index_df, args.window_length, args.num_windows, pd.Timestamp(args.as_of) if args.as_of else None)
    family_lookup = load_family_lookup()
    candidates = candidate_config_paths(Path(args.candidate_root), args.top_n)

    result_rows = []
    holding_frames = []
    for _, window in windows.iterrows():
        entry = pd.Timestamp(window["entry_date"])
        print(f"\n=== Window {window['window_id']}: entry={entry.date()} test={pd.Timestamp(window['start_date']).date()}..{pd.Timestamp(window['end_date']).date()} ===", flush=True)
        for candidate, config_path in candidates:
            cfg = retrained_config_for_window(
                candidate=candidate,
                base_config_path=config_path,
                family_lookup=family_lookup,
                matrix_path=matrix_path,
                cache_root=cache_root,
                entry_date=entry,
                final_fit=bool(args.final_fit),
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
    write_report(out_dir, windows, summary_df, window_results)

    print("\n=== Rolling stability summary ===")
    print(summary_df.to_string(index=False))
    print(f"Wrote rolling stability outputs to {out_dir}")


if __name__ == "__main__":
    main()

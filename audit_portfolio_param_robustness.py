"""
Audit rounded-parameter and neighborhood stability for live-tuned portfolios.

For each candidate's best live portfolio config, this script evaluates:

- the exact tuned parameters,
- rounded parameter variants,
- local neighborhood perturbations.

It reuses the strict 13-window live-style inputs and does not retrain models.
The goal is to identify whether the selected portfolio parameters live in a
stable plateau or a narrow validation spike.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_recent_windows import load_index, load_matrix, load_prices
from live_portfolio_fine_tune_3_7_13 import (
    CandidateInput,
    candidate_window_inputs,
    evaluate_candidate_params,
    final_rank_scores,
    load_windows,
    markdown_table,
)
from rolling_single_model_stability import load_family_lookup
from tune_portfolio_grid import load_json


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_SOURCE_RUN = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "live_walk_forward_13w"
DEFAULT_LIVE_TUNE = ROOT / "experiments" / "research_families" / "fine_tune_top12" / "live_portfolio_fine_tune_3_7_13"
DEFAULT_OUT = DEFAULT_LIVE_TUNE / "robustness_audit"


def candidate_from_row(row: pd.Series, candidate_root: Path) -> CandidateInput:
    name = str(row["candidate"])
    config_path = candidate_root / name / "best_finetuned_config.json"
    return CandidateInput(name=name, config_path=config_path, base_config=load_json(config_path))


def nearest(values: list[float], x: float) -> float:
    return min(values, key=lambda v: (abs(v - x), v))


def param_tuple(
    weight: float,
    top_k: int,
    cap: float,
    risk: bool,
    weighting: str,
    mode: str = "rank_average",
) -> tuple[float, int, float, bool, str, str]:
    return (round(float(weight), 6), int(top_k), round(float(cap), 6), bool(risk), str(weighting), str(mode))


def rounded_and_neighbor_params(row: pd.Series, include_toggle: bool) -> list[tuple[float, int, float, bool, str, str]]:
    weight = float(row["model0_weight"])
    top_k = int(row["top_k"])
    cap = float(row["internal_max_weight"])
    risk = bool(row["risk_filter_enabled"])
    weighting = str(row["weighting"])
    mode = str(row["score_combination"])

    params: set[tuple[float, int, float, bool, str, str]] = set()
    params.add(param_tuple(weight, top_k, cap, risk, weighting, mode))

    rounded_weights = sorted(set([0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 1.0]))
    rounded_caps = sorted(set([0.03, 0.032, 0.033333, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10]))
    rounded_topks = sorted(set([30, 31, 33, 35, 37, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]))

    rounded_base = rounded_param(row)
    params.add(rounded_base)

    weight_values = sorted(set([max(0.0, min(1.0, weight + d)) for d in [-0.05, -0.025, -0.01, 0.0, 0.01, 0.025, 0.05]]))
    weight_values.extend([rounded_base[0]])
    topk_values = sorted(set([max(30, min(100, top_k + d)) for d in [-5, -2, -1, 0, 1, 2, 5]] + [rounded_base[1]]))
    cap_values = sorted(set([max(0.03, min(0.10, cap + d)) for d in [-0.006, -0.003, -0.001, 0.0, 0.001, 0.003, 0.006]] + [rounded_base[2]]))
    risk_values = [risk]
    weighting_values = [weighting]
    if include_toggle:
        risk_values = [False, True]
        weighting_values = ["rank", "equal"]

    for w in weight_values:
        for k in topk_values:
            for c in cap_values:
                for r in risk_values:
                    for style in weighting_values:
                        # Equal weighting ignores cap unless cap is below 1/top_k;
                        # keep the rounded effective cap for interpretability.
                        effective_cap = max(0.03, min(0.10, 1.0 / k)) if style == "equal" else c
                        params.add(param_tuple(w, k, effective_cap, r, style, mode))
    return sorted(params)


def rounded_param(row: pd.Series) -> tuple[float, int, float, bool, str, str]:
    weight = float(row["model0_weight"])
    top_k = int(row["top_k"])
    cap = float(row["internal_max_weight"])
    risk = bool(row["risk_filter_enabled"])
    weighting = str(row["weighting"])
    mode = str(row["score_combination"])

    rounded_weights = sorted(set([0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 1.0]))
    rounded_caps = sorted(set([0.03, 0.032, 0.033333, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10]))
    rounded_topks = sorted(set([30, 31, 33, 35, 37, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]))
    return param_tuple(
        nearest(rounded_weights, weight),
        int(nearest([float(x) for x in rounded_topks], float(top_k))),
        nearest(rounded_caps, cap),
        risk,
        weighting,
        mode,
    )


def classify(summary: dict[str, Any]) -> str:
    median_delta = float(summary["median_delta_raw_score"])
    pct_within = float(summary["pct_within_raw_0_001"])
    rounded_delta = float(summary["rounded_delta_raw_score"])
    if rounded_delta > -0.001 and pct_within >= 0.35 and median_delta > -0.002:
        return "stable"
    if rounded_delta > -0.0025 and pct_within >= 0.20:
        return "acceptable"
    if rounded_delta <= -0.004:
        return "spiky"
    return "fragile"


def write_report(out_dir: Path, ranked: pd.DataFrame, summary: pd.DataFrame) -> None:
    columns = [
        "candidate",
        "robustness_class",
        "exact_rank",
        "exact_final_score",
        "rounded_final_score",
        "rounded_delta_final_score",
        "exact_raw_score",
        "rounded_delta_raw_score",
        "median_delta_raw_score",
        "median_final_score",
        "median_delta_final_score",
        "p25_final_score",
        "pct_within_0_02",
        "pct_within_0_05",
        "pct_within_raw_0_001",
        "pct_within_raw_0_0025",
        "neighbor_count",
    ]
    lines = [
        "# Portfolio Parameter Robustness Audit",
        "",
        "## Summary",
        "",
        markdown_table(summary, columns),
        "",
        "## Exact And Rounded Rows",
        "",
    ]
    exact_round = ranked[ranked["audit_kind"].isin(["exact", "rounded"])].copy()
    lines.append(
        markdown_table(
            exact_round,
            [
                "candidate",
                "audit_kind",
                "final_score",
                "mean_3w",
                "mean_7w",
                "mean_13w",
                "hit_7w",
                "worst_13w",
                "model0_weight",
                "top_k",
                "internal_max_weight",
                "risk_filter_enabled",
                "weighting",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `stable`: rounded and nearby settings remain close to the exact tuned score.",
            "- `acceptable`: some degradation, but not a single-parameter spike.",
            "- `spiky`: exact setting is strong but rounded/nearby settings degrade sharply.",
            "- `fragile`: weak or unstable neighborhood.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-tune-dir", default=str(DEFAULT_LIVE_TUNE))
    parser.add_argument("--candidate-root", default=str(ROOT / "experiments" / "research_families" / "fine_tune_top12" / "candidates"))
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--num-windows", type=int, default=13)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--include-toggle", action="store_true", help="Also toggle risk_filter and rank/equal around each candidate.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    live_tune_dir = Path(args.live_tune_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.read_csv(live_tune_dir / "leaderboard.csv")
    if args.top_n is not None:
        leaderboard = leaderboard.head(args.top_n)

    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    prices = load_prices(Path(args.prices))
    index_df = load_index(Path(args.index))
    windows = load_windows(Path(args.source_run), args.num_windows)
    family_lookup = load_family_lookup()

    parts = []
    for _, row in leaderboard.iterrows():
        candidate = candidate_from_row(row, Path(args.candidate_root))
        inputs = candidate_window_inputs(
            candidate=candidate,
            windows=windows,
            matrix=matrix,
            prices=prices,
            index_df=index_df,
            source_run=Path(args.source_run),
            matrix_path=matrix_path,
            family_lookup=family_lookup,
            skip_existing=bool(args.skip_existing),
        )
        params = rounded_and_neighbor_params(row, include_toggle=bool(args.include_toggle))
        evaluated = evaluate_candidate_params(
            candidate=candidate,
            inputs=inputs,
            params_iter=params,
            keep_top=len(params),
            hit_scale=0.01,
            round_name="robustness",
            progress_every=0,
        )
        exact_key = param_tuple(
            float(row["model0_weight"]),
            int(row["top_k"]),
            float(row["internal_max_weight"]),
            bool(row["risk_filter_enabled"]),
            str(row["weighting"]),
            str(row["score_combination"]),
        )
        rounded_key = rounded_param(row)
        evaluated["audit_kind"] = "neighbor"
        for idx, erow in evaluated.iterrows():
            key = param_tuple(
                float(erow["model0_weight"]),
                int(erow["top_k"]),
                float(erow["internal_max_weight"]),
                bool(erow["risk_filter_enabled"]),
                str(erow["weighting"]),
                str(erow["score_combination"]),
            )
            if key == exact_key:
                evaluated.at[idx, "audit_kind"] = "exact"
            elif key == rounded_key:
                evaluated.at[idx, "audit_kind"] = "rounded"
        parts.append(evaluated)
        print(f"[audit] {candidate.name}: {len(evaluated)} variants", flush=True)

    all_rows = pd.concat(parts, ignore_index=True)
    ranked = final_rank_scores(all_rows)

    summary_rows = []
    for candidate, group in ranked.groupby("candidate", sort=False):
        exact = group[group["audit_kind"] == "exact"].iloc[0]
        rounded_rows = group[group["audit_kind"] == "rounded"]
        rounded = rounded_rows.iloc[0] if not rounded_rows.empty else exact
        deltas = group["final_score"] - float(exact["final_score"])
        raw_deltas = group["raw_combined_score"] - float(exact["raw_combined_score"])
        item = {
            "candidate": candidate,
            "exact_rank": int(exact.name + 1),
            "exact_final_score": float(exact["final_score"]),
            "rounded_final_score": float(rounded["final_score"]),
            "rounded_delta_final_score": float(rounded["final_score"] - exact["final_score"]),
            "exact_raw_score": float(exact["raw_combined_score"]),
            "rounded_raw_score": float(rounded["raw_combined_score"]),
            "rounded_delta_raw_score": float(rounded["raw_combined_score"] - exact["raw_combined_score"]),
            "median_final_score": float(group["final_score"].median()),
            "median_delta_final_score": float(deltas.median()),
            "median_raw_score": float(group["raw_combined_score"].median()),
            "median_delta_raw_score": float(raw_deltas.median()),
            "p25_final_score": float(group["final_score"].quantile(0.25)),
            "p25_raw_score": float(group["raw_combined_score"].quantile(0.25)),
            "min_final_score": float(group["final_score"].min()),
            "max_final_score": float(group["final_score"].max()),
            "pct_within_0_02": float((deltas >= -0.02).mean()),
            "pct_within_0_05": float((deltas >= -0.05).mean()),
            "pct_within_raw_0_001": float((raw_deltas >= -0.001).mean()),
            "pct_within_raw_0_0025": float((raw_deltas >= -0.0025).mean()),
            "neighbor_count": int(len(group)),
        }
        item["robustness_class"] = classify(item)
        summary_rows.append(item)
    class_order = {"stable": 0, "acceptable": 1, "fragile": 2, "spiky": 3}
    summary = pd.DataFrame(summary_rows)
    summary["_class_order"] = summary["robustness_class"].map(class_order).fillna(99)
    summary = summary.sort_values(["_class_order", "exact_final_score"], ascending=[True, False]).drop(columns=["_class_order"])

    ranked.to_csv(out_dir / "all_neighbor_results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_report(out_dir, ranked, summary)
    print(summary.to_string(index=False))
    print(f"Wrote robustness audit to {out_dir}")


if __name__ == "__main__":
    main()

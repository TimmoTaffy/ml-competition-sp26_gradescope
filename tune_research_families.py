"""
Run controlled research-family tuning for the CSI500 XGB workflow.

The search space is deliberately structured:
1. feature-ranking family,
2. feature-count pair,
3. XGB model preset,
4. portfolio-construction grid.

Held-out test is never evaluated here. The script trains from scratch, caches
candidate outputs, and compares every candidate on the same portfolio-validation
split using a global selection score.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from evaluate_validation_portfolios import common_dates, load_prediction_scores, summarize
from train_model import default_config_path, train_one_profile
from tune_portfolio_grid import (
    add_selection_score,
    evaluate_scores,
    load_json,
    make_grid_config,
    parse_float_list,
    parse_int_list,
    parse_str_list,
)


ROOT = Path(__file__).parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "experiments" / "research_families"

FAMILY_NAMES = ("balanced", "stable", "recent", "economic", "short_aggressive")
MODEL_PRESETS = ("conservative", "current", "capacity")
COUNT_PAIRS = {
    "compact": {"general": 60, "short_history": 30},
    "medium": {"general": 90, "short_history": 60},
    "wide": {"general": 120, "short_history": 90},
    "extra_wide": {"general": 150, "short_history": 120},
}

FAMILY_SPECS = {
    "balanced": {
        "weights": None,
        "split": {},
        "single_factor": {"recent_fraction": 0.50},
    },
    "stable": {
        "weights": {
            "rank_ic_stability": 0.35,
            "top_decile_excess": 0.15,
            "group_ablation": 0.25,
            "worst_window": 0.20,
            "missing_penalty": 0.05,
        },
        "split": {"max_folds": 8},
        "single_factor": {"recent_fraction": 0.50},
    },
    "recent": {
        "weights": {
            "recent_top_decile_excess": 0.35,
            "recent_rank_ic": 0.20,
            "hit_rate": 0.20,
            "group_ablation": 0.15,
            "worst_window": 0.05,
            "missing_penalty": 0.05,
        },
        "split": {"validation_days": 10, "step_days": 10, "max_folds": 10},
        "single_factor": {"recent_fraction": 0.35},
    },
    "economic": {
        "weights": {
            "rank_ic_stability": 0.20,
            "top_decile_excess": 0.15,
            "group_ablation": 0.35,
            "worst_window": 0.15,
            "missing_penalty": 0.05,
        },
        "split": {"max_folds": 8},
        "single_factor": {"recent_fraction": 0.50},
        "top_features_per_group": {"general": 12, "short_history": 10},
        "extra_candidate_groups": ["quality", "valuation"],
    },
    "short_aggressive": {
        "weights": {
            "recent_top_decile_excess": 0.40,
            "recent_rank_ic": 0.20,
            "hit_rate": 0.20,
            "group_ablation": 0.10,
            "worst_window": 0.05,
            "missing_penalty": 0.05,
        },
        "split": {"validation_days": 8, "step_days": 8, "max_folds": 10},
        "single_factor": {"recent_fraction": 0.30},
    },
}

BROAD_GRID = {
    "weights": "1.0,0.85,0.7,0.55,0.4,0.25,0.1,0.0",
    "top_k": "30,40,50,70,100",
    "internal_caps": "0.03,0.05,0.08,0.10",
    "risk_filter": "true,false",
    "weighting": "rank,equal",
    "score_combination": "rank_average",
}

FINE_GRID = {
    "weights": "1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0",
    "top_k": "30,35,40,45,50,60,70,80,100",
    "internal_caps": "0.03,0.04,0.05,0.06,0.08,0.10",
    "risk_filter": "true,false",
    "weighting": "rank,equal",
    "score_combination": "rank_average",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    count_pair: str
    model_preset: str
    general_n: int
    short_n: int


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def apply_model_preset(config: dict, preset: str, profile: str) -> dict:
    cfg = copy.deepcopy(config)
    params = cfg["model"]["params"]

    if preset == "current":
        return cfg

    if preset == "conservative":
        params["max_depth"] = min(int(params.get("max_depth", 3)), 2)
        params["n_estimators"] = max(180, int(round(int(params.get("n_estimators", 350)) * 0.75)))
        params["learning_rate"] = min(float(params.get("learning_rate", 0.04)), 0.04)
        params["subsample"] = min(float(params.get("subsample", 0.85)), 0.80)
        params["colsample_bytree"] = min(float(params.get("colsample_bytree", 0.85)), 0.80)
        params["min_child_weight"] = max(25, int(round(float(params.get("min_child_weight", 20)) * 1.5)))
        params["reg_lambda"] = max(10.0, float(params.get("reg_lambda", 8.0)) * 1.5)
        params["reg_alpha"] = max(0.0, float(params.get("reg_alpha", 0.0)))
        cfg["model"]["early_stopping_rounds"] = max(
            25,
            int(round(int(cfg["model"].get("early_stopping_rounds", 30)) * 0.8)),
        )
        if profile == "short_history":
            cfg["split"]["train_lookback_days"] = min(int(cfg["split"].get("train_lookback_days") or 100), 80)
        return cfg

    if preset == "capacity":
        params["max_depth"] = min(4, max(3, int(params.get("max_depth", 3)) + 1))
        params["n_estimators"] = int(round(int(params.get("n_estimators", 350)) * 1.25))
        params["learning_rate"] = max(0.025, float(params.get("learning_rate", 0.04)) * 0.9)
        params["subsample"] = min(0.90, max(float(params.get("subsample", 0.85)), 0.85))
        params["colsample_bytree"] = min(0.90, max(float(params.get("colsample_bytree", 0.85)), 0.85))
        params["min_child_weight"] = max(8, int(round(float(params.get("min_child_weight", 20)) * 0.75)))
        params["reg_lambda"] = max(3.0, float(params.get("reg_lambda", 8.0)) * 0.75)
        cfg["model"]["early_stopping_rounds"] = int(
            round(int(cfg["model"].get("early_stopping_rounds", 30)) * 1.2)
        )
        if profile == "short_history":
            lookback = cfg["split"].get("train_lookback_days")
            cfg["split"]["train_lookback_days"] = int(round((lookback or 100) * 1.2))
        return cfg

    raise ValueError(f"Unknown model preset: {preset}")


def build_experiment_ensemble_config(base_ensemble: dict, model_parent: Path) -> dict:
    cfg = copy.deepcopy(base_ensemble)
    for spec in cfg["models"]:
        spec["dir"] = relative_to_root(model_parent / spec["name"])
    return cfg


def profile_dir_name(profile: str, family: str) -> str:
    return f"{profile}_{family}"


def base_feature_selection_config(profile: str) -> dict:
    suffix = "general" if profile == "general" else "short_history"
    return load_json(CONFIG_DIR / f"feature_selection_{suffix}.json")


def build_family_feature_selection_config(profile: str, family: str) -> dict:
    if family not in FAMILY_SPECS:
        raise ValueError(f"Unknown ranking family: {family}")
    cfg = copy.deepcopy(base_feature_selection_config(profile))
    spec = FAMILY_SPECS[family]
    cfg["profile"] = profile_dir_name(profile, family)
    cfg["split"].update(spec.get("split", {}))
    cfg["single_factor"].update(spec.get("single_factor", {}))
    if spec.get("weights") is not None:
        cfg["selection"]["weights"] = copy.deepcopy(spec["weights"])
    cfg["selection"]["max_features"] = 180 if profile == "general" else 140
    cfg["selection"]["min_features"] = min(cfg["selection"]["min_features"], 50 if profile == "general" else 35)
    if "top_features_per_group" in spec:
        cfg["selection"]["top_features_per_group"] = int(spec["top_features_per_group"][profile])
    for group in spec.get("extra_candidate_groups", []):
        if group not in cfg["candidate_groups"]:
            cfg["candidate_groups"].append(group)
    return cfg


def run_feature_selection_family(
    profile: str,
    family: str,
    out_dir: Path,
    matrix: str,
    feature_list: str,
    skip_existing: bool,
) -> Path:
    family_dir = ROOT / "experiments" / "feature_selection" / profile_dir_name(profile, family)
    ranking = family_dir / "selected_features_with_scores.csv"
    dependencies = [Path(matrix), Path(feature_list)]
    is_fresh = ranking.exists() and all(
        ranking.stat().st_mtime >= dep.stat().st_mtime
        for dep in dependencies
        if dep.exists()
    )
    if skip_existing and is_fresh:
        print(f"[skip] feature selection {profile}/{family}: {ranking}")
        return ranking

    config_dir = out_dir / "feature_selection_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{profile}_{family}.json"
    config = build_family_feature_selection_config(profile, family)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    cmd = [
        sys.executable,
        "feature_selection.py",
        "--profile",
        profile,
        "--config",
        str(config_path),
        "--matrix",
        matrix,
        "--feature-list",
        feature_list,
        "--out",
        str(family_dir),
    ]
    print(f"\n[feature_selection:{profile}/{family}] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ranking


def ranked_feature_path(profile: str, family: str) -> Path:
    return ROOT / "experiments" / "feature_selection" / profile_dir_name(profile, family) / "selected_features_with_scores.csv"


def feature_set_path(out_dir: Path, profile: str, family: str, count: int) -> Path:
    return out_dir / "feature_sets" / f"{profile}_{family}_top{count}.txt"


def materialize_feature_set(profile: str, family: str, count: int, out_dir: Path) -> Path:
    ranking_path = ranked_feature_path(profile, family)
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Missing ranking file: {ranking_path}. Run with --run-feature-selection first."
        )
    df = pd.read_csv(ranking_path)
    if "combined_score" in df.columns:
        df = df.sort_values("combined_score", ascending=False)
    features = df["feature"].dropna().astype(str).drop_duplicates().tolist()
    if len(features) < count:
        raise ValueError(f"{ranking_path} has {len(features)} features, cannot create top{count}")
    path = feature_set_path(out_dir, profile, family, count)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(features[:count]) + "\n")
    return path


def enumerate_candidates(families: list[str], count_pairs: list[str], model_presets: list[str]) -> list[Candidate]:
    candidates = []
    for family, count_pair, preset in product(families, count_pairs, model_presets):
        counts = COUNT_PAIRS[count_pair]
        name = f"{family}_{count_pair}_{preset}"
        candidates.append(
            Candidate(
                name=name,
                family=family,
                count_pair=count_pair,
                model_preset=preset,
                general_n=int(counts["general"]),
                short_n=int(counts["short_history"]),
            )
        )
    return candidates


def restrict_candidates(
    candidates: list[Candidate],
    candidates_from: str | None,
    top_n: int,
    max_candidates: int | None,
    random_state: int,
) -> list[Candidate]:
    selected = candidates
    if candidates_from:
        prior = pd.read_csv(candidates_from)
        names = set(prior.head(top_n)["candidate"].astype(str))
        selected = [c for c in selected if c.name in names]
    if max_candidates is not None and len(selected) > max_candidates:
        rng = random.Random(random_state)
        selected = rng.sample(selected, max_candidates)
        selected = sorted(selected, key=lambda c: c.name)
    return selected


def train_candidate(
    candidate: Candidate,
    matrix_path: Path,
    out_dir: Path,
    as_of: str | None,
    skip_existing: bool,
    force: bool,
) -> Path:
    candidate_dir = out_dir / "candidates" / candidate.name
    model_parent = candidate_dir / "models"
    config_dir = candidate_dir / "configs"
    ensemble_path = candidate_dir / "ensemble_config.json"
    done_paths = [
        model_parent / "general" / "portfolio_validation_predictions.csv",
        model_parent / "short_history" / "portfolio_validation_predictions.csv",
        ensemble_path,
    ]
    if skip_existing and not force and all(p.exists() for p in done_paths):
        print(f"[skip] train {candidate.name}")
        return ensemble_path

    config_dir.mkdir(parents=True, exist_ok=True)
    model_parent.mkdir(parents=True, exist_ok=True)

    feature_paths = {
        "general": materialize_feature_set("general", candidate.family, candidate.general_n, out_dir),
        "short_history": materialize_feature_set("short_history", candidate.family, candidate.short_n, out_dir),
    }
    for profile in ("general", "short_history"):
        base_config = load_json(default_config_path(profile, "current"))
        tuned_config = apply_model_preset(base_config, candidate.model_preset, profile)
        config_path = config_dir / f"model_{profile}.json"
        config_path.write_text(json.dumps(tuned_config, ensure_ascii=False, indent=2) + "\n")
        train_one_profile(
            profile=profile,
            matrix_path=matrix_path,
            config_path=config_path,
            out_dir=model_parent / profile,
            as_of=pd.Timestamp(as_of) if as_of else None,
            feature_path_override=feature_paths[profile],
            final_fit=False,
        )

    base_ensemble = load_json(CONFIG_DIR / "ensemble_two_model.json")
    ensemble_config = build_experiment_ensemble_config(base_ensemble, model_parent)
    ensemble_path.write_text(json.dumps(ensemble_config, ensure_ascii=False, indent=2) + "\n")
    return ensemble_path


def grid_for_stage(stage: str, args: argparse.Namespace) -> dict[str, str]:
    base = copy.deepcopy(FINE_GRID if stage == "fine" else BROAD_GRID)
    for key in ["weights", "top_k", "internal_caps", "risk_filter", "weighting", "score_combination"]:
        value = getattr(args, key)
        if value:
            base[key] = value
    return base


def evaluate_candidate_grid(
    candidate: Candidate,
    ensemble_path: Path,
    matrix: pd.DataFrame,
    target: str,
    split: str,
    grid: dict[str, str],
    out_dir: Path,
    skip_existing: bool,
    force: bool,
) -> pd.DataFrame:
    candidate_out = out_dir / "candidates" / candidate.name
    grid_path = candidate_out / "grid_results.csv"
    if skip_existing and not force and grid_path.exists():
        return pd.read_csv(grid_path)

    base_config = load_json(ensemble_path)
    filename = "portfolio_validation_predictions.csv" if split == "portfolio_validation" else "validation_predictions.csv"
    score_frames = load_prediction_scores(base_config["models"], target, filename)
    dates = common_dates(score_frames, target, int(base_config["portfolio"]["min_stocks"]))
    if len(dates) == 0:
        raise ValueError(f"No common dates for {candidate.name}")

    risk_values = [x.lower() in {"1", "true", "yes", "on"} for x in parse_str_list(grid["risk_filter"])]
    rows = []
    best_config = None
    best_daily = None
    best_holdings = None

    for model_weight, top_k, cap, risk_enabled, weighting, score_combination in product(
        parse_float_list(grid["weights"]),
        parse_int_list(grid["top_k"]),
        parse_float_list(grid["internal_caps"]),
        risk_values,
        parse_str_list(grid["weighting"]),
        parse_str_list(grid["score_combination"]),
    ):
        cfg = make_grid_config(base_config, model_weight, top_k, cap, risk_enabled, weighting, score_combination)
        daily, holdings = evaluate_scores(score_frames, matrix, dates, target, cfg)
        summary = summarize(daily.assign(variant="ensemble_grid"))
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "candidate": candidate.name,
                "ranking_family": candidate.family,
                "feature_count_pair": candidate.count_pair,
                "model_preset": candidate.model_preset,
                "general_top_n": candidate.general_n,
                "short_top_n": candidate.short_n,
                "model0_weight": float(model_weight),
                "model1_weight": float(1.0 - model_weight),
                "top_k": int(top_k),
                "internal_max_weight": float(cap),
                "risk_filter_enabled": bool(risk_enabled),
                "weighting": weighting,
                "score_combination": score_combination,
            }
        )
        rows.append(row)

    ranked = add_selection_score(pd.DataFrame(rows))
    if ranked.empty:
        raise ValueError(f"No valid grid rows for {candidate.name}")

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
    best_daily, best_holdings = evaluate_scores(score_frames, matrix, dates, target, best_config)

    candidate_out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(grid_path, index=False)
    best_daily.to_csv(candidate_out / "best_per_date_results.csv", index=False)
    best_holdings.to_csv(candidate_out / "best_portfolio_holdings.csv", index=False)
    (candidate_out / "best_config.json").write_text(json.dumps(best_config, ensure_ascii=False, indent=2) + "\n")
    print(
        f"[grid] {candidate.name}: best mean={best['mean_portfolio_excess']:.6f}, "
        f"worst={best['worst_day']:.6f}, hit={best['hit_rate']:.3f}"
    )
    return ranked


def best_per_candidate(ranked: pd.DataFrame) -> pd.DataFrame:
    ordered = ranked.sort_values(
        ["selection_score", "mean_portfolio_excess", "worst_day"],
        ascending=[False, False, False],
    )
    return ordered.groupby("candidate", sort=False).head(1).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> list[str]:
    if n is not None:
        df = df.head(n)
    if df.empty:
        return ["No rows."]
    cols = [c for c in columns if c in df.columns]
    out = df[cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return lines


def write_report(out_dir: Path, args: argparse.Namespace, candidates: list[Candidate], ranked: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    columns = [
        "candidate",
        "selection_score",
        "mean_portfolio_excess",
        "cumulative_portfolio_excess",
        "hit_rate",
        "worst_day",
        "mean_rank_ic",
        "ranking_family",
        "feature_count_pair",
        "model_preset",
        "general_top_n",
        "short_top_n",
        "model0_weight",
        "model1_weight",
        "top_k",
        "internal_max_weight",
        "risk_filter_enabled",
        "weighting",
    ]
    lines = [
        "# Research Families Report",
        "",
        f"- Stage: `{args.stage}`",
        f"- Target: `{args.target}`",
        f"- Split: `{args.split}`",
        f"- Candidate count: `{len(candidates)}`",
        "- Held-out test: not used.",
        "- Global selection score is recomputed after merging all candidate grid rows.",
        "- Selection score: `0.45 * mean_excess_rank + 0.25 * worst_day_rank + 0.20 * hit_rate_rank + 0.10 * rank_ic_rank`.",
        "",
        "## Best Candidate Per Family Shape",
        "",
    ]
    lines.extend(markdown_table(leaderboard, columns, n=30))
    lines.extend(["", "## Global Top Configs", ""])
    lines.extend(markdown_table(ranked, columns, n=40))
    lines.extend(
        [
            "",
            "## Usage Notes",
            "",
            "- Treat this as portfolio-validation evidence, not final proof.",
            "- Prefer clusters of nearby strong configs over a single isolated winner.",
            "- Use `--candidates-from ... --top-n ... --stage fine` to refine broad-search leaders.",
            "- Do not use held-out test inside this script.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_global_best_configs(out_dir: Path, leaderboard: pd.DataFrame) -> None:
    for idx, row in leaderboard.iterrows():
        candidate_dir = out_dir / "candidates" / str(row["candidate"])
        ensemble_path = candidate_dir / "ensemble_config.json"
        if not ensemble_path.exists():
            continue
        base_config = load_json(ensemble_path)
        cfg = make_grid_config(
            base_config,
            float(row["model0_weight"]),
            int(row["top_k"]),
            float(row["internal_max_weight"]),
            bool(row["risk_filter_enabled"]),
            str(row["weighting"]),
            str(row["score_combination"]),
        )
        cfg["_global_selection_metadata"] = {
            "candidate": str(row["candidate"]),
            "selection_score": float(row["selection_score"]),
            "mean_portfolio_excess": float(row["mean_portfolio_excess"]),
            "cumulative_portfolio_excess": float(row["cumulative_portfolio_excess"]),
            "hit_rate": float(row["hit_rate"]),
            "worst_day": float(row["worst_day"]),
            "mean_rank_ic": float(row["mean_rank_ic"]),
        }
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "global_best_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
        )
        if idx == 0:
            (out_dir / "best_global_config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DATA_DIR / "final_training_matrix.parquet"))
    parser.add_argument("--feature-list", default=str(DATA_DIR / "final_feature_columns.txt"))
    parser.add_argument("--target", default="target_excess_5d")
    parser.add_argument("--split", choices=["portfolio_validation", "model_validation"], default="portfolio_validation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--stage", choices=["broad", "fine"], default="broad")
    parser.add_argument("--families", default="balanced,stable,recent,economic,short_aggressive")
    parser.add_argument("--count-pairs", default="compact,medium,wide,extra_wide")
    parser.add_argument("--model-presets", default="conservative,current,capacity")
    parser.add_argument("--run-feature-selection", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--candidates-from", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=2601)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--top-k", default=None)
    parser.add_argument("--internal-caps", default=None)
    parser.add_argument("--risk-filter", default=None)
    parser.add_argument("--weighting", default=None)
    parser.add_argument("--score-combination", default=None)
    args = parser.parse_args()

    families = parse_list(args.families)
    count_pairs = parse_list(args.count_pairs)
    model_presets = parse_list(args.model_presets)
    for family in families:
        if family not in FAMILY_NAMES:
            raise ValueError(f"Unknown family {family}; choose from {FAMILY_NAMES}")
    for count_pair in count_pairs:
        if count_pair not in COUNT_PAIRS:
            raise ValueError(f"Unknown count pair {count_pair}; choose from {sorted(COUNT_PAIRS)}")
    for preset in model_presets:
        if preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset {preset}; choose from {MODEL_PRESETS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(args.matrix)

    if args.run_feature_selection:
        for family in families:
            for profile in ("general", "short_history"):
                run_feature_selection_family(
                    profile=profile,
                    family=family,
                    out_dir=out_dir,
                    matrix=args.matrix,
                    feature_list=args.feature_list,
                    skip_existing=args.skip_existing and not args.force,
                )

    candidates = enumerate_candidates(families, count_pairs, model_presets)
    candidates = restrict_candidates(
        candidates,
        candidates_from=args.candidates_from,
        top_n=args.top_n,
        max_candidates=args.max_candidates,
        random_state=args.random_state,
    )
    if not candidates:
        raise ValueError("No candidates selected")

    matrix = pd.read_parquet(matrix_path)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["stock_code"] = matrix["stock_code"].astype(str).str.zfill(6)
    grid = grid_for_stage(args.stage, args)

    all_rows = []
    for candidate in candidates:
        print(f"\n[candidate] {candidate.name}")
        ensemble_path = train_candidate(
            candidate=candidate,
            matrix_path=matrix_path,
            out_dir=out_dir,
            as_of=args.as_of,
            skip_existing=args.skip_existing,
            force=args.force,
        )
        rows = evaluate_candidate_grid(
            candidate=candidate,
            ensemble_path=ensemble_path,
            matrix=matrix,
            target=args.target,
            split=args.split,
            grid=grid,
            out_dir=out_dir,
            skip_existing=args.skip_existing,
            force=args.force,
        )
        all_rows.append(rows.drop(columns=["selection_score"], errors="ignore"))

    combined = pd.concat(all_rows, ignore_index=True)
    ranked = add_selection_score(combined)
    leaderboard = best_per_candidate(ranked)
    ranked.to_csv(out_dir / "all_grid_results.csv", index=False)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    write_global_best_configs(out_dir, leaderboard)
    write_report(out_dir, args, candidates, ranked, leaderboard)

    manifest = {
        "stage": args.stage,
        "target": args.target,
        "split": args.split,
        "families": families,
        "count_pairs": count_pairs,
        "model_presets": model_presets,
        "candidate_count": len(candidates),
        "grid": grid,
        "heldout_test_used": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print("\nResearch-family leaderboard:")
    print(leaderboard.head(20).to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

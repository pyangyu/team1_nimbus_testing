#!/usr/bin/env python3
"""Build accuracy-versus-compute curves and paired method contrasts.

Each run input is one dataset's canonical JSONL result file (the sweep's
``jobs.jsonl`` or BudgetThinker's normalized ``results.jsonl``). A JSON config
assigns readable method names and declares the policy contrasts to report.

Example config::

    {
      "runs": {
        "base": "results/base/aime2026/jobs.jsonl",
        "ours": "results/ours/aime2026/jobs.jsonl",
        "budgetthinker": "results/budgetthinker-aime-512/results.jsonl"
      },
      "contrasts": [
        {
          "name": "trained budget-cut token vs base silent",
          "left": {"run": "ours", "policy": "budget_cut_token", "cut_point": 0.5, "cut_size": 0.25},
          "right": {"run": "base", "policy": "silent", "cut_point": 0.5, "cut_size": 0.25}
        },
        {
          "name": "base fixed budget vs BudgetThinker",
          "left": {"run": "base", "policy": "static_budget", "budget_tokens": 512},
          "right": {"run": "budgetthinker", "policy": "fixed_budget", "budget_tokens": 512}
        }
      ]
    }

Run the example after replacing its paths with completed runs:

    python gpu/compare_method_runs.py --config gpu/comparison_config.example.json --out results/comparison/aime2026

This is an offline model-behavior analysis. It does not measure serving
throughput, latency, or SLO attainment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from budgetthinker_adapter import read_jsonl, write_jsonl


def _load_runs(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Load result files and require each run to represent one dataset/manifest."""
    runs = {}
    for name, raw_path in config["runs"].items():
        rows = read_jsonl(Path(raw_path))
        if not rows:
            raise ValueError(f"run {name!r} contains no rows: {raw_path}")
        datasets = {row["dataset"] for row in rows}
        fingerprints = {row.get("manifest_fingerprint") for row in rows}
        if len(datasets) != 1 or len(fingerprints) != 1 or None in fingerprints:
            raise ValueError(f"run {name!r} mixes datasets/manifests or lacks a manifest fingerprint")
        runs[name] = rows
    return runs


def _selector_matches(row: dict[str, Any], selector: dict[str, Any]) -> bool:
    """Match a row against a policy plus any supplied cut or budget coordinates."""
    for field in ("policy", "cut_point", "cut_size", "budget_tokens"):
        if field not in selector:
            continue
        expected = selector[field]
        observed = row.get(field)
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            if observed is None or not math.isclose(float(observed), float(expected), abs_tol=1e-8):
                return False
        elif observed != expected:
            return False
    return True


def _select_arm(run_rows: list[dict[str, Any]], selector: dict[str, Any], label: str) -> dict[str, dict]:
    """Return selected rows indexed by sample ID, rejecting duplicate rows."""
    selected = {}
    for row in run_rows:
        if not _selector_matches(row, selector):
            continue
        sample_id = row.get("sample_id", row.get("sid"))
        if sample_id in selected:
            raise ValueError(f"contrast arm {label!r} has duplicate sample ID {sample_id}")
        selected[sample_id] = row
    if not selected:
        raise ValueError(f"contrast arm {label!r} selected no rows")
    return selected


def _cluster_ci(rows: list[dict[str, Any]], value_field: str, seed: int,
                iterations: int) -> tuple[float, float, float]:
    """Bootstrap a mean by resampling question IDs and retaining all their samples."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(float(row[value_field]))
    question_ids = sorted(grouped)
    values = [value for group in grouped.values() for value in group]
    estimate = statistics.mean(values)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(iterations):
        draw = [question_ids[rng.randrange(len(question_ids))] for _ in question_ids]
        sample = [value for question_id in draw for value in grouped[question_id]]
        bootstrap.append(statistics.mean(sample))
    bootstrap.sort()
    return estimate, bootstrap[int(0.025 * iterations)], bootstrap[min(iterations - 1, int(0.975 * iterations))]


def _summarize_arm(rows: list[dict[str, Any]], seed: int,
                   iterations: int) -> dict[str, Any]:
    """Summarize accuracy CI, actual token cost, and lexical behavior metrics."""
    accuracy, lower, upper = _cluster_ci(rows, "correct", seed, iterations)
    return {
        "n_samples": len(rows),
        "n_questions": len({row["question_id"] for row in rows}),
        "accuracy_pct": round(100 * accuracy, 2),
        "accuracy_ci95_pct": [round(100 * lower, 2), round(100 * upper, 2)],
        "mean_thinking_tokens": round(statistics.mean(row["thinking_tokens"] for row in rows), 2),
        "mean_answer_tokens": round(statistics.mean(row["answer_tokens"] for row in rows), 2),
        "mean_answer_cap": round(statistics.mean(row.get("answer_cap", 0) for row in rows), 2),
        "mean_total_tokens": round(statistics.mean(row["total_tokens"] for row in rows), 2),
        "reasoning_leakage_pct": round(100 * statistics.mean(
            bool(row["reasoning_leakage"]) for row in rows), 2),
        "mean_rechecks": round(statistics.mean(row["recheck_count"] for row in rows), 3),
        "mean_approach_switches": round(statistics.mean(
            row["approach_switch_count"] for row in rows), 3),
        "latency_measured": all(row.get("latency_seconds") is not None for row in rows),
    }


def _build_accuracy_curve(runs: dict[str, list[dict[str, Any]]],
                          iterations: int, seed: int) -> list[dict[str, Any]]:
    """Produce one accuracy-vs-actual-token point per run and experimental arm."""
    curve = []
    for run_name, rows in runs.items():
        groups = defaultdict(list)
        for row in rows:
            coordinates = (row.get("policy", row.get("arm")), row.get("cut_point"),
                           row.get("cut_size"), row.get("budget_tokens"))
            groups[coordinates].append(row)
        for index, (coordinates, arm_rows) in enumerate(sorted(groups.items(), key=str)):
            policy, cut_point, cut_size, budget_tokens = coordinates
            summary = _summarize_arm(arm_rows, seed + index, iterations)
            curve.append({
                "run": run_name, "policy": policy, "cut_point": cut_point,
                "cut_size": cut_size, "budget_tokens": budget_tokens, **summary,
            })
    return curve


def compare_methods(config: dict[str, Any], output_dir: Path,
                    iterations: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Write a combined accuracy/token curve and configured paired contrasts.

    Contrasts require exact sample-ID, question-ID, dataset, and manifest
    fingerprint alignment. This prevents a misleading paired claim when runs
    used different question text or gold answers.
    """
    runs = _load_runs(config)
    fingerprints = {rows[0]["manifest_fingerprint"] for rows in runs.values()}
    datasets = {rows[0]["dataset"] for rows in runs.values()}
    if len(fingerprints) != 1 or len(datasets) != 1:
        raise ValueError("all runs in one comparison must share dataset and manifest fingerprint")

    curve = _build_accuracy_curve(runs, iterations, seed)
    contrasts = []
    for index, contrast in enumerate(config.get("contrasts", [])):
        left_name, right_name = contrast["left"]["run"], contrast["right"]["run"]
        left = _select_arm(runs[left_name], contrast["left"], left_name)
        right = _select_arm(runs[right_name], contrast["right"], right_name)
        if left.keys() != right.keys():
            missing_left = sorted(right.keys() - left.keys())
            missing_right = sorted(left.keys() - right.keys())
            raise ValueError(
                f"contrast {contrast['name']!r} is unpaired; "
                f"missing left={missing_left[:5]}, missing right={missing_right[:5]}"
            )
        paired_rows = []
        for sample_id in sorted(left):
            left_row, right_row = left[sample_id], right[sample_id]
            if left_row["question_id"] != right_row["question_id"]:
                raise ValueError(f"question ID mismatch for sample {sample_id}")
            paired_rows.append({
                "question_id": left_row["question_id"],
                "sample_id": sample_id,
                "left_correct": bool(left_row["correct"]),
                "right_correct": bool(right_row["correct"]),
                "accuracy_difference": int(bool(left_row["correct"])) -
                                       int(bool(right_row["correct"])),
                "total_token_difference": int(left_row["total_tokens"]) -
                                          int(right_row["total_tokens"]),
            })
        difference, lower, upper = _cluster_ci(
            paired_rows, "accuracy_difference", seed + index, iterations)
        token_difference = statistics.mean(row["total_token_difference"] for row in paired_rows)
        contrasts.append({
            "name": contrast["name"],
            "left_run": left_name, "left_policy": contrast["left"]["policy"],
            "right_run": right_name, "right_policy": contrast["right"]["policy"],
            "n_paired_samples": len(paired_rows),
            "accuracy_difference_pp": round(100 * difference, 2),
            "accuracy_difference_ci95_pp": [round(100 * lower, 2), round(100 * upper, 2)],
            "mean_total_token_difference": round(token_difference, 2),
        })

    summary = {
        "dataset": next(iter(datasets)),
        "manifest_fingerprint": next(iter(fingerprints)),
        "accuracy_vs_actual_tokens": curve,
        "paired_contrasts": contrasts,
        "throughput_note": "This model-behavior analysis does not measure serving throughput, latency, or SLOs.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "accuracy_vs_actual_tokens.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8")
    with (output_dir / "accuracy_vs_actual_tokens.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        columns = ["run", "policy", "cut_point", "cut_size", "budget_tokens",
                   "n_samples", "n_questions", "accuracy_pct", "accuracy_ci95_pct",
                   "mean_thinking_tokens", "mean_answer_tokens", "mean_answer_cap",
                   "mean_total_tokens",
                   "reasoning_leakage_pct", "mean_rechecks", "mean_approach_switches"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in curve:
            writer.writerow({column: row.get(column) for column in columns})
    write_jsonl(output_dir / "paired_contrasts.jsonl", contrasts)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Load a comparison config and write the combined report files."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON mapping run names to result files and defining contrasts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bootstrap_iterations < 100:
        parser.error("--bootstrap-iterations must be at least 100")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = compare_methods(config, args.out, args.bootstrap_iterations, args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pair Qwen static-budget sweep rows with normalized BudgetThinker outputs.

This comparison is for fixed-budget inference, not mid-generation cuts. It
requires the same manifest sample IDs and the same absolute thinking cap.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from budgetthinker_adapter import read_jsonl, write_jsonl


def _select_budget_rows(rows: list[dict[str, Any]], policy: str, budget: int) -> dict[str, dict]:
    """Index one policy's rows by stable sample ID and reject duplicate records."""
    selected = {}
    for row in rows:
        if row.get("policy") != policy or row.get("budget_tokens") != budget:
            continue
        sample_id = row["sample_id"]
        if sample_id in selected:
            raise ValueError(f"duplicate result row for {policy}: {sample_id}")
        selected[sample_id] = row
    return selected


def _cluster_bootstrap(differences: list[dict], iterations: int, seed: int) -> tuple[float, float, float]:
    """Compute a paired mean difference CI by resampling whole questions."""
    grouped = defaultdict(list)
    for row in differences:
        grouped[row["question_id"]].append(float(row["difference"]))
    question_ids = sorted(grouped)
    if not question_ids:
        raise ValueError("no paired question samples are available")

    observed = statistics.mean(value for values in grouped.values() for value in values)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled_ids = [question_ids[rng.randrange(len(question_ids))]
                       for _ in question_ids]
        values = [value for question_id in sampled_ids for value in grouped[question_id]]
        estimates.append(statistics.mean(values))
    estimates.sort()
    lower = estimates[int(0.025 * iterations)]
    upper = estimates[min(iterations - 1, int(0.975 * iterations))]
    return observed, lower, upper


def _method_summary(rows: list[dict]) -> dict[str, Any]:
    """Summarize accuracy, token use, and lexical metrics for one aligned method."""
    if not rows:
        raise ValueError("cannot summarize an empty method result set")
    return {
        "n_samples": len(rows),
        "n_questions": len({row["question_id"] for row in rows}),
        "accuracy_pct": round(100 * statistics.mean(bool(row["correct"]) for row in rows), 2),
        "mean_thinking_tokens": round(statistics.mean(row["thinking_tokens"] for row in rows), 2),
        "mean_answer_tokens": round(statistics.mean(row["answer_tokens"] for row in rows), 2),
        "mean_total_tokens": round(statistics.mean(row["total_tokens"] for row in rows), 2),
        "reasoning_leakage_pct": round(100 * statistics.mean(
            bool(row["reasoning_leakage"]) for row in rows), 2),
        "mean_rechecks": round(statistics.mean(row["recheck_count"] for row in rows), 3),
        "mean_approach_switches": round(statistics.mean(
            row["approach_switch_count"] for row in rows), 3),
        "latency_measured": all(row.get("latency_seconds") is not None for row in rows),
    }


def compare_fixed_budget_runs(
    qwen_jobs_path: Path,
    budgetthinker_results_path: Path,
    budget_tokens: int,
    output_dir: Path,
    iterations: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pair rows from two independent runs and write a confidence-interval report.

    The function requires exact sample-ID equality for the chosen budget. It
    deliberately refuses an unpaired comparison rather than silently dropping
    missing questions or samples.
    """
    qwen_rows = _select_budget_rows(read_jsonl(qwen_jobs_path), "static_budget", budget_tokens)
    budgetthinker_rows = _select_budget_rows(
        read_jsonl(budgetthinker_results_path), "fixed_budget", budget_tokens)
    if not qwen_rows or not budgetthinker_rows:
        raise ValueError(f"no rows found for fixed budget {budget_tokens}")
    if qwen_rows.keys() != budgetthinker_rows.keys():
        missing_qwen = sorted(budgetthinker_rows.keys() - qwen_rows.keys())
        missing_budgetthinker = sorted(qwen_rows.keys() - budgetthinker_rows.keys())
        raise ValueError(
            "sample IDs do not match; "
            f"missing Qwen={missing_qwen[:5]}, missing BudgetThinker={missing_budgetthinker[:5]}"
        )
    qwen_example = next(iter(qwen_rows.values()))
    budgetthinker_example = next(iter(budgetthinker_rows.values()))
    if qwen_example["dataset"] != budgetthinker_example["dataset"]:
        raise ValueError("Qwen and BudgetThinker results come from different datasets")
    if qwen_example.get("manifest_fingerprint") != budgetthinker_example.get("manifest_fingerprint"):
        raise ValueError("Qwen and BudgetThinker results do not share the same manifest")

    paired = []
    for sample_id in sorted(qwen_rows):
        qwen = qwen_rows[sample_id]
        budgetthinker = budgetthinker_rows[sample_id]
        if qwen["question_id"] != budgetthinker["question_id"]:
            raise ValueError(f"question ID mismatch for sample {sample_id}")
        paired.append({
            "question_id": qwen["question_id"],
            "sample_id": sample_id,
            "qwen_correct": bool(qwen["correct"]),
            "budgetthinker_correct": bool(budgetthinker["correct"]),
            "difference": int(bool(qwen["correct"])) - int(bool(budgetthinker["correct"])),
        })

    difference, lower, upper = _cluster_bootstrap(paired, iterations, seed)
    summary = {
        "dataset": qwen_example["dataset"],
        "manifest_fingerprint": qwen_example["manifest_fingerprint"],
        "budget_tokens": budget_tokens,
        "comparison": "Qwen static-budget minus BudgetThinker fixed-budget",
        "qwen": _method_summary(list(qwen_rows.values())),
        "budgetthinker": _method_summary(list(budgetthinker_rows.values())),
        "paired_accuracy_difference_pp": round(100 * difference, 2),
        "paired_accuracy_difference_ci95_pp": [round(100 * lower, 2), round(100 * upper, 2)],
        "paired_sample_count": len(paired),
        "bootstrap_unit": "question; all samples for one question are resampled together",
        "throughput_note": "Request latency and throughput are not measured by this offline evaluator.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "paired_samples.jsonl", paired)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-jobs", type=Path, required=True)
    parser.add_argument("--budgetthinker-results", type=Path, required=True)
    parser.add_argument("--budget-tokens", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bootstrap_iterations < 100:
        parser.error("--bootstrap-iterations must be at least 100")
    summary = compare_fixed_budget_runs(
        args.qwen_jobs, args.budgetthinker_results, args.budget_tokens,
        args.out, args.bootstrap_iterations, args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

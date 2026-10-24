#!/usr/bin/env python3
"""Run the unmodified BudgetThinker evaluator against a shared Lifeline manifest.

This adapter creates the local JSONL format expected by BudgeThinker's
``evaluation/data_loader.py``, invokes its official evaluation entry point in
an isolated process, then normalizes its per-question samples for paired
analysis. It does not patch or edit the BudgetThinker checkout.

The official evaluator requires the repository's custom vLLM 0.7.3 setup and
``<remaining>`` token support. Run this script with that environment's Python,
not the current Qwen3 sweep environment.

Example (run from the repository root, inside the prepared BudgetThinker env):
    python gpu/budgetthinker_adapter.py --manifest results/budget_drop_sweep/aime2026/manifest.json --budgetthinker-repo BudgeThinker --model Xin-Rui/BudgetThinker-1.5b --budget-tokens 2048 --cuda-devices 0 --out results/comparison/budgetthinker
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import thinking_budget_bench as TB
from budget_drop_sweep import behavior_metrics
from evaluation_contract import manifest_fingerprint


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all complete JSONL rows from a BudgetThinker output file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write normalized JSONL rows, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the shared evaluation manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "dataset", "samples_per_problem", "questions", "samples"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing required fields: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError(f"unsupported manifest schema version: {manifest['schema_version']}")
    return manifest


def prepare_budgetthinker_dataset(
    manifest_path: Path,
    data_root: Path,
    data_name: str = "math500",
) -> Path:
    """Create BudgetThinker's local ``data/<name>/test.jsonl`` input file.

    The official parser understands ``math500`` records with ``question``,
    ``gt_cot``, and ``gt`` fields. Each manifest question is represented once;
    ``n_sampling`` in the official runner creates its sample replications.
    A sidecar index preserves the original Lifeline question IDs.

    Args:
        manifest_path: Shared manifest created by the sweep.
        data_root: Root passed to BudgetThinker's ``--data_dir`` option.
        data_name: Dataset parser key; use ``math500`` for math-style answers.
    Returns:
        Path to the prepared ``test.jsonl`` file.
    """
    manifest = load_manifest(manifest_path)
    questions = manifest["questions"]
    if not questions:
        raise ValueError("manifest has no selected questions")

    rows = []
    index_map = []
    for question in questions:
        gold = str(question["gold"])
        boxed_gold = gold if "\\boxed" in gold else f"\\boxed{{{gold}}}"
        rows.append({
            "idx": int(question["question_index"]),
            "question": question["question"],
            "gt_cot": boxed_gold,
            "gt": gold,
        })
        index_map.append({
            "idx": int(question["question_index"]),
            "question_id": question["question_id"],
        })

    test_path = data_root / data_name / "test.jsonl"
    write_jsonl(test_path, rows)
    write_jsonl(data_root / data_name / "lifeline_index_map.jsonl", index_map)
    return test_path


def _as_list(value: Any) -> list[Any]:
    """Normalize evaluator fields that may be scalar for one sample or lists."""
    return value if isinstance(value, list) else [value]


def _answer_segments(response: str) -> tuple[str, str]:
    """Split a completion at its final boxed answer for heuristic behavior metrics."""
    answer_start = response.rfind("\\boxed")
    if answer_start < 0:
        return response, ""
    return response[:answer_start], response[answer_start:]


def normalize_budgetthinker_outputs(
    manifest_path: Path,
    evaluator_output: Path,
    normalized_output: Path,
    tokenizer,
    budget_tokens: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """Explode official per-question sample lists into paired result rows.

    The official evaluator stores ``code`` and ``pred`` arrays per question.
    Array position is mapped to the manifest's sample index, yielding the same
    ``question_id#sample=N`` identity used by the Lifeline sweep.
    """
    manifest = load_manifest(manifest_path)
    questions_by_idx = {int(question["question_index"]): question
                        for question in manifest["questions"]}
    sample_count = int(manifest["samples_per_problem"])
    manifest_hash = manifest_fingerprint(manifest)
    source_rows = read_jsonl(evaluator_output)
    normalized = []

    for source in source_rows:
        idx = int(source["idx"])
        if idx not in questions_by_idx:
            raise ValueError(f"BudgetThinker output contains unknown question index {idx}")
        question = questions_by_idx[idx]
        responses = _as_list(source.get("code", []))
        predictions = _as_list(source.get("pred", []))
        if len(responses) != sample_count or len(predictions) != sample_count:
            raise ValueError(
                f"question {question['question_id']} returned {len(responses)} responses and "
                f"{len(predictions)} predictions; expected {sample_count}"
            )

        for sample_index, (response_value, prediction_value) in enumerate(zip(responses, predictions)):
            response = str(response_value or "")
            prediction = str(prediction_value or "")
            graded_answer = (prediction if "\\boxed" in prediction
                             else f"\\boxed{{{prediction}}}")
            reasoning_text, answer_text = _answer_segments(response)
            metrics = TB.grade(manifest["dataset"], graded_answer, question["gold"])
            behavior = behavior_metrics(reasoning_text, answer_text)
            thinking_tokens = len(tokenizer.encode(reasoning_text, add_special_tokens=False))
            answer_tokens = len(tokenizer.encode(answer_text, add_special_tokens=False))
            sample_id = f"{question['question_id']}#sample={sample_index}"
            normalized.append({
                "schema_version": 1,
                "dataset": manifest["dataset"],
                "manifest_fingerprint": manifest_hash,
                "method": "budgetthinker",
                "model": model_name,
                "policy": "fixed_budget",
                "budget_tokens": budget_tokens,
                "question_id": question["question_id"],
                "sample_id": sample_id,
                "sample_index": sample_index,
                "question_index": idx,
                "correct": metrics,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "total_tokens": thinking_tokens + answer_tokens,
                "reasoning_leakage": behavior["reasoning_leakage"],
                "recheck_count": behavior["recheck_count"],
                "approach_switch_count": behavior["approach_switch_count"],
                "behavior_evidence": behavior["behavior_evidence"],
                "prediction": prediction,
                "response": response,
                "latency_seconds": None,
            })

    write_jsonl(normalized_output, normalized)
    return normalized


def _official_output_path(
    output_dir: Path,
    model_path: str,
    data_name: str,
    split: str,
    sample_count: int,
    seed: int,
    temperature: float,
    budget_tokens: int,
) -> Path:
    """Reconstruct the official evaluator's output filename convention."""
    model_name = model_path.rstrip("/").split("/")[-1]
    prefix = f"{split}_deepseek3_{-1}_seed{seed}_t{temperature}"
    filename = f"{prefix}_s0_e-1_b{budget_tokens}_original.jsonl"
    return output_dir / model_name / "deepseek3" / data_name / filename


def _require_official_vllm(evaluation_dir: Path) -> None:
    """Fail early unless this interpreter uses the vLLM version in upstream docs."""
    version_probe = subprocess.run(
        [sys.executable, "-c", "import vllm; print(vllm.__version__)"],
        cwd=evaluation_dir, capture_output=True, text=True)
    if version_probe.returncode != 0:
        raise RuntimeError(
            "Could not import vLLM in this Python environment. Activate the "
            "BudgetThinker evaluation environment before running this adapter.\n"
            f"{version_probe.stderr.strip()}"
        )
    version = version_probe.stdout.strip().splitlines()[-1]
    if version != "0.7.3":
        raise RuntimeError(
            f"BudgetThinker's patched evaluator requires vLLM 0.7.3; found {version}. "
            "Do not run it in the Qwen3 sweep's vLLM environment."
        )
    source_probe = subprocess.run(
        [sys.executable, "-c",
         "import inspect, vllm.worker.model_runner as m; print(inspect.getsourcefile(m))"],
        cwd=evaluation_dir, capture_output=True, text=True)
    if source_probe.returncode != 0:
        raise RuntimeError(f"Could not inspect vLLM ModelRunner: {source_probe.stderr.strip()}")
    model_runner_path = Path(source_probe.stdout.strip().splitlines()[-1])
    source = model_runner_path.read_text(encoding="utf-8")
    if "151665" not in source or "remaining" not in source:
        raise RuntimeError(
            f"vLLM 0.7.3 is installed but its ModelRunner lacks the BudgetThinker "
            f"remaining-token patch: {model_runner_path}. Apply BudgeThinker/to_replace first."
        )


def run_budgetthinker(args: argparse.Namespace) -> Path:
    """Prepare shared questions, invoke the official evaluator, normalize output."""
    repo = args.budgetthinker_repo.resolve()
    evaluation_dir = repo / "evaluation"
    runner = evaluation_dir / "remaining_eval_multi_process.py"
    if not runner.is_file():
        raise FileNotFoundError(f"BudgetThinker evaluator not found: {runner}")
    _require_official_vllm(evaluation_dir)

    out_dir = args.out.resolve()
    data_root = out_dir / "input_data"
    prepare_budgetthinker_dataset(args.manifest.resolve(), data_root, args.data_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The upstream evaluator requires these environment variables at import time.
    env = os.environ.copy()
    run_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", out_dir.name)
    env.update({
        "CUDA_VISIBLE_DEVICES": args.cuda_devices,
        "DATA_NAME": args.data_name,
        "remaining": "8ratio",
        "tip": "8ratio",
        "stage": "2",
        "BUDGET": str(args.budget_tokens),
        "model": str(out_dir / "logs"),
        "modelname": str(out_dir / "metadata"),
        "mode": f"lifeline-budgetthinker-{run_tag}",
    })
    Path(env["model"]).mkdir(parents=True, exist_ok=True)
    Path(env["modelname"]).mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(runner),
        "--model_name_or_path", args.model,
        "--data_names", args.data_name,
        "--data_dir", str(data_root),
        "--output_dir", str(out_dir / "official_output"),
        "--split", args.split,
        "--prompt_type", "deepseek3",
        "--num_test_sample", "-1",
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
        "--n_sampling", str(args.samples),
        "--top_p", str(args.top_p),
        "--start", "0",
        "--end", "-1",
        "--use_safetensors",
        "--save_outputs",
        "--use_vllm",
        "--overwrite",
        "--max_tokens_per_call", str(args.budget_tokens),
    ]
    subprocess.run(command, cwd=evaluation_dir, env=env, check=True)

    raw_output = _official_output_path(
        out_dir / "official_output", args.model, args.data_name, args.split,
        args.samples, args.seed, args.temperature, args.budget_tokens)
    if not raw_output.is_file():
        raise FileNotFoundError(f"BudgetThinker output not found: {raw_output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    normalized_path = out_dir / "results.jsonl"
    normalize_budgetthinker_outputs(
        args.manifest.resolve(), raw_output, normalized_path, tokenizer,
        args.budget_tokens, args.model)
    print(f"Normalized BudgetThinker results: {normalized_path}")
    return normalized_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse BudgetThinker adapter options for an isolated official run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budgetthinker-repo", type=Path, default=Path("BudgeThinker"))
    parser.add_argument("--model", required=True,
                        help="BudgetThinker checkpoint path or Hugging Face model ID")
    parser.add_argument("--budget-tokens", type=int, required=True)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--data-name", default="math500",
                        help="BudgetThinker parser key; use math500 for math-style answers")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run BudgetThinker's official inference and write normalized result rows."""
    args = parse_args(argv)
    if args.budget_tokens < 1:
        raise ValueError("budget_tokens must be positive")
    run_budgetthinker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

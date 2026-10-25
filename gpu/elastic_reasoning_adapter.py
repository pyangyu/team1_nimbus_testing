#!/usr/bin/env python3
"""Evaluate an Elastic-Reasoning checkpoint on a shared Lifeline manifest.

This adapter leaves the Elastic-Reasoning submodule untouched. It prepares the
Parquet schema expected by ``verl.trainer.main_generation``, runs the upstream
E1 evaluator in the Elastic environment, and converts its responses to the
shared per-question/sample result schema.

Elastic E1 fixes separate thinking and solution budgets at generation start.
It is therefore a fixed-budget baseline, not a mid-generation budget-drop
method. The answer/solution budget is recorded separately from thinking tokens.

Run with the Python environment installed from Elastic-Reasoning/verl. Example
from the Lifeline repository root:

    python gpu/elastic_reasoning_adapter.py \\
      --manifest results/budget_drop_sweep/aime2026/manifest.json \\
      --elastic-repo Elastic-Reasoning --model Salesforce/E1-Math-1.5B \\
      --thinking-budget 512 --solution-budget 20 --cuda-devices 0 \\
      --out results/elastic-aime-512
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import thinking_budget_bench as TB
from budget_drop_sweep import behavior_metrics
from evaluation_contract import manifest_fingerprint


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a versioned evaluation manifest and validate its required fields."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "dataset", "samples_per_problem", "questions", "samples"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError(f"unsupported manifest schema version: {manifest['schema_version']}")
    return manifest


def make_elastic_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert frozen questions to Elastic's prompt/reward-model row format."""
    rows = []
    for question in manifest["questions"]:
        rows.append({
            "question_id": question["question_id"],
            "question_index": int(question["question_index"]),
            "prompt": [{"role": "user", "content": question["question"] + TB.INSTR}],
            "data_source": "lighteval/MATH",
            "reward_model": {"ground_truth": str(question["gold"])},
        })
    return rows


def write_elastic_parquet(manifest_path: Path, parquet_path: Path) -> Path:
    """Write the shared questions in the Parquet schema Elastic's evaluator expects.

    Args:
        manifest_path: Frozen question/sample manifest shared with other methods.
        parquet_path: Destination input file for ``data.path`` in main_generation.
    Returns:
        The destination Parquet path.
    """
    import pandas as pd

    manifest = load_manifest(manifest_path)
    rows = make_elastic_rows(manifest)
    if not rows:
        raise ValueError("manifest contains no questions")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return parquet_path


def _split_e1_response(response: str, thinking_end_token: str) -> tuple[str, str]:
    """Separate Elastic's thinking and solution spans at its E1 delimiter."""
    if thinking_end_token and thinking_end_token in response:
        thinking, solution = response.split(thinking_end_token, 1)
        return thinking, solution
    # Retain the response for scoring if a tokenizer hides or omits the delimiter.
    return response, ""


def _as_response_list(value: Any) -> list[str]:
    """Normalize Parquet list columns across pandas/Arrow versions."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Elastic response column is not a list of samples")
    return [str(response or "") for response in value]


def normalize_elastic_output(
    manifest_path: Path,
    parquet_path: Path,
    normalized_path: Path,
    model: str,
    thinking_budget: int,
    solution_budget: int,
    thinking_end_token_id: int = 151649,
) -> list[dict[str, Any]]:
    """Normalize Elastic's response arrays to paired Lifeline result rows."""
    import pandas as pd
    from transformers import AutoTokenizer

    manifest = load_manifest(manifest_path)
    questions = {question["question_id"]: question for question in manifest["questions"]}
    fingerprint = manifest_fingerprint(manifest)
    frame = pd.read_parquet(parquet_path)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    end_token = tokenizer.convert_ids_to_tokens(thinking_end_token_id)
    if not isinstance(end_token, str):
        end_token = ""

    normalized = []
    for _, source in frame.iterrows():
        question_id = str(source["question_id"])
        if question_id not in questions:
            raise ValueError(f"Elastic output has unknown question ID: {question_id}")
        responses = _as_response_list(source["responses"])
        if len(responses) != int(manifest["samples_per_problem"]):
            raise ValueError(
                f"Elastic returned {len(responses)} samples for {question_id}; "
                f"expected {manifest['samples_per_problem']}"
            )

        question = questions[question_id]
        for sample_index, response in enumerate(responses):
            thinking, solution = _split_e1_response(response, end_token)
            normalized.append({
                "schema_version": 1,
                "dataset": manifest["dataset"],
                "manifest_fingerprint": fingerprint,
                "method": "elastic_reasoning",
                "model": model,
                "policy": "elastic_e1_fixed",
                "budget_tokens": thinking_budget + solution_budget,
                "allocated_thinking_tokens": thinking_budget,
                "allocated_solution_tokens": solution_budget,
                "question_id": question_id,
                "sample_id": f"{question_id}#sample={sample_index}",
                "sample_index": sample_index,
                "question_index": int(source["question_index"]),
                "correct": TB.grade("math500", response, question["gold"]),
                "thinking_tokens": len(tokenizer.encode(thinking, add_special_tokens=False)),
                "answer_tokens": len(tokenizer.encode(solution, add_special_tokens=False)),
                "answer_cap": solution_budget,
                "total_tokens": len(tokenizer.encode(response, add_special_tokens=False)),
                **behavior_metrics(thinking, solution),
                "response": response,
                "latency_seconds": None,
            })

    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with normalized_path.open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return normalized


def run_elastic(args: argparse.Namespace) -> Path:
    """Prepare a matched Parquet input and invoke Elastic's official E1 evaluator."""
    repo = args.elastic_repo.resolve()
    verl_dir = repo / "verl"
    entrypoint = verl_dir / "verl" / "trainer" / "main_generation.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Elastic generation entry point not found: {entrypoint}")

    manifest = load_manifest(args.manifest.resolve())
    if args.samples != manifest["samples_per_problem"]:
        raise ValueError("--samples must match the shared manifest's samples_per_problem")
    out_dir = args.out.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = write_elastic_parquet(args.manifest.resolve(), out_dir / "input.parquet")
    output_path = out_dir / "elastic_responses.parquet"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    env["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    gpu_count = len([device for device in args.cuda_devices.split(",") if device.strip()])
    command = [
        sys.executable, "-m", "verl.trainer.main_generation",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={gpu_count}",
        f"data.path={input_path}",
        "data.prompt_key=prompt",
        "data.data_source_key=data_source",
        "data.reward_model_key=reward_model",
        f"data.n_samples={args.samples}",
        f"data.batch_size={args.batch_size}",
        f"data.output_path={output_path}",
        f"model.path={args.model}",
        "rollout.name=vllm",
        f"rollout.temperature={args.temperature}",
        f"rollout.top_p={args.top_p}",
        f"rollout.gpu_memory_utilization={args.gpu_util}",
        f"rollout.tensor_model_parallel_size={args.tensor_parallel_size}",
        f"rollout.response_length={args.thinking_budget + args.solution_budget + 128}",
        "rollout.e1_mode=True",
        f"rollout.e1_thinking_length={args.thinking_budget}",
        f"rollout.e1_solution_length={args.solution_budget}",
    ]
    subprocess.run(command, cwd=verl_dir, env=env, check=True)
    if not output_path.is_file():
        raise FileNotFoundError(f"Elastic output was not created: {output_path}")

    normalized_path = out_dir / "results.jsonl"
    normalize_elastic_output(
        args.manifest.resolve(), output_path, normalized_path, args.model,
        args.thinking_budget, args.solution_budget, args.thinking_end_token_id)
    print(f"Normalized Elastic results: {normalized_path}")
    return normalized_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse launcher, budget, and vLLM settings for the Elastic evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--elastic-repo", type=Path, default=Path("Elastic-Reasoning"))
    parser.add_argument("--model", required=True, help="Elastic checkpoint path or Hugging Face ID")
    parser.add_argument("--thinking-budget", type=int, required=True)
    parser.add_argument("--solution-budget", type=int, required=True)
    parser.add_argument("--thinking-end-token-id", type=int, default=151649)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-util", type=float, default=0.80)
    parser.add_argument("--attention-backend", default="XFORMERS")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.thinking_budget, args.solution_budget, args.samples,
           args.tensor_parallel_size, args.batch_size) < 1:
        parser.error("budgets, sample count, parallel size, and batch size must be positive")
    if args.temperature == 0 and args.samples != 1:
        parser.error("Elastic's greedy evaluation requires --samples 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run Elastic E1 inference and write common-schema results."""
    run_elastic(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

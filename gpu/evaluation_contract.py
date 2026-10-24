"""Shared manifest schema for paired budget-policy evaluations.

The manifest is model- and backend-independent. Each runner consumes the same
question IDs, question text, gold answers, and per-question sample IDs, while
remaining free to use its own prompt template and tokenizer.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


def build_evaluation_manifest(
    dataset: str,
    problems: list[dict[str, Any]],
    samples_per_problem: int,
    seed: int,
) -> dict[str, Any]:
    """Build a deterministic manifest shared by all inference backends.

    Args:
        dataset: Stable dataset label, such as ``aime2026`` or ``math500``.
        problems: Selected records containing ``id``, ``q``, and ``gold``.
        samples_per_problem: Number of generation samples assigned to each item.
        seed: Seed used to select the problem subset.
    Returns:
        JSON-serializable manifest with question records and sample identities.
    """
    if samples_per_problem < 1:
        raise ValueError("samples_per_problem must be positive")

    question_ids = [str(problem["id"]) for problem in problems]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("problem IDs in an evaluation manifest must be unique")

    questions = []
    samples = []
    for problem_index, problem in enumerate(problems):
        question_id = str(problem["id"])
        questions.append({
            "question_index": problem_index,
            "question_id": question_id,
            "question": str(problem["q"]),
            "gold": str(problem["gold"]),
        })
        for sample_index in range(samples_per_problem):
            samples.append({
                "sample_id": f"{question_id}#sample={sample_index}",
                "question_id": question_id,
                "sample_index": sample_index,
            })

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "seed": seed,
        "selection": "uniform without replacement, selected before generation",
        "samples_per_problem": samples_per_problem,
        "selected_problem_count": len(questions),
        "selected_problem_ids": question_ids,
        "questions": questions,
        "samples": samples,
    }


def problems_from_manifest(
    manifest: dict[str, Any],
    population: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Recover normalized question records from a previously frozen manifest.

    Reusing these question texts and IDs avoids re-sampling or silently
    changing dataset records when evaluating another checkpoint.
    """
    if manifest.get("schema_version") is None and population is not None:
        # Legacy manifests stored IDs but not question text or gold answers.
        question_ids = [str(question_id) for question_id in manifest.get("selected_problem_ids", [])]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("legacy manifest contains duplicate question IDs")
        by_id = {str(problem["id"]): problem for problem in population}
        missing_ids = [question_id for question_id in question_ids if question_id not in by_id]
        if missing_ids:
            raise ValueError(f"legacy manifest question IDs are missing from the dataset: {missing_ids[:5]}")
        return [{"id": question_id, "q": str(by_id[question_id]["q"]),
                 "gold": str(by_id[question_id]["gold"])} for question_id in question_ids]

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema version: {manifest.get('schema_version')}")
    questions = manifest.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("manifest must contain a non-empty questions list")

    problems = []
    seen_ids = set()
    for question in questions:
        required = ("question_id", "question", "gold")
        if any(field not in question for field in required):
            raise ValueError(f"manifest question is missing one of {required}")
        question_id = str(question["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"manifest contains duplicate question ID {question_id}")
        seen_ids.add(question_id)
        problems.append({"id": question_id, "q": str(question["question"]),
                         "gold": str(question["gold"])})

    if manifest.get("selected_problem_count") != len(problems):
        raise ValueError("manifest selected_problem_count does not match its question list")
    return problems


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for exact question/sample alignment."""
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

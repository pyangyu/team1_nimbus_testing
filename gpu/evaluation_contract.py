"""Shared manifest schema for paired budget-policy evaluations.

The manifest is model- and backend-independent. Each runner consumes the same
question IDs, question text, gold answers, and per-question sample IDs, while
remaining free to use its own prompt template and tokenizer.
"""
from __future__ import annotations

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

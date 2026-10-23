"""Fast tests for sweep helpers and the Phase B orchestration.

The tests use small synthetic records and fake model components. They check
data selection, grid construction, metrics, deterministic statistics, and
output writing without requiring vLLM or a GPU.
"""
import json
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

from budget_drop_sweep import (
    behavior_metrics,
    bootstrap_cluster_mean,
    make_arm_specs,
    run_drop_tests,
    stable_seed,
    summarize,
    uniform_sample,
)


class BudgetDropSweepTests(unittest.TestCase):
    """Behavioral checks for sampling, arm generation, metrics, and report output."""

    def test_uniform_selection_is_repeatable_and_without_replacement(self):
        """Selection must be deterministic for a seed and contain unique questions."""
        # The same seed must choose the same unbiased subset, with no duplicate problems.
        population = [{"id": str(index)} for index in range(20)]
        first = uniform_sample(population, 8, 42)
        second = uniform_sample(population, 8, 42)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len({row["id"] for row in first}), 8)

    def test_grid_contains_cut_silent_announced_and_resample_arms(self):
        """The grid must include the immediate, silent, resample, and both warning arms."""
        # Each cut point contributes one immediate-cut arm and four window-based arms.
        specs = make_arm_specs([0.25, 0.5], [0.1])
        arms = {spec["arm"] for spec in specs}
        self.assertEqual(arms, {"cut_now", "silent", "resample", "warn_soft", "warn_count"})
        self.assertEqual(len(specs), 2 * (1 + 4))

    def test_invalid_budget_grid_is_rejected(self):
        """Reject a grace window that would extend beyond the natural trace."""
        # A grace window cannot extend beyond the trace's full natural-thinking length.
        with self.assertRaises(ValueError):
            make_arm_specs([0.8], [0.25])

    def test_behavior_metrics_count_explicit_markers(self):
        """Check that known phrases trigger each documented lexical behavior metric."""
        # These are intentionally lexical checks, so each expected marker is explicit.
        metrics = behavior_metrics(
            "Let's check the answer. Another approach is to factor the expression.",
            "First, we need to calculate it. The answer is 42.")
        self.assertTrue(metrics["reasoning_leakage"])
        self.assertEqual(metrics["recheck_count"], 1)
        self.assertEqual(metrics["approach_switch_count"], 1)

    def test_cluster_bootstrap_is_deterministic_and_keeps_problem_groups(self):
        """Repeated bootstrap calls with the same seed must return identical estimates."""
        # Two samples from one problem stay together when problem-level bootstrapping runs.
        rows = [
            {"id": "a", "value": 1}, {"id": "a", "value": 0},
            {"id": "b", "value": 1}, {"id": "b", "value": 1},
        ]
        first = bootstrap_cluster_mean(rows, "value", 7, iterations=200)
        second = bootstrap_cluster_mean(rows, "value", 7, iterations=200)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.75)

    def test_arm_seed_is_stable(self):
        """The same identity has the same seed, while different arms get different seeds."""
        # Stable seeds let a resumed run reproduce a given sample/arm independently of process state.
        self.assertEqual(stable_seed(5, "sample-arm"), stable_seed(5, "sample-arm"))
        self.assertNotEqual(stable_seed(5, "sample-arm"), stable_seed(5, "other-arm"))

    def test_drop_pipeline_writes_every_requested_arm(self):
        """Exercise arm prompt construction through scoring and summary-file creation."""
        # Keep orchestration coverage local: these fakes avoid loading vLLM or a GPU.
        class FakeTokenizer:
            """Map characters to integer IDs so the test does not need a model tokenizer."""

            def encode(self, text, add_special_tokens=False):
                """Encode each character reversibly for the fake generation pipeline."""
                return [ord(char) + 10 for char in text]

            def decode(self, ids, skip_special_tokens=True):
                """Decode the reversible character IDs produced by encode()."""
                return "".join(chr(token - 10) for token in ids)

        class FakeEngine:
            """Mimic continuation and answer calls made by the real engine adapter."""

            def generate(self, prompts, max_tokens, seeds, stop_ids=None):
                """Return a short continuation or a fixed correct answer per prompt."""
                if stop_ids:
                    return [([300, 301], "stop") for _ in prompts]
                answer = "The answer is \\boxed{42}. First, we calculate it."
                ids = [ord(char) + 10 for char in answer]
                return [(ids, "stop") for _ in prompts]

        tokenizer = FakeTokenizer()
        t_open, t_close = 1, 2
        reasoning = [ord(char) + 10 for char in "We can solve this directly."]
        # This one synthetic trace is enough to exercise prefix slicing and all arm builders.
        record = {
            "sid": "toy#sample=0", "id": "toy", "sample": 0,
            "p": {"gold": "42"}, "prompt": [500],
            "gen": [t_open] + reasoning + [t_close], "start": 1,
            "L": len(reasoning), "capped": False,
        }
        args = SimpleNamespace(trace_batch_size=1, seed=3, hard_cut=False,
                               max_len=10000, answer_max=128)
        specs = make_arm_specs([0.25], [0.25])
        # Temporary output keeps the test isolated and exercises both JSONL and summary export.
        with TemporaryDirectory() as temp_dir:
            run_drop_tests([record], specs, FakeEngine(), tokenizer, t_open, t_close,
                           tokenizer.encode, args, Path(temp_dir))
            from budget_drop_sweep import read_jsonl
            rows = read_jsonl(Path(temp_dir) / "jobs.jsonl")
            out_dir = Path(temp_dir)
            config = {
                "dataset": "toy", "model": "fake", "selected_problem_count": 1,
                "samples": 1, "cut_points": [0.25], "cut_sizes": [0.25],
                "behavior_metrics_note": "heuristic", "resample_control_note": "fresh run",
            }
            (out_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (out_dir / "natural.jsonl").write_text(
                json.dumps({"id": "toy", "sid": "toy#sample=0", "correct": True}) + "\n",
                encoding="utf-8")
            summary = summarize(out_dir)
            csv_exists = (out_dir / "accuracy_vs_cut.csv").exists()

        # Every requested arm should be scored, then appear in the aggregate and paired reports.
        self.assertEqual({row["key"] for row in rows}, {spec["key"] for spec in specs})
        self.assertTrue(all(row["correct"] for row in rows))
        self.assertTrue(all(row["reasoning_leakage"] for row in rows))
        self.assertEqual(len(summary["accuracy_vs_cut"]), len(specs))
        self.assertEqual(len(summary["paired_accuracy_differences"]), 3)
        self.assertTrue(csv_exists)


if __name__ == "__main__":
    unittest.main()
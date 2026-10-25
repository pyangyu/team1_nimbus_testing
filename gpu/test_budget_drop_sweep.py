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
    parse_args,
    run_drop_tests,
    stable_seed,
    summarize,
    uniform_sample,
)
from budgetthinker_adapter import (
    normalize_budgetthinker_outputs,
    prepare_budgetthinker_dataset,
)
from compare_method_runs import compare_methods
from evaluation_contract import build_evaluation_manifest, problems_from_manifest
from elastic_reasoning_adapter import _as_response_list, make_elastic_rows


class BudgetDropSweepTests(unittest.TestCase):
    """Behavioral checks for sampling, arm generation, metrics, and report output."""

    def test_shared_manifest_contains_stable_problem_and_sample_ids(self):
        """All evaluation backends should receive the same per-question samples."""
        problems = [
            {"id": "q1", "q": "Question one", "gold": "11"},
            {"id": "q2", "q": "Question two", "gold": "22"},
        ]
        manifest = build_evaluation_manifest("toy", problems, 2, seed=9)

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["selected_problem_ids"], ["q1", "q2"])
        self.assertEqual(
            [sample["sample_id"] for sample in manifest["samples"]],
            ["q1#sample=0", "q1#sample=1", "q2#sample=0", "q2#sample=1"],
        )

    def test_shared_manifest_rejects_duplicate_problem_ids(self):
        """Duplicate IDs would make paired comparisons ambiguous, so reject them."""
        duplicate_problems = [
            {"id": "same", "q": "Question one", "gold": "11"},
            {"id": "same", "q": "Question two", "gold": "22"},
        ]
        with self.assertRaises(ValueError):
            build_evaluation_manifest("toy", duplicate_problems, 1, seed=9)

    def test_frozen_manifest_reconstructs_exact_questions(self):
        """Later model runs must use the stored text and IDs rather than re-sample."""
        problems = [{"id": "q1", "q": "Original question", "gold": "42"}]
        manifest = build_evaluation_manifest("toy", problems, 3, seed=11)
        self.assertEqual(problems_from_manifest(manifest), problems)

    def test_legacy_manifest_resolves_ids_from_population(self):
        """Older ID-only manifests remain reusable when their source data is available."""
        legacy = {"dataset": "toy", "selected_problem_ids": ["q2", "q1"]}
        population = [
            {"id": "q1", "q": "Question one", "gold": "11"},
            {"id": "q2", "q": "Question two", "gold": "22"},
        ]
        self.assertEqual(
            problems_from_manifest(legacy, population),
            [population[1], population[0]],
        )

    def test_budgetthinker_data_adapter_preserves_problem_ids(self):
        """Convert the shared question manifest to the official local JSONL format."""
        problems = [{"id": "q1", "q": "Question one", "gold": "42"}]
        manifest = build_evaluation_manifest("toy", problems, 2, seed=9)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            test_path = prepare_budgetthinker_dataset(manifest_path, root / "data")
            from budgetthinker_adapter import read_jsonl
            rows = read_jsonl(test_path)
            index_map = read_jsonl(root / "data" / "math500" / "lifeline_index_map.jsonl")

        self.assertEqual(rows[0]["idx"], 0)
        self.assertEqual(rows[0]["question"], "Question one")
        self.assertEqual(rows[0]["gt_cot"], "\\boxed{42}")
        self.assertEqual(index_map[0]["question_id"], "q1")

    def test_elastic_data_adapter_preserves_manifest_and_reward_fields(self):
        """Elastic receives the same question and gold answer in its Parquet row schema."""
        manifest = build_evaluation_manifest(
            "aime2026", [{"id": "q1", "q": "Question one", "gold": "42"}], 2, seed=4)
        rows = make_elastic_rows(manifest)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question_id"], "q1")
        self.assertEqual(rows[0]["prompt"][0]["role"], "user")
        self.assertIn("Question one", rows[0]["prompt"][0]["content"])
        self.assertEqual(rows[0]["reward_model"]["ground_truth"], "42")
        self.assertEqual(rows[0]["data_source"], "lighteval/MATH")

    def test_elastic_response_lists_accept_array_like_parquet_values(self):
        """Parquet list columns may load as array-like objects rather than Python lists."""
        class ArrayLike:
            def tolist(self):
                return ["reasoning one", "reasoning two"]

        self.assertEqual(_as_response_list(ArrayLike()), ["reasoning one", "reasoning two"])

    def test_budgetthinker_output_normalizes_to_paired_sample_rows(self):
        """Explode official per-question sample arrays into shared sample IDs."""
        manifest = build_evaluation_manifest(
            "math500", [{"id": "q1", "q": "Question one", "gold": "42"}], 2, seed=9)

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.json"
            output_path = root / "official.jsonl"
            normalized_path = root / "normalized.jsonl"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path.write_text(json.dumps({
                "idx": 0,
                "code": ["Let me verify the result. \\\\boxed{42}",
                         "Another approach fails. \\\\boxed{41}"],
                "pred": ["42", "41"],
            }) + "\n", encoding="utf-8")
            rows = normalize_budgetthinker_outputs(
                manifest_path, output_path, normalized_path, FakeTokenizer(), 1024,
                "fake-budgetthinker")

        self.assertEqual([row["sample_id"] for row in rows],
                         ["q1#sample=0", "q1#sample=1"])
        self.assertEqual([row["correct"] for row in rows], [True, False])
        self.assertEqual([row["answer_cap"] for row in rows], [20, 20])
        self.assertEqual(rows[0]["recheck_count"], 1)
        self.assertEqual(rows[1]["approach_switch_count"], 1)

    def test_method_comparison_builds_actual_token_curve_and_cross_arm_contrast(self):
        """A configured trained-announced vs base-silent contrast uses paired samples."""
        fingerprint = "frozen-test-manifest"

        def row(question_id, sample_index, policy, correct, total_tokens):
            return {
                "schema_version": 1, "dataset": "toy",
                "manifest_fingerprint": fingerprint,
                "method": "fake-model", "policy": policy,
                "question_id": question_id,
                "sample_id": f"{question_id}#sample={sample_index}",
                "sample_index": sample_index, "cut_point": 0.5,
                "cut_size": 0.25, "budget_tokens": None,
                "correct": correct, "thinking_tokens": total_tokens - 5,
                "answer_tokens": 5, "total_tokens": total_tokens,
                "reasoning_leakage": False, "recheck_count": 0,
                "approach_switch_count": 0, "latency_seconds": None,
            }

        base_rows = [row("q1", 0, "silent", True, 80),
                     row("q1", 1, "silent", False, 82),
                     row("q2", 0, "silent", False, 75),
                     row("q2", 1, "silent", False, 77)]
        trained_rows = [row("q1", 0, "budget_cut_token", True, 70),
                row("q1", 1, "budget_cut_token", True, 72),
                row("q2", 0, "budget_cut_token", False, 68),
                row("q2", 1, "budget_cut_token", True, 70)]

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path, trained_path = root / "base.jsonl", root / "trained.jsonl"
            base_path.write_text("".join(json.dumps(item) + "\n" for item in base_rows),
                                 encoding="utf-8")
            trained_path.write_text("".join(json.dumps(item) + "\n" for item in trained_rows),
                                    encoding="utf-8")
            config = {
                "runs": {"base": str(base_path), "trained": str(trained_path)},
                "contrasts": [{
                    "name": "trained budget-cut token vs base silent",
                    "left": {"run": "trained", "policy": "budget_cut_token",
                             "cut_point": 0.5, "cut_size": 0.25},
                    "right": {"run": "base", "policy": "silent",
                              "cut_point": 0.5, "cut_size": 0.25},
                }],
            }
            summary = compare_methods(config, root / "report", iterations=200, seed=3)

        self.assertEqual(len(summary["accuracy_vs_actual_tokens"]), 2)
        self.assertEqual(summary["paired_contrasts"][0]["n_paired_samples"], 4)
        self.assertEqual(summary["paired_contrasts"][0]["accuracy_difference_pp"], 50.0)
        self.assertEqual(summary["paired_contrasts"][0]["mean_total_token_difference"], -8.5)

    def test_method_comparison_rejects_different_manifest_fingerprints(self):
        """Runs must use identical question text/gold, not merely the same IDs."""
        base = {"schema_version": 1, "dataset": "toy", "policy": "silent",
                "cut_point": 0.5, "cut_size": 0.1, "budget_tokens": None,
                "question_id": "q1", "sample_id": "q1#sample=0",
                "correct": True, "thinking_tokens": 10, "answer_tokens": 2,
                "total_tokens": 12, "reasoning_leakage": False,
                "recheck_count": 0, "approach_switch_count": 0,
                "latency_seconds": None}
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for name, fingerprint in (("base", "manifest-a"), ("ours", "manifest-b")):
                path = root / f"{name}.jsonl"
                path.write_text(json.dumps({**base, "manifest_fingerprint": fingerprint}) + "\n",
                                encoding="utf-8")
                paths.append(path)
            config = {"runs": {"base": str(paths[0]), "ours": str(paths[1])}}
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                compare_methods(config, root / "report", iterations=200)

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
        self.assertEqual(arms, {"cut_now", "silent", "resample", "warn_soft",
                    "warn_count", "budget_cut_token"})
        self.assertEqual(len(specs), 2 * (1 + 5))

    def test_grid_adds_absolute_static_budget_arms(self):
        """Fixed-budget controls are distinct arms keyed by absolute token cap."""
        specs = make_arm_specs([0.5], [0.1], [512, 1024])
        fixed = [spec for spec in specs if spec["arm"] == "static_budget"]
        self.assertEqual([spec["budget_tokens"] for spec in fixed], [512, 1024])
        self.assertEqual([spec["key"] for spec in fixed],
                         ["static_budget|b=512", "static_budget|b=1024"])

    def test_fixed_budget_only_mode_omits_fractional_cut_grid(self):
        """A fixed-budget comparison can run without the fractional cut arms."""
        args = parse_args(["--fixed-budget-only", "--fixed-budgets", "512"])
        self.assertEqual(args.cut_points, [])
        self.assertEqual(args.cut_sizes, [])
        self.assertEqual(len(make_arm_specs(args.cut_points, args.cut_sizes,
                                            args.fixed_budgets)), 1)

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
                return "".join(chr(token - 10) for token in ids if token >= 10)

        class FakeEngine:
            """Mimic continuation and answer calls made by the real engine adapter."""

            def __init__(self):
                self.prompts = []

            def generate(self, prompts, max_tokens, seeds, stop_ids=None):
                """Return a short continuation or a fixed correct answer per prompt."""
                self.prompts.extend(prompts)
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
            "dataset": "toy",
            "manifest_fingerprint": "toy-fingerprint",
            "p": {"gold": "42"}, "prompt": [500],
            "gen": [t_open] + reasoning + [t_close], "start": 1,
            "L": len(reasoning), "capped": False,
        }
        args = SimpleNamespace(trace_batch_size=1, seed=3, hard_cut=False, model="fake",
                       budget_cut_template="<budget_cut left={W}>",
                               max_len=10000, answer_max=128, fixed_answer_max=20)
        specs = make_arm_specs([0.25], [0.25], [8])
        fake_engine = FakeEngine()
        # Temporary output keeps the test isolated and exercises both JSONL and summary export.
        with TemporaryDirectory() as temp_dir:
            run_drop_tests([record], specs, fake_engine, tokenizer, t_open, t_close,
                           tokenizer.encode, args, Path(temp_dir))
            from budget_drop_sweep import read_jsonl
            rows = read_jsonl(Path(temp_dir) / "jobs.jsonl")
            out_dir = Path(temp_dir)
            config = {
                "dataset": "toy", "model": "fake", "selected_problem_count": 1,
                "samples": 1, "cut_points": [0.25], "cut_sizes": [0.25],
                "fixed_budgets": [8],
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
        decoded_prompts = [tokenizer.decode(prompt) for prompt in fake_engine.prompts]
        self.assertTrue(any("<budget_cut left=" in prompt for prompt in decoded_prompts))
        self.assertEqual(len(summary["accuracy_vs_cut"]), len(specs))
        self.assertEqual(len(summary["paired_accuracy_differences"]), 4)
        self.assertTrue(csv_exists)


if __name__ == "__main__":
    unittest.main()
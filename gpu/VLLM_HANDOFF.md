# vLLM Baseline Handoff

## Repository setup

The research baselines are pinned as Git submodules. A fresh clone should use:

```bash
git clone --recurse-submodules <lifeline-repository-url>
```

For an existing clone:

```bash
git submodule update --init --recursive
```

The root `.gitmodules` registers both `BudgeThinker` and `Elastic-Reasoning`. Do not edit files inside either upstream submodule as part of the Lifeline adapters.

## Keep inference environments separate

These runners depend on different vLLM stacks. Do not install them into one Python environment.

| Runner | vLLM / Transformers notes |
|---|---|
| Current Qwen sweep | Existing Qwen environment; prior benchmark runs use vLLM 0.31. |
| BudgetThinker | Its README requires a patched vLLM 0.7.3 `ModelRunner` and Transformers 4.46.1. The adapter checks the vLLM version and patch before launching. |
| Elastic-Reasoning | Its `verl/pyproject.toml` pins `vllm<=0.6.3` and `transformers<4.48`. Install and run it in the Elastic repository's own environment. |

Elastic's documented setup is:

```bash
cd Elastic-Reasoning
pip install -e ./verl
pip install -e .
```

The Elastic adapter launches `verl.trainer.main_generation` with E1 mode enabled. The checkpoint, CUDA devices, tensor-parallel size, and separate thinking/solution budgets are explicit CLI arguments.

## Evaluation sequence

1. Generate the frozen question/sample manifest with the Qwen sweep. Keep this manifest unchanged for every checkpoint and method in a comparison.
2. Run the Qwen fixed-budget controls and/or mid-generation cut arms using `--manifest-dir`.
3. Run BudgetThinker in its patched 0.7.3 environment with `gpu/budgetthinker_adapter.py`.
4. Run Elastic in its own environment with `gpu/elastic_reasoning_adapter.py`.
5. Use `gpu/compare_method_runs.py` with `gpu/comparison_config.example.json` to produce accuracy-versus-actual-token curves and paired contrasts.

Example Elastic fixed-budget run from the Lifeline repository root, inside Elastic's environment:

```bash
python gpu/elastic_reasoning_adapter.py \
  --manifest results/budget_drop_sweep/aime2026/manifest.json \
  --elastic-repo Elastic-Reasoning \
  --model Salesforce/E1-Math-1.5B \
  --thinking-budget 512 --solution-budget 20 \
  --cuda-devices 0 --tensor-parallel-size 1 --samples 8 \
  --out results/elastic-aime-fixed-512
```

Elastic E1 allocates a fixed thinking budget and a separate solution budget from the start. It is a fixed-budget baseline, not a mid-generation cut method. The output records the two allocations separately, actual thinking/solution token use, total tokens, accuracy, and lexical behavior metrics.

## Fair-comparison constraints

- All runs must use the same manifest fingerprint, question IDs, gold answers, and sample IDs. The comparison tools reject mismatches.
- Compare methods at matched configured budgets, and also inspect actual generated token counts; fixed caps can produce different actual token use.
- BudgetThinker and Elastic use different published checkpoints from Qwen3-4B. These are external system baselines, not controlled same-base training comparisons.
- A model trained by Part 4 can be evaluated with `--model <checkpoint>` and `--budget-cut-template`; the template must exactly match the token format used in its training data.
- This model-behavior harness does not establish serving throughput, latency, SLO attainment, or goodput. Those remain Part 1 measurements.
- The current machine has not run real BudgetThinker or Elastic inference. Run those commands only on a compatible GPU host with the corresponding isolated environment and checkpoints installed.

#!/usr/bin/env bash
# Sudden budget-drop test (about 25 min on an RTX 4090 with Qwen3-4B).
# Run from anywhere:  bash gpu/run_drop.sh            (full run)
#                     bash gpu/run_drop.sh --limit 20 (quick check)
#                     bash gpu/run_drop.sh --mock     (no GPU, checks the pipeline)
cd "$(dirname "$0")/.." || exit 1
export VLLM_USE_FLASHINFER_SAMPLER=0
python gpu/budget_drop_bench.py --traces results/gpu_hard --out results/gpu_drop "$@" && \
python analyze_drop.py results/gpu_drop

#!/usr/bin/env bash
# Re-score the AIME budget-drop test with "final answer only": after </think> every arm, natural included,
# may only write the final answer, so the model cannot redo its reasoning in the answer.
# Reuses the natural traces in Downloads/aime_drop, so no thinking is regenerated (about 2 h on an RTX 4090).
#   bash /mnt/c/Users/Matthew/Downloads/run_aime_final.sh
#   bash /mnt/c/Users/Matthew/Downloads/run_aime_final.sh --resume     continue after a stop
DL=/mnt/c/Users/Matthew/Downloads
LOG="$DL/lifeline_aime_final_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "== start $(date) =="
if ! source ~/venv/bin/activate; then echo "!! ~/venv not found"; exit 1; fi
[ -f "$DL/aime_drop/phase_a.jsonl" ] || { echo "!! Downloads/aime_drop/phase_a.jsonl not found"; exit 1; }
mkdir -p ~/lifeline && cp "$DL/aime_drop_bench.py" "$DL/budget_drop_bench.py" "$DL/thinking_budget_bench.py" ~/lifeline/ \
  && cd ~/lifeline || exit 1
export VLLM_USE_FLASHINFER_SAMPLER=0
python aime_drop_bench.py --traces "$DL/aime_drop" --out "$DL/aime_final" --answer-mode final "$@"
status=$?
echo "== finished $(date) with status $status | results in Downloads/aime_final =="

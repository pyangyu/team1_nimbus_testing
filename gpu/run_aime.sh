#!/usr/bin/env bash
# Budget-drop test on AIME 2026 (new problems Qwen3 cannot have seen), 8 samples per problem, in 4 rounds.
#   bash /mnt/c/Users/Matthew/Downloads/run_aime.sh                    full run, overnight
#   bash /mnt/c/Users/Matthew/Downloads/run_aime.sh --resume           continue after a stop
#   bash /mnt/c/Users/Matthew/Downloads/run_aime.sh --samples 4        shorter run (2 rounds)
# Results go straight to Downloads/aime_drop, so a stopped run keeps every finished round.
DL=/mnt/c/Users/Matthew/Downloads
LOG="$DL/lifeline_aime_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "== start $(date) =="
if ! source ~/venv/bin/activate; then echo "!! ~/venv not found"; exit 1; fi
mkdir -p ~/lifeline && cp "$DL/aime_drop_bench.py" "$DL/budget_drop_bench.py" "$DL/thinking_budget_bench.py" ~/lifeline/ \
  && cd ~/lifeline || exit 1
export VLLM_USE_FLASHINFER_SAMPLER=0
python aime_drop_bench.py --out "$DL/aime_drop" "$@"
status=$?
echo "== finished $(date) with status $status | results in Downloads/aime_drop =="

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$HOME/venv/bin/activate" ]]; then
    source "$HOME/venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
export VLLM_USE_FLASHINFER_SAMPLER=0

"$PYTHON" gpu/budget_drop_bench.py --sweep --dataset all \
    --aime-problems 30 --math-problems 100 --samples 8 \
    --cut-points 0.25 0.5 0.75 --cut-sizes 0.1 0.25 \
    --out results/budget_drop_sweep "$@"
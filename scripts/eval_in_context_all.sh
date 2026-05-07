#!/usr/bin/env bash
# Run base Gemma in `in_context` mode + the two trained checkpoints in
# `ttt_paper` mode, against every task in `benchmark/configs/benchmark.yaml`
# (RULER + HELMET). Steps run in serial (one GPU). Each step's output goes
# to eval_logs/<n>_*.log; failures don't stop later steps — we always get
# whatever data is collectable.
#
# Token comes from $HF_TOKEN inherited from the caller's env. Do NOT write
# the token into this file.
#
# Usage (run from the project root):
#
#   HF_TOKEN=hf_xxx nohup ./scripts/eval_in_context_all.sh \
#     > eval_logs/_overall_in_context.log 2>&1 &
#   disown $!
#   echo "PID: $!"
#
# Monitor:
#   tail -f eval_logs/_overall_in_context.log         # high-level progress
#   tail -f eval_logs/06_in_context_base.log          # current step detail
#   ps -ef | grep eval_in_context_all                 # is it alive
#
# Kill:
#   pkill -f eval_in_context_all.sh; pkill -f benchmark.scripts.evaluate
#
# WARNING: step 1 (base Gemma in_context) overwrites the existing
# `summary/gemma-3-1b-it__in_context__full.json` and the per-task .jsonl
# files under `raw/full/gemma-3-1b-it__in_context/`. Back them up first if
# you want to keep the May 4 results:
#
#   cp benchmark/results/summary/gemma-3-1b-it__in_context__full.json{,.may4}
#   cp -r benchmark/results/raw/full/gemma-3-1b-it__in_context{,.may4}
#
# Steps 2 and 3 are ttt_paper mode; the model_name has a `-ttt` suffix
# (e.g. `gemma-3-1b-it-ttt-longalpaca-full-ttt`) so they overwrite the
# May 4 ttt_paper results. Back those up too if you want them:
#   cp benchmark/results/summary/gemma-3-1b-it-ttt-longalpaca-full-ttt__ttt_paper__full.json{,.may4}
#
# Rough wall time: ~5h total (prior single in_context run was ~1h43m).

set -u

PY=/home/htn002/continual-learning-research/In-Place-Test-Time-Training/.venv/bin/python
ROOT=/home/htn002/continual-learning-research/In-Place-Test-Time-Training
PRED_IC=benchmark.eval.gemma3_predictors:gemma3_in_context_factory
PRED_TTT=benchmark.eval.gemma3_predictors:gemma3_ttt_paper_factory

LONG=changminbark/gemma-3-1b-it-ttt-longalpaca-full
TINY=hungngo04/gemma-3-1b-it-ttt-tinystories-500k
BASE=google/gemma-3-1b-it

cd "$ROOT" || exit 99
mkdir -p eval_logs

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "[fatal] no HF_TOKEN / HUGGING_FACE_HUB_TOKEN in env (Gemma3 base config is gated)" >&2
  exit 2
fi

run() {
  local label="$1"; shift
  local logfile="$1"; shift
  echo "===== $(date -Is) START $label =====" >&2
  # tee -> live terminal output + persisted log; pipefail so we capture the
  # python rc, not tee's. PYTHONUNBUFFERED so Python flushes per line.
  set -o pipefail
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "eval_logs/$logfile"
  local rc=${PIPESTATUS[0]}
  set +o pipefail
  echo "===== $(date -Is) END   $label (rc=$rc) =====" >&2
}

run "1/3 in_context base"         06_in_context_base.log \
  env GEMMA3_BASE_MODEL_ID="$BASE" \
  "$PY" -u -m benchmark.scripts.evaluate --profile full --predictor "$PRED_IC"

run "2/3 ttt_paper  longalpaca"   07_ttt_paper_longalpaca.log \
  env GEMMA3_TTT_MODEL_ID="$LONG" \
  "$PY" -u -m benchmark.scripts.evaluate --profile full --predictor "$PRED_TTT"

run "3/3 ttt_paper  tinystories"  08_ttt_paper_tinystories.log \
  env GEMMA3_TTT_MODEL_ID="$TINY" \
  "$PY" -u -m benchmark.scripts.evaluate --profile full --predictor "$PRED_TTT"

echo "===== $(date -Is) ALL DONE =====" >&2

#!/bin/bash
# End-to-end launcher for the MusicCaps figurative-caption experiment (E1).
#
# Usage on the cluster:
#
#   cd ~/reps/clap-mtg/scripts/musiccaps_figurative
#   bash run.sh generate   # step 1: LLM rewrites (GPU; 4 GPU × vLLM)
#   bash run.sh clean      # step 2: validate + build per-level files (CPU)
#   bash run.sh evaluate <cfg_file> <model_tag>  # step 3: retrieval eval
#
# Steps 1 and 3 each need a GPU. Step 2 is CPU-only and fast.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-help}"
shift || true

REWRITER_MODEL="${REWRITER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
OUT_ROOT="${OUT_ROOT:-musiccaps_figurative}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"

case "$MODE" in
generate)
  python generate_captions.py \
    --split test \
    --out-path "${OUT_ROOT}/raw_levels.jsonl" \
    --batch-size "${BATCH_SIZE}" \
    --model "${REWRITER_MODEL}" \
    --tensor-parallel "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    "$@"
  ;;

clean)
  python postprocess.py \
    --in-path "${OUT_ROOT}/raw_levels.jsonl" \
    --out-dir "${OUT_ROOT}"
  python build_dataset.py \
    --levels-path "${OUT_ROOT}/levels.jsonl" \
    --out-dir "${OUT_ROOT}/per_level"
  python inspect_examples.py \
    --levels-path "${OUT_ROOT}/levels.jsonl" \
    --n 5
  ;;

evaluate)
  CFG_FILE="${1:?usage: run.sh evaluate <cfg_file> <model_tag>}"
  MODEL_TAG="${2:?usage: run.sh evaluate <cfg_file> <model_tag>}"
  shift 2 || true
  python evaluate_curve.py \
    --cfg-file "${CFG_FILE}" \
    --model-tag "${MODEL_TAG}" \
    --levels-path "${OUT_ROOT}/levels.jsonl" \
    "$@"
  python evaluate_pairs.py \
    --cfg-file "${CFG_FILE}" \
    --model-tag "${MODEL_TAG}" \
    --levels-path "${OUT_ROOT}/levels.jsonl" \
    "$@"
  ;;

help | *)
  cat <<EOF
Usage: run.sh <mode> [args...]

Modes:
  generate                               Step 1: LLM rewrite all MusicCaps test captions
  clean                                  Step 2: validate + build per-level caption files
  evaluate <cfg_file> <model_tag> [...]  Step 3: run curve + pairs eval for one model

Env vars:
  REWRITER_MODEL  LLM for the rewrite step (default: meta-llama/Llama-3.3-70B-Instruct)
  OUT_ROOT        Output root (default: musiccaps_figurative)
  BATCH_SIZE      vLLM batch size (default: 16)
  MAX_MODEL_LEN   vLLM max model len (default: 2048)
  TENSOR_PARALLEL vLLM tensor parallel size (default: 4)

Evaluate multiple models by rerunning step 3 with different cfg/tag pairs, e.g.:

  bash run.sh evaluate cfg/exp/R04/config.gin R04
  bash run.sh evaluate cfg/exp/R05/config.gin R05
  bash run.sh evaluate cfg/exp/R02/config.gin R02
  bash run.sh evaluate cfg/exp/R03/config.gin R03

EOF
  ;;
esac

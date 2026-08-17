#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=batch_eval_checkpoints_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

# Evaluate all checkpoints of one or more models on downstream tasks.
#
# Usage:
#   sbatch slurm_eval_checkpoints.sh <model_id> [model_id2 ...]
#   sbatch slurm_eval_checkpoints.sh --use_audio_type_token <model_id>
#   sbatch slurm_eval_checkpoints.sh --steps 20000,40000,60000 <model_id>
#
# Options (must come before model IDs):
#   --use_audio_type_token   Pass --use_audio_type_token to evaluate_all.sh
#   --steps <list>           Comma-separated list of target steps. For each
#                            target, the available checkpoint closest to it is
#                            selected. Default: evaluate every checkpoint.

set -e

# --- Parse arguments ---
USE_ATT=""
TARGET_STEPS=""
MODELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --use_audio_type_token) USE_ATT="--use_audio_type_token"; shift ;;
    --steps) TARGET_STEPS="$2"; shift 2 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "Usage: sbatch slurm_eval_checkpoints.sh [--use_audio_type_token] <model_id> [model_id2 ...]"
  exit 1
fi

# --- Environment setup ---
export scr=/scratch/<group>/

export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}
DATASETS="gtzan fma_small musiccaps song_describer dimsim"
CKPT_BASE=/projects/<group>/logs/text-audio

echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Audio type token: ${USE_ATT:-no}"

for model_id in "${MODELS[@]}"; do
  CKPT_DIR=${CKPT_BASE}/${model_id}/checkpoints
  AVAILABLE_STEPS=$(ls ${CKPT_DIR}/*.ckpt | grep -oP 'step=\K\d+' | sort -un)

  if [ -n "$TARGET_STEPS" ]; then
    # For each requested target, pick the closest available step.
    STEPS=$(python3 -c "
import sys
avail = [int(s) for s in '''$AVAILABLE_STEPS'''.split()]
targets = [int(t) for t in '$TARGET_STEPS'.split(',')]
picked = sorted({min(avail, key=lambda s: abs(s - t)) for t in targets})
print(' '.join(str(s) for s in picked))
")
  else
    STEPS="$AVAILABLE_STEPS"
  fi

  echo "================================================"
  echo "Model: ${model_id}"
  echo "Checkpoints found: ${STEPS}"
  echo "================================================"

  for step in $STEPS; do
    echo "------------------------------------------------"
    echo "Processing model: ${model_id} (step ${step})"
    echo "------------------------------------------------"
    ./evaluate_all.sh "$model_id" \
      $USE_ATT \
      --step "$step" \
      --datasets $DATASETS \
      --output_dir "$OUTPUT_DIR"
  done
done

#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=batch_eval_lsrcmb68_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

export scr=/scratch/<group>/

# Offline transformers
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

MODEL_ID=R13
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}
CKPT_BASE=/projects/<group>/logs/text-audio
DATASETS_CKPT="gtzan fma_small musiccaps song_describer dimsim"

echo "=============================================="
echo "Full evaluation pipeline for model: ${MODEL_ID}"
echo "=============================================="

# 1. Last checkpoint — full evaluation (all tasks including probing)
echo ""
echo ">>> PHASE 1: Last checkpoint (all tasks)"
echo ""
./evaluate_all.sh "$MODEL_ID" --output_dir "$OUTPUT_DIR"

# 2. Checkpoint-wise — non-probing tasks only
echo ""
echo ">>> PHASE 2: Checkpoint-wise evaluation"
echo ""
CKPT_DIR=${CKPT_BASE}/${MODEL_ID}/checkpoints
AVAILABLE_STEPS=$(ls ${CKPT_DIR}/*.ckpt | grep -oP 'step=\K\d+' | sort -un)

echo "Available checkpoints: ${AVAILABLE_STEPS}"

for step in $AVAILABLE_STEPS; do
  echo "------------------------------------------------"
  echo "Evaluating step=${step}"
  echo "------------------------------------------------"
  ./evaluate_all.sh "$MODEL_ID" \
    --step "$step" \
    --datasets $DATASETS_CKPT \
    --output_dir "$OUTPUT_DIR"
done

# 3. Averaged last 3 checkpoints — full evaluation
echo ""
echo ">>> PHASE 3: Averaged last 3 checkpoints (all tasks)"
echo ""
./evaluate_all.sh "$MODEL_ID" --output_dir "$OUTPUT_DIR" --avg_last_n 3

echo ""
echo "=============================================="
echo "All evaluations complete for model: ${MODEL_ID}"
echo "=============================================="

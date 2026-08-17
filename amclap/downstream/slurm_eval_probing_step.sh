#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=eval_probing_step_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

# Evaluate probing tasks at a specific checkpoint step.
# Usage: sbatch slurm_eval_probing_step.sh <model_id> <step>
# Example: sbatch slurm_eval_probing_step.sh R14 40000

set -e

MODEL_ID=${1:?Usage: sbatch slurm_eval_probing_step.sh <model_id> <step>}
STEP=${2:?Usage: sbatch slurm_eval_probing_step.sh <model_id> <step>}

export scr=/scratch/<group>/

# Offline transformers
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}
DATASETS_PROBING="mtt jamendo_genre jamendo_instrument jamendo_moodtheme mgphot_autotagging mgphot_regression"

echo "=============================================="
echo "Probing evaluation for model: ${MODEL_ID} at step=${STEP}"
echo "Datasets: ${DATASETS_PROBING}"
echo "Output: ${OUTPUT_DIR}/${MODEL_ID}/step=${STEP}"
echo "=============================================="

./evaluate_all.sh "$MODEL_ID" \
  --step "$STEP" \
  --datasets $DATASETS_PROBING \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Probing evaluation complete for model: ${MODEL_ID} at step=${STEP}"
echo "=============================================="

#!/bin/bash
# Submit one SLURM job per model for parallel downstream evaluation.
#
# Usage:
#   ./slurm_batch_eval_parallel.sh                          # evaluate default models
#   ./slurm_batch_eval_parallel.sh model1 model2 model3     # evaluate specific models
#   ./slurm_batch_eval_parallel.sh --use_audio_type_token model1 model2  # with audio type token

set -e

USE_AUDIO_TYPE_TOKEN=false

# Parse flags
while [[ $# -gt 0 && "$1" =~ ^-- ]]; do
  case $1 in
  --use_audio_type_token)
    USE_AUDIO_TYPE_TOKEN=true
    shift
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# Models: use arguments if provided, otherwise use defaults
if [ $# -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(R02 R03 R04 R05 R06 R07 ny6g2bzr R08 z0r7jh5a 2cvx96fi f9l9z22m 8jlpuoge)
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}

AUDIO_TOKEN_FLAG=""
if [ "$USE_AUDIO_TYPE_TOKEN" = true ]; then
  AUDIO_TOKEN_FLAG="--use_audio_type_token"
fi

echo "Submitting ${#MODELS[@]} evaluation jobs..."

for model_id in "${MODELS[@]}"; do
  JOB_ID=$(sbatch \
    --account=<group> \
    --partition=acc \
    --qos=acc_resa \
    --nodes=1 \
    --cpus-per-task=20 \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --time=04:00:00 \
    --output="eval_${model_id}_%j_output.txt" \
    --mail-type=all \
    --mail-user=<email> \
    --job-name="eval_${model_id}" \
    --wrap="
set -e

export scr=/scratch/<group>/
export HF_HOME=\${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
export HF_DATASETS_CACHE=\$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

cd ${SCRIPT_DIR}
./evaluate_all.sh ${model_id} --output_dir ${OUTPUT_DIR} ${AUDIO_TOKEN_FLAG}
" 2>&1)

  echo "  ${model_id}: ${JOB_ID}"
done

echo "All jobs submitted."

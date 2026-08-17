#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=batch_eval_checkpoints_multi_%j_output.txt
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

# R07:
# R06:
# R05:
# ny6g2bzr: LA TT, quotes+mu+so
# R08: sigmoid, quotes+mu+so
# z0r7jh5a: LeJEPA, quotes+mu+so
# f9l9z22m: LpJPEA, quotes+mu+so
# 8jlpuoge: LpJPEA J, quotes+mu+so
MODELS=(R07 R06 R05)

OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}
DATASETS="gtzan fma_small musiccaps song_describer dimsim"
CKPT_BASE=/projects/<group>/logs/text-audio

echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS}"
echo "Output directory: ${OUTPUT_DIR}"

for model_id in "${MODELS[@]}"; do
  CKPT_DIR=${CKPT_BASE}/${model_id}/checkpoints
  STEPS=$(ls ${CKPT_DIR}/*.ckpt | grep -oP 'step=\K\d+' | sort -un)

  echo "================================================"
  echo "Model: ${model_id}"
  echo "Checkpoints found: ${STEPS}"
  echo "================================================"

  for step in $STEPS; do
    echo "------------------------------------------------"
    echo "Processing model: ${model_id} (step ${step})"
    echo "------------------------------------------------"
    ./evaluate_all.sh "$model_id" \
      --use_audio_type_token \
      --step "$step" \
      --datasets $DATASETS \
      --output_dir "$OUTPUT_DIR"
  done
done

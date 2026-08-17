#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=batch_eval_token_ablation_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

# Audio-type-token ablation on non-probing downstream tasks.
# For each paper model we evaluate at a pinned checkpoint both with and
# without --use_audio_type_token, writing results to two separate output roots
# so the variants never overwrite each other.
#
#   with_token    -> /scratch/<group>/downstream_results_with_token
#   without_token -> /scratch/<group>/downstream_results_without_token
#
# Layout per variant:
#   <root>/<model_id>/step=<N>/<task>/results.json (or caption.json)

set -e

export scr=/scratch/<group>/
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

cd ~/reps/clap-mtg/src/downstream

DATASETS="gtzan fma_small musiccaps song_describer dimsim"
WITH_DIR=/scratch/<group>/downstream_results_with_token
WITHOUT_DIR=/scratch/<group>/downstream_results_without_token

# Models at their last checkpoint.
# Override with env var LAST_CKPT_MODELS="a b c" to restrict to a subset.
LAST_CKPT_MODELS=(${LAST_CKPT_MODELS:-R01 R02 R03 R04 R05 R06 R07 R08 c0u3izks o26r43l0})
# Models with trained TE, pinned to step 60000.
TE_MODELS=(${TE_MODELS:-R13 R14})
TE_STEP=60000

CKPT_BASE=/projects/<group>/logs/text-audio

run_pair() {
  local model_id=$1
  local step=$2
  echo "================================================"
  echo "Model: ${model_id} @ step=${step}"
  echo "================================================"

  for variant in without_token with_token; do
    if [ "$variant" = "with_token" ]; then
      out_root=$WITH_DIR
      flag="--use_audio_type_token"
    else
      out_root=$WITHOUT_DIR
      flag=""
    fi
    echo "---- variant: ${variant} (out=${out_root}) ----"
    ./evaluate_all.sh "$model_id" \
      $flag \
      --step "$step" \
      --datasets $DATASETS \
      --output_dir "$out_root"
  done
}

for model_id in "${LAST_CKPT_MODELS[@]}"; do
  last_step=$(ls ${CKPT_BASE}/${model_id}/checkpoints/*.ckpt 2>/dev/null \
              | grep -oP 'step=\K\d+' | sort -un | tail -1)
  if [ -z "$last_step" ]; then
    echo "WARN: no checkpoints found for ${model_id}, skipping"
    continue
  fi
  run_pair "$model_id" "$last_step"
done

for model_id in "${TE_MODELS[@]}"; do
  run_pair "$model_id" "$TE_STEP"
done

echo "================================================"
echo "Token-ablation evaluation complete."
echo "With token:    ${WITH_DIR}"
echo "Without token: ${WITHOUT_DIR}"
echo "================================================"

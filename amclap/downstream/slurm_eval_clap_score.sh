#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=clap_score_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

export scr=/scratch/<group>/
export pro=/projects/<group>/

# Offline HF cache
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

BASE_DIR=${BASE_DIR:-/projects/<group>/logs/text-audio}
MUSICEVAL_DIR=${MUSICEVAL_DIR:-/scratch/<group>/generative_eval/musiceval/MusicEval-full}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}
DEVICE=${DEVICE:-cuda:0}
SEGMENT_SIZE=${SEGMENT_SIZE:-10.0}
NEW_FREQ=${NEW_FREQ:-24000}

MODELS=(R01 R04 R05 R07 R14)

cd /home/<user>/reps/clap-mtg/src/downstream

for MODEL_ID in "${MODELS[@]}"; do
  echo "================================================"
  echo "CLAP-score evaluation: ${MODEL_ID}"
  echo "================================================"

  MODEL_DIR="${BASE_DIR}/${MODEL_ID}/checkpoints/"
  CFG_FILE=$(find "${MODEL_DIR}" -name "*.gin" | head -n 1)
  if [ -z "$CFG_FILE" ]; then
    echo "Skip ${MODEL_ID}: no .gin found in ${MODEL_DIR}"
    continue
  fi

  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_ID}"
  mkdir -p "${MODEL_OUTPUT_DIR}"

  AUDIO_TOKEN_FLAG=""
  if [ "${MODEL_ID}" = "R04" ]; then
    AUDIO_TOKEN_FLAG="--use_audio_type_token"
  fi

  python downstream_clap_score.py \
    --cfg_file "${CFG_FILE}" \
    --data_dir "${MUSICEVAL_DIR}" \
    --device "${DEVICE}" \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --segment_size "${SEGMENT_SIZE}" \
    --new_freq "${NEW_FREQ}" \
    ${AUDIO_TOKEN_FLAG}
done

echo "CLAP-score evaluation complete. Results under ${OUTPUT_DIR}/<model>/musiceval/"

#!/bin/bash
# Batch-extract audio encoders from CLAP checkpoints into omar-rq format.
#
# For each model folder, finds the .ckpt and .gin files, extracts the audio
# encoder weights, and saves them as omar-rq compatible model.ckpt + config.gin
# in a subdirectory called `audio_encoder/`.
#
# Usage:
#   bash scripts/batch_extract_audio_encoders.sh

set -euo pipefail

BASE_DIR="/scratch/<user>/bsc-a3/text-audio"
# bayc7zny
MODEL_FOLDERS=(R09 rgpxysf5 1zpfl121 f5v6nb3l)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_audio_encoder.py"

for folder in "${MODEL_FOLDERS[@]}"; do
  model_dir="${BASE_DIR}/${folder}/checkpoints/"
  echo "=================================================="
  echo "Processing: ${folder}"
  echo "=================================================="

  if [ ! -d "${model_dir}" ]; then
    echo "ERROR: Directory not found: ${model_dir}"
    continue
  fi

  # Find the single .ckpt and .gin files
  ckpt_file=$(find "${model_dir}" -maxdepth 1 -name "*.ckpt" | head -1)
  gin_file=$(find "${model_dir}" -maxdepth 1 -name "*.gin" | head -1)

  if [ -z "${ckpt_file}" ]; then
    echo "ERROR: No .ckpt file found in ${model_dir}"
    continue
  fi

  if [ -z "${gin_file}" ]; then
    echo "ERROR: No .gin file found in ${model_dir}"
    continue
  fi

  echo "  ckpt: ${ckpt_file}"
  echo "  config: ${gin_file}"

  output_dir="${model_dir}/audio_encoder"
  echo "  output: ${output_dir}"

  python "${EXTRACT_SCRIPT}" \
    --clap_ckpt "${ckpt_file}" \
    --clap_config "${gin_file}" \
    --output_dir "${output_dir}"

  echo ""
done

echo "Done!"

#!/bin/bash
set -e

# GTZAN zero-shot classification evaluation

BASE_DIR=${BASE_DIR:-/projects/<group>/logs/text-audio}
DEVICE=${DEVICE:-cuda:0}

for model_id in R09; do
    model_dir=${BASE_DIR}/${model_id}/checkpoints/
    cfg_file=$(find ${model_dir} -name "*.gin" | head -n 1)

    echo "Processing ${cfg_file}"

    python downstream_gtzan.py \
        --cfg_file ${cfg_file} \
        --device ${DEVICE}
done

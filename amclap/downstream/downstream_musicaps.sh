#! /bin/bash
set -e

# base_dir=/path/to/logs/text-audio
base_dir=/projects/<group>/logs/text-audio

# 8ixm1n41 -> large, 30s  with dt
# nqudnej0 -> small, 10s, with big bs and fs data
# nc9vud2p -> large, 10s, with fs, msd, pse, and dt
# R09 -> small, 10s, with fs, msd, pse, and dt

# nqudnej0 2hvrpfkf R09 nc9vud2p 99eycym5

# JEPA results
# zt4fi6z0 -> 󰘧 = 1. MRR: 0.006135
# chuqau95 -> 󰘧 = 0.02. MRR: 0.0105866
# 1znovffx -> 󰘧 = 0.001. MRR: 0.0056797

for model_id in pmn4dy8p; do
  model_dir=${base_dir}/${model_id}/checkpoints/
  # find the cfg file here
  cfg_file=$(find ${model_dir} -name "*.gin")

  echo "processing" ${cfg_file}

  python downstream_retrieval.py \
    --device cuda:0 \
    --data_dir /scratch/<group>/downstream_datasets/music_caps/ \
    --cfg_file ${cfg_file} \
    --audio-setup clap

done

#! /bin/bash
set -e

base_dir=/home/<user>/reps/clap-mtg/model_weights

# 8ixm1n41 -> large, 30s  with dt
# nqudnej0 -> small, 10s, with big bs and fs data
# m3q4ll2o -> large, 10s, with fs, msd, and pse

# nqudnej0 2hvrpfkf R09 nc9vud2p
# for model_id in 99eycym5 2ujn9q8i uxde4ynq; do
for model_id in 99eycym5 2ujn9q8i; do

  model_dir=${base_dir}/${model_id}/checkpoints/
  # find the cfg file here
  cfg_file=$(find ${model_dir} -name "config_*.gin")
  #
  echo "processing" ${cfg_file}

  python retrieval_demo_preprocess.py \
    --device cuda:0 \
    --data_dir /mnt/projects/discotube/youtube-downloads/ \
    --cfg_file ${cfg_file} \
    --id_list id_list
done

### BASELINE

# python retrieval_demo_preprocess.py \
#   --device cuda:0 \
#   --data_dir /mnt/projects/discotube/youtube-downloads/ \
#   --model_type laion_clap \
#   --id_list id_list

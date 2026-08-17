#! /bin/bash
set -e

# base_dir=/data/shared/<user>/data/logs/text-audio
base_dir=/mnt/projects/mtg_text_audio/wandb/text-audio

# 8ixm1n41 -> large, 30s  with dt
# nqudnej0 -> small, 10s, with big bs and fs data
# m3q4ll2o -> large, 10s, with fs, msd, and pse

# nqudnej0 2hvrpfkf R09 nc9vud2p
# for model_id in 99eycym5 2ujn9q8i; do
# xch7hzmk
for model_id in xch7hzmk; do
  model_dir=${base_dir}/${model_id}/checkpoints/
  # find the cfg file here
  cfg_file=$(find ${model_dir} -name "*.gin")
  #
  echo "processing" ${cfg_file}

  python downstream_musicaps.py \
    --device cuda:0 \
    --data_dir /mnt/projects/mtg_text_audio/audio_embs/omar-rq-multifeature-25hz-fsq/discotube_5s/ \
    --dataset all_music \
    --cfg_file ${cfg_file} \
    --audio-setup frozen
done

### BASELINE

# python downstream_musicaps.py \
#   --device cuda:0 \
#   --data_dir /data/shared/<user>/data/discotube/mmaps/ \
#   --model_type laion_clap \
#   --dataset all_music

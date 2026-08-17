#! /bin/bash
set -e

# base_dir=/path/to/logs/text-audio
# base_dir=/mnt/projects/mtg_text_audio/wandb/text-audio
base_dir=/projects/<group>/logs/text-audio

# clap-style model ids: nqudnej0 lgq191pk 2hvrpfkf R09 nc9vud2p 99eycym5 2ujn9q8i uxde4ynq
# clamp3-style model ids: lmi0l33r xch7hzmk

# for model_id in mdhx1kv9 vuv93oxt yb3unncj drdy9rxe; do
# clamp3 style models -> 4zr1tct9 ihjcue9e

for model_id in s2b6fjyi; do

  model_dir=${base_dir}/${model_id}/checkpoints/

  # find the cfg file here
  cfg_file=$(find ${model_dir} -name "*.gin")

  echo "processing" ${cfg_file}

  python downstream_retrieval.py \
    --device cuda:0 \
    --data_dir /scratch/<group>/downstream_datasets/song_describer \
    --dataset song_describer \
    --cfg_file ${cfg_file} \
    --audio-setup clap

  # --data_dir /mnt/projects/mtg_text_audio/song_describer \
done

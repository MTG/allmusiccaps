set -e

input_dir="/data/shared/<user>/data/freesound/sounds/"
output_dir="/data/shared/<user>/data/freesound/mmaps/"

python preproc.py \
  --n-tasks 64 \
  --input-dir ${input_dir} \
  --output-dir ${output_dir} \
  --sample-rate 16000 \
  --filelist filelist_freesound_c

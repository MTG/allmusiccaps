set -e

input_dir="/data/shared/<user>/data/discotube/mp4/"
output_dir="/data/shared/<user>/data/discotube/mmaps/"

python preproc.py \
  --n-tasks 128 \
  --input-dir ${input_dir} \
  --output-dir ${output_dir} \
  --sample-rate 16000

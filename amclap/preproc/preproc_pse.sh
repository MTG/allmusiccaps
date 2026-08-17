set -e

input_dir="/data/shared/<user>/data/pse/pse_data_december_2024/data/"
output_dir="/data/shared/<user>/data/pse/mmaps/pse_data_december_2024/data/"

python preproc.py \
  --n-tasks 128 \
  --input-dir ${input_dir} \
  --output-dir ${output_dir} \
  --sample-rate 16000

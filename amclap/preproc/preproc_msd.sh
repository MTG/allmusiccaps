#!/bin/bash

#SBATCH --job-name ext_msd
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

source /projects/<group>/envs/clap/bin/activate

input_dir="/projects/<group>/msd"
output_dir="/scratch/<group>/mmaps_msd"
filelsit="/projects/<group>/msd/filelist"

python preproc.py \
  --n-tasks 80 \
  --input-dir ${input_dir} \
  --output-dir ${output_dir} \
  --sample-rate 16000 \
  --filelist ${filelsit}

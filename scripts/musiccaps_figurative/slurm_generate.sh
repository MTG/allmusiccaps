#!/bin/bash
# SLURM job for E1 step 1: generate figurative captions with vLLM
#
# Usage:
#   cd ~/reps/clap-mtg/scripts/musiccaps_figurative
#   sbatch slurm_generate.sh

#SBATCH --job-name E1_generate
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=E1_generate_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

source /projects/<group>/envs/vllm/bin/activate

export HF_HOME=/scratch/<group>/HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_DATASETS_OFFLINE=1

cd ~/reps/clap-mtg/scripts/musiccaps_figurative

bash run.sh generate

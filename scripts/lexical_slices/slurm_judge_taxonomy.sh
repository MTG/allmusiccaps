#!/bin/bash
#SBATCH --job-name=judge_taxonomy
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=exp_%x_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

export scr=/scratch/<group>/
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /projects/<group>/envs/vllm/bin/activate

cd /home/<user>/reps/clap-mtg

python scripts/lexical_slices/judge_taxonomy_vllm.py \
  --tensor-parallel 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-tokens 80

#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=2
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=exp_%x_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

CONFIG_NAME=$1
CONFIG_FILE="cfg/config_${CONFIG_NAME}.gin"

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SRUN_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE

export scr=/scratch/<group>/

# Offline transformers
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WANDB setup
export WANDB_API_KEY=<your-wandb-api-key>

source /projects/<group>/envs/clap/bin/activate

echo "Running experiment: $CONFIG_NAME"
echo "Config file: $CONFIG_FILE"

srun python src/train.py $CONFIG_FILE

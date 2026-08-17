#!/bin/bash

#SBATCH --job-name lejepa_mpnet_base_v2_ssl_large_cos
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=2
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SRUN_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE

export scr=/scratch/<group>/
export pro=/projects/<group>/

# Offline transformers
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# WANDB setup
export WANDB_API_KEY=<your-wandb-api-key>

source /projects/<group>/envs/clap/bin/activate

srun python src/train.py cfg/config_lejepa_mpnet_base_v2_ssl_mp_10s_large_clap_dt__lr_5e-4__l2e-1__cos.gin

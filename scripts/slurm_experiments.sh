#!/bin/bash
#
# SLURM launcher for CLAP experiments
# Usage: scripts/slurm_experiments.sh <config_name>
# Example: scripts/slurm_experiments.sh E1_1_mert
#
# Available configs:
#   E1_1_mert      - MERT-v1-95M audio encoder
#   E1_2_htsat     - HTS-AT audio encoder
#   E1_3_maest     - MEST audio encoder
#   E2_1_roberta   - RoBERTa-base text encoder
#   E2_2_xlmr      - XLM-RoBERTa-base text encoder
#   E2_3_modernbert - ModernBERT-base text encoder
#   E3_1_m4rag     - M4-RAG training data
#   E3_2_msd       - MSD-only training data
#   E3_3_dt_msd_m4rag - DT v1 + MSD + M4-RAG (equal proportions)
#   E3_4_dt_msd_m4rag_lr1e-4 - DT v1 + MSD + M4-RAG (LR=1e-4)
#   E3_5_dt_msd_m4rag_fs_pse - DT v1 + MSD + M4-RAG + FS + PSE (sound=20%)
#   E3_6_dt_v2_msd_m4rag - DT v2 + MSD + M4-RAG (equal proportions)

# Check argument
if [ -z "$1" ]; then
  echo "Usage: scripts/slurm_experiments.sh <config_name>"
  echo "Example: scripts/slurm_experiments.sh E1_1_mert"
  exit 1
fi

CONFIG_NAME=$1
CONFIG_FILE="cfg/config_${CONFIG_NAME}.gin"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file not found: $CONFIG_FILE"
  exit 1
fi

sbatch --job-name="$CONFIG_NAME" scripts/slurm_job.sh "$CONFIG_NAME"

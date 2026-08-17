#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=batch_eval_avg_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

export scr=/scratch/<group>/

# Offline transformers
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/clap/bin/activate

# Evaluate averaged-checkpoint models (last 3 checkpoints)
MODELS=(R14 ny6g2bzr)
AVG_LAST_N=3
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}

echo "Starting averaged-checkpoint evaluation (last ${AVG_LAST_N})"
echo "Models: ${MODELS[*]}"
echo "Output directory: $OUTPUT_DIR"

for model_id in "${MODELS[@]}"; do
  echo "------------------------------------------------"
  echo "Processing model: $model_id (avg last ${AVG_LAST_N})"
  echo "------------------------------------------------"
  ./evaluate_all.sh "$model_id" --output_dir "$OUTPUT_DIR" --avg_last_n "$AVG_LAST_N"
done

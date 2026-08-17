#!/bin/bash
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=batch_eval_%j_output.txt
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

# XXXXXXXXX: DT pretrain)
# id, loss, data, audio_type
# R09, InfoNCE, DTV1+MSD+FS+PSE, no
# 1zpfl121, InfoNCE, DTV2+MSD+FS+PSE no
# R04  InfoNCE, DTV1+MSD+R4+FS+PSE, yes
# rgpxysf5, InfoNCE, DTV1+MSD+R4, no
# f5v6nb3l, InfoNCE/Sigreg, DTV1+MSD+R4, no
# 6si461i0, mixed views
# kqbtt80w, lambd sched

# Models to evaluate
# R05 R03 R02 R06 R07 R04

MODELS=(c0u3izks o26r43l0)
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}

# Ensure we are in the directory containing this script
# cd "$(dirname "$0")"

echo "Starting batch evaluation for models: ${MODELS[*]}"
echo "Output directory: $OUTPUT_DIR"

for model_id in "${MODELS[@]}"; do
  echo "------------------------------------------------"
  echo "Processing model: $model_id"
  echo "------------------------------------------------"
  ./evaluate_all.sh "$model_id" --output_dir "$OUTPUT_DIR" --use_audio_type_token
done

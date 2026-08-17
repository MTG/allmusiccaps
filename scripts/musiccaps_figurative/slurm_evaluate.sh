#!/bin/bash
# SLURM job for E1 step 3: evaluate a CLAP model on the figurative MusicCaps
# curve (Exp B) and the paired level_1 ↔ level_k retrieval (Exp A).
#
# Usage:
#   cd ~/reps/clap-mtg/scripts/musiccaps_figurative
#   sbatch slurm_evaluate.sh <model_id>
#
# Submits one GPU job per model. Example to launch all 4 Table 1 models:
#
#   for m in R04 R05 R03 R02; do
#     sbatch slurm_evaluate.sh "$m"
#   done

#SBATCH --job-name E1_evaluate
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=E1_evaluate_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -eo pipefail

MODEL_ID="${1:?usage: sbatch slurm_evaluate.sh <model_id>}"

# Activate before enabling -u: the venv activate script references
# CONDA_PREFIX unconditionally and trips `set -u`.
source /projects/<group>/envs/clap/bin/activate
set -u

# Offline HF cache (same as slurm_batch_eval.sh)
export scr=/scratch/<group>/
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd ~/reps/clap-mtg/scripts/musiccaps_figurative

# Paths that mirror evaluate_all.sh / slurm_batch_eval.sh defaults.
BASE_DIR=${BASE_DIR:-/projects/<group>/logs/text-audio}
MUSICCAPS_DIR=${MUSICCAPS_DIR:-/scratch/<group>/downstream_datasets/music_caps}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}

MODEL_DIR="${BASE_DIR}/${MODEL_ID}/checkpoints/"
CFG_FILE=$(find "${MODEL_DIR}" -name "*.gin" | head -n 1)

if [ -z "$CFG_FILE" ]; then
  echo "Error: Could not find .gin config file in ${MODEL_DIR}"
  exit 1
fi

# Results land under downstream_results/figurative/<model_id>/ so that
# evaluate_curve.py / evaluate_pairs.py create their per-model subdirs in a
# shared "figurative" namespace next to the existing music_caps / song_describer
# trees.
FIG_OUT_DIR="${OUTPUT_DIR}/figurative"
MODEL_OUT_DIR="${FIG_OUT_DIR}/${MODEL_ID}"

echo "=============================================="
echo "E1 evaluation: ${MODEL_ID}"
echo "Config:  ${CFG_FILE}"
echo "Data:    ${MUSICCAPS_DIR}"
echo "Output:  ${MODEL_OUT_DIR}"
echo "=============================================="

# --- Experiment B: sensitivity curve ---------------------------------------
python evaluate_curve.py \
  --cfg-file "${CFG_FILE}" \
  --model-tag "${MODEL_ID}" \
  --levels-path musiccaps_figurative/levels.jsonl \
  --data-dir "${MUSICCAPS_DIR}" \
  --out-dir "${FIG_OUT_DIR}"

# --- Experiment A: paired literal ↔ figurative retrieval -------------------
python evaluate_pairs.py \
  --cfg-file "${CFG_FILE}" \
  --model-tag "${MODEL_ID}" \
  --levels-path musiccaps_figurative/levels.jsonl \
  --data-dir "${MUSICCAPS_DIR}" \
  --out-dir "${FIG_OUT_DIR}"

echo "=============================================="
echo "E1 evaluation complete: ${MODEL_ID}"
echo "=============================================="

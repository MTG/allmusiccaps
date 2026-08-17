#!/bin/bash
# Launch SLURM jobs to evaluate losses-group models at step=40000.
# Run from the cluster login node: bash scripts/launch_step40k_eval.sh
#
# Submits one job per model. Models with only probing tasks missing get
# a targeted dataset list; R14 gets the full evaluation.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/../src/downstream" && pwd)"
STEP=40000

SBATCH_OPTS=(
  --account=<group>
  --partition=acc
  --qos=acc_resa
  --nodes=1
  --cpus-per-task=20
  --ntasks-per-node=1
  --gres=gpu:1
  --time=24:00:00
  --mail-type=all
  --mail-user=<email>
)

ENV_SETUP='
export scr=/scratch/<group>/
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source /projects/<group>/envs/clap/bin/activate
'

PROBING_DATASETS="mtt jamendo_genre jamendo_instrument jamendo_moodtheme mgphot_regression"
ALL_DATASETS="gtzan fma_small musiccaps song_describer dimsim ${PROBING_DATASETS}"

# Models missing only probing tasks at step=40000
for mid in ny6g2bzr R08 z0r7jh5a f9l9z22m; do
  echo "Submitting ${mid} (probing only, step=${STEP})"
  sbatch "${SBATCH_OPTS[@]}" \
    --job-name="eval_${mid}_s${STEP}" \
    --output="eval_${mid}_step${STEP}_%j.txt" \
    --wrap "${ENV_SETUP}
cd ${SCRIPT_DIR}
./evaluate_all.sh ${mid} --step ${STEP} --datasets ${PROBING_DATASETS}
"
done

# R14: missing all tasks at step=40000
mid=R14
echo "Submitting ${mid} (all tasks, step=${STEP})"
sbatch "${SBATCH_OPTS[@]}" \
  --job-name="eval_${mid}_s${STEP}" \
  --output="eval_${mid}_step${STEP}_%j.txt" \
  --wrap "${ENV_SETUP}
cd ${SCRIPT_DIR}
./evaluate_all.sh ${mid} --step ${STEP} --datasets ${ALL_DATASETS}
"

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"

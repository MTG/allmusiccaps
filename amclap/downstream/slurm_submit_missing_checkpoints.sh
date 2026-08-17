#!/bin/bash
# Submit one slurm job per (model, checkpoint) for the missing downstream
# evaluations needed by scripts/plot_training_dynamics_probing.py.
#
# Each (model, step) gets its own job so they can fan out across the queue.
# Audio type token is NOT used.
#
# Run from src/downstream/ on the login node:
#   ./slurm_submit_missing_checkpoints.sh                # submit all missing
#   ./slurm_submit_missing_checkpoints.sh --dry-run      # print sbatch lines only
#   ./slurm_submit_missing_checkpoints.sh --model o26r43l0 [--step 20000]
#
# Coverage targeted by this script:
#   o26r43l0, c0u3izks: full suite (probing + non-probing) at all 8 checkpoints
#   z0r7jh5a: probing-only at all 8 checkpoints (non-probing already exists)
#   R07, R13, R14, R08: probing-only at the 6 missing steps

set -e

DRY_RUN=false
ONLY_MODEL=""
ONLY_STEP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --model)   ONLY_MODEL="$2"; shift 2 ;;
    --step)    ONLY_STEP="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Datasets
PROBING="mtt jamendo_genre jamendo_instrument jamendo_moodtheme mgphot_autotagging mgphot_regression"
NONPROBING="gtzan fma_small musiccaps song_describer dimsim"
FULL="$NONPROBING $PROBING"

# (model, comma-separated steps, datasets)
JOBS=(
  "R07|21024,78840,99864,120888,140000,149796|$PROBING"
  "R13|20000,80000,100000,120000,140000,149796|$PROBING"
  "R14|20000,80000,100000,120000,140000,149796|$PROBING"
  "R08|20000,80000,100000,120000,140000,149796|$PROBING"
  "o26r43l0|20000,40000,60000,80000,100000,120000,140000,149796|$FULL"
  "c0u3izks|20000,40000,60000,80000,100000,120000,140000,149796|$FULL"
  "z0r7jh5a|20000,40000,60000,80000,100000,120000,140000,149796|$PROBING"
)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

submit_one() {
  local model_id=$1
  local step=$2
  local datasets=$3

  local job_name="eval_${model_id}_${step}"
  local cmd=(
    sbatch
    --account=<group>
    --partition=acc
    --qos=acc_resa
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task=20
    --gres=gpu:1
    --time=5:00:00
    --job-name="$job_name"
    --output="eval_${model_id}_step${step}_%j.out"
    --mail-type=fail
    --mail-user=<email>
    --wrap="cd ${SCRIPT_DIR} && \
export scr=/scratch/<group>/ && \
export HF_HOME=\${scr}HF_HOME/ && \
export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub && \
export HF_DATASETS_CACHE=\$HF_HOME/datasets && \
export TRANSFORMERS_OFFLINE=1 && \
export HF_HUB_OFFLINE=1 && \
source /projects/<group>/envs/clap/bin/activate && \
./evaluate_all.sh ${model_id} --step ${step} --datasets ${datasets} --output_dir /scratch/<group>/downstream_results"
  )

  if $DRY_RUN; then
    printf '%s ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

count=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_id steps datasets <<<"$entry"
  if [ -n "$ONLY_MODEL" ] && [ "$ONLY_MODEL" != "$model_id" ]; then
    continue
  fi
  IFS=',' read -ra step_list <<<"$steps"
  for step in "${step_list[@]}"; do
    if [ -n "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$step" ]; then
      continue
    fi
    submit_one "$model_id" "$step" "$datasets"
    count=$((count + 1))
  done
done

echo ""
echo "Submitted $count job(s). Dry-run=${DRY_RUN}."

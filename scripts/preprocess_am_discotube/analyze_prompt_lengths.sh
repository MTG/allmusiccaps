#!/bin/bash

#SBATCH --job-name inference_vllm
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

# Offline transformers
# export HF_HOME=${scr}HF_HOME/
# export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /projects/<group>/envs/vllm/bin/activate

python analyze_prompt_lengths.py \
  --data-path ../../notebooks/allmusic_youtube_discogs_reviews_merged_sets.pkl \
  --show-longest \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --max-model-len 2016 \
  --max-samples -1 \
  --use-vllm

## Leave 32 tokens For margin

echo ""
echo "Done! Results saved to: $OUT_PATH"

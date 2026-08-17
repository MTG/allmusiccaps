#!/bin/bash
# Run music description generation with Transformers backend

#!/bin/bash

#SBATCH --job-name infer
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>

set -e

# Offline transformers
export HF_HOME=${scr}HF_HOME/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"

# Default values
DATA_PATH="../../notebooks/allmusic_youtube_discogs_reviews.pkl"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME="MaziyarPanahi/calme-2.1-qwen2.5-72b"
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
N_SAMPLES=32
BATCH_SIZE=8
SEED=42
OUT_PATH="quick_test_outputs_transformers.jsonl"
QUANTIZATION=4

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running model_comp_transformers.py"
echo "=================================="
echo "Data path:    $DATA_PATH"
echo "Model:        $MODEL_NAME"
echo "Samples:      $N_SAMPLES"
echo "Batch size:   $BATCH_SIZE"
echo "Output:       $OUT_PATH"
echo "Quantization: ${QUANTIZATION}"
echo "=================================="

python model_comp_transformers.py \
  --data-path "$DATA_PATH" \
  --model-name "$MODEL_NAME" \
  --n-samples "$N_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --out-path "$OUT_PATH" \
  --quantization-bits "$QUANTIZATION" \
  "$@"

echo ""
echo "Done! Results saved to: $OUT_PATH"

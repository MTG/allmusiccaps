#!/bin/bash
# Run music description generation with vLLM backend

#!/bin/bash

set -e

# Default values
DATA_PATH="../../notebooks/allmusic_youtube_discogs_reviews.pkl"
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME="MaziyarPanahi/calme-2.1-qwen2.5-72b"
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
N_SAMPLES=64
BATCH_SIZE=16
SEED=42
OUT_PATH="quick_test_outputs_vllm.jsonl"

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running model_comp_vllm.py"
echo "=================================="
echo "Data path:    $DATA_PATH"
echo "Model:        $MODEL_NAME"
echo "Samples:      $N_SAMPLES"
echo "Batch size:   $BATCH_SIZE"
echo "Output:       $OUT_PATH"
echo "=================================="

python model_comp_vllm.py \
  --data-path "$DATA_PATH" \
  --model "$MODEL_NAME" \
  --n-samples "$N_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  "$@"

echo ""
echo "Done! Results saved to: $OUT_PATH"

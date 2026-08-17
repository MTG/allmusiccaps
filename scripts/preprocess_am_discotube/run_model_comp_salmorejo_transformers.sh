#!/bin/bash
# Run music description generation with Transformers backend

#!/bin/bash

set -e

# Default values
DATA_PATH="../../notebooks/allmusic_youtube_discogs_reviews.pkl"
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME="MaziyarPanahi/calme-2.1-qwen2.5-72b"
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
N_SAMPLES=8
BATCH_SIZE=4
SEED=42
OUT_PATH="quick_test_outputs_transformers.jsonl"
QUANTIZATION=0

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

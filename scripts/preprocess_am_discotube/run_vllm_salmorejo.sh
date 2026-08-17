#!/bin/bash
# Run music description generation with vLLM backend

#!/bin/bash

set -e

# Default values
DATA_PATH="../../notebooks/allmusic_youtube_discogs_reviews_merged_sets.pkl"
# MODEL_NAME examples:
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME="MaziyarPanahi/calme-2.1-qwen2.5-72b"
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
N_SAMPLES=16
BATCH_SIZE=16
SEED=42
OUT_PATH="quick_test_outputs_vllm.jsonl"
OUT_DIR="sample_outputs_vllm"

# vLLM engine/model settings
TOKENIZER=""
TENSOR_PARALLEL=1
DTYPE="auto"
TRUST_REMOTE_CODE=false
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.90
ENABLE_PREFIX_CACHING=false
ENFORCE_EAGER=false
MAX_NUM_SEQS=256

# Inference controls
MAX_TOKENS=512
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=-1
NUM_CANDIDATES=1
PRESENCE_PENALTY=0.0
FREQUENCY_PENALTY=0.0
REPETITION_PENALTY=1.0
CHECK_ARTIST_LEAKAGE=false
FORCE=false

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running inference_vllm.py"
echo "=================================="
echo "Data path:    $DATA_PATH"
echo "Model:        $MODEL_NAME"
echo "Samples:      $N_SAMPLES"
echo "Batch size:   $BATCH_SIZE"
echo "Out JSONL:    $OUT_PATH"
echo "Out dir:      $OUT_DIR"
echo "TP Size:      $TENSOR_PARALLEL"
echo "Max tokens:   $MAX_TOKENS"
echo "Temp/TopP/K:  $TEMPERATURE / $TOP_P / $TOP_K"
echo "=================================="

python inference_vllm.py \
  --data-path "$DATA_PATH" \
  --model "$MODEL_NAME" \
  $([ -n "$TOKENIZER" ] && echo "--tokenizer \"$TOKENIZER\"") \
  --tensor-parallel "$TENSOR_PARALLEL" \
  --dtype "$DTYPE" \
  $([ "$TRUST_REMOTE_CODE" = "true" ] && echo "--trust-remote-code") \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  $([ "$ENABLE_PREFIX_CACHING" = "true" ] && echo "--enable-prefix-caching") \
  $([ "$ENFORCE_EAGER" = "true" ] && echo "--enforce-eager") \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --n-samples "$N_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --out-path "$OUT_PATH" \
  --out-dir "$OUT_DIR" \
  $([ "$FORCE" = "true" ] && echo "--force") \
  $([ "$CHECK_ARTIST_LEAKAGE" = "true" ] && echo "--check-artist-leakage") \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --n "$NUM_CANDIDATES" \
  --presence-penalty "$PRESENCE_PENALTY" \
  --frequency-penalty "$FREQUENCY_PENALTY" \
  --repetition-penalty "$REPETITION_PENALTY" \
  "$@"

echo ""
echo "Done! Results saved to: $OUT_PATH"

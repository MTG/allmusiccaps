#!/bin/bash
# Local debug script for CLAP training
# Usage:
#   ./scripts/run_debug_local.sh                    # Basic CPU run
#   ./scripts/run_debug_local.sh --gpu              # GPU run
#   ./scripts/run_debug_local.sh --max_steps=50     # Extended run
#   ./scripts/run_debug_local.sh --batch_size=8     # Larger batch
#   ./scripts/run_debug_local.sh --gin_param='CLAP.lr=1e-5'  # Custom gin param

set -e

# Default values
USE_GPU=false
MAX_STEPS=""
BATCH_SIZE=""
GIN_PARAMS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --max_steps=*)
            MAX_STEPS="${1#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --gin_param=*)
            GIN_PARAMS+=("${1#*=}")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu] [--max_steps=N] [--batch_size=N] [--gin_param='...']"
            exit 1
            ;;
    esac
done

# Set environment variables
export WANDB_MODE=offline
export DEBUG_LOG_DIR="${DEBUG_LOG_DIR:-./debug_logs/}"

# Create log directory if needed
mkdir -p "$DEBUG_LOG_DIR"

# Build gin bindings string
GIN_BINDINGS=""

if [ "$USE_GPU" = true ]; then
    GIN_BINDINGS="$GIN_BINDINGS --gin_bindings='train.params[\"accelerator\"]=\"gpu\"'"
    GIN_BINDINGS="$GIN_BINDINGS --gin_bindings='train.params[\"precision\"]=\"bf16-mixed\"'"
fi

if [ -n "$MAX_STEPS" ]; then
    GIN_BINDINGS="$GIN_BINDINGS --gin_bindings='train.params[\"max_steps\"]=$MAX_STEPS'"
fi

if [ -n "$BATCH_SIZE" ]; then
    GIN_BINDINGS="$GIN_BINDINGS --gin_bindings='DummyTextAudioDataModule.batch_size=$BATCH_SIZE'"
fi

for param in "${GIN_PARAMS[@]}"; do
    GIN_BINDINGS="$GIN_BINDINGS --gin_bindings='$param'"
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run training
cd "$PROJECT_ROOT"
eval python src/train.py cfg/debug/config_debug_local.gin $GIN_BINDINGS

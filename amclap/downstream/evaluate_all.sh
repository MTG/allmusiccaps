#!/bin/bash
set -e

# Unified evaluation script - calls existing evaluation scripts for all datasets
#
# Usage:
#   ./evaluate_all.sh <model_id>
#   ./evaluate_all.sh <model_id> --datasets gtzan musiccaps
#   ./evaluate_all.sh <model_id> --device cuda:1

# Configuration
BASE_DIR=${BASE_DIR:-/projects/<group>/logs/text-audio}
GTZAN_DIR=${GTZAN_DIR:-/scratch/<group>/downstream_datasets/gtzan}
MUSICCAPS_DIR=${MUSICCAPS_DIR:-/scratch/<group>/downstream_datasets/music_caps}
SONG_DESCRIBER_DIR=${SONG_DESCRIBER_DIR:-/scratch/<group>/downstream_datasets/song_describer}
FMA_SMALL_DIR=${FMA_SMALL_DIR:-/scratch/<group>/downstream_datasets/fma_small}
DIMSIM_DIR=${DIMSIM_DIR:-/scratch/<group>/downstream_datasets/dimsim}
MTT_DIR=${MTT_DIR:-/scratch/<group>/downstream_datasets/magnatagatune}
JAMENDO_DIR=${JAMENDO_DIR:-/scratch/<group>/downstream_datasets/mtg-jamendo}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/<group>/downstream_results}
DEVICE=${DEVICE:-cuda:0}

# Audio config (set these based on your model)
SEGMENT_SIZE=${SEGMENT_SIZE:-10.0}
NEW_FREQ=${NEW_FREQ:-24000}
USE_AUDIO_TYPE_TOKEN=${USE_AUDIO_TYPE_TOKEN:-false}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-false}

# Default: evaluate all datasets
DATASETS="gtzan fma_small musiccaps song_describer dimsim mtt jamendo_genre jamendo_instrument jamendo_moodtheme mgphot_autotagging mgphot_regression"
CKPT_STEP=""
AVG_LAST_N=""

# Parse arguments
MODEL_ID=$1
shift || true

while [[ $# -gt 0 ]]; do
  case $1 in
  --datasets)
    shift
    DATASETS=""
    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
      DATASETS="$DATASETS $1"
      shift
    done
    ;;
  --device)
    DEVICE="$2"
    shift 2
    ;;
  --output_dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
  --segment_size)
    SEGMENT_SIZE="$2"
    shift 2
    ;;
  --new_freq)
    NEW_FREQ="$2"
    shift 2
    ;;
  --use_audio_type_token)
    USE_AUDIO_TYPE_TOKEN=true
    shift
    ;;
  --force_recompute)
    FORCE_RECOMPUTE=true
    shift
    ;;
  --step)
    CKPT_STEP="$2"
    shift 2
    ;;
  --avg_last_n)
    AVG_LAST_N="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

if [ -z "$MODEL_ID" ]; then
  echo "Usage: $0 <model_id> [options]"
  echo ""
  echo "Options:"
  echo "  --datasets <list>        Datasets to evaluate (gtzan fma_small musiccaps song_describer dimsim mtt jamendo_genre jamendo_instrument jamendo_moodtheme mgphot_autotagging mgphot_regression)"
  echo "  --device <device>        CUDA device (default: cuda:0)"
  echo "  --output_dir <path>      Output directory for results"
  echo "  --segment_size <float>   Audio segment size in seconds (default: 10.0)"
  echo "  --new_freq <int>         Audio sample rate in Hz (default: 24000)"
  echo "  --use_audio_type_token   Add [MUSIC] prefix to text queries"
  echo "  --force_recompute        Delete existing embeddings and recompute from scratch"
  echo "  --step <int>             Evaluate a specific step checkpoint (e.g. 20000)"
  echo "  --avg_last_n <int>       Average the last N checkpoints before evaluation"
  echo ""
  echo "Examples:"
  echo "  $0 R09"
  echo "  $0 R09 --datasets gtzan fma_small musiccaps"
  echo "  $0 R09 --segment_size 10.0 --new_freq 24000 --use_audio_type_token"
  echo "  $0 R09 --step 20000 --datasets gtzan"
  exit 1
fi

# Find config file
MODEL_DIR="${BASE_DIR}/${MODEL_ID}/checkpoints/"
CFG_FILE=$(find ${MODEL_DIR} -name "*.gin" | head -n 1)

if [ -z "$CFG_FILE" ]; then
  echo "Error: Could not find .gin config file in ${MODEL_DIR}"
  exit 1
fi

# Create model-specific output directory
if [ -n "$CKPT_STEP" ]; then
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_ID}/step=${CKPT_STEP}"
elif [ -n "$AVG_LAST_N" ]; then
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_ID}/avg_last_${AVG_LAST_N}"
else
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_ID}"
fi

# Build ckpt step flag
CKPT_STEP_FLAG=""
if [ -n "$CKPT_STEP" ]; then
  CKPT_STEP_FLAG="--ckpt_step $CKPT_STEP"
fi

# Build avg_last_n flag
AVG_LAST_N_FLAG=""
if [ -n "$AVG_LAST_N" ]; then
  AVG_LAST_N_FLAG="--avg_last_n $AVG_LAST_N"
fi

# Build audio type token flag
AUDIO_TOKEN_FLAG=""
if [ "$USE_AUDIO_TYPE_TOKEN" = true ]; then
  AUDIO_TOKEN_FLAG="--use_audio_type_token"
fi

# Build force recompute flag
FORCE_RECOMPUTE_FLAG=""
if [ "$FORCE_RECOMPUTE" = true ]; then
  FORCE_RECOMPUTE_FLAG="--force_recompute"
fi

echo "=============================================="
echo "Evaluating model: ${MODEL_ID}"
echo "Config: ${CFG_FILE}"
echo "Datasets: ${DATASETS}"
echo "Device: ${DEVICE}"
echo "Output: ${MODEL_OUTPUT_DIR}"
echo "Segment size: ${SEGMENT_SIZE}s"
echo "Sample rate: ${NEW_FREQ}Hz"
echo "Use audio type token: ${USE_AUDIO_TYPE_TOKEN}"
if [ -n "$CKPT_STEP" ]; then
  echo "Checkpoint step: ${CKPT_STEP}"
fi
if [ -n "$AVG_LAST_N" ]; then
  echo "Averaging last: ${AVG_LAST_N} checkpoints"
fi
echo "=============================================="

# GTZAN
if [[ "$DATASETS" == *"gtzan"* ]]; then
  echo ""
  echo ">>> GTZAN (Zero-shot Classification)"
  python downstream_gtzan.py "$CFG_FILE" ../../cfg/downstream/gtzan_zsl.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/gtzan_zsl" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# FMA Small
if [[ "$DATASETS" == *"fma_small"* ]]; then
  echo ""
  echo ">>> FMA Small (Zero-shot Classification)"
  python downstream_fma_small.py "$CFG_FILE" ../../cfg/downstream/fma_small_zsl.gin \
    --metadata_file "${FMA_SMALL_DIR}/fma_metadata/tracks.csv" \
    --output_dir "${MODEL_OUTPUT_DIR}/fma_small_zsl" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MusicCaps
if [[ "$DATASETS" == *"musiccaps"* ]]; then
  echo ""
  echo ">>> MusicCaps (Audio-Text Retrieval)"
  python downstream_retrieval.py \
    --device "$DEVICE" \
    --data_dir "$MUSICCAPS_DIR" \
    --dataset music_caps \
    --cfg_file "$CFG_FILE" \
    --audio-setup clap \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# SongDescriber
if [[ "$DATASETS" == *"song_describer"* ]]; then
  echo ""
  echo ">>> SongDescriber (Audio-Text Retrieval)"
  python downstream_retrieval.py \
    --device "$DEVICE" \
    --data_dir "$SONG_DESCRIBER_DIR" \
    --dataset song_describer \
    --cfg_file "$CFG_FILE" \
    --audio-setup clap \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MagnaTagATune
if [[ "$DATASETS" == *"mtt"* ]]; then
  echo ""
  echo ">>> MagnaTagATune (Autotagging)"
  python downstream_mtt.py "$CFG_FILE" ../../cfg/downstream/mtt_autotagging.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/mtt_autotagging" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# DimSim
if [[ "$DATASETS" == *"dimsim"* ]]; then
  echo ""
  echo ">>> DimSim (Music Similarity)"
  python downstream_dimsim.py \
    --cfg_file "$CFG_FILE" \
    --audio_dir "$DIMSIM_DIR/audio" \
    --metadata_file "$DIMSIM_DIR/metadata/clean-dim-sim.csv" \
    --device "$DEVICE" \
    --output_dir "${MODEL_OUTPUT_DIR}/dimsim" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MTG-Jamendo Genre
if [[ "$DATASETS" == *"jamendo_genre"* ]]; then
  echo ""
  echo ">>> MTG-Jamendo Genre (Autotagging)"
  python downstream_jamendo.py "$CFG_FILE" ../../cfg/downstream/jamendo_genre.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/jamendo_genre" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MTG-Jamendo Instrument
if [[ "$DATASETS" == *"jamendo_instrument"* ]]; then
  echo ""
  echo ">>> MTG-Jamendo Instrument (Autotagging)"
  python downstream_jamendo.py "$CFG_FILE" ../../cfg/downstream/jamendo_instrument.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/jamendo_instrument" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MTG-Jamendo Mood/Theme
if [[ "$DATASETS" == *"jamendo_moodtheme"* ]]; then
  echo ""
  echo ">>> MTG-Jamendo Mood/Theme (Autotagging)"
  python downstream_jamendo.py "$CFG_FILE" ../../cfg/downstream/jamendo_moodtheme.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/jamendo_moodtheme" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MGPHot Autotagging
if [[ "$DATASETS" == *"mgphot_autotagging"* ]]; then
  echo ""
  echo ">>> MGPHot (Autotagging)"
  python downstream_mgphot_autotagging.py "$CFG_FILE" ../../cfg/downstream/mgphot_autotagging.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/mgphot_autotagging" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

# MGPHot Regression
if [[ "$DATASETS" == *"mgphot_regression"* ]]; then
  echo ""
  echo ">>> MGPHot (Genome Regression)"
  python downstream_mgphot_regression.py "$CFG_FILE" ../../cfg/downstream/mgphot_regression.gin \
    --output_dir "${MODEL_OUTPUT_DIR}/mgphot_regression" \
    --segment_size "$SEGMENT_SIZE" \
    --new_freq "$NEW_FREQ" \
    $AUDIO_TOKEN_FLAG \
    $FORCE_RECOMPUTE_FLAG \
    $CKPT_STEP_FLAG \
    $AVG_LAST_N_FLAG
fi

echo ""
echo "=============================================="
echo "Evaluation complete for model: ${MODEL_ID}"
echo "Results saved to: ${MODEL_OUTPUT_DIR}"
echo "=============================================="

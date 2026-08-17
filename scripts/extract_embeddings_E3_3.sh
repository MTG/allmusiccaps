#!/bin/bash

# Define common variables
CONFIG="cfg/config_embedding_extraction_E3_3.gin"
SCRIPT="src/extract_omarrq_audio_embeddings.py"

# Function to submit a job
submit_job() {
  JOB_NAME=$1
  DATA_DIR=$2
  OUT_DIR=$3
  FILELIST=$4

  echo "Submitting job ${JOB_NAME}..."
  echo "  Data: ${DATA_DIR}"
  echo "  Out:  ${OUT_DIR}"
  echo "  Filelist: ${FILELIST}"

  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=<group>
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /projects/<group>/envs/clap/bin/activate

# Create output directory
mkdir -p ${OUT_DIR}

echo "Starting extraction for ${JOB_NAME}"
echo "Data Dir: ${DATA_DIR}"
echo "Out Dir: ${OUT_DIR}"
echo "Filelist: ${FILELIST}"

python ${SCRIPT} ${CONFIG} \\
    --gin-binds="AudioEmbeddingDataModule.data_dir='${DATA_DIR}'" \\
    --gin-binds="AudioEmbeddingDataModule.filelist_path='${FILELIST}'" \\
    --gin-binds="extract_embeddings.out_dir='${OUT_DIR}'"

echo "Job finished"
EOT
}

# Ensure logs directory exists
mkdir -p logs

# Submit Job 1: DiscoTube + M4-RAG
# Assuming filelist exists at some path or empty string if default
submit_job "extract_dt_m4rag" \
  "/scratch/<group>/mmaps_discotube/" \
  "/scratch/<group>/embeddings/omar-rq-multicodebook/discotube/" \
  "/scratch/<group>/discotube/metadata/filelist_allm_v1_v2_m4rag" # Update this with actual filelist path if known, or leave empty

# Submit Job 2: MSD
submit_job "extract_msd" \
  "/scratch/<group>/mmaps_msd/mp3/" \
  "/scratch/<group>/embeddings/omar-rq-multicodebook/msd/" \
  "/scratch/<group>/mmaps_msd/filelist_mmap.txt" # Update this with actual filelist path if known

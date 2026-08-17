#!/bin/bash

# for subset in aa ab ac ad; do
for subset in ab ac; do
  sbatch <<EOT
#!/bin/bash

#SBATCH --job-name extract_msd_${subset}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=debug_%j_output.txt

module load Anaconda3
source ~/.bashrc
conda activate mclap

python extract_omarrq_audio_embeddings.py ../cfg/config_clap_embedding_extraction.gin \
  --gin-binds="AudioEmbeddingDataModule.filelist_path = '/projects/<group>/projects/mtg_text_audio/metadata/msd/filelist_${subset}'" \
  --gin-binds='AudioEmbeddingDataModule.data_dir = "/projects/<group>/audio/incoming/millionsong-audio/mp3"' \
  --gin-binds='extract_embeddings.out_dir = "/projects/<group>/projects/mtg_text_audio/audio_embs/omar-rq-multifeature-25hz-fsq/msd_5s/"'
EOT
done

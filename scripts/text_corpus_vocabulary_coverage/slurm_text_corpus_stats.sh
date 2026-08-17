#!/bin/bash
#SBATCH --job-name=text_corpus_stats
#SBATCH --output=text_corpus_stats_%j.out
#SBATCH --error=text_corpus_stats_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=<group>
#SBATCH --cpus-per-task=4
#SBATCH --qos=acc_resa

source /projects/<group>/envs/clap/bin/activate

export HF_HOME=/scratch/<group>/HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_DATASETS_OFFLINE=1

cd ~/reps/clap-mtg/scripts/text_corpus_vocabulary_coverage

python text_corpus_stats.py --out-path text_corpus_stats.json

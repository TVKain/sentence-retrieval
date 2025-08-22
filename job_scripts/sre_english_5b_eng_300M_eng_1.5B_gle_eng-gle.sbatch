#!/bin/bash
#SBATCH --job-name=sre_english_5b_eng_300M_eng_1.5B_gle_eng-gle
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/sre_english_5b_eng_300M_eng_1.5B_gle_eng-gle-%j.out

# Base path to model checkpoints
BASE_PATH="/home/khanh/sla/sla_cpt/qwen2.5-0.5b_english_wiki_300M_en_1.5Birish_corpus"

# Languages
BASE_LANG="eng_Latn"
TARGET_LANG="gle_Latn"

# Artifact folder with Job ID
ARTIFACT_FOLDER="../json_artifacts/sre_english_5b_eng_300M_eng_1.5B_gle_eng-gle_${SLURM_JOB_ID}"

# Create the artifact folder
mkdir -p "$ARTIFACT_FOLDER"

# Loop through each checkpoint folder in BASE_PATH
for CKPT in "$BASE_PATH"/checkpoint-*; do
    echo "Running experiment for $CKPT"

    python ../sr_experiment.py \
        --model_path "$CKPT" \
        --base "$BASE_LANG" \
        --target "$TARGET_LANG" \
        --json-artifact-out "$ARTIFACT_FOLDER"
done

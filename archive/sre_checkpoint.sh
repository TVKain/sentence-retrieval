#!/bin/bash
#SBATCH --job-name=sre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/%j-sre.out

ENV_FILE=$1

if [ -z "$ENV_FILE" ]; then
    echo "Error: No environment file specified"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found"
    exit 1
fi

# Load the specified environment file
source "$ENV_FILE"

# Create the artifact folder if not exists
mkdir -p "$ARTIFACT_FOLDER"

# Loop through each checkpoint in BASE_PATH
for checkpoint_dir in "$BASE_PATH"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        echo "Running experiment for checkpoint: $checkpoint_dir"

        # Run the Python script for this checkpoint
        python ../sr_experiment.py \
            --model_path "$checkpoint_dir" \
            --base "$BASE_LANG" \
            --target "$TARGET_LANG" \
            --json-artifact-out "$ARTIFACT_FOLDER" 
    fi
done

echo "SRE CHECKPOINT JOB DONE"
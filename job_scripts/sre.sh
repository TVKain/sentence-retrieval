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

# Create the artifact folder
mkdir -p "$ARTIFACT_FOLDER"

export PYTHONUNBUFFERED=1

# Run the Python script
python ../sr_experiment.py \
    --model_path "$BASE_PATH" \
    --base "$BASE_LANG" \
    --target "$TARGET_LANG" \
    --artifact-out "$ARTIFACT_FOLDER" 

echo "SRE SINGLE JOB DONE"
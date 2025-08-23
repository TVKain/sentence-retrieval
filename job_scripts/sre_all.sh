#!/bin/bash
#SBATCH --job-name=sre-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/%j-sre-all.out


# Run all experiments in `envs` folder

ENVS_DIR="envs"

if [ ! -d "$ENVS_DIR" ]; then
    echo "Error: Environments folder $ENVS_DIR not found"
    exit 1
fi

# Loop over all env files
for ENV_FILE in "$ENVS_DIR"/*.sh; do
    if [ ! -f "$ENV_FILE" ]; then
        echo "No environment files found in $ENVS_DIR"
        exit 1
    fi

    echo "Running experiment for $ENV_FILE..."
    source "$ENV_FILE"

    mkdir -p "$ARTIFACT_FOLDER"

    python ../sr_experiment.py \
        --model_path "$BASE_PATH" \
        --base "$BASE_LANG" \
        --target "$TARGET_LANG" \
        --artifact-out "$ARTIFACT_FOLDER"

    echo "Done: $ENV_FILE"
done

echo "ALL SRE JOBS DONE"
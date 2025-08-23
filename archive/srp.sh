#!/bin/bash
#SBATCH --job-name=srp               # Job name
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --partition=physical-gpu              # Partition/queue
#SBATCH --mem=32G                             # RAM
#SBATCH --output=logs/%j-srp.out     # Stdout log

ENV_FILE=$1

if [ -z "$ENV_FILE" ]; then
    echo "Error: No environment file specified"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found"
    exit 1
fi

source $ENV_FILE

echo $JSON_FOLDER

JSON_FILES=("$JSON_FOLDER"/*.json)  # Array of all JSON files in folder

echo "Found ${#JSON_FILES[@]} JSON files. Generating plot..."
python ../sr_plot.py --json "${JSON_FILES[@]}" --plot-artifact-out "$SAVE_FOLDER"

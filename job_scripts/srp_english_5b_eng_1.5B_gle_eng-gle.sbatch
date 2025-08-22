#!/bin/bash
#SBATCH --job-name=sr_plot_all                # Job name
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --partition=physical-gpu              # Partition/queue
#SBATCH --mem=32G                             # RAM
#SBATCH --output=logs/sr_plot_all-%j.out     # Stdout log

# --- User variables ---
JSON_FOLDER="../json_artifacts/sre_english_5b_eng_1.5B_gle_eng-gle"  # Folder containing JSON artifacts
SAVE_FOLDER="../plot_artifacts"                                           # Folder to save plots and metadata

# --- Run the plotting script on all JSON files ---
JSON_FILES=("$JSON_FOLDER"/*.json)  # Array of all JSON files in folder

echo "Found ${#JSON_FILES[@]} JSON files. Generating plot..."
python ../sr_plot.py --json "${JSON_FILES[@]}" --plot-artifact-out "$SAVE_FOLDER"

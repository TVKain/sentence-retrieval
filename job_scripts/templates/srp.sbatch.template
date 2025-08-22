#!/bin/bash
#SBATCH --job-name=<JOB_NAME>             # Job name
#SBATCH --cpus-per-task=<CPUS>           # CPU cores
#SBATCH --partition=<PARTITION>          # Partition/queue
#SBATCH --mem=<MEMORY>                   # RAM
#SBATCH --time=<TIME>                     # Max runtime (optional)
#SBATCH --output=logs/<JOB_NAME>-%j.out  # Stdout log

# --- User variables (replace or export before submitting) ---
JSON_FILES="<JSON_FILES>"                # Space-separated list of JSON artifacts
SAVE_FOLDER="<SAVE_FOLDER>"              # Optional: folder to save plots and metadata

# --- Run the plotting script ---
python <SCRIPT_PATH>/sr_plot.py \
    --json $JSON_FILES \
    --plot-artifact-out $SAVE_FOLDER

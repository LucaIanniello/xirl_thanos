#!/bin/bash
#SBATCH --job-name=wandb_sync
#SBATCH --output=wandb_sync.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100GB

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xirl

# Path to your zip
ZIPFILE=/home/liannello/wandb_to_sync/all_runs.zip
ZIPDIR=$(dirname "$ZIPFILE")

cd "$ZIPDIR"

# Unzip the runs (will extract into the same directory as the zip)
unzip -n "$ZIPFILE"

# Loop through each run directory and sync
for run_dir in offline-run-*; do
    if [ -d "$run_dir" ]; then
        echo ">>> Syncing $run_dir"
        wandb sync "$run_dir"
    fi
done

echo "All runs have been synced."

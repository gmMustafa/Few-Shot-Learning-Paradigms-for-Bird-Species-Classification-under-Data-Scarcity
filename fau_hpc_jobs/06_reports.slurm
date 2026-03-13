#!/bin/bash -l
#SBATCH --job-name=fsl_reports
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python
source ~/venvs/fsl/bin/activate

cd "$HOME/FSL-Code/Project/code"
mkdir -p logs

echo "=== Job started on $(date) ==="
python compare_results.py
python plots/kshot_trend.py
python plots/per_class_heatmap.py
python plots/tsne_plot.py

echo "=== Job finished on $(date) ==="

#!/bin/bash -l
#SBATCH --job-name=relationnet
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
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
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

cd train_evaluate/relationnet
python train.py
python evaluate.py --k-shot 1
python evaluate.py --k-shot 3
python evaluate.py --k-shot 5

echo "=== Job finished on $(date) ==="

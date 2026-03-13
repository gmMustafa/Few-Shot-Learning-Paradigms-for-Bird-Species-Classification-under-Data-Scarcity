#!/bin/bash -l
#SBATCH --job-name=baseline_b15
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
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

cd train_evaluate/baseline/resNet50_b15
python train.py
python evaluate.py
python evaluate_novel_zeroshot.py
python finetune_novel_kshot.py
python evaluate_novel_ncc.py --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --k-shot 5 --distance manhattan

echo "=== Job finished on $(date) ==="

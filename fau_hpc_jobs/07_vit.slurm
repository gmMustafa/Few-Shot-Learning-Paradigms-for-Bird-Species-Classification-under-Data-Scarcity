#!/bin/bash -l
#SBATCH --job-name=vit_backbone
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python
source ~/venvs/fsl/bin/activate

cd "$HOME/FSL-Code/Project/code"
mkdir -p logs

# ---------------------------------------------------------------
# DeiT-S pretrained weights (transferred from local machine).
# Download on a machine with internet:
#   python -c "import timm, torch; m=timm.create_model('deit_small_patch16_224',pretrained=True,num_classes=0); torch.save(m.state_dict(), 'deit_small_patch16_224.pth')"
# Then: scp deit_small_patch16_224.pth <user>@<hpc>:~/
# ---------------------------------------------------------------
DEIT_WEIGHTS="pretrained_weights/deit_small_patch16_224.pth"

echo "=== Job started on $(date) ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
python -c "import timm; print('timm version:', timm.__version__)"
if [ -f "$DEIT_WEIGHTS" ]; then
    echo "DeiT-S weights found at $DEIT_WEIGHTS"
else
    echo "WARNING: DeiT-S weights NOT found at $DEIT_WEIGHTS — will attempt HuggingFace download (may fail on compute nodes)"
fi

# ---------------------------------------------------------------
# Stage 7A — DeiT-S Supervised Baseline (B15 and B10)
# ---------------------------------------------------------------
echo "=== Stage 7A: DeiT-S supervised baseline ==="
cd train_evaluate/vit_backbone/baseline
python train.py                                                           \
    --weights-path "$DEIT_WEIGHTS"                                        # B15/N5
python train.py --config ../../../configs/vit_baseline_b10_config.yaml   \
    --weights-path "$DEIT_WEIGHTS"                                        # B10/N10
cd ../../..

# ---------------------------------------------------------------
# Stage 7B — DeiT-S NCC Novel Evaluation
# ---------------------------------------------------------------
echo "=== Stage 7B: DeiT-S NCC (B15/N5) ==="
cd train_evaluate/vit_backbone/ncc
python evaluate_novel_ncc.py --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --k-shot 5 --distance manhattan

echo "=== Stage 7B: DeiT-S NCC (B10/N10) ==="
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 5 --distance manhattan
cd ../../..

# ---------------------------------------------------------------
# Stage 7C — DeiT-S ProtoNet Episodic Training + Evaluation
# ---------------------------------------------------------------
echo "=== Stage 7C: DeiT-S ProtoNet (B15/N5) ==="
cd train_evaluate/vit_backbone/proto_net
python train.py --weights-path "$DEIT_WEIGHTS"

python evaluate.py --k-shot 1 --distance euclidean
python evaluate.py --k-shot 1 --distance manhattan
python evaluate.py --k-shot 3 --distance euclidean
python evaluate.py --k-shot 3 --distance manhattan
python evaluate.py --k-shot 5 --distance euclidean
python evaluate.py --k-shot 5 --distance manhattan

echo "=== Stage 7C: DeiT-S ProtoNet (B10/N10) ==="
python train.py --config ../../../configs/vit_proto_configs_10b_10n.yaml  \
                --weights-path "$DEIT_WEIGHTS"

python evaluate.py  --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 1 --distance euclidean
python evaluate.py  --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 1 --distance manhattan
python evaluate.py  --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 3 --distance euclidean
python evaluate.py  --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 3 --distance manhattan
python evaluate.py  --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 5 --distance euclidean
python evaluate.py  --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 5 --distance manhattan
cd ../../..

echo "=== Job finished on $(date) ==="

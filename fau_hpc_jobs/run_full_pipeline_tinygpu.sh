#!/bin/bash -l
#SBATCH --job-name=fsl_full_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=logs/fsl_full_eval_%j.out
#SBATCH --error=logs/fsl_full_eval_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

set -e
set -u

echo "============================================================"
echo "JOB START"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA check:"
nvidia-smi || true
echo "============================================================"

 cd ~/Few-Shot-Learning-Paradigms-for-Bird-Species-Classification-under-Data-Scarcity

mkdir -p logs

# Load Python environment
module load python
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
# Replace this with your actual conda env name if different
conda activate fsl_py311

echo "Python:"
python --version
which python

echo "============================================================"
echo "Stage 2 — NCC re-evaluate (b15)"
echo "============================================================"
cd train_evaluate/baseline/resNet50_bX
python evaluate_novel_ncc.py --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --k-shot 5 --distance manhattan
cd ../../..

echo "============================================================"
echo "Stage 2B — NCC re-evaluate (b10)"
echo "============================================================"
cd train_evaluate/baseline/resNet50_bX
python evaluate_novel_ncc.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 5 --distance manhattan
cd ../../..

echo "============================================================"
echo "Stage 3 — ProtoNet re-evaluate (both splits)"
echo "============================================================"
cd train_evaluate/proto_net
python evaluate.py --k-shot 1 --distance euclidean
python evaluate.py --k-shot 1 --distance manhattan
python evaluate.py --k-shot 3 --distance euclidean
python evaluate.py --k-shot 3 --distance manhattan
python evaluate.py --k-shot 5 --distance euclidean
python evaluate.py --k-shot 5 --distance manhattan

python evaluate.py --config ../../configs/proto_configs_10b_10n.yaml --k-shot 1 --distance euclidean
python evaluate.py --config ../../configs/proto_configs_10b_10n.yaml --k-shot 1 --distance manhattan
python evaluate.py --config ../../configs/proto_configs_10b_10n.yaml --k-shot 3 --distance euclidean
python evaluate.py --config ../../configs/proto_configs_10b_10n.yaml --k-shot 3 --distance manhattan
python evaluate.py --config ../../configs/proto_configs_10b_10n.yaml --k-shot 5 --distance euclidean
python evaluate.py --config ../../configs/proto_configs_10b_10n.yaml --k-shot 5 --distance manhattan
cd ../..

echo "============================================================"
echo "Stage 4 — MatchingNet re-evaluate (both splits)"
echo "============================================================"
cd train_evaluate/matchingnet
python evaluate.py --k-shot 1
python evaluate.py --k-shot 3
python evaluate.py --k-shot 5

python evaluate.py --config ../../configs/matchingnet_configs_10b_10n.yaml --k-shot 1
python evaluate.py --config ../../configs/matchingnet_configs_10b_10n.yaml --k-shot 3
python evaluate.py --config ../../configs/matchingnet_configs_10b_10n.yaml --k-shot 5
cd ../..

echo "============================================================"
echo "Stage 5 — RelationNet (b10 re-evaluate + b15 first-time run)"
echo "============================================================"
cd train_evaluate/relationnet

# b15/n5 — NEW
python train.py
python evaluate.py --k-shot 1
python evaluate.py --k-shot 3
python evaluate.py --k-shot 5

# b10/n10 — already trained
python evaluate.py --config ../../configs/relationnet_configs_10b_10n.yaml --k-shot 1
python evaluate.py --config ../../configs/relationnet_configs_10b_10n.yaml --k-shot 3
python evaluate.py --config ../../configs/relationnet_configs_10b_10n.yaml --k-shot 5
cd ../..

echo "============================================================"
echo "Stage 6 — Generate all outputs"
echo "============================================================"
python compare_results.py
python plots/kshot_trend.py
python plots/per_class_heatmap.py
python plots/tsne_plot.py --model protonet    --config configs/proto_configs.yaml
python plots/tsne_plot.py --model matchingnet --config configs/matchingnet_configs.yaml
python plots/tsne_plot.py --model relationnet --config configs/relationnet_configs.yaml
python plots/tsne_plot.py --model protonet    --config configs/proto_configs_10b_10n.yaml
python plots/tsne_plot.py --model matchingnet --config configs/matchingnet_configs_10b_10n.yaml
python plots/tsne_plot.py --model relationnet --config configs/relationnet_configs_10b_10n.yaml

echo "============================================================"
echo "Stage 7 — DeiT-S ViT backbone (NCC + ProtoNet, both splits)"
echo "============================================================"
cd train_evaluate/vit_backbone/baseline
python train.py
python train.py --config ../../../configs/vit_baseline_b10_config.yaml
cd ../../..

cd train_evaluate/vit_backbone/ncc
python evaluate_novel_ncc.py --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --k-shot 5 --distance manhattan

python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 1 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 1 --distance manhattan
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 3 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 3 --distance manhattan
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 5 --distance euclidean
python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 5 --distance manhattan
cd ../../..

cd train_evaluate/vit_backbone/proto_net
python train.py
python evaluate.py --k-shot 1 --distance euclidean
python evaluate.py --k-shot 1 --distance manhattan
python evaluate.py --k-shot 3 --distance euclidean
python evaluate.py --k-shot 3 --distance manhattan
python evaluate.py --k-shot 5 --distance euclidean
python evaluate.py --k-shot 5 --distance manhattan

python train.py    --config ../../../configs/vit_proto_configs_10b_10n.yaml
python evaluate.py --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 1 --distance euclidean
python evaluate.py --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 1 --distance manhattan
python evaluate.py --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 3 --distance euclidean
python evaluate.py --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 3 --distance manhattan
python evaluate.py --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 5 --distance euclidean
python evaluate.py --config ../../../configs/vit_proto_configs_10b_10n.yaml --k-shot 5 --distance manhattan
cd ../../..

echo "============================================================"
echo "JOB FINISHED SUCCESSFULLY"
echo "Date: $(date)"
echo "============================================================"
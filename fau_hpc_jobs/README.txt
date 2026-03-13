FAU HPC Slurm job pack for the FSL thesis pipeline
==================================================

Files included:
- 01_baseline_fs.slurm
- 02_baseline_b15.slurm
- 03_protonet.slurm
- 04_matchingnet.slurm
- 05_relationnet.slurm
- 06_reports.slurm
- submit_all.sh

Assumptions:
- Project path on HPC: $HOME/FSL-Code/Project/code
- Python virtual environment: ~/venvs/fsl
- Python module is available via: module load python
- GPU partition: a100
- Each training/eval script is single-GPU

How to use:
1. Copy these files to your FAU HPC working directory.
2. Make the launcher executable:
   chmod +x submit_all.sh
3. Submit everything:
   ./submit_all.sh

Useful commands:
- squeue -u $USER
- scontrol show job <JOB_ID>
- scancel <JOB_ID>

Before running real jobs, test one script interactively:
- salloc --gres=gpu:a100:1 --partition=a100 --time=00:30:00
- module load python
- source ~/venvs/fsl/bin/activate
- cd $HOME/FSL-Code/Project/code/train_evaluate/proto_net
- python evaluate.py --k-shot 1 --distance euclidean

Adjust if needed:
- If your cluster uses a different GPU partition, change:
  #SBATCH --partition=a100
  #SBATCH --gres=gpu:a100:1
- If you use conda instead of venv, replace the activation line.
- If your project path differs, update the cd line in all scripts.

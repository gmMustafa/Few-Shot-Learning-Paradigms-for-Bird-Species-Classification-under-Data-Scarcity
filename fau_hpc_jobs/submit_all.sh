#!/bin/bash
set -euo pipefail

jid1=$(sbatch --parsable 01_baseline_fs.slurm)
echo "Submitted 01_baseline_fs.slurm as job $jid1"

jid2=$(sbatch --parsable 02_baseline_b15.slurm)
echo "Submitted 02_baseline_b15.slurm as job $jid2"

jid3=$(sbatch --parsable 03_protonet.slurm)
echo "Submitted 03_protonet.slurm as job $jid3"

jid4=$(sbatch --parsable 04_matchingnet.slurm)
echo "Submitted 04_matchingnet.slurm as job $jid4"

jid5=$(sbatch --parsable 05_relationnet.slurm)
echo "Submitted 05_relationnet.slurm as job $jid5"

jid6=$(sbatch --parsable --dependency=afterok:${jid1}:${jid2}:${jid3}:${jid4}:${jid5} 06_reports.slurm)
echo "Submitted 06_reports.slurm as dependent job $jid6"

echo "All jobs submitted successfully."
echo "Monitor with: squeue -u \$USER"

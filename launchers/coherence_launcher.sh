#!/bin/bash -l
#SBATCH --job-name=coherence
#SBATCH --output=coherence.log
#SBATCH --open-mode=append
#
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --time=0-120:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#

module load lang/Python

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"

python coherence_analysis.py
#!/bin/bash
# 
#SBATCH --job-name=ours_tran
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --output=slurm_log/rush_10.log


# load module
module load miniconda3
source activate py3.10

# Run command
python ./model_code/ours_rush.py

#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1

module load python/3.9-anaconda
module load cudnn/8.6.0.163-11.8
module load cuda/11.8.0
module load git/2.35.2
source /home/hpc/b114cb/b114cb13/.env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=0

python ESM_features_batch.py

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

pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

python ESM_features_batch.py

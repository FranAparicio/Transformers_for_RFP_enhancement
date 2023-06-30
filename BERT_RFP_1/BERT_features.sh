#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1

module load python/3.9-anaconda
module load cudnn/8.6.0.163-11.8
module load cuda/11.8.0
source /home/hpc/b114cb/b114cb13/.env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=0

cp RFPs3max.csv RFPs3max_embed.csv
python BERT_features.py RFPs3.fasta RFPs3max_embed.csv output_filtered_BERTx450_5e-7_10

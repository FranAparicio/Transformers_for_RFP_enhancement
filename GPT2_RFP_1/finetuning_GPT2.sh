#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1

module load python/3.9-anaconda
module load cudnn/8.6.0.163-11.8
module load cuda/11.8.0
source /home/hpc/b114cb/b114cb13/.env/bin/activate

python run_clm.py \
	--model_name_or_path="nferruz/ProtGPT2" \
	--train_file="final_set_coll_RFPs3_filtered.txt" \
	--validation_split_percentage="5" \
	--tokenizer_name="nferruz/ProtGPT2" \
	--do_train \
	--do_eval \
	--output_dir="output_filtered_GPTx200_5e-6_5" \
	--learning_rate="5e-6" \
	--per_device_train_batch_size="1" \
	--logging_steps="50" \
        --eval_steps="50" \
	--save_steps="500" \
        --num_train_epochs="200" \
	--evaluation_strategy="steps"

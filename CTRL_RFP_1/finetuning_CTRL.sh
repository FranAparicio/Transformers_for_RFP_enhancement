#!/bin/bash -l
#
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1

module load python/3.9-anaconda
module load cudnn/8.6.0.163-11.8
module load cuda/11.8.0
source /home/hpc/b114cb/b114cb13/.env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=0

python 5.run_clm-post.py \
	--tokenizer_name="nferruz/1.24.3.1" \
	--do_train \
	--do_eval \
	--output_dir="./output_CTRLx30_0.8e-04_good" \
	--overwrite_output_dir \
	--evaluation_strategy="steps" \
	--eval_steps="5" \
	--logging_steps="5" \
	--save_steps="500" \
	--num_train_epochs="35" \
	--per_device_train_batch_size="1" \
	--per_device_eval_batch_size="4" \
	--cache_dir '.' \
	--save_total_limit="2" \
	--learning_rate="0.8e-04" \
	--dataloader_drop_last="True" \
	--model_type="gpt2" \
	--config_name="AI4PD/ZymCTRL" \
	--gradient_accumulation_steps="4"

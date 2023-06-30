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

python run_mlm.py \
        --output_dir="./output_filtered_BERTx450_1e-6_10" \
	--overwrite_output_dir \
	--model_name_or_path="Rostlab/prot_bert_bfd" \
        --config_name="Rostlab/prot_bert_bfd" \
        --tokenizer_name="Rostlab/prot_bert_bfd" \
        --train_file="final_set_RFPs3_filtered.txt" \
        --validation_split_percentage="10" \
        --max_seq_length="512" \
        --per_device_train_batch_size="1" \
        --per_device_eval_batch_size="1" \
        --adafactor \
        --learning_rate="1e-6" \
	--logging_steps="200" \
	--eval_steps="200" \
	--save_steps="1500" \
	--num_train_epochs="450" \
	--evaluation_strategy="steps" \
	--line_by_line True \
	--do_train \
	--do_eval \
	--seed="42"

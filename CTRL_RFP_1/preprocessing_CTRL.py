import random
import transformers
from transformers import AutoTokenizer

# 1. Read the source file
with open('RFPs3_filtered.fasta', 'r') as fn:
    data = fn.readlines()
    fn.close()

# Put sequences into dictionary
sequences={}
for line in data:
    if '>' in line:
        name = line.strip()
        sequences[name] = ['1.24.3.1'] # modify with the actual EC class.
        continue
    sequences[name].append(line.strip())

# Process fasta files to be in single string - run this part only if the fastas were formated to 60 characters
"""
processed_sequences = {}
for name, sequence in sequences.items():
    processed_sequences[f"{sequence[0]};{name}"] = ''.join([x for x in sequence[1:]])
"""

# Shuffle sequences
sequences_list = [(key,value) for key,value in sequences.items()]
random.shuffle(sequences_list)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL')

# the objective is to get here strings, that when tokenized, will span a window length of 1024.
# for each sequence group its length and untokenized string

print("procesing dataset")
processed_dataset = []
for i in sequences_list:
    # length of the control code
    label = i[0].split(';')[0]
    sequence = i[1]#.strip() !!! It gave me problems
    separator = '<sep>'
    control_code_length = len(tokenizer(label+separator)['input_ids'])
    available_space = 1021 - control_code_length # It is not 1024 because '<|endoftext|>', and start and end

    # Option 1: the sequence is larger than the available space (3-4% of sequences in BRENDA are over 1024)
    if len(sequence) > available_space:
        total_length = control_code_length + len(sequence[:available_space]) + 1
        seq = f"{label}{separator}{sequence[:available_space]}<|endoftext|>"
        processed_dataset.append((total_length, seq))

    # Option 2 & 3: The sequence fits in the block_size space with or without padding
    else:
        total_length = control_code_length + len(sequence) + 3
        # in this case the sequence does not fit with the start/end tokens
        seq = f"{label}{separator}<start>{sequence}<end><|endoftext|>"
        processed_dataset.append((total_length, seq))

# Helper function to group sequences
def grouper(iterable):
    prev = None
    group = ''
    total_sum = 0
    for item in iterable:
        if prev is None or item[0] + total_sum < 1025:
            group += item[1]
            total_sum += item[0]
        else:
            total_sum = item[0]
            yield group
            group = item[1]
        prev = item
    if group:
        total_sum = 0
        yield group

# Group sequences
print("grouping processed dataset")
grouped_dataset=dict(enumerate(grouper(processed_dataset),1))

# Save the processed file out
fn = open("./1.24.3.1_processed.txt",'w')
for key,value in grouped_dataset.items():
    fn.write(value)
    fn.write("\n")
fn.close()

fn = open("./1.24.3.1_processed.txt",'w')
for key,value in grouped_dataset.items():
    padding_len = 1024 - len(tokenizer(value)['input_ids'])
    padding = "<pad>"*padding_len
    print(len(tokenizer(value+padding)['input_ids']))
    fn.write(value+padding)
    fn.write
    fn.write("\n")
fn.close()

from datasets import load_dataset
from transformers.testing_utils import CaptureLogger

# Load the tokenizer again
# tokenizer = AutoTokenizer.from_pretrained('/agh/projects/noelia/NLP/zymCTRL/dataset_preparation/tokenizer')
# I masked it because it wasn't finding the repository. So I will leave the tokenizer previously loaded.

#Load the data files
data_files = {}
dataset_args = {}
validation_split_percentage = 10 # for a split 90/10
data_files["train"] = './1.24.3.1_processed.txt'
extension = "text"
raw_datasets = load_dataset(extension, data_files=data_files, cache_dir='.', **dataset_args)
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

# Load datasets using the HF datasets library:
raw_datasets["train"] = load_dataset(extension,
                data_files=data_files,
                split=f"train[{validation_split_percentage}%:]",
                cache_dir='.',
                **dataset_args,)

raw_datasets["validation"] = load_dataset(extension,
                                          data_files=data_files,
                                          split=f"train[:{validation_split_percentage}%]",
                                          cache_dir='.',
                                          **dataset_args,)

def tokenize_function(examples):
    " This function tokenizes input"
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples["text"])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
        )
    return output

# tokenize in parallel
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=32,
    remove_columns=['text'],
    load_from_cache_file = False,
    desc="Running tokenizer on dataset",
)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

train_dataset.save_to_disk('./dataset/train')
eval_dataset.save_to_disk('./dataset/eval')

# This has saved the datasets tokenized. Now we need to group them into the block size of 1024
from datasets import load_from_disk

train_dataset = load_from_disk('./dataset/train') # The original format was: ./8.1.1.1/dataset/train, but it doesn't exit !!!!
eval_dataset = load_from_disk('./dataset/eval') # Same as before !!!!

from datasets.dataset_dict import DatasetDict
tokenized_datasets = DatasetDict()

tokenized_datasets["train"] = train_dataset
tokenized_datasets["validation"] = eval_dataset

block_size = 1024
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop,
    # you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=124,
    load_from_cache_file=False,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

train_dataset.save_to_disk('./dataset/train2')
eval_dataset.save_to_disk('./dataset/eval2')

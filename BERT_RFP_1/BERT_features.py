import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import pandas as pd
import os
import requests
import sys
from tqdm.auto import tqdm

# Set the arguments and files
if len(sys.argv) < 2:
    print("Usage: python script.py input_file output_file model")
    sys.exit(1)
input_file = sys.argv[1]
output_file = sys.argv[2]
model = sys.argv[3]

# Load the vocabulary and model
tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)
model = AutoModel.from_pretrained(model)

# Load the model into the GPU if available
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

# Load and preprocess sequences from the set
with open(input_file,"r") as fasta:
        sequences = []
        for line in fasta:
                if line.startswith(">"):
                        continue
                else:
                        sequence = " ".join(line.strip())
                        sequences.append(sequence)

sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

# Extract the sequences features and convert the output to numpy if needed
embeddings = fe(sequences)

embeddings = np.array(embeddings,dtype=object)
print(embeddings.shape)

# Append the sequences at the end of each csv line (embedding column)
csv = pd.read_csv(output_file)
embeddings_df = pd.DataFrame(embeddings, columns=["embedding"])
csv = pd.concat([csv, embeddings_df], axis=1)
csv.to_csv(output_file,index=False)


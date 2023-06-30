import torch
import esm

# Path to multifasta file
fasta_file="/home/atuin/b114cb/b114cb13/RFPs3.fasta"

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Read multifasta file and extract sequences
def read_multifasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = None
        sequence = ''
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if sequence != '':
                    sequences.append((header, sequence))
                    sequence = ''
                header = line[1:]
            else:
                sequence += line
        if sequence != '':
            sequences.append((header, sequence))
    return sequences

# Prepare data from multifasta file
data = read_multifasta(fasta_file)
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))


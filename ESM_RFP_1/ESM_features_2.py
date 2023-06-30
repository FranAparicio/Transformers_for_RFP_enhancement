import torch
import esm

# Path to multifasta file
fasta_file = "/home/atuin/b114cb/b114cb13/RFPs3.fasta"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Read multifasta file and extract sequences
def read_multifasta(file_path):
    sequences = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
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

# Split data into smaller batches
batch_size = 8
data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

# Process each batch separately
for batch in data_batches:
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Transfer data to GPU
    batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations (on GPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33].to("cpu")

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # Transfer sequence representations to GPU
    sequence_representations = torch.stack(sequence_representations).to(device)

    # Look at the unsupervised self-attention map contact predictions
    import matplotlib.pyplot as plt
    for (_, seq), tokens_len, attention_contacts in zip(batch, batch_lens, results["contacts"]):
        plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        plt.title(seq)
        plt.show()


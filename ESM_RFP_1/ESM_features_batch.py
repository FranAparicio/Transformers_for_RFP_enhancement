import torch
import esm
import os

# Path to multifasta file
fasta_file = "/home/atuin/b114cb/b114cb13/ESM_RFP_1/CTRL_library_500.fasta"
batch_size = 5  # Number of sequences to process per batch
output_dir = "ESM_embeddings_CTRL_big_test"

# Load ESM-2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # Example set
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# model, alphabet = esm.pretrained.esm2_t48_15B_UR50D() # Runs out of memory

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
                header = line[1:].lstrip('\ufeff') # Remove the BOM character
            else:
                sequence += line
        if sequence != '':
            sequences.append((header, sequence))
    return sequences

# Prepare data from multifasta file
data = read_multifasta(fasta_file)

# Create directory to save embeddings
os.makedirs(output_dir, exist_ok=True)

# Process sequences in batches
for batch_start in range(0, len(data), batch_size):
    batch_data = data[batch_start:batch_start+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
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

    # Look at the unsupervised self-attention map contact predictions
    for j, (header, seq) in enumerate(batch_data):
        # plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        # plt.title(seq)
        # plt.show()
        print(f"Header: {header}")
        print(f"Sequence: {seq}")
        print()

        # Save representation to file
        index = batch_start + j
        filename = os.path.join(output_dir, f"{data[index][0]}.pt")
        torch.save(sequence_representations[j], filename)


import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math

# Setting the desired variables
label = "1.24.3.1"
fine_model = "./output_CTRLx35_0.8e-04_good"
folder = "./CTRL_denovo_final/"
batch_size = 100

def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

def main(label, model,special_tokens,device,tokenizer):
    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
           do_sample=True,
           num_return_sequences=10) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.

    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]
    return sequences

if __name__=='__main__':
    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(fine_model) # change to ZymCTRL location
    model = GPT2LMHeadModel.from_pretrained(fine_model).to(device) # change to ZymCTRL location
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    # change to the appropriate BRENDA EC classes
    labels=[label] # oxidoreductases. You can put as many labels as you want.

    for label in tqdm(labels):
        # We'll run 100 batches per label. 20 sequences will be generated per batch.
        for i in range(0,batch_size):
            sequences = main(label, model, special_tokens, device, tokenizer)
            for key,value in sequences.items():
                for index, val in enumerate(value):
                    # Sequences will be saved with the name of the label followed by the batch index,
                    # and the order of the sequence in that batch.
                    fn = open(f"{folder}/{label}_{i}_{index}.fasta", "w")
                    fn.write(f'>{label}_{i}_{index}\t{val[1]}\n{val[0]}')
                    fn.close()


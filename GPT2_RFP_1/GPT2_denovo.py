from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import math

# Setting the folders and variables
model_trained = "./output_filtered_GPTx150_5e-6_5"
output_file = "./GPT2_library_large_4.fasta"
batches = 250

# Setting the device
if torch.cuda.is_available():
        device = torch.device("cuda")
else:
     	device = torch.device("cpu")

# Setting the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_trained)
model = AutoModelForCausalLM.from_pretrained(model_trained).to(device)

# Function to calculate perplexities
def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

# Printing the perplexity of each sequence in a csv format
protgpt2 = pipeline('text-generation', model=model_trained)

def generatingBatch(output_file):
	seqs = protgpt2("<|endoftext|>", max_length=1024, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=20, eos_token_id=0)
	new_dict = {}
	for entry in seqs:
		sequence = entry["generated_text"].strip()
		perplexity = calculatePerplexity(sequence, model, tokenizer)
		new_dict[perplexity] = sequence[13:].replace("\n",'')
	sorted_dict = {k: new_dict[k] for k in sorted(new_dict, key=int)}
	top_protein = next(iter(sorted_dict.items()))
	header, sequence = top_protein
	output = open(output_file,"a")
	output.write(f'>{header}\n')
	output.write(sequence + "\n")
	output.close()

for batch in range(batches):
	generatingBatch(output_file)

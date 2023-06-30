from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import math

# Setting the folders and variables
model_trained = "./output_filtered_GPTx150_5e-6_5"
output_folder = "./GPT_denovo_set_final_quick/"
batches = 50
sequences_per_batch = 20

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_trained)
model = AutoModelForCausalLM.from_pretrained(model_trained).to(device)

# Function to calculate perplexities
def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

# Generate and sort a batch of sequences
def generate_and_sort_batch(batch_num, sequences_per_batch):
        protgpt2 = pipeline('text-generation', model=model_trained)
        seqs = protgpt2("<|endoftext|>", max_length=1024, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=sequences_per_batch, eos_token_id=0)
        sorted_seqs = sorted(seqs, key=lambda x: calculatePerplexity(x['generated_text'].strip()[13:].replace("\n",""), model, tokenizer))

        for i, entry in enumerate(sorted_seqs):
            sequence = entry["generated_text"].strip()[13:].replace("\n","")
            perplexity = calculatePerplexity(sequence, model, tokenizer)
            header = f"{perplexity:.4f}"
            filename = f"{output_folder}_{batch_num}_{i}.fasta"
            with open(filename, "w") as output_file:
                output_file.write(f">{header}\n")
                output_file.write(sequence + "\n")

# Generate batches of sequences
for batch in range(batches):
    generate_and_sort_batch(batch, sequences_per_batch)

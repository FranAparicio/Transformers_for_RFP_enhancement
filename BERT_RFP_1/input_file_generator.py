import re

input_file = "final_set.fasta"

def multifasta(input_file):
	# Read the input .fasta
	output_file = re.sub(".fasta",".txt",input_file)
	with open(input_file) as fasta:
		sequences = ""
		for line in fasta:
			if line.startswith(">"):
				continue
			else:
				sequence = " ".join(line.strip())
				sequence = re.sub(r"[UZOB]", "X", sequence)
				sequences += "[CLS] " + sequence + " [SEP]\n"
	# Write the output .txt
	output_file = re.sub(".fasta",".txt",input_file)
	with open(output_file, "w") as txt:
		formatted_sequences=(sequences).strip()
		txt.write(formatted_sequences)

multifasta(input_file)

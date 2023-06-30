import re

input_file = "final_set.fasta"

def multifasta(input_file):
	# Read the input .fasta
	output_file = re.sub(".fasta",".txt",input_file)
	with open(input_file) as fasta:
		sequences = ""
		for line in fasta:
			if line.startswith(">"):
				sequences += "<|endoftext|>" + "\n"
			else:
				sequences += line.strip() + "\n"
	# Write the output .txt
	output_file = re.sub(".fasta",".txt",input_file)
	with open(output_file, "w") as txt:
		txt.write(sequences)

multifasta(input_file)

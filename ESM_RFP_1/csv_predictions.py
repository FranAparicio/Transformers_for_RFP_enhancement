from Bio import SeqIO
import csv

input_file = "GPT2_library_500.fasta"  # Replace with the path to your multifasta file
output_file = "GPT2_library_500.csv"  # Replace with the desired output file path

def generate_csv_from_multifasta(input_file, output_file):
    sequences = list(SeqIO.parse(input_file, "fasta"))

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ["perplexity", "ID", "sequence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i, sequence in enumerate(sequences):
            writer.writerow({"perplexity": sequence.description,
                             "ID": i + 1,
                             "sequence": str(sequence.seq)})

generate_csv_from_multifasta(input_file, output_file)


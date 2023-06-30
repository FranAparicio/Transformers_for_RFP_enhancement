import re

# Read the contents of the text file
with open('GPT2_library_500.fasta', 'r') as file:
    data = file.read()

# Extract perplexity values from the headers using regular expressions
perplexities = re.findall(r'>([\d.]+)', data)

# Convert perplexity values to floats
perplexities = [float(p) for p in perplexities]

# Calculate statistics
minimum = min(perplexities)
maximum = max(perplexities)
mean = sum(perplexities) / len(perplexities)
above_2_count = sum(1 for p in perplexities if p > 2)

# Find the maximum sequence
max_index = perplexities.index(maximum)
sequences = re.findall(r'>[\d.]+\n([A-Z]+)', data)
max_sequence = sequences[max_index]

# Print the statistics
print("Perplexity Statistics:")
print("Minimum: {:.6f}".format(minimum))
print("Maximum: {:.6f}".format(maximum))
print("Mean: {:.6f}".format(mean))
print("Number of Perplexity Values Above 2: {}".format(above_2_count))
print("Maximum Sequence: {}".format(max_sequence))


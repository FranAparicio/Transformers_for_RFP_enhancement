import os
import csv
import torch
import shutil

# Relevant files and directories
directory = 'ESM_embeddings_CTRL_big_test'
input_csv_file = 'CTRL_library_500.csv'
output_csv_file = 'CTRL_library_500_ESM_big_test.csv'

# Copy the original CSV file to the output file
shutil.copyfile(input_csv_file, output_csv_file)

# Read the existing CSV file
data = []
with open(output_csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    fieldnames = csv_reader.fieldnames
    data = list(csv_reader)

# Iterate through the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a PyTorch model file
    if filename.endswith('.pt'):
        file_path = os.path.join(directory, filename)
        # Extract the UUID from the filename (excluding the file extension)
        perplexity = os.path.splitext(filename)[0]

        # Load the embedding dictionary from the file
        embeddings = torch.load(file_path)
        embedding = embeddings.tolist()  # Extract the embedding as a list

        # Find the corresponding row in the CSV file based on the UUID
        for row in data:
            if row['perplexity'] == perplexity:
                # Append the embedding as a new column
                row['embedding'] = embedding
                break

# Write the updated data to the CSV file
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames + ['embedding'])
    writer.writeheader()
    writer.writerows(data)

print("Embeddings appended to the CSV file successfully!")


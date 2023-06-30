import os
import csv
import torch
import shutil

# Relevant files and directories
directory = 'ESM_embeddings_CTRL_big_test'

# Iterate through the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a PyTorch model file
    if filename.endswith('.pt'):
        file_path = os.path.join(directory, filename)
        # Extract the UUID from the filename (excluding the file extension)
        uuid = os.path.splitext(filename)[0]

        # Load the embedding dictionary from the file
        embedding_dict = torch.load(file_path)
        print(embedding_dict)
"""
        embedding = embedding_dict['mean_representations'][33].tolist()  # Extract the embedding as a list

        # Find the corresponding row in the CSV file based on the UUID
        for row in data:
            if row['uuid'] == uuid:
                # Append the embedding as a new column
                row['embedding'] = embedding
                break

# Write the updated data to the CSV file
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames + ['embedding'])
    writer.writeheader()
    writer.writerows(data)

print("Embeddings appended to the CSV file successfully!")

"""

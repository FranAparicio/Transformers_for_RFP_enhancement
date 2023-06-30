import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Selecting the input file and parameter of interest
data_df = pd.read_csv("/home/atuin/b114cb/b114cb13/RFPs3max_embed.csv")
column_of_interest = "states.0.em_max"

# Set hyperparameters
input_size = 1024
hidden_size = 256
output_size = 1
learning_rate = 0.001
num_epochs = 50

# Dropping those rows which have a missing value for the column of interest
print("Shape before dropping:")
print(data_df.shape)
data_df.dropna(subset=[column_of_interest], inplace=True)
print("Shape after dropping:")
print(data_df.shape)

# Loading the vectors and parameters
input_vectors = data_df["embedding"].values
input_parameters = data_df[column_of_interest].values
print(input_parameters)
new_vectors = []
for x in input_vectors:
    arr = ast.literal_eval (x)
    CLS_token_index = 0
    new_vectors.append(arr[CLS_token_index])
input_vectors = np.array(new_vectors)

# Print the updated shapes
print("Updated shapes:")
print("Input vectors:", input_vectors.shape)
print("Input parameters:", input_parameters.shape)

# Split the data into training and evaluation sets
train_vectors, eval_vectors, train_parameters, eval_parameters = train_test_split(
    input_vectors, input_parameters, test_size=0.2, random_state=42
)

# Define the feedforward neural network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
model = FeedForwardNN(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert the training vectors and parameter values to tensors
train_vectors_tensor = torch.tensor(train_vectors, dtype=torch.float32).to(device)
train_parameters_tensor = torch.tensor(train_parameters, dtype=torch.float32).to(device)

# Create a DataLoader to handle batching of data of training data
train_dataset = torch.utils.data.TensorDataset(train_vectors_tensor, train_parameters_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Convert the evaluation vectors and parameter values to tensors
eval_vectors_tensor = torch.tensor(eval_vectors, dtype=torch.float32).to(device)
eval_parameters_tensor = torch.tensor(eval_parameters, dtype=torch.float32).to(device)

# Create a DataLoader for the evaluation set
eval_dataset = torch.utils.data.TensorDataset(eval_vectors_tensor, eval_parameters_tensor)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Training loop
train_losses = []
eval_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch_vectors, batch_parameters in train_dataloader:
        # Forward pass
        outputs = model(batch_vectors)
        loss = criterion(outputs, batch_parameters)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

   # Evaluation phase
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch_vectors, batch_parameters in eval_dataloader:
            # Forward pass
            outputs = model(batch_vectors)
            loss = criterion(outputs, batch_parameters)
            eval_loss += loss.item()

    # Calculate average evaluation loss
    avg_eval_loss = eval_loss / len(eval_dataloader)

    # Print the loss after each epoch
    train_losses.append(loss.item())
    eval_losses.append(avg_eval_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Eval Loss: {avg_eval_loss}")

# Save the trained model
torch.save(model.state_dict(), "trained_qy_predictor.pth")

# Calculate predictions on the evaluation set
model.eval()
with torch.no_grad():
    predictions = []
    for batch_vectors, batch_parameters in eval_dataloader:
        outputs = model(batch_vectors)
        predictions.extend(outputs.cpu().numpy().flatten())
    predictions = np.array(predictions)

# Plot the learning curve
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, eval_losses, label='Eval Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the predicted vs. actual values
plt.scatter(eval_parameters, predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.grid(True)

# Calculate R^2 score and MSE
r2 = r2_score(eval_parameters, predictions)
mse = mean_squared_error(eval_parameters, predictions)

# Add MSE and R^2 to the plot
plt.text(plt.xlim()[0], plt.ylim()[1], f"R^2 = {r2:.4f}\nMSE = {mse:.4f}", ha="left", va="top", fontsize=10)

# Show the plot
plt.show()

# Function to print predicted and real values for evaluation set
def print_predictions(model, eval_dataloader):
    model.eval()
    with torch.no_grad():
        for batch_vectors, batch_parameters in eval_dataloader:
            # Forward pass
            outputs = model(batch_vectors)
            predictions = outputs.cpu().numpy().flatten()
            real_values = batch_parameters.cpu().numpy().flatten()

            # Print predicted and real values
            for pred, real in zip(predictions, real_values):
                print(f"Predicted: {pred}, Real: {real}")

# Call the function to print predictions on evaluation set
print("Predictions on Evaluation Set:")
print_predictions(model, eval_dataloader)

# Calculate R^2 score and MSE
r2 = r2_score(eval_parameters, predictions)
mse = mean_squared_error(eval_parameters, predictions)

# Print the evaluation metrics
print("R^2 score:", r2)
print("MSE:", mse)


import json  # Import the json modul
import pickle  # Import pickle modul
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# To Load pre-trained embeddings
with open("fasttext_embeddings_25.pkl", "rb") as f:
    fasttext_embeddings = pickle.load(f)


# To Extract local context (left and right neighbors of the aspect term)
def extract_local_context(entry):
    aspect_index = entry["index"]  # the Position of the aspect term in the sentence
    tokens = entry["tokens"]  # the List of tokens in the sentence

    # will Get the left and right neighbors
    left_context = [tokens[aspect_index - 1]] if aspect_index > 0 else []
    right_context = [tokens[aspect_index + 1]] if aspect_index < len(tokens) - 1 else []

    # Combine left and right context
    context_tokens = left_context + right_context

    # Convert tokens to embeddings (use zero vector if token not in embeddings)
    context_embeddings = [
        fasttext_embeddings.get(token, np.zeros(25)) for token in context_tokens
    ]

    # Pad context to ensure it has exactly 2 embeddings (left + right)
    while len(context_embeddings) < 2:
        context_embeddings.append(np.zeros(25))

    # Get embedding for the aspect term
    aspect_embedding = fasttext_embeddings.get(entry["aspect_terms"][0], np.zeros(25))

    return np.array(context_embeddings), np.array(aspect_embedding)


# Function to process the dataset
def process_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Map sentiment labels to integers
    label_mapping = {"positive": 0, "negative": 1, "neutral": 2, "conflict": 3}

    X_context, X_aspect, y = [], [], []
    for entry in data:
        context, aspect = extract_local_context(entry)
        X_context.append(context)
        X_aspect.append(aspect)
        y.append(label_mapping[entry["polarity"]])

    return (
        np.array(X_context, dtype=np.float32),
        np.array(X_aspect, dtype=np.float32),
        np.array(y),
    )


# Load training and validation data
X_train_context, X_train_aspect, y_train = process_data("train_task_2.json")
X_val_context, X_val_aspect, y_val = process_data("val_task_2.json")

# Convert data to PyTorch tensors and create DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train_context), torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val_context), torch.tensor(y_val, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define the Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(
            hidden_dim, 1
        )  # Linear layer to compute attention scores

    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_dim)
        attention_scores = self.attention(x)  # Compute attention scores
        attention_weights = torch.softmax(attention_scores, dim=1)  # Apply softmax
        context_vector = torch.sum(attention_weights * x, dim=1)  # Weighted sum
        return context_vector


# Define the RNN Model with Attention
class RNNClassifierWithAttention(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, output_dim=4):
        super(RNNClassifierWithAttention, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)  # RNN layer
        self.attention = Attention(hidden_dim)  # Attention layer
        self.fc1 = nn.Linear(hidden_dim, 32)  # Fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(32, output_dim)  # Output layer

    def forward(self, x):
        rnn_out, _ = self.rnn(
            x
        )  # RNN output: (batch_size, sequence_length, hidden_dim)
        attention_out = self.attention(rnn_out)  # Apply attention
        out = self.fc1(attention_out)  # Pass through fully connected layer
        out = self.relu(out)  # Apply ReLU
        out = self.fc2(out)  # Final output
        return out


# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = RNNClassifierWithAttention().to(device)
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Optimizer

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(20):  # Train for 20 epochs
    model.train()  # Set model to training mode
    total_train_loss = 0

    # Training phase
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move data to device
        optimizer.zero_grad()  # Clear gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_train_loss += loss.item()  # Accumulate loss

    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store training loss

    # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)  # Compute validation loss
            total_val_loss += loss.item()  # Accumulate validation loss

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == batch_y).sum().item()  # Count correct predictions
            total += batch_y.size(0)  # Total number of samples

    # Calculate average validation loss and accuracy
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # Store validation loss
    val_accuracy = correct / total  # Validation accuracy

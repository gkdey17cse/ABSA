import torch
import torch.nn as nn


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

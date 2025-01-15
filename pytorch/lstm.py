# LSTMs are a type of RNN used for handling long-term dependencies in sequential data. LSTMs solve the vanishing gradient problem in regular RNNs.

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple LSTM architecture
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # Initial hidden state
        c0 = torch.zeros(1, x.size(0), hidden_size)  # Initial cell state
        out, (hn, cn) = self.lstm(x, (h0, c0))  # Forward pass
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

# Example dataset (time series)
input_size = 1  # Example input size (1 feature)
hidden_size = 50  # Number of hidden units
output_size = 1  # Example output size (1 value)

model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop (use real sequential data for practical cases)
for epoch in range(100):  # Number of epochs
    inputs = torch.randn(16, 10, input_size)  # Random input (batch_size, sequence_length, input_size)
    targets = torch.randn(16, output_size)  # Random target

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

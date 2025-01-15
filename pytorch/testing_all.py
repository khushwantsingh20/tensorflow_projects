import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize the network
input_size = 4  # Example: 4 features
hidden_size = 8  # Number of neurons in hidden layer
output_size = 3  # Example: 3 output classes
model = SimpleNN(input_size, hidden_size, output_size)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example training loop
for epoch in range(100):  # Number of epochs
    # Example input and target
    inputs = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)  # Example data
    targets = torch.tensor([0], dtype=torch.long)  # Example label

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

# Test the network
with torch.no_grad():
    test_input = torch.tensor([[6.7, 3.1, 4.4, 1.4]], dtype=torch.float32)
    prediction = model(test_input)
    print("Predicted class:", torch.argmax(prediction, dim=1).item())

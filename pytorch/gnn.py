"""Graph Neural Networks (GNNs) are designed for processing data represented as 
graphs. This can include tasks like node classification, link prediction, and graph 
classification. GNNs are widely used in areas like social network analysis, recommendation
systems, and bioinformatics.
Code: Simple Graph Neural Network (GNN) using PyTorch Geometric"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

# Define a simple GCN (Graph Convolutional Network)
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))  # Graph convolution layer 1
        x = torch.relu(self.conv2(x, edge_index))  # Graph convolution layer 2
        return x

# Example graph data (use real graph data in practice)
x = torch.randn(10, 3)  # Node features (10 nodes with 3 features each)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # Edges (from node to node)
y = torch.randint(0, 2, (10,))  # Node labels (binary classification)

model = GCN(in_channels=3, hidden_channels=16, out_channels=2)
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

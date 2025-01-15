"""Transformers are a class of models introduced for sequence-to-sequence tasks 
(like machine translation) and have become the foundation for many state-of-the-art models in NLP 
(e.g., BERT, GPT). Unlike RNNs and LSTMs, transformers rely entirely on attention mechanisms, 
enabling them to process entire sequences in parallel and capture long-range dependencies more efficiently."""

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_classes):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=2
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded)
        output = output.mean(dim=1)  # Pooling over the sequence length
        output = self.fc(output)
        return output

# Example parameters
input_dim = 1000  # Vocabulary size
embed_dim = 128   # Embedding dimension
num_heads = 8     # Number of attention heads
num_classes = 2   # Binary classification

model = SimpleTransformer(input_dim, embed_dim, num_heads, num_classes)
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop (use real sequential data for practical cases)
for epoch in range(100):  # Number of epochs
    inputs = torch.randint(0, input_dim, (32, 10))  # Random sequence data (batch_size, seq_len)
    targets = torch.randint(0, num_classes, (32,))  # Random targets

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

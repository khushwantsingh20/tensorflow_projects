
"""Autoencoders are a type of neural network used for unsupervised learning.
They are primarily used for dimensionality reduction, denoising, and feature extraction.
An autoencoder consists of an encoder and a decoder.
The encoder maps the input into a lower-dimensional latent space, and the decoder reconstructs 
the input from this compressed representation."""


import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # Output should be between 0 and 1 (image pixel range)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example training loop for autoencoder
model = Autoencoder()
criterion = nn.MSELoss()  # Mean squared error for reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example data (using random data for illustration)
for epoch in range(100):  # Number of epochs
    inputs = torch.randn(64, 784)  # Example input (batch_size, input_size)
    noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)  # Adding noise

    optimizer.zero_grad()

    outputs = model(noisy_inputs)
    loss = criterion(outputs, inputs)  # Compare to the clean input
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

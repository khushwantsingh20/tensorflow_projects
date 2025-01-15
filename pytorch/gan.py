# GANs are used for generating new data instances that resemble a given dataset.
# This typically consists of two networks: a Generator and a Discriminator.

import torch
import torch.nn as nn
import torch.optim as optim

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()  # Output between -1 and 1 (image pixel range)
        )

    def forward(self, z):
        return self.fc(z)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1 (real or fake)
        )

    def forward(self, x):
        return self.fc(x)

# Instantiate models, loss function, and optimizers
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Example training loop (use real data in practice)
for epoch in range(100):  # Number of epochs
    real_images = torch.randn(64, 784)  # Example random real data
    z = torch.randn(64, 100)  # Random noise for generator

    # Train Discriminator
    optimizer_d.zero_grad()
    real_labels = torch.ones(64, 1)
    fake_labels = torch.zeros(64, 1)

    outputs_real = discriminator(real_images)
    loss_real = criterion(outputs_real, real_labels)

    fake_images = generator(z)
    outputs_fake = discriminator(fake_images.detach())
    loss_fake = criterion(outputs_fake, fake_labels)

    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer_d.step()

    # Train Generator
    optimizer_g.zero_grad()
    outputs_fake = discriminator(fake_images)
    loss_g = criterion(outputs_fake, real_labels)
    loss_g.backward()
    optimizer_g.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

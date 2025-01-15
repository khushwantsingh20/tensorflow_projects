"""Siamese Networks are used for tasks where you need to measure the similarity between two
inputs, often applied in problems like face verification, signature verification, and one-shot
learning. 
A Siamese network consists of two identical subnetworks with shared weights."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np  # Import numpy for generating random indices

# Define the Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(128 * 26 * 26, 256)  # Update the input size after convolution
        self.fc2 = nn.Linear(256, 128)  # A smaller final layer before similarity computation
        self.fc3 = nn.Linear(128, 1)  # Output layer

    def forward_one(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        print(f"Shape after conv2: {x.shape}")  # Debug: print shape after convolution
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        # Cosine similarity measure
        similarity = F.cosine_similarity(out1, out2)
        return similarity

# Prepare the MNIST dataset and dataloaders for training
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SiameseNetwork()
criterion = nn.MSELoss()  # Mean squared error loss for similarity regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example of creating pairs of images for the Siamese network
def create_pairs(dataset, num_pairs=1000):
    pairs = []
    labels = []
    for _ in range(num_pairs):
        img1, label1 = dataset[np.random.randint(len(dataset))]
        img2, label2 = dataset[np.random.randint(len(dataset))]
        if label1 == label2:  # Positive pair (same class)
            pairs.append((img1, img2))
            labels.append(1)
        else:  # Negative pair (different class)
            pairs.append((img1, img2))
            labels.append(0)
    return pairs, torch.tensor(labels)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for i, (input1, input2) in enumerate(train_loader):
        optimizer.zero_grad()

        # Create pairs and get similarity labels (1 for similar, 0 for not similar)
        pairs, labels = create_pairs(train_dataset)

        # Convert to tensor and move to the device if needed
        input1, input2 = torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])

        # Forward pass
        similarity = model(input1, input2)

        # Compute loss
        loss = criterion(similarity, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/10], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

print("Finished Training")

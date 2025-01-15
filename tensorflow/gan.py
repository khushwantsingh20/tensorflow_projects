import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 255.0  # Normalize data
X_train = X_train.reshape(X_train.shape[0], 784)

# Generator Model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(784, activation='tanh'))  # Output 28x28 images flattened to 784 pixels
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential()
    model.add(Dense(1024, input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))  # Output probability (real or fake)
    return model

# Combine Generator and Discriminator into a GAN
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False  # We only train the generator when using the combined model
    model.add(discriminator)
    return model

# Compile the discriminator and GAN models
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

generator = build_generator()

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Train GAN
def train_gan(epochs, batch_size=128):
    batch_count = X_train.shape[0] // batch_size
    
    for epoch in range(epochs):
        for _ in range(batch_count):
            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, 100))
            generated_images = generator.predict(noise)
            
            # Get real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]
            
            # Labels for real and fake images
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator via the combined model (discriminator is frozen)
            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = gan.train_on_batch(noise, real_labels)
        
        print(f"{epoch+1}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

        # Generate and save images at regular intervals
        if (epoch + 1) % 100 == 0:
            plot_generated_images(epoch + 1)

# Plot generated images
def plot_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

# Start training GAN
train_gan(epochs=10000, batch_size=128)

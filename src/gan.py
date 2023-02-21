import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, LSTM
from keras.optimizers import Adam


def create_generator(data):
    """Create and compile the generator model"""
    model = Sequential()
    model.add(LSTM(units=256, input_shape=data.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(4, activation="tanh"))
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001, beta_1=0.5)
    )
    return model


def create_discriminator(data):
    """Create and compile the discriminator model"""
    model = Sequential()
    model.add(LSTM(units=256, input_shape=data.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dense(512, input_dim=4))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
    )
    return model


def create_gan(generator, discriminator):
    """Create and compile the GAN model"""
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
    )
    return model


def generate_fake_data(generator, n_samples):
    """Generate fake data using the generator model"""
    noise = np.random.normal(0, 1, size=(n_samples, 100))
    fake_data = generator.predict(noise)
    return fake_data


def train_gan(generator, discriminator, gan, data, n_epochs=10000, batch_size=128):
    """Train the GAN model"""
    for i in range(n_epochs):
        # Train discriminator on real data
        real_data = data[np.random.randint(0, data.shape[0], size=batch_size)]
        real_labels = np.ones((batch_size, 1))
        discriminator_loss_real = discriminator.train_on_batch(real_data, real_labels)

        # Train discriminator on fake data
        fake_data = generate_fake_data(generator, batch_size)
        fake_labels = np.zeros((batch_size, 1))
        discriminator_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        # Train GAN
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        gan_labels = np.ones((batch_size, 1))
        gan_loss = gan.train_on_batch(noise, gan_labels)

        # Print progress
        if i % 100 == 0:
            print(
                f"Epoch: {i}, Discriminator Loss Real: {discriminator_loss_real}, Discriminator Loss Fake: {discriminator_loss_fake}, GAN Loss: {gan_loss}"
            )


def generate_csvs(generator, n_csvs=4, n_rows=250):
    for i in range(n_csvs):
        fake_data = generate_fake_data(generator, n_rows)
        df = pd.DataFrame(fake_data, columns=["track", "note", "velocity", "time"])
        df.to_csv(f"generated_data_{i}.csv", index=False)


real_data = []
for filename in os.listdir("assets/csvs/realCSVs"):
    if filename.endswith(".csv"):
        csv_path = os.path.join("assets/csvs/realCSVs", filename)
        df = pd.read_csv(csv_path)
        real_data.append(df.values)
real_data = np.concatenate(real_data, axis=0)


# Set up models
generator = create_generator(real_data)
discriminator = create_discriminator(real_data)
gan = create_gan(generator, discriminator)

train_gan(generator, discriminator, gan, real_data, n_epochs=10000, batch_size=128)

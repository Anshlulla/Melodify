import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np
import os
import pickle

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5
SPECTOGRAM_PATH = r"C:\Users\jmdgo\saved_data\spectrograms"
TARGET_SHAPE = (256, 752)


def load_data(spectrograms_path, target_shape):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
            if spectrogram.shape[:2] != target_shape:
                spectrogram = tf.image.resize(spectrogram, target_shape).numpy()
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(256, 752, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_data(SPECTOGRAM_PATH, TARGET_SHAPE)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")

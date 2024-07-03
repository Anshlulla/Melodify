import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import pickle

class Autoencoder:
    """
    Autoencoder represents a deep convolutional autoencoder architecture
    with mirrored encoder and decoder parts.
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape  # [w,h,c]
        self.conv_filters = conv_filters  # []
        self.conv_kernels = conv_kernels  # []
        self.conv_strides = conv_strides  # []
        self.latent_space_dim = latent_space_dim  # int
        self._num_conv_layers = len(conv_filters)
        self.encoder = None
        self.decoder = None
        self.model = None
        self._shape_before_bottleneck = None
        self._model_input = None
        self._build()

    def summary(self):
        """
        Prints a summary of the autoencoder model.
        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        """
        Builds the autoencoder model.
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        """
        Builds the encoder part of the autoencoder.
        """
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        """
        Adds the encoder input layer.
        """
        return layers.Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        """
        Creates all convolutional blocks in the encoder.
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Adds a convolutional block to a graph of layers, consisting of
        Conv2D, ReLU layers, and BatchNormalization .
        """
        layer_number = layer_index + 1
        conv_layer = layers.Conv2D(filters=self.conv_filters[layer_index],
                                   kernel_size=self.conv_kernels[layer_index],
                                   strides=self.conv_strides[layer_index],
                                   padding="same",
                                   name=f"encoder_conv_layer_{layer_number}")
        x = conv_layer(x)
        x = layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, conv_layers):
        """
        Flatten the convolutional layers and
        add the bottleneck layer(Dense Layer).
        """
        # keep track of shapes before flattening data, to make it easy to code the decoder part
        self._shape_before_bottleneck = K.int_shape(conv_layers)[1:]  # [batch_size, w, h, c] -> ignore batch_size
        x = layers.Flatten(name="bottleneck")(conv_layers)
        x = layers.Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

    def _build_decoder(self):
        """
        Builds the decoder part of the autoencoder.
        """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        """
        Adds the decoder input layer.
        """
        return layers.Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        """
        Adds a dense layer to the decoder.
        """
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = layers.Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        """
        Adds a reshape layer to the decoder.
        """
        reshape_layer = layers.Reshape(target_shape=self._shape_before_bottleneck, name="decoder_reshape")(dense_layer)
        return reshape_layer

    def _add_conv_transpose_layers(self, reshape_layer):
        """
        Creates all convolutional transpose blocks in the decoder.
        """
        x = reshape_layer
        # loop through all the conv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        """
        Adds a convolutional transpose block to a graph of layers, consisting of
        Conv2DTranspose, ReLU layers, and BatchNormalization .
        """
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = layers.Conv2DTranspose(filters=self.conv_filters[layer_index],
                                                      kernel_size=self.conv_kernels[layer_index],
                                                      strides=self.conv_strides[layer_index],
                                                      padding="same",
                                                      name=f"decoder_conv_transpose_layer_{layer_number}")
        x = conv_transpose_layer(x)
        x = layers.ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{layer_number}")(x)
        return x

    def _add_decoder_output(self, conv_transpose_layers):
        """
        Adds the decoder output layer.
        """
        conv_transpose_layer = layers.Conv2DTranspose(filters=1,
                                                      # since we need to ensure our ouptut matches the input i.e. the grayscale images(colour channels=1)
                                                      kernel_size=self.conv_kernels[0],
                                                      strides=self.conv_strides[0],
                                                      padding="same",
                                                      name=f"decoder_conv_transpose_layer_{self._num_conv_layers}")
        x = conv_transpose_layer(conv_transpose_layers)
        output_layer = layers.Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def compile(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        mse_loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer,
                           loss=mse_loss)

    def train(self, X_train, batch_size, epochs):
        self.model.fit(X_train, X_train,
                       # input and output same since we want the autoencoder to output something similar (if not same) to input
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True)

    def save(self, save_folder="."):
        """
        Saves the model to a directory.
        """
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_it_doesnt_exist(self, folder):
        """
        Creates a folder/directory if it does not exist.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        """
        Saves the parameters of the model.
        """
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        """
        Saves the weights of the model.
        """
        save_path = os.path.join(save_folder, ".weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_folder="."):
        """
        Loads the model from a directory.
        """
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, ".weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        """
        Loads the weights of the model.
        """
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        """
        Reconstructs the given images.
        """
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations


if __name__ == "__main__":
    autoencoder = Autoencoder(input_shape=(28, 28, 1),
                              conv_filters=[32, 64, 64, 64],
                              conv_kernels=[3, 3, 3, 3],
                              conv_strides=[1, 2, 2, 1],  # strides=2 suggests down-sampling our data
                              latent_space_dim=2)
    autoencoder.summary()
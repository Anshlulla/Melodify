# Generative-Music
## Autoencoder
The Autoencoder class represents a deep convolutional autoencoder architecture with mirrored encoder and decoder parts.

### Key Features:
* **Input Shape**: (28, 28, 1)
* **Encoder Layers**: Convolutional layers with ReLU activation and Batch Normalization. Converts input images into a compressed representation in the latent space.
* **Bottleneck**: Dense layer for dimensionality reduction to a specified latent space dimension.
* **Decoder Layers**: Convolutional transpose layers to reconstruct input from latent space. Reconstructs images from the latent space back to the original input shape.
* **Loss Function**: Mean Squared Error (MSE) used for reconstruction loss.
* **Training**: Uses MSE loss to minimize the difference between input and reconstructed images.
* **Saving** **and** **Loading**: Supports saving and loading of model weights and parameters.
  


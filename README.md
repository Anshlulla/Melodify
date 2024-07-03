# Generative-Music
## Autoencoder
The Autoencoder class represents a deep convolutional autoencoder architecture with mirrored encoder and decoder parts.

### Key Features:
* **Encoder Layers**: Convolutional layers with ReLU activation and Batch Normalization. Converts input images into a compressed representation in the latent space.
* **Bottleneck**: Dense layer for dimensionality reduction to a specified latent space dimension.
* **Decoder Layers**: Convolutional transpose layers to reconstruct input from latent space. Reconstructs images from the latent space back to the original input shape.
* **Loss Function**: Mean Squared Error (MSE) used for reconstruction loss.
* **Training**: Uses MSE loss to minimize the difference between input and reconstructed images.
* **Saving** **and** **Loading**: Supports saving and loading of model weights and parameters.

## Preprocessing Pipeline:
This pipeline facilitates preprocessing of audio data for further analysis or machine learning tasks, ensuring uniformity in input sizes and scaling of spectrogram data for model training or other applications.

* **Loader** (Loader class): Responsible for loading an audio file with specified sample rate, duration, and mono/stereo preference using librosa.
* **Padder** (Padder class): Handles padding of the audio signal to match a desired length using numpy's np.pad() function.
* **Log Spectrogram Extractor** (LogSpectrogramExtractor class): Computes the log-scaled spectrogram of the audio signal using Short-Time Fourier Transform (STFT) provided by librosa.
* **MinMax Normalizer** (MinMaxNormaliser class): Normalizes the extracted log spectrogram values to a specified range (typically [0, 1]) and can also perform denormalization.
* **Saver** (Saver class): Saves the processed log spectrograms as numpy arrays (.npy files) and stores min-max normalization values for each spectrogram.
* **Preprocessing Pipeline** (PreprocessingPipeline class): Orchestrates the entire preprocessing process for a directory of audio files:
  1. Loads each audio file.
  2. Checks if padding is needed and applies padding if necessary.
  3. Extracts the log spectrogram.
  4. Normalizes the spectrogram.
  5. Saves the normalized spectrogram and stores associated min-max values.

### Usage in __main__:
* Configuration: Sets parameters like FRAME_SIZE, HOP_LENGTH, DURATION, SAMPLE_RATE, MONO, and directories for saving spectrograms and min-max values.
* Instantiation: Creates instances of Loader, Padder, LogSpectrogramExtractor, MinMaxNormaliser, and Saver.
* Pipeline Execution: Initializes a PreprocessingPipeline with instantiated components.
Calls preprocess() method on the pipeline instance to process audio files located in AUDIO_FILES_DIR.
  


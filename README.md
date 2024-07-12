# Generative-Music

## Dataset

The dataset used for this project is the [GTZAN Dataset for Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data). This dataset contains 1000 audio tracks, each 30 seconds long, categorized into 10 different genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock.

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


## Train:
* **Constants and Configuration**:
  1. LEARNING_RATE, BATCH_SIZE, EPOCHS: Hyperparameters for training.
  2. SPECTOGRAM_PATH: Path to the directory containing spectrogram data.
  3. TARGET_SHAPE: Desired shape for spectrogram images after resizing.

* **Loading Spectrograms**: The load_fsdd function iterates through spectrogram files in SPECTOGRAM_PATH, loads each file using np.load, adds a channel dimension (np.expand_dims), and resizes the spectrogram if its shape doesn't match TARGET_SHAPE.

* **Training Function**: The train function initializes an Autoencoder model with specific architecture parameters (input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim).
It then compiles the model, trains it on x_train data, and returns the trained autoencoder model instance.

* **Main Execution**: 
  1. X_train is loaded using load_fsdd.
  2. The train function is called to train an autoencoder on x_train.
  3. The trained autoencoder model is saved using autoencoder.save("model").


## SoundGenerator
The SoundGenerator class facilitates the generation of audio signals from spectrograms using an autoencoder and additional processing steps.
The SoundGenerator class serves as a bridge between spectrogram data generated or reconstructed by an autoencoder and the actual audio signals that can be played or analyzed further.
* **Generate**:
  1. Inputs: Takes spectrograms and min_max_values (containing normalization information) as input.
  2. Functionality: Uses the autoencoder (ae) to reconstruct spectrograms from the input.
  3. Returns: Generates audio signals from the reconstructed spectrograms using the convert_spectrograms_to_audio method and returns these signals along with latent representations from the autoencoder.

* **Convert Spectrograms to Audio**:

  1. Inputs: Accepts a list of reconstructed spectrograms and corresponding min_max_values.
  2. Functionality:
      *  Reshapes logarithmic spectrograms.
      * Applies denormalization using a MinMaxNormaliser.
      * Converts denormalized spectrograms to linear amplitude spectrograms.
      * Uses the Griffin-Lim algorithm (via librosa.istft) to convert spectrograms into time-domain audio signals.
  3. Returns: A list of generated audio signals corresponding to each input spectrogram.


## **Generate**:
This script facilitates the generation of audio signals from spectrograms using a trained autoencoder, focusing on loading, processing, and saving audio data. It demonstrates a streamlined process for converting spectrograms into playable audio files using deep learning techniques.

* **Load Data**: Loads spectrograms from a directory, adjusts their shape, and returns them along with their file paths.
* **Select Spectrograms**: Randomly selects a specified number of spectrograms, retrieves their min-max normalization values, and returns them.
* **Save Signals**: Saves audio signals as .wav files in a specified directory.

1. Loads a trained autoencoder (ae) and initializes a SoundGenerator instance for audio generation from spectrograms.
2. Loads min-max normalization values from a file (MIN_MAX_VALUES_PATH).
3. Processes spectrograms: loads, selects a subset, generates audio signals from them using sound_generator, and saves both the generated and original audio signals (signals, original_signals) to designated directories (SAVE_DIR_GENERATED, SAVE_DIR_ORIGINAL).



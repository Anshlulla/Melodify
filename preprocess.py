import numpy as np
import librosa
import os
import pickle

class Loader:
    """
    Loader is responsible for loading an audio file.
    """
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0] # returns a tuple: (signal, sample_rate)
        return signal


class Padder:
    """
    Padder is responsible for applying padding to an array.
    """
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        """
        eg: [1,2,3] with 2 items -> [0,0,1,2,3]
        """
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode) # insert/append num_missing_items at the beginning of the array
        return padded_array

    def right_pad(self, array, num_missing_items):
        """
        eg: [1,2,3] with 2 items -> [1,2,3,0,0]
        """
        padded_array = np.pad(array, (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """
    Extracts Log Spectrograms (in dB) from a time-series signal.
    """
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        # short-time fourier transform
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]  # (1 + (frame_size / 2), num_frames)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    """
    Applies min-max normalization to an array. Using range [0,1].
    """
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def normalise(self, array):
        normalised_array = (array - array.min()) / (array.max() - array.min()) # (x - xmin) / (xmax - xmin)
        normalised_array = normalised_array * (self.max_val - self.min_val) + self.min_val
        return normalised_array

    def denormalise(self, normalised_array, original_min_of_array, original_max_of_array):
        array = (normalised_array - self.min_val) / (self.max_val - self.min_val)
        array = array * (original_max_of_array - original_min_of_array) + original_min_of_array
        return array


class Saver:
    """
    Responsible for saving the features, and the min max values which will further be used during reconstruction.
    """
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
        self._ensure_dir_exists(self.feature_save_dir)
        self._ensure_dir_exists(self.min_max_values_save_dir)

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1] # returns [head, tail]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(min_max_values, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(min_max_values, f)

    @staticmethod
    def _ensure_dir_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


class PreprocessingPipeline:
    """
    Preprocesses the audio files in a directory, applying the following steps to each file:
        1. Load the file
        2. Pad the signal (if necessary)
        3. Extracting log spectrograms from the signal
        4. Normalize spectrogram
        5. Save the normalized spectrogram

    Storing the min max values for all the log spectrograms for reconstructing the signal.
    """
    def __init__(self, loader, padder, extractor, normaliser, saver):
        self.loader = loader
        self.padder = padder
        self.extractor = extractor
        self.normaliser = normaliser
        self.saver = saver
        self.min_max_values = {} # {save_path: {"min": min_val, "max": max_val}}
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def preprocess(self, audio_file_dir):
        for root, _, files in os.walk(audio_file_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self._process_file(file_path)
                    print(f"Processed file: {file_path}")
                except Exception as e:
                    print(f"Could not process file {file_path}: {e}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_required(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        normalised_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(normalised_feature, file_path)
        self._store_min_max_values(save_path, feature.min(), feature.max())

    def _is_padding_required(self, signal):
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_values(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 5 # in seconds
    SAMPLE_RATE = 22050
    MONO = True

    # Relative paths
    SPECTOGRAMS_SAVE_DIR = os.path.join(os.getcwd(), "saved_data", "spectrograms")
    MIN_MAX_VALUES_SAVE_DIR = os.path.join(os.getcwd(), "saved_data", "min_max_values")
    AUDIO_FILES_DIR = r"C:\Users\jmdgo\Downloads\archive (9)\Data\genres_original"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTOGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    # preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline(loader, padder, log_spectrogram_extractor, min_max_normaliser, saver)

    preprocessing_pipeline.preprocess(AUDIO_FILES_DIR)

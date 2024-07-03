import os
import pickle
import numpy as np
import soundfile as sf

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = r"C:\Users\jmdgo\Downloads\archive (9)\Data\genres_original"
SAVE_DIR_GENERATED = r"C:\Users\jmdgo\saved_data\samples\generated"
MIN_MAX_VALUES_PATH = r"C:\Users\jmdgo\saved_data\min_max_values\min_max_values.pkl"

def load_data(spectrograms_path, target_shape=(256, 752)):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            spectrogram = adjust_spectrogram_shape(spectrogram, target_shape)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train, file_paths

def adjust_spectrogram_shape(spectrogram, target_shape):
    target_bins, target_frames = target_shape
    bins, frames = spectrogram.shape

    if bins < target_bins:
        pad_bins = target_bins - bins
        spectrogram = np.pad(spectrogram, ((0, pad_bins), (0, 0)), mode='constant')
    elif bins > target_bins:
        spectrogram = spectrogram[:target_bins, :]

    if frames < target_frames:
        pad_frames = target_frames - frames
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_frames)), mode='constant')
    elif frames > target_frames:
        spectrogram = spectrogram[:, :target_frames]

    return spectrogram

def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate=22050):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # initialise sound generator
    ae = Autoencoder.load("model")
    sound_generator = SoundGenerator(ae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_data(SPECTOGRAM_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, 5)

    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectograms_to_audio(sampled_specs, sampled_min_max_values)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)

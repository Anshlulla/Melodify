class SoundGenerator:
    """
    SoundGenerator is responsible for generating audios from spectograms
    """
    def __init__(self, ae, hop_length):
        self.ae = ae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectograms, min_max_values):
        generated_spectograms, latent_representations = self.ae.reconstruct(spectograms)
        signals = self.convert_spectograms_to_audio(generated_spectograms, min_max_values)
        return signals, latent_representations

    def convert_spectograms_to_audio(self, spectograms, min_max_values):
        signals = []
        for spectogram, min_max_value in zip(spectograms, min_max_values):
            # reshape the log spectogram
            log_spectogram = spectogram[:,:,0]
            # apply denormalisation
            denormalised_log_spec = self._min_max_normaliser.denormalise(log_spectogram,
                                                                          min_max_value["min"], min_max_value["max"])
            # log spectogram -> spectogram
            spec = librosa.db_to_amplitude(denormalised_log_spec)
            # apply Griffin-Lim algorithm (used inverse short-time fourier transform)
            signal = librosa.istft(spec, hop_length=self.hop_length)
            # append signal to signals list
            signals.append(signal)

        return signals
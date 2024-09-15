import numpy as np
from scipy.io import wavfile
from scipy import signal

# Load the audio file
filename = 'test.wav'
sample_rate, audio_signal = wavfile.read(filename)

# Check the number of channels in the audio signal
num_channels = audio_signal.shape[1] if audio_signal.ndim == 2 else 1

# Generate a noise signal with the same number of channels as the audio signal
noise_signal = np.random.randn(audio_signal.shape[0], num_channels)

# Set the desired signal-to-noise ratio (SNR) in dB
desired_snr_db = 5

# Calculate the noise amplitude based on the desired SNR
signal_power = np.mean(audio_signal**2, axis=0)
noise_power = signal_power / (10**(desired_snr_db/10))
noise_amplitude = np.sqrt(noise_power) - 90


# Mix the audio signal with the noise signal
noisy_signal = audio_signal + (noise_amplitude * noise_signal)

# Clip the noisy signal to the range [-1, 1]
noisy_signal = np.clip(noisy_signal, -1, 1)

# Export the noisy audio signal
wavfile.write('noisy_audio.wav', sample_rate, (noisy_signal * 32767).astype(np.int16))




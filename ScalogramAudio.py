import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet, spectrogram

# Path to the WAV file
wav_file_path = r"C:\Users\19mlf3\Desktop\MixPre-107u.wav"

# Read the WAV file
sample_rate, data = wav.read(wav_file_path)

# If stereo, take the first channel
if len(data.shape) > 1:
    data = data[:, 0]

# Detect where the signal is non-negligible and trim
threshold = 0.01 * np.max(np.abs(data))  # Define threshold as 1% of the maximum amplitude
nonzero_indices = np.where(np.abs(data) > threshold)[0]
start_idx, end_idx = nonzero_indices[0], nonzero_indices[-1]

# Adjust start and end to remove any additional gaps
extra_trim_time = 0.25  # Time in seconds to trim from start and end
extra_trim_samples = int(extra_trim_time * sample_rate)
start_idx = max(start_idx + extra_trim_samples, 0)
end_idx = min(end_idx - extra_trim_samples, len(data) - 1)

# Extract the trimmed signal
trimmed_data = data[start_idx:end_idx + 1]

# Downsampling to make data more manageable
downsample_factor = 10
downsampled_data = trimmed_data[::downsample_factor]
downsampled_rate = sample_rate // downsample_factor

# Time vectors, resetting to start at 0
trimmed_time = np.linspace(0, len(trimmed_data) / sample_rate, len(trimmed_data))
downsampled_time = np.linspace(0, len(downsampled_data) / downsampled_rate, len(downsampled_data))

# Continuous Wavelet Transform (CWT)
scales = np.arange(1, 256)
cwt_coefficients = cwt(downsampled_data, morlet, scales)

# Short-Time Fourier Transform (STFT)
frequencies, times, stft_spectrogram = spectrogram(downsampled_data, fs=downsampled_rate)

# Align STFT times to start at 0
times = times - times[0]

# Create a grid layout with adjusted width ratios
fig = plt.figure(figsize=(12, 8))
grid = fig.add_gridspec(3, 4, width_ratios=[100, 1, 25, 1], height_ratios=[1, 1, 1])  # Adjusted width ratio

# Time-domain plot (Analyzed Signal)
ax1 = fig.add_subplot(grid[0, :2])
ax1.plot(trimmed_time, trimmed_data, color='blue')
ax1.set_title("Analyzed Signal")
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("")  
ax1.set_xlim(0, trimmed_time[-1])  
ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Add numbers to the x-ticks of the first plot (time-domain signal)
ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

# Wavelet scalogram
ax2 = fig.add_subplot(grid[1, :2])
im2 = ax2.imshow(
    np.abs(cwt_coefficients),
    aspect='auto',
    extent=[0, downsampled_time[-1], scales[-1], scales[0]],
    cmap="jet",
)
ax2.set_title("Wavelet Scalogram")
ax2.set_ylabel("Scales")
ax2.set_xlabel("")  

# Keep tick marks but remove tick labels from the second plot (wavelet scalogram)
ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)

# Colorbar for wavelet scalogram (Attached to the plot)
fig.colorbar(im2, ax=ax2, label="Magnitude", orientation='vertical', fraction=0.02, pad=0.04)

# STFT spectrogram
ax3 = fig.add_subplot(grid[2, :2])
im3 = ax3.pcolormesh(
    times,
    frequencies,
    10 * np.log10(stft_spectrogram),
    shading='auto',
    cmap="jet",
)
ax3.set_title("STFT Spectrogram")
ax3.set_xlabel("Time [seconds]")
ax3.set_ylabel("Frequency [Hz]")
ax3.set_xlim(0, times[-1])  

# Colorbar for STFT spectrogram (Attached to the plot)
fig.colorbar(im3, ax=ax3, label="Power [dB]", orientation='vertical', fraction=0.02, pad=0.04)

# Adjust layout and display
plt.tight_layout()
plt.show()

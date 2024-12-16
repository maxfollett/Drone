import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# Function to load data from the SEMP file
def load_semp_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        try:
            time, amplitude = map(float, line.split())
            data.append(amplitude)
        except ValueError:
            continue  # Skip lines that cannot be parsed

    if not data:
        raise ValueError("No valid numerical data found in the file.")

    return np.array(data), len(lines) / (time if time > 0 else 1)  # Estimate sampling rate

# File SEMP path
file_path = '/Users/19mlf3/Desktop/AA.S0004.PRE.semp'
file_name = os.path.basename(file_path)

# Load the SEMP file
data, rate = load_semp_file(file_path)

# Normalize the data to the range [-1, 1]
max_val = np.max(np.abs(data))
if max_val != 0:
    data = data / max_val

# Create a time vector
time = np.arange(0, len(data)) / rate

# Perform Continuous Wavelet Transform (CWT)
wavelet = 'cmor3.0-1.0'
scales = np.arange(1, 512)
coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1/rate)

# Fourier Transform to compute the amplitude spectrum
N = len(data)
fft_values = fft(data)
fft_freqs = fftfreq(N, 1 / rate)
positive_freqs = fft_freqs[:N // 2]
positive_amplitudes = 2.0 / N * np.abs(fft_values[:N // 2])

# Wavelet shape
wavelet_function, _ = pywt.ContinuousWavelet(wavelet).wavefun(level=10)

# Plotting results
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'Analysis of {file_name}', fontsize=16)

# 1. Time-domain signal (bottom left)
ax1 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
ax1.plot(time, data)
ax1.set_title('Time Domain Signal')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Normalized Amplitude')

# 2. Wavelet Scalogram (top left)
ax2 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
im = ax2.imshow(np.abs(coefficients), extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
                cmap='jet', aspect='auto', vmax=np.max(np.abs(coefficients)) * 0.5)
ax2.set_title('Wavelet Scalogram (CWT)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_yscale('log')
fig.colorbar(im, ax=ax2, label='Magnitude')

# 3. Fourier Amplitude Spectrum (top right, sideways)
ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
ax3.plot(positive_amplitudes, positive_freqs)
ax3.set_title('Fourier Amplitude Spectrum')
ax3.set_xlabel('Amplitude')
ax3.set_ylabel('Frequency [Hz]')
ax3.set_yscale('log')
ax3.set_xscale('linear')

# 4. Wavelet Shape (bottom right)
ax4 = plt.subplot2grid((3, 3), (2, 2))
ax4.plot(wavelet_function)
ax4.set_title('Wavelet Shape')
ax4.set_xlabel('Wavelet Coefficient')
ax4.set_ylabel('Amplitude')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include suptitle
plt.show()

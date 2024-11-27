import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet

# Path to the new WAV file
wav_file_path = r"C:\Users\19mlf3\Desktop\ThrottleUp_ThrottleDown.WAV"

# Read the WAV file
sample_rate, data = wav.read(wav_file_path)

# Check if the data is stereo or mono
if len(data.shape) > 1:
    data = data[:, 0]  # Use the first channel if stereo

# Check the first 10 values to ensure it's not all zeros
print("First 10 values of the raw data:", data[:10])

# Plot the waveform of the original audio data
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, len(data) / sample_rate, len(data)), data)
plt.title("Original Audio Signal")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.show()

# Downsampling to make it more manageable (optional)
downsample_factor = 10  # You can adjust this factor
downsampled_data = data[::downsample_factor]

# Check the first 10 values of the downsampled data
print("Downsampled data (first 10 values):", downsampled_data[:10])

# Create a time vector for plotting
time_vector = np.linspace(0, len(downsampled_data) / (sample_rate / downsample_factor), len(downsampled_data))

# Plot the downsampled audio signal
plt.figure(figsize=(10, 6))
plt.plot(time_vector, downsampled_data)
plt.title("Downsampled Audio Signal")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.show()

# Perform Continuous Wavelet Transform (CWT) using Morlet wavelet
# You can adjust the width or other parameters if needed
coefficients = cwt(downsampled_data, morlet, np.arange(1, 256))

# Check the shape of the coefficients matrix
print(f"CWT coefficients shape: {coefficients.shape}")

# Plot a CWT scalogram (time-frequency representation)
plt.figure(figsize=(12, 6))
plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, time_vector[-1], 1, 255])
plt.colorbar(label="Magnitude")
plt.title("Continuous Wavelet Transform (CWT) Scalogram")
plt.xlabel("Time [seconds]")
plt.ylabel("Scale")
plt.show()

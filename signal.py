import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import urllib.request
from scipy.io.wavfile import write

# Step 1: Download audio from GitHub
url = 'https://raw.githubusercontent.com/erciitb/Convener_assignment_resources/refs/heads/main/signal/modulated_noisy_audio.wav'
filename = 'modulated_noisy_audio.wav'
urllib.request.urlretrieve(url, filename)
print("✅ Audio downloaded!")

# Step 2: Load audio file
sample_rate, data = wavfile.read(filename)

# Step 3: If stereo, use only one channel
if len(data.shape) > 1:
    data = data[:, 0]

# Step 4: Normalize if it's 16-bit audio
if data.dtype == np.int16:
    data = data / 32768.0

# Step 5: Apply FFT to check the spectrum
fft_spectrum = np.fft.fft(data)
freq = np.fft.fftfreq(len(fft_spectrum), d=1/sample_rate)

# Step 6: Visualize spectrum (optional)
plt.figure(figsize=(10, 5))
plt.plot(freq[:len(freq)//2], np.abs(fft_spectrum[:len(freq)//2]))
plt.title("Frequency Spectrum of Modulated Noisy Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xticks(np.arange(0, 22000, 1000)) 
plt.tight_layout()
plt.show()

# Step 7: Bandpass filter around 10 kHz
filtered_spectrum = fft_spectrum.copy()
low_cutoff = 9200
high_cutoff = 10800
for i in range(len(freq)):
    if not (low_cutoff <= abs(freq[i]) <= high_cutoff):
        filtered_spectrum[i] = 0

# Step 8: Inverse FFT to get filtered time-domain signal
cleaned_data = np.fft.ifft(filtered_spectrum).real
cleaned_data = cleaned_data * 32768
cleaned_data = cleaned_data.astype(np.int16)
write("cleaned_audio.wav", sample_rate, cleaned_data)

# Step 9: Load cleaned file and normalize
sample_rate, cleaned_data = wavfile.read("cleaned_audio.wav")
if cleaned_data.dtype == np.int16:
    cleaned_data = cleaned_data.astype(np.float32) / 32768.0

# Step 10: Demodulate with cosine at 10 kHz
carrier_freq = 10000  # 10 kHz
t = np.arange(len(cleaned_data)) / sample_rate
demodulated_signal = cleaned_data * np.cos(2 * np.pi * carrier_freq * t)

# Step 11: Envelope detection and low-pass filtering
envelope = np.abs(demodulated_signal)
demod_fft = np.fft.fft(envelope)
freqs = np.fft.fftfreq(len(envelope), d=1/sample_rate)
for i in range(len(freqs)):
    if abs(freqs[i]) > 3000:  # Low-pass filter below 3kHz
        demod_fft[i] = 0

# Step 12: Back to time domain and save
final_audio = np.fft.ifft(demod_fft).real
final_audio = final_audio / np.max(np.abs(final_audio))
final_audio = (final_audio * 32767).astype(np.int16)
write("demodulated_cos_10000Hz.wav", sample_rate, final_audio)

print("✅ Demodulation complete! Output saved as 'demodulated_cos_10000Hz.wav'.")

# Optional: play the audio if using Jupyter
from IPython.display import Audio
Audio("demodulated_cos_10000Hz.wav")

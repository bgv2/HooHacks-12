from scipy.io import wavfile
import numpy as np
import torchaudio

def load_audio(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    return sample_rate, audio_data

def normalize_audio(audio_data):
    audio_data = audio_data.astype(np.float32)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    return audio_data

def reduce_noise(audio_data, noise_factor=0.1):
    noise = np.random.randn(len(audio_data))
    noisy_audio = audio_data + noise_factor * noise
    return noisy_audio

def save_audio(file_path, sample_rate, audio_data):
    torchaudio.save(file_path, torch.tensor(audio_data).unsqueeze(0), sample_rate)

def process_audio(file_path, output_path):
    sample_rate, audio_data = load_audio(file_path)
    normalized_audio = normalize_audio(audio_data)
    denoised_audio = reduce_noise(normalized_audio)
    save_audio(output_path, sample_rate, denoised_audio)
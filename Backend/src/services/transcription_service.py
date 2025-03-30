from typing import List
import torchaudio
import torch
from generator import load_csm_1b, Segment

class TranscriptionService:
    def __init__(self, model_device: str = "cpu"):
        self.generator = load_csm_1b(device=model_device)

    def transcribe_audio(self, audio_path: str) -> str:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = self._resample_audio(audio_tensor, sample_rate)
        transcription = self.generator.generate_transcription(audio_tensor)
        return transcription

    def _resample_audio(self, audio_tensor: torch.Tensor, orig_freq: int) -> torch.Tensor:
        target_sample_rate = self.generator.sample_rate
        if orig_freq != target_sample_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor.squeeze(0), orig_freq=orig_freq, new_freq=target_sample_rate)
        return audio_tensor

    def transcribe_audio_stream(self, audio_chunks: List[torch.Tensor]) -> str:
        combined_audio = torch.cat(audio_chunks, dim=1)
        transcription = self.generator.generate_transcription(combined_audio)
        return transcription
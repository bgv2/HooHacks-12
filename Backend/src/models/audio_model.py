from dataclasses import dataclass
import torch

@dataclass
class AudioModel:
    model: torch.nn.Module
    sample_rate: int

    def __post_init__(self):
        self.model.eval()

    def process_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            processed_audio = self.model(audio_tensor)
        return processed_audio

    def resample_audio(self, audio_tensor: torch.Tensor, target_sample_rate: int) -> torch.Tensor:
        if self.sample_rate != target_sample_rate:
            resampled_audio = torchaudio.functional.resample(audio_tensor, orig_freq=self.sample_rate, new_freq=target_sample_rate)
            return resampled_audio
        return audio_tensor

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
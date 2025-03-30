from dataclasses import dataclass
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from src.llm.generator import load_csm_1b

@dataclass
class TextToSpeechService:
    generator: any

    def __init__(self, device: str = "cuda"):
        self.generator = load_csm_1b(device=device)

    def text_to_speech(self, text: str, speaker: int = 0) -> torch.Tensor:
        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=10000,
        )
        return audio

    def save_audio(self, audio: torch.Tensor, file_path: str):
        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
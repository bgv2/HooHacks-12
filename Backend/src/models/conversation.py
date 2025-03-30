from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        # Ensure audio is a tensor if provided
        if self.audio is not None and not isinstance(self.audio, torch.Tensor):
            self.audio = torch.tensor(self.audio, dtype=torch.float32)

@dataclass
class Conversation:
    context: List[str] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    current_speaker: Optional[int] = None

    def add_message(self, message: str, speaker: int):
        self.context.append(f"Speaker {speaker}: {message}")
        self.current_speaker = speaker

    def add_segment(self, segment: Segment):
        self.segments.append(segment)
        self.context.append(f"Speaker {segment.speaker}: {segment.text}")
        self.current_speaker = segment.speaker

    def get_context(self) -> List[str]:
        return self.context
    
    def get_segments(self) -> List[Segment]:
        return self.segments

    def clear_context(self):
        self.context.clear()
        self.segments.clear()
        self.current_speaker = None

    def get_last_message(self) -> Optional[str]:
        if self.context:
            return self.context[-1]
        return None
        
    def get_last_segment(self) -> Optional[Segment]:
        if self.segments:
            return self.segments[-1]
        return None
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Conversation:
    context: List[str] = field(default_factory=list)
    current_speaker: Optional[int] = None

    def add_message(self, message: str, speaker: int):
        self.context.append(f"Speaker {speaker}: {message}")
        self.current_speaker = speaker

    def get_context(self) -> List[str]:
        return self.context

    def clear_context(self):
        self.context.clear()
        self.current_speaker = None

    def get_last_message(self) -> Optional[str]:
        if self.context:
            return self.context[-1]
        return None
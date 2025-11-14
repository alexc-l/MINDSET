from abc import ABC, abstractmethod
from typing import Dict, Any

class EQModule(ABC):
    @abstractmethod
    def modulate(
        self,
        raw_text: str,
        eq_score: float,
        emotional_cues: Dict[str, Any],
    ) -> str:
        """Apply EQ-driven emotional constraints / empathy adjustments."""
        pass
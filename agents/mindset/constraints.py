from abc import ABC, abstractmethod
from typing import Dict, Any

class CognitiveEmotionalConstraints(ABC):
    @abstractmethod
    def apply(
        self,
        step_output: str,
        personality_profile: Dict[str, Any],
    ) -> str:
        """Inject human-like limits (e.g. empathy threshold, info overload)."""
        pass
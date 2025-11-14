from abc import ABC, abstractmethod
from typing import Any, Dict

class MetaProcess(ABC):
    """
    One atomic reasoning step (e.g. “dominant function perception”,
    “System-1 intuition”, “Openness-driven ideation” …).
    """

    @abstractmethod
    def execute(
        self,
        question: str,
        options: str,
        personality_profile: Dict[str, Any],
        constraints: Dict[str, Any],
        include_metadata: bool = False,
        **extra: Any,
    ) -> str:
        """
        Returns the raw text output of this meta-step.
        Concrete subclasses will fill the prompt, call the LLM, and return the response.
        """
        pass
from abc import ABC, abstractmethod
from typing import Dict, Any

class IQModule(ABC):
    @abstractmethod
    def modulate(
        self,
        raw_text: str,
        iq_score: float,
        cognitive_constraints: Dict[str, Any],
    ) -> str:
        """Scale reasoning depth, enforce bounded rationality, etc."""
        pass
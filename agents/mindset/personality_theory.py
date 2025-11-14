from abc import ABC, abstractmethod
from typing import Dict, Any

from agents.mindset.process_combination import ProcessCombination


class PersonalityTheory(ABC):
    """
    Encapsulates everything needed for ONE personality model
    (MBTI, Big-Five, HEXACO, DISC, Dual-Process …).
    """

    @abstractmethod
    def build_process_combination(self) -> ProcessCombination:
        """Return a ready-to-use ProcessCombination for this theory."""
        pass

    @abstractmethod
    def predict_from_demographics(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Theory-guided multi-layer rule-based prediction.
        Returns a dict that will be merged into the global personality_profile.
        """
        pass
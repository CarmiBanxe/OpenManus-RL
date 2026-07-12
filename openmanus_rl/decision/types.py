"""
Type definitions for decision theory
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DecisionOption:
    """Represents a decision option with its predicted outcomes"""
    action: str
    outcomes: List[Dict[str, Any]]
    probabilities: List[float]
    
    def __post_init__(self):
        if len(self.outcomes) != len(self.probabilities):
            raise ValueError("Outcomes and probabilities must have the same length")
        
        # Normalize probabilities
        total = sum(self.probabilities)
        if total > 0:
            self.probabilities = [p / total for p in self.probabilities]
        else:
            # Equal probabilities if not provided
            self.probabilities = [1.0 / len(self.outcomes)] * len(self.outcomes)

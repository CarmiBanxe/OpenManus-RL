"""
Optimal stopping models
"""

import logging
import math
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OptimalStopping:
    """
    Implements optimal stopping models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        logger.info("OptimalStopping initialized")
    
    def secretary_problem(self, candidates: List[float]) -> int:
        """
        Solve the secretary problem using the 37% rule
        
        Args:
            candidates: List of candidate values
            
        Returns:
            Index of selected candidate
        """
        n = len(candidates)
        if n < 2:
            return 0  # Only one candidate
        
        # Sample size (approximately n/e)
        sample_size = max(1, int(n / math.e))
        
        # Find best candidate in sample
        best_in_sample = max(candidates[:sample_size]) if sample_size > 0 else float('-inf')
        
        # Find first candidate after sample that's better than best in sample
        for i in range(sample_size, n):
            if candidates[i] > best_in_sample:
                return i
        
        # If no candidate is better, select the last one
        return n - 1
    
    def optimal_stopping_threshold(
        self, 
        values: List[float], 
        discount_factor: float = 0.95
    ) -> List[float]:
        """
        Calculate optimal stopping thresholds for a sequence of values
        
        Args:
            values: List of values
            discount_factor: Discount factor for future values
            
        Returns:
            List of thresholds for each position
        """
        n = len(values)
        if n == 0:
            return []
        
        # Initialize thresholds
        thresholds = [0.0] * n
        
        # Start from the end
        thresholds[-1] = values[-1]
        
        # Work backwards
        for i in range(n - 2, -1, -1):
            # Expected value of continuing
            expected_continue = discount_factor * thresholds[i + 1]
            
            # Value of stopping now
            value_stop = values[i]
            
            # Optimal threshold is the maximum of stopping or continuing
            thresholds[i] = max(value_stop, expected_continue)
        
        return thresholds
    
    def parking_problem(self, spots: List[bool]) -> int:
        """
        Solve the parking problem (select the best available spot)
        
        Args:
            spots: List of booleans indicating if a spot is available
            
        Returns:
            Index of selected spot
        """
        n = len(spots)
        if n == 0:
            return -1
        
        # Find first available spot
        for i, available in enumerate(spots):
            if available:
                return i
        
        # No available spots
        return -1
    
    def search_with_unknown_distribution(
        self, 
        values: List[float], 
        max_samples: int = 10
    ) -> int:
        """
        Optimal stopping when distribution is unknown
        
        Args:
            values: List of values to sample from
            max_samples: Maximum number of samples to take
            
        Returns:
            Index of selected value
        """
        n = len(values)
        if n == 0:
            return -1
        
        # Sample up to max_samples values
        sample_size = min(max_samples, n)
        
        # Take samples
        samples = values[:sample_size]
        
        # Estimate distribution parameters (mean and variance)
        mean = sum(samples) / len(samples) if samples else 0
        variance = sum((x - mean) ** 2 for x in samples) / len(samples) if samples else 0
        
        # Calculate threshold based on estimated distribution
        # For simplicity, we'll use mean + standard deviation
        std_dev = math.sqrt(variance)
        threshold = mean + std_dev
        
        # Find first value after samples that exceeds threshold
        for i in range(sample_size, n):
            if values[i] > threshold:
                return i
        
        # If no value exceeds threshold, select the last one
        return n - 1

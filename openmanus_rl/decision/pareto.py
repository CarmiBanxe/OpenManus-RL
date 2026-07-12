"""
Pareto optimization for multi-criteria decision making
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ParetoOptimizer:
    """
    Finds Pareto optimal solutions for multi-criteria decision making
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        logger.info("ParetoOptimizer initialized")
    
    def find_pareto_front(self, solutions: List[List[float]]) -> List[int]:
        """
        Find Pareto optimal solutions
        
        Args:
            solutions: List of solutions, each represented by a list of criteria values
            
        Returns:
            List of indices of Pareto optimal solutions
        """
        if not solutions:
            return []
        
        # Track dominated solutions
        dominated = set()
        
        # Check each solution against all others
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol2, sol1):
                    dominated.add(i)
        
        # Return indices of non-dominated solutions
        pareto_front = [i for i in range(len(solutions)) if i not in dominated]
        
        logger.debug(f"Found {len(pareto_front)} Pareto optimal solutions out of {len(solutions)}")
        return pareto_front
    
    def _dominates(self, sol1: List[float], sol2: List[float]) -> bool:
        """
        Check if sol1 dominates sol2
        
        Args:
            sol1: First solution
            sol2: Second solution
            
        Returns:
            True if sol1 dominates sol2
        """
        if len(sol1) != len(sol2):
            return False
        
        # sol1 dominates sol2 if it is at least as good in all criteria
        # and strictly better in at least one criterion
        at_least_as_good = all(c1 >= c2 for c1, c2 in zip(sol1, sol2))
        strictly_better = any(c1 > c2 for c1, c2 in zip(sol1, sol2))
        
        return at_least_as_good and strictly_better
    
    def pareto_ranking(self, solutions: List[List[float]]) -> List[int]:
        """
        Assign Pareto ranks to solutions
        
        Args:
            solutions: List of solutions, each represented by a list of criteria values
            
        Returns:
            List of ranks for each solution (0 is best/Pareto front)
        """
        if not solutions:
            return []
        
        ranks = [None] * len(solutions)
        remaining_solutions = set(range(len(solutions)))
        current_rank = 0
        
        while remaining_solutions:
            # Find Pareto front of remaining solutions
            remaining_solutions_list = [solutions[i] for i in remaining_solutions]
            pareto_front_indices = self.find_pareto_front(remaining_solutions_list)
            
            # Map back to original indices
            pareto_front = [list(remaining_solutions)[i] for i in pareto_front_indices]
            
            # Assign current rank to Pareto front
            for idx in pareto_front:
                ranks[idx] = current_rank
            
            # Remove Pareto front from remaining solutions
            remaining_solutions -= set(pareto_front)
            
            current_rank += 1
        
        return ranks
    
    def crowding_distance(
        self, 
        solutions: List[List[float]], 
        pareto_front: List[int]
    ) -> List[float]:
        """
        Calculate crowding distance for solutions in Pareto front
        
        Args:
            solutions: List of solutions
            pareto_front: Indices of solutions in Pareto front
            
        Returns:
            List of crowding distances for each solution in Pareto front
        """
        if not pareto_front:
            return []
        
        # Initialize distances
        distances = [0.0] * len(pareto_front)
        
        # Number of criteria
        if not solutions:
            return distances
        
        num_criteria = len(solutions[0])
        
        # Calculate crowding distance for each criterion
        for j in range(num_criteria):
            # Sort by current criterion
            sorted_indices = sorted(pareto_front, key=lambda i: solutions[i][j])
            
            # Set infinite distance for boundary solutions
            distances[pareto_front.index(sorted_indices[0])] = float('inf')
            distances[pareto_front.index(sorted_indices[-1])] = float('inf')
            
            # Calculate distance for interior solutions
            if len(sorted_indices) > 2:
                min_val = solutions[sorted_indices[0]][j]
                max_val = solutions[sorted_indices[-1]][j]
                
                if max_val - min_val > 0:
                    for k in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[k]
                        prev_idx = sorted_indices[k - 1]
                        next_idx = sorted_indices[k + 1]
                        
                        distance = (solutions[next_idx][j] - solutions[prev_idx][j]) / (max_val - min_val)
                        distances[pareto_front.index(idx)] += distance
        
        return distances

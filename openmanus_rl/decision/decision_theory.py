"""
Integrated Decision Theory Module
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

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

class UtilityCalculator:
    """Calculates utility for outcomes based on various utility functions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.utility_function = self._get_utility_function()
        logger.info(f"UtilityCalculator initialized with function: {self.utility_function.__name__}")
    
    def _get_utility_function(self):
        function_name = self.config.get('function', 'linear')
        if function_name == 'linear':
            return self._linear_utility
        elif function_name == 'logarithmic':
            return self._logarithmic_utility
        elif function_name == 'exponential':
            return self._exponential_utility
        else:
            logger.warning(f"Unknown utility function: {function_name}, using linear")
            return self._linear_utility
    
    def _linear_utility(self, value: float) -> float:
        return value
    
    def _logarithmic_utility(self, value: float) -> float:
        return np.log(max(0, value) + 1)
    
    def _exponential_utility(self, value: float) -> float:
        return np.exp(value) - 1
    
    def utility(self, outcome: Dict[str, Any]) -> float:
        if 'utility' in outcome:
            return outcome['utility']
        
        if 'reward' in outcome:
            return self.utility_function(outcome['reward'])
        
        utility = 0.0
        weights = self.config.get('attribute_weights', {})
        
        for attribute, value in outcome.items():
            if attribute in ['action', 'probabilities']:
                continue
            weight = weights.get(attribute, 1.0)
            utility += weight * self.utility_function(value)
        
        return utility
    
    def expected_utility(self, action: str, outcomes: List[Dict[str, Any]], probabilities: List[float]) -> float:
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have the same length")
        
        if not outcomes:
            return 0.0
        
        utilities = [self.utility(outcome) for outcome in outcomes]
        expected_utility = sum(u * p for u, p in zip(utilities, probabilities))
        
        logger.debug(f"Expected utility for {action}: {expected_utility}")
        return expected_utility

class ParetoOptimizer:
    """Finds Pareto optimal solutions for multi-criteria decision making"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info("ParetoOptimizer initialized")
    
    def find_pareto_front(self, solutions: List[List[float]]) -> List[int]:
        if not solutions:
            return []
        
        dominated = set()
        
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol2, sol1):
                    dominated.add(i)
        
        pareto_front = [i for i in range(len(solutions)) if i not in dominated]
        
        logger.debug(f"Found {len(pareto_front)} Pareto optimal solutions out of {len(solutions)}")
        return pareto_front
    
    def _dominates(self, sol1: List[float], sol2: List[float]) -> bool:
        if len(sol1) != len(sol2):
            return False
        
        at_least_as_good = all(c1 >= c2 for c1, c2 in zip(sol1, sol2))
        strictly_better = any(c1 > c2 for c1, c2 in zip(sol1, sol2))
        
        return at_least_as_good and strictly_better

class BehavioralModel:
    """Implements behavioral decision making models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.reference_point = self.config.get('reference_point', 0.0)
        self.loss_aversion = self.config.get('loss_aversion', 2.0)
        
        logger.info(f"BehavioralModel initialized with reference_point: {self.reference_point}")
    
    def adjust_option(self, option: DecisionOption, state: Dict[str, Any]) -> DecisionOption:
        adjusted_outcomes = []
        for outcome in option.outcomes:
            adjusted_outcome = outcome.copy()
            
            if 'reward' in adjusted_outcome:
                adjusted_outcome['reward'] = self.prospect_utility(adjusted_outcome['reward'])
            
            adjusted_outcomes.append(adjusted_outcome)
        
        return DecisionOption(
            action=option.action,
            outcomes=adjusted_outcomes,
            probabilities=option.probabilities.copy()
        )
    
    def prospect_utility(self, outcome: float) -> float:
        gain_loss = outcome - self.reference_point
        
        if gain_loss >= 0:
            return gain_loss ** 0.88
        else:
            return -self.loss_aversion * (-gain_loss) ** 0.88

class OptimalStopping:
    """Implements optimal stopping models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info("OptimalStopping initialized")
    
    def secretary_problem(self, candidates: List[float]) -> int:
        n = len(candidates)
        if n < 2:
            return 0
        
        sample_size = max(1, int(n / math.e))
        best_in_sample = max(candidates[:sample_size]) if sample_size > 0 else float('-inf')
        
        for i in range(sample_size, n):
            if candidates[i] > best_in_sample:
                return i
        
        return n - 1

class DecisionFramework:
    """Core decision-making framework that combines multiple decision theory approaches"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.utility_calculator = UtilityCalculator(self.config.get('utility', {}))
        self.pareto_optimizer = ParetoOptimizer(self.config.get('pareto', {}))
        self.behavioral_model = BehavioralModel(self.config.get('behavioral', {}))
        self.optimal_stopping = OptimalStopping(self.config.get('stopping', {}))
        
        logger.info("DecisionFramework initialized with config: %s", self.config)
    
    def make_decision(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]] = None,
        approach: str = "expected_utility"
    ) -> Dict[str, Any]:
        logger.info(f"Making decision with approach: {approach}")
        
        if approach == "expected_utility":
            return self._expected_utility_approach(state, options, criteria)
        elif approach == "pareto":
            return self._pareto_approach(state, options, criteria)
        elif approach == "behavioral":
            return self._behavioral_approach(state, options, criteria)
        elif approach == "optimal_stopping":
            return self._optimal_stopping_approach(state, options, criteria)
        else:
            raise ValueError(f"Unknown decision approach: {approach}")
    
    def _expected_utility_approach(self, state: Dict[str, Any], options: List[DecisionOption], criteria: Optional[List[str]]) -> Dict[str, Any]:
        utilities = []
        
        for option in options:
            utility = self.utility_calculator.expected_utility(
                option.action, option.outcomes, option.probabilities
            )
            utilities.append(utility)
        
        max_idx = utilities.index(max(utilities))
        optimal_option = options[max_idx]
        
        return {
            "decision": optimal_option.action,
            "approach": "expected_utility",
            "utility": utilities[max_idx],
            "all_utilities": utilities,
            "reasoning": "Selected option with maximum expected utility"
        }
    
    def _pareto_approach(self, state: Dict[str, Any], options: List[DecisionOption], criteria: Optional[List[str]]) -> Dict[str, Any]:
        if not criteria:
            criteria = ["reward", "cost", "time"]
        
        options_criteria = []
        for option in options:
            criteria_values = []
            for criterion in criteria:
                values = [outcome.get(criterion, 0) for outcome in option.outcomes]
                avg_value = sum(values) / len(values) if values else 0
                criteria_values.append(avg_value)
            options_criteria.append(criteria_values)
        
        pareto_indices = self.pareto_optimizer.find_pareto_front(options_criteria)
        pareto_options = [options[i] for i in pareto_indices]
        
        if len(pareto_options) == 1:
            selected_option = pareto_options[0]
            reasoning = "Single Pareto optimal option found"
        else:
            weights = self.config.get('pareto', {}).get('criteria_weights', [1.0] * len(criteria))
            selected_option = self._select_from_pareto(pareto_options, criteria, weights)
            reasoning = f"Selected from {len(pareto_options)} Pareto optimal options using weighted criteria"
        
        return {
            "decision": selected_option.action,
            "approach": "pareto",
            "pareto_front_size": len(pareto_options),
            "reasoning": reasoning
        }
    
    def _behavioral_approach(self, state: Dict[str, Any], options: List[DecisionOption], criteria: Optional[List[str]]) -> Dict[str, Any]:
        adjusted_options = []
        for option in options:
            adjusted_option = self.behavioral_model.adjust_option(option, state)
            adjusted_options.append(adjusted_option)
        
        if self.config.get('behavioral', {}).get('use_satisficing', False):
            threshold = self.config.get('behavioral', {}).get('satisficing_threshold', 0.5)
            selected_option = self.behavioral_model.satisficing(adjusted_options, threshold)
            reasoning = f"Selected first satisficing option with threshold {threshold}"
        else:
            utilities = []
            for option in adjusted_options:
                utility = self.utility_calculator.expected_utility(
                    option.action, option.outcomes, option.probabilities
                )
                utilities.append(utility)
            
            max_idx = utilities.index(max(utilities))
            selected_option = adjusted_options[max_idx]
            reasoning = "Selected option with maximum behaviorally-adjusted utility"
        
        return {
            "decision": selected_option.action,
            "approach": "behavioral",
            "reasoning": reasoning
        }
    
    def _optimal_stopping_approach(self, state: Dict[str, Any], options: List[DecisionOption], criteria: Optional[List[str]]) -> Dict[str, Any]:
        if len(options) < 3:
            return self._expected_utility_approach(state, options, criteria)
        
        utilities = []
        for option in options:
            utility = self.utility_calculator.expected_utility(
                option.action, option.outcomes, option.probabilities
            )
            utilities.append(utility)
        
        selected_index = self.optimal_stopping.secretary_problem(utilities)
        selected_option = options[selected_index]
        
        return {
            "decision": selected_option.action,
            "approach": "optimal_stopping",
            "reasoning": f"Applied optimal stopping rule, selected option at index {selected_index}"
        }
    
    def _select_from_pareto(self, pareto_options: List[DecisionOption], criteria: List[str], weights: List[float]) -> DecisionOption:
        best_option = None
        best_score = float('-inf')
        
        for option in pareto_options:
            score = 0.0
            for i, criterion in enumerate(criteria):
                values = [outcome.get(criterion, 0) for outcome in option.outcomes]
                avg_value = sum(values) / len(values) if values else 0
                
                if i < len(weights):
                    score += weights[i] * avg_value
                else:
                    score += avg_value
            
            if score > best_score:
                best_score = score
                best_option = option
        
        return best_option

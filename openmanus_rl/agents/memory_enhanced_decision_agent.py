"""
Memory-enhanced decision agent for OpenManus RL
"""

import logging
from typing import Dict, Any, List, Optional
from .decision_agent import DecisionAgent
from ..memory.summarized_memory import SummarizedMemory

logger = logging.getLogger(__name__)

class MemoryEnhancedDecisionAgent(DecisionAgent):
    """
    Decision agent enhanced with memory capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize memory
        memory_config = config.get('memory', {}) if config else {}
        self.memory_system = SummarizedMemory(memory_config)
        
        logger.info("MemoryEnhancedDecisionAgent initialized")
    
    def select_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Select an action using decision theory with memory
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        # Get relevant memories
        relevant_memories = self.memory_system.get_relevant_memories(state)
        
        # Enhance state with memory information
        enhanced_state = {
            **state,
            "relevant_memories": relevant_memories,
            "memory_count": len(relevant_memories)
        }
        
        # Use parent class to select action
        action = super().select_action(enhanced_state, available_actions)
        
        # Store decision in memory
        self.memory_system.add_memory({
            "state": state,
            "action": action,
            "available_actions": available_actions
        })
        
        return action
    
    def update_policy(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """
        Update policy based on experience and store in memory
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
        """
        # Use parent class to update policy
        super().update_policy(state, action, reward, next_state)
        
        # Store experience in memory
        self.memory_system.add_memory({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })
    
    def get_memory_summary(self) -> str:
        """
        Get a summary of stored memories
        
        Returns:
            Memory summary string
        """
        return self.memory_system.get_summary()
    
    def forget_old_memories(self, threshold: float = 0.1):
        """
        Forget old or irrelevant memories
        
        Args:
            threshold: Relevance threshold for forgetting
        """
        self.memory_system.forget(threshold)
        logger.info(f"Forgotten memories with relevance below {threshold}")

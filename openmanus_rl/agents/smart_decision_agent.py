"""
Smart Decision Agent with Ollama integration
"""

import logging
import httpx
import json
from typing import Dict, Any, List, Optional
from ..decision.decision_theory import DecisionFramework, DecisionOption

logger = logging.getLogger(__name__)

class SmartDecisionAgent:
    """
    Smart decision agent with Ollama integration and simplified architecture
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize decision framework
        self.decision_framework = DecisionFramework(self.config.get('decision', {}))
        
        # Ollama configuration
        self.ollama_base_url = self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        self.ollama_model = self.config.get('ollama', {}).get('model', 'qwen2.5:7b-instruct')
        
        # Default decision approach
        self.default_approach = self.config.get('default_approach', 'expected_utility')
        
        # Learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.experience_history = []
        
        logger.info(f"SmartDecisionAgent initialized with approach: {self.default_approach}")
    
    def select_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Select an action using decision theory with Ollama
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        logger.info(f"Selecting action from {len(available_actions)} options")
        
        # Convert actions to decision options
        options = []
        for action in available_actions:
            outcomes = self._predict_outcomes(state, action)
            probabilities = [1.0] * len(outcomes)
            
            option = DecisionOption(
                action=action,
                outcomes=outcomes,
                probabilities=probabilities
            )
            options.append(option)
        
        # Make decision
        decision_result = self.decision_framework.make_decision(
            state=state,
            options=options,
            approach=self.default_approach
        )
        
        selected_action = decision_result['decision']
        
        # Store experience
        self.experience_history.append({
            'state': state,
            'action': selected_action,
            'decision_result': decision_result,
            'timestamp': self._get_timestamp()
        })
        
        logger.info(f"Selected action: {selected_action}")
        return selected_action
    
    def _predict_outcomes(self, state: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
        """
        Predict outcomes using Ollama LLM with fallback
        
        Args:
            state: Current state
            action: Action to predict outcomes for
            
        Returns:
            List of predicted outcomes
        """
        try:
            return self._predict_with_ollama(state, action)
        except Exception as e:
            logger.warning(f"Failed to predict with Ollama: {str(e)}. Using fallback.")
            return self._fallback_prediction(state, action)
    
    def _predict_with_ollama(self, state: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
        """Predict outcomes using Ollama"""
        prompt = f"""
Predict 3 possible outcomes for the action: {action}
Current state: {state}

For each outcome, provide:
- reward (0.0-1.0): how beneficial the outcome is
- cost (0.0-1.0): how costly the outcome is
- time (0.0-1.0): how time-consuming the outcome is

Return as JSON array:
[{{"reward": 0.8, "cost": 0.3, "time": 0.5}}, {{"reward": 0.5, "cost": 0.2, "time": 0.3}}, {{"reward": 0.2, "cost": 0.1, "time": 0.1}}]
"""
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status code {response.status_code}")
            
            data = response.json()
            # Handle different response formats from Ollama
            if isinstance(data, str):
                try:
                    outcomes = json.loads(data)
                except json.JSONDecodeError:
                    outcomes = []
            elif isinstance(data, dict):
                if "response" in data:
                    try:
                        outcomes = json.loads(data["response"])
                    except json.JSONDecodeError:
                        outcomes = []
                else:
                    outcomes = []
            else:
                outcomes = []
        """Fallback prediction when Ollama is unavailable"""
        # Base outcomes
        outcomes = [
            {"reward": 0.6, "cost": 0.4, "time": 0.5},
            {"reward": 0.4, "cost": 0.3, "time": 0.3},
            {"reward": 0.2, "cost": 0.2, "time": 0.2}
        ]
        
        # Adjust based on action type
        if "search" in action.lower() or "analyze" in action.lower():
            outcomes[0]["reward"] *= 1.3
            outcomes[0]["time"] *= 1.2
        elif "generate" in action.lower() or "create" in action.lower():
            outcomes[0]["reward"] *= 1.2
            outcomes[0]["cost"] *= 1.1
        elif "optimize" in action.lower() or "improve" in action.lower():
            outcomes[0]["reward"] *= 1.1
            outcomes[0]["time"] *= 0.8
        
        # Adjust based on state
        if state.get("urgency", 0) > 0.7:
            for outcome in outcomes:
                outcome["time"] *= 0.8
        
        return outcomes
    
    def update_policy(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """
        Update policy based on experience
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
        """
        # Find the experience in history
        for i, experience in enumerate(self.experience_history):
            if (experience['state'] == state and 
                experience['action'] == action):
                
                # Update with actual reward
                self.experience_history[i]['actual_reward'] = reward
                
                # Simple learning: adjust approach if reward is low
                if reward < 0.3:
                    approaches = ['expected_utility', 'pareto', 'behavioral', 'optimal_stopping']
                    current_idx = approaches.index(self.default_approach)
                    self.default_approach = approaches[(current_idx + 1) % len(approaches)]
                    logger.info(f"Changed decision approach to {self.default_approach} due to low reward")
                
                break
    
    def get_decision_explanation(self, state: Dict[str, Any], action: str) -> Optional[str]:
        """Get explanation for a decision"""
        for experience in self.experience_history:
            if (experience['state'] == state and 
                experience['action'] == action and
                'decision_result' in experience):
                
                decision_result = experience['decision_result']
                return f"Approach: {decision_result.get('approach', 'unknown')}, " \
                       f"Reasoning: {decision_result.get('reasoning', 'no reasoning provided')}"
        
        return None
    
    def get_detailed_explanation(self, state: Dict[str, Any], action: str) -> str:
        """Get detailed explanation using Ollama"""
        basic_explanation = self.get_decision_explanation(state, action) or "No explanation available"
        
        try:
            prompt = f"""
Explain why the action '{action}' was chosen in state {state}.
Basic explanation: {basic_explanation}

Provide a concise explanation focusing on:
1. The decision theory approach used
2. Key factors that influenced the decision
3. Why this action was preferred over alternatives
"""
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["response"].strip()
                else:
                    return basic_explanation
                    
        except Exception as e:
            logger.warning(f"Failed to get detailed explanation: {str(e)}")
            return basic_explanation
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.experience_history:
            return {"message": "No experience recorded yet"}
        
        total_experiences = len(self.experience_history)
        rewarded_experiences = [e for e in self.experience_history if 'actual_reward' in e]
        
        if not rewarded_experiences:
            return {
                "total_experiences": total_experiences,
                "rewarded_experiences": 0,
                "message": "No rewards recorded yet"
            }
        
        rewards = [e['actual_reward'] for e in rewarded_experiences]
        avg_reward = sum(rewards) / len(rewards)
        
        approach_counts = {}
        for experience in self.experience_history:
            approach = experience['decision_result'].get('approach', 'unknown')
            approach_counts[approach] = approach_counts.get(approach, 0) + 1
        
        return {
            "total_experiences": total_experiences,
            "rewarded_experiences": len(rewarded_experiences),
            "average_reward": avg_reward,
            "current_approach": self.default_approach,
            "approach_usage": approach_counts
        }

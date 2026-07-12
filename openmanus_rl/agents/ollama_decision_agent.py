"""
Ollama-enhanced decision agent for OpenManus RL
"""

import logging
import httpx
from typing import Dict, Any, List, Optional
from .decision_agent import DecisionAgent

logger = logging.getLogger(__name__)

class OllamaDecisionAgent(DecisionAgent):
    """
    Decision agent enhanced with Ollama LLM for outcome prediction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Ollama configuration
        self.ollama_base_url = self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        self.ollama_model = self.config.get('ollama', {}).get('model', 'qwen2.5:7b-instruct')
        
        logger.info(f"OllamaDecisionAgent initialized with model: {self.ollama_model}")
    
    def _predict_outcomes(self, state: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
        """
        Predict outcomes using Ollama LLM
        
        Args:
            state: Current state
            action: Action to predict outcomes for
            
        Returns:
            List of predicted outcomes
        """
        # If Ollama is not available, fall back to simple prediction
        try:
            return self._predict_outcomes_with_ollama(state, action)
        except Exception as e:
            logger.warning(f"Failed to predict outcomes with Ollama: {str(e)}. Using simple prediction.")
            return super()._predict_outcomes(state, action)
    
    def _predict_outcomes_with_ollama(self, state: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
        """
        Predict outcomes using Ollama LLM
        
        Args:
            state: Current state
            action: Action to predict outcomes for
            
        Returns:
            List of predicted outcomes
        """
        # Create prompt for Ollama
        prompt = f"""
You are an AI assistant that predicts outcomes for actions in a given state.

Current state: {state}
Action to perform: {action}

Predict 3 possible outcomes for this action, each with a reward value (0.0 to 1.0), cost value (0.0 to 1.0), and time value (0.0 to 1.0).

Return your response as a JSON array with the following structure:
[
  {{"reward": 0.8, "cost": 0.3, "time": 0.5}},
  {{"reward": 0.5, "cost": 0.2, "time": 0.3}},
  {{"reward": 0.2, "cost": 0.1, "time": 0.1}}
]

Only return the JSON array, no other text.
"""
        
        # Call Ollama API
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
            
            # Parse the response
            try:
                import json
                outcomes = json.loads(data["response"])
                
                # Validate outcomes
                if not isinstance(outcomes, list) or len(outcomes) != 3:
                    raise ValueError("Invalid response format")
                
                for outcome in outcomes:
                    if not all(key in outcome for key in ["reward", "cost", "time"]):
                        raise ValueError("Missing required keys in outcome")
                
                return outcomes
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Ollama response: {str(e)}")
                raise Exception("Invalid response from Ollama")
    
    def get_decision_explanation_with_ollama(self, state: Dict[str, Any], action: str) -> str:
        """
        Get detailed explanation for a decision using Ollama
        
        Args:
            state: State where decision was made
            action: Action that was selected
            
        Returns:
            Detailed explanation string
        """
        # Get basic explanation
        basic_explanation = self.get_decision_explanation(state, action) or "No explanation available"
        
        # Get detailed explanation from Ollama
        try:
            prompt = f"""
You are an AI assistant that explains decision-making reasoning.

State: {state}
Action: {action}
Basic explanation: {basic_explanation}

Provide a detailed explanation for why this action was chosen, considering the decision theory approach used and the predicted outcomes.

Keep your explanation concise and focused on the key factors that influenced the decision.
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
                    logger.warning(f"Ollama API returned status code {response.status_code}")
                    return basic_explanation
                    
        except Exception as e:
            logger.warning(f"Failed to get detailed explanation from Ollama: {str(e)}")
            return basic_explanation

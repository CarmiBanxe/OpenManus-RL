"""
Memory Management Utility for OpenManus RL
Optimized for local Ollama instance with qwen2.5:7b-instruct
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory configuration and optimization for OpenManus RL."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize memory manager.
        
        Args:
            config_path: Path to memory configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "memory_config.yaml"
        )
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load memory configuration from file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load memory config: {e}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "memory": {
                "use_summary": True,
                "model": "qwen2.5:7b-instruct",
                "endpoint": "http://localhost:11434",
                "api_key": None,
                "summary_concurrency": 1,
                "timeout_s": 60,
                "max_history_length": 20,
                "summary_threshold": 5,
                "environments": {
                    "webshop": {"prompt_style": "webshop", "max_summary_length": 700},
                    "alfworld": {"prompt_style": "alfred", "max_summary_length": 600},
                    "default": {"prompt_style": "alfred", "max_summary_length": 600}
                }
            },
            "advanced": {
                "enable_compression": True,
                "cache_summaries": True,
                "fallback_to_recent": True,
                "fallback_steps": 3
            }
        }
    
    def get_memory_config(self, env_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory configuration for a specific environment.
        
        Args:
            env_type: Environment type (webshop, alfworld, etc.)
            
        Returns:
            Dictionary with memory configuration
        """
        config = self.config["memory"].copy()
        
        # Apply environment-specific settings
        if env_type and env_type.lower() in config["environments"]:
            env_config = config["environments"][env_type.lower()]
            config.update(env_config)
        elif "default" in config["environments"]:
            default_config = config["environments"]["default"]
            config.update(default_config)
            
        return config
    
    def optimize_for_system(self, vram_gb: int = 8, ram_gb: int = 64) -> Dict[str, Any]:
        """
        Optimize memory configuration for system specifications.
        
        Args:
            vram_gb: Available VRAM in GB
            ram_gb: Available RAM in GB
            
        Returns:
            Optimized configuration dictionary
        """
        config = self.config.copy()
        
        # Adjust concurrency based on system resources
        if vram_gb < 8:
            config["memory"]["summary_concurrency"] = 1
            config["memory"]["timeout_s"] = 90  # Longer timeout for slower systems
        elif vram_gb >= 16:
            config["memory"]["summary_concurrency"] = 2
            
        # Adjust history length based on RAM
        if ram_gb < 32:
            config["memory"]["max_history_length"] = 10
        elif ram_gb >= 64:
            config["memory"]["max_history_length"] = 30
            
        return config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Deep merge updates
            config = self.config
            for key, value in updates.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
            
            self.config = config
            return self.save_config(config)
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False

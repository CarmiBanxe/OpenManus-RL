"""Ollama engine wrapper.

Provides a wrapper for Ollama that's compatible with OpenManus's engine interface.
"""

from typing import Callable, Optional
import ollama


def create_llm_engine(model_string: str = "qwen2.5:7b-instruct", is_multimodal: bool = False, base_url: Optional[str] = None) -> Callable[[str], str]:
    """Create an Ollama-based LLM engine.
    
    Args:
        model_string: The model name to use with Ollama
        is_multimodal: Whether the model supports multimodal inputs (not used for now)
        base_url: The base URL for Ollama (not used, Ollama connects directly)
        
    Returns:
        A callable that maps prompt -> text using Ollama
    """
    def _engine(prompt: str) -> str:
        # Use Ollama directly
        response = ollama.chat(
            model=model_string,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.get("message", {}).get("content", "").strip()

    return _engine
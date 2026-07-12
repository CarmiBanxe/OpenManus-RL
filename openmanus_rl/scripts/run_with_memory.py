#!/usr/bin/env python3
"""
Run OpenManus RL with enhanced memory management
Optimized for local Ollama instance with qwen2.5:7b-instruct
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add the parent directory to the path so we can import openmanus_rl modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openmanus_rl.utils.memory_manager import MemoryManager
from openmanus_rl.memory.summarized_memory import SummarizedMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_memory(args) -> Dict[str, Any]:
    """Setup memory configuration based on arguments."""
    manager = MemoryManager(args.config)
    
    # Get base config
    if args.env_type:
        config = manager.get_memory_config(args.env_type)
    else:
        config = manager.get_memory_config()
    
    # Optimize for system if requested
    if args.optimize:
        config = manager.optimize_for_system(args.vram, args.ram)
    
    # Override with command line arguments
    if args.endpoint:
        config["endpoint"] = args.endpoint
    if args.model:
        config["model"] = args.model
    if args.concurrency:
        config["summary_concurrency"] = args.concurrency
    if args.timeout:
        config["timeout_s"] = args.timeout
    
    logger.info(f"Memory configuration: {config}")
    return config


def main():
    """Main function to run OpenManus RL with enhanced memory."""
    parser = argparse.ArgumentParser(description="Run OpenManus RL with enhanced memory management")
    
    # Memory configuration
    parser.add_argument("--config", type=str, default=None,
                        help="Path to memory configuration file")
    parser.add_argument("--env-type", type=str, default=None,
                        help="Environment type (webshop, alfworld, etc.)")
    
    # Model settings
    parser.add_argument("--endpoint", type=str, default="http://localhost:11434",
                        help="LLM API endpoint")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct",
                        help="LLM model name")
    
    # Performance settings
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Summary generation concurrency")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Request timeout in seconds")
    
    # System optimization
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize for system specifications")
    parser.add_argument("--vram", type=int, default=8,
                        help="Available VRAM in GB")
    parser.add_argument("--ram", type=int, default=64,
                        help="Available RAM in GB")
    
    # Task settings
    parser.add_argument("--task", type=str, required=True,
                        help="Task to run")
    parser.add_argument("--task-dir", type=str, default=None,
                        help="Directory containing task files")
    
    args = parser.parse_args()
    
    # Setup memory
    memory_config = setup_memory(args)
    
    # Initialize memory manager
    memory = SummarizedMemory()
    
    # Here you would integrate with your actual OpenManus RL runner
    # This is a placeholder for the integration point
    logger.info(f"Running task {args.task} with enhanced memory management")
    logger.info(f"Memory config: {memory_config}")
    
    # Example of how to use the memory in your actual code:
    # 
    # from openmanus_rl.agents import OpenManusAgent
    # agent = OpenManusAgent(memory=memory, memory_config=memory_config)
    # agent.run_task(args.task, task_dir=args.task_dir)
    
    logger.info("Task completed successfully")


if __name__ == "__main__":
    main()

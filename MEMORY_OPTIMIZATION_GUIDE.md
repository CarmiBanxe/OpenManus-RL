# Guide to Maximizing Memory Functionality in OpenManus RL

This guide explains how to maximize the memory functionality in OpenManus RL with your local Ollama setup.

## Quick Start

### 1. Basic Usage

Run a task with optimized memory settings:

```bash
cd ~/OpenManus
python ./openmanus_rl/scripts/run_with_memory.py --task your_task_name --env-type webshop
```

### 2. Advanced Usage

For system-specific optimization:

```bash
python ./openmanus_rl/scripts/run_with_memory.py \
    --task your_task_name \
    --env-type alfworld \
    --optimize \
    --vram 8 \
    --ram 64 \
    --concurrency 1
```

## Configuration Options

### Memory Settings

- `use_summary`: Enable/disable memory summarization (default: true)
- `model`: LLM model to use (default: qwen2.5:7b-instruct)
- `endpoint`: LLM API endpoint (default: http://localhost:11434)
- `summary_concurrency`: Number of parallel summary generations (default: 1)
- `timeout_s`: Request timeout in seconds (default: 60)
- `max_history_length`: Maximum steps before summarization (default: 20)
- `summary_threshold`: Minimum steps before attempting summarization (default: 5)

### Environment-Specific Settings

Different environments have different optimal memory configurations:

- **Webshop**: Optimized for product search and selection
- **ALFWorld**: Optimized for household task completion
- **Default**: General-purpose configuration

## Performance Optimization

### For Your System (8GB VRAM, 64GB RAM)

The default configuration is already optimized for your system, but you can further tune it:

1. **Reduce concurrency** if you experience instability:
   ```bash
   --concurrency 1
   ```

2. **Increase timeout** if summaries are timing out:
   ```bash
   --timeout 90
   ```

3. **Reduce history length** if memory usage is high:
   ```yaml
   memory:
     max_history_length: 15
   ```

### Monitoring Memory Usage

Monitor memory usage with:

```bash
watch -n 1 'nvidia-smi && free -h'
```

## Integration with Existing Code

To integrate the enhanced memory with your existing OpenManus RL code:

```python
from openmanus_rl.memory.summarized_memory import SummarizedMemory
from openmanus_rl.utils.memory_manager import MemoryManager

# Initialize memory manager
manager = MemoryManager()
config = manager.get_memory_config("webshop")

# Initialize memory
memory = SummarizedMemory()

# Use in your agent
agent = YourAgentClass(memory=memory, memory_config=config)
```

## Tips for Maximizing Memory Effectiveness

1. **Use environment-specific configurations** for better results
2. **Monitor system resources** during operation
3. **Adjust concurrency** based on your system's capabilities
4. **Use shorter prompts** for faster processing
5. **Enable caching** to reduce repeated summarizations
6. **Regularly check summary quality** and adjust parameters as needed

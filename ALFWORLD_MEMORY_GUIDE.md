# Using Enhanced Memory with OpenManus RL

## Quick Start

To use the enhanced memory system with ALFWorld, you need to:

1. Set up ALFWorld data:
   ```bash
   export ALFWORLD_DATA=/path/to/alfworld/data
   ```

2. Run ALFWorld with enhanced memory:
   ```bash
   cd ~/OpenManus
   python ./scripts/rollout/run_alfworld_rollout_with_memory.py \
     --env_name alfworld \
     --batch_size 1 \
     --total_envs 10 \
     --max_steps 50 \
     --use_summary \
     --summary_endpoint http://localhost:11434 \
     --summary_model qwen2.5:7b-instruct \
     --summary_concurrency 1
   ```

## Memory Configuration

The enhanced memory system provides:
- Automatic summarization of long interaction histories
- Optimized prompts for different environments
- Configurable concurrency and timeouts
- Fallback behavior when summarization fails

## Troubleshooting

If you encounter issues with ALFWorld data:
1. Make sure ALFWORLD_DATA environment variable is set
2. Check that the data directory contains the required json_2.1.1 folder
3. Verify that the data files are accessible

For memory-related issues:
1. Ensure Ollama is running on localhost:11434
2. Check that the qwen2.5:7b-instruct model is installed
3. Monitor system resources during operation

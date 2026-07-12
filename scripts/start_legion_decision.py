"""
Стартовый скрипт для Legion Decision Framework.
"""
import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legion_decision.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


async def start_legion_decision() -> None:
    logger.info("Starting Legion Decision Framework")

    for d in ('logs', 'data', 'models'):
        os.makedirs(d, exist_ok=True)

    from openmanus_rl.integration.legion_integration import LegionOptimizedAgent
    from openmanus_rl.integration.legion_osint_integration import LegionOSINTEnhancedAgent
    from openmanus_rl.integration.legion_voice_integration import LegionVoicePipeline

    config = {
        'model_name': os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct'),
        'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'gpu_memory_limit': 8 * 1024 ** 3,
        'ram_limit': 64 * 1024 ** 3,
    }

    _agent = LegionOptimizedAgent(config)
    _voice = LegionVoicePipeline(config)
    osint_agent = LegionOSINTEnhancedAgent(config)

    logger.info("Legion Decision Framework running — Sprint 2 will wire the main loop")

    # Sprint 2: добавить event loop / API server здесь

    await osint_agent.cleanup()
    logger.info("Legion Decision Framework stopped")


if __name__ == "__main__":
    asyncio.run(start_legion_decision())

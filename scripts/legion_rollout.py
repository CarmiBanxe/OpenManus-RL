"""
Rollout script для Legion Decision Framework.
Тестирует интеграцию с OpenManus-RL в средах AlfWorld, GAIA, WebShop.
"""
import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from openmanus_rl.integration.legion_integration import LegionOptimizedAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legion_rollout.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class LegionRolloutManager:
    """Управляет rollout процедурами для Legion Decision Framework."""

    def __init__(
        self,
        env_name: str = "alfworld",
        model_name: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        batch_size: int = 4,
        total_envs: int = 10,
        max_steps: int = 30,
        concurrency: int = 2,
    ) -> None:
        self.env_name = env_name
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.total_envs = total_envs
        self.max_steps = max_steps
        self.concurrency = concurrency

        self.agent = LegionOptimizedAgent({
            'model_name': self.model_name,
            'base_url': self.base_url,
            'gpu_memory_limit': 8 * 1024 ** 3,
            'ram_limit': 64 * 1024 ** 3,
        })

        self.stats: Dict[str, Any] = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'avg_steps_per_episode': 0.0,
            'avg_reward_per_episode': 0.0,
        }
        logger.info(f"LegionRolloutManager initialized for {env_name}")

    async def run_rollout(self) -> Dict[str, Any]:
        """Запуск rollout процедуры."""
        logger.info(f"Starting rollout: env={self.env_name}, model={self.model_name}")
        env = await self._load_environment()

        for episode in range(self.total_envs):
            logger.info(f"Episode {episode + 1}/{self.total_envs}")
            await self._run_episode(env, episode)

        self._calculate_stats()
        self._save_results()
        return self.stats

    async def _load_environment(self) -> Any:
        if self.env_name == "alfworld":
            from openmanus_rl.envs.alfworld import AlfWorldEnv  # type: ignore[import]
            return AlfWorldEnv(model_name=self.model_name, base_url=self.base_url)
        if self.env_name == "gaia":
            from openmanus_rl.envs.gaia import GAIAEnv  # type: ignore[import]
            return GAIAEnv(model_name=self.model_name, base_url=self.base_url)
        if self.env_name == "webshop":
            from openmanus_rl.envs.webshop import WebShopEnv  # type: ignore[import]
            return WebShopEnv(model_name=self.model_name, base_url=self.base_url)
        raise ValueError(f"Unknown environment: {self.env_name}")

    async def _run_episode(self, env: Any, episode_num: int) -> None:
        state = await env.reset()
        episode_reward = 0.0
        episode_steps = 0
        success = False

        for step in range(self.max_steps):
            action = self.agent.select_action(state, env.get_available_actions())
            next_state, reward, done, _info = await env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.stats['total_steps'] += 1

            logger.debug(
                f"Episode {episode_num + 1}, Step {step + 1}: reward={reward:.4f}, done={done}"
            )

            if done:
                success = reward > 0
                break
            state = next_state

        self.stats['total_episodes'] += 1
        if success:
            self.stats['successful_episodes'] += 1
        else:
            self.stats['failed_episodes'] += 1
        self.stats['total_reward'] += episode_reward
        logger.info(
            f"Episode {episode_num + 1}: steps={episode_steps}, "
            f"reward={episode_reward:.4f}, success={success}"
        )

    def _calculate_stats(self) -> None:
        n = self.stats['total_episodes']
        if n > 0:
            self.stats['avg_steps_per_episode'] = self.stats['total_steps'] / n
            self.stats['avg_reward_per_episode'] = self.stats['total_reward'] / n

    def _save_results(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/legion_rollout_{self.env_name}_{ts}.json"
        results = {
            'config': {
                'env_name': self.env_name,
                'model_name': self.model_name,
                'base_url': self.base_url,
                'batch_size': self.batch_size,
                'total_envs': self.total_envs,
                'max_steps': self.max_steps,
                'concurrency': self.concurrency,
            },
            'stats': self.stats,
        }
        with open(path, 'w') as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"Results saved to {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description='Legion Rollout Manager')
    parser.add_argument('--env', type=str, default='alfworld',
                        choices=['alfworld', 'gaia', 'webshop'])
    parser.add_argument('--model', type=str, default='qwen2.5:7b-instruct')
    parser.add_argument('--base_url', type=str, default='http://localhost:11434')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--total_envs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--concurrency', type=int, default=2)
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)

    manager = LegionRolloutManager(
        env_name=args.env,
        model_name=args.model,
        base_url=args.base_url,
        batch_size=args.batch_size,
        total_envs=args.total_envs,
        max_steps=args.max_steps,
        concurrency=args.concurrency,
    )
    results = await manager.run_rollout()

    print("\nRollout Results:")
    print(f"  Total Episodes:         {results['total_episodes']}")
    print(f"  Successful Episodes:    {results['successful_episodes']}")
    print(f"  Failed Episodes:        {results['failed_episodes']}")
    print(f"  Avg Steps per Episode:  {results['avg_steps_per_episode']:.2f}")
    print(f"  Avg Reward per Episode: {results['avg_reward_per_episode']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

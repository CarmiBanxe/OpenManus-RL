# ПРОМПТ ДЛЯ CLAUDE CODE: РЕАЛИЗАЦИЯ СПРИНТА 1

## КОНТЕКСТ ПРОЕКТА

Ты — Claude Code, работающий в репозитории OpenManus-RL (~/OpenManus).
Проект — локальный AI-движок принятия решений для системы Legion (8GB VRAM, 64GB RAM, AMD GPU с ROCm).
Все модели запускаются локально через Ollama на localhost:11434 или vLLM.

## ЦЕЛЬ СПРИНТА 1

Полная интеграция фреймворка принятия решений в инфраструктуру OpenManus-RL с добавлением базового голосового интерфейса и OSINT-возможностей.

## ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ

### 1. ИНФРАСТРУКТУРНЫЙ СЛОЙ

#### 1.1. Docker окружение для OpenManus-RL

**Создать файл:** `docker/legion-decision/Dockerfile`
```dockerfile
FROM rocm/pytorch:latest

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv \
    git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-legion.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-legion.txt

WORKDIR /app
COPY . /app/
RUN pip install -e .

EXPOSE 8000 8080 8082 5009

CMD ["python", "scripts/start_legion_decision.py"]
```

**Создать файл:** `docker/legion-decision/docker-compose.yml`
```yaml
version: '3.8'

services:
  legion-decision:
    build: .
    container_name: legion-decision
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - VLLM_BASE_URL=http://host.docker.internal:8082
      - WHISPER_MODEL=faster-whisper-large-v3-turbo
      - TTS_MODEL=kokoro-82m
      - GPU_DEVICE=0
      - ROCM_PATH=/opt/rocm
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    ports:
      - "8000:8000"
      - "8080:8080"
      - "8082:8082"
      - "5009:5009"
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    privileged: true
    restart: unless-stopped
    depends_on:
      - spiderfoot

  spiderfoot:
    image: spiderfoot/spiderfoot:latest
    container_name: spiderfoot
    ports:
      - "5009:5009"
    restart: unless-stopped
```

#### 1.2. Rollout процедуры для тестирования

**Создать файл:** `scripts/legion_rollout.py`
```python
"""
Rollout script для Legion Decision Framework.
Тестирует интеграцию с OpenManus-RL в средах AlfWorld, GAIA, WebShop.
"""
import os
import json
import asyncio
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

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
    ):
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

    async def _load_environment(self):
        if self.env_name == "alfworld":
            from openmanus_rl.envs.alfworld import AlfWorldEnv
            return AlfWorldEnv(model_name=self.model_name, base_url=self.base_url)
        elif self.env_name == "gaia":
            from openmanus_rl.envs.gaia import GAIAEnv
            return GAIAEnv(model_name=self.model_name, base_url=self.base_url)
        elif self.env_name == "webshop":
            from openmanus_rl.envs.webshop import WebShopEnv
            return WebShopEnv(model_name=self.model_name, base_url=self.base_url)
        raise ValueError(f"Unknown environment: {self.env_name}")

    async def _run_episode(self, env: Any, episode_num: int) -> None:
        state = await env.reset()
        episode_reward = 0.0
        episode_steps = 0
        success = False

        for step in range(self.max_steps):
            action = await self.agent.select_action(state, env.get_available_actions())
            next_state, reward, done, info = await env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.stats['total_steps'] += 1

            logger.debug(f"Episode {episode_num + 1}, Step {step + 1}: reward={reward:.4f}, done={done}")

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
        logger.info(f"Episode {episode_num + 1}: steps={episode_steps}, reward={episode_reward:.4f}, success={success}")

    def _calculate_stats(self) -> None:
        if self.stats['total_episodes'] > 0:
            self.stats['avg_steps_per_episode'] = self.stats['total_steps'] / self.stats['total_episodes']
            self.stats['avg_reward_per_episode'] = self.stats['total_reward'] / self.stats['total_episodes']

    def _save_results(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f'logs/legion_rollout_{self.env_name}_{ts}.json'
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
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description='Legion Rollout Manager')
    parser.add_argument('--env', type=str, default='alfworld', choices=['alfworld', 'gaia', 'webshop'])
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
    print(f"  Total Episodes:              {results['total_episodes']}")
    print(f"  Successful Episodes:         {results['successful_episodes']}")
    print(f"  Failed Episodes:             {results['failed_episodes']}")
    print(f"  Avg Steps per Episode:       {results['avg_steps_per_episode']:.2f}")
    print(f"  Avg Reward per Episode:      {results['avg_reward_per_episode']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. АГЕНТ ПРИНЯТИЯ РЕШЕНИЙ

#### 2.1. LegionOptimizedAgent

**Создать файл:** `openmanus_rl/integration/legion_integration.py`
```python
"""
Интеграция фреймворка принятия решений с системой Legion.
"""
import gc
import logging
from typing import Dict, Any, List, Optional

from openmanus_rl.agents.smart_decision_agent import SmartDecisionAgent

logger = logging.getLogger(__name__)


class LegionModelDistributor:
    """Оптимизация распределения моделей между GPU и CPU."""

    def __init__(
        self,
        gpu_memory_limit: int = 8 * 1024 ** 3,
        ram_limit: int = 64 * 1024 ** 3,
    ):
        self.gpu_memory_limit = gpu_memory_limit
        self.ram_limit = ram_limit
        self.model_requirements: Dict[str, Dict[str, int]] = {
            'qwen2.5-7b':    {'gpu':  5 * 1024**3, 'ram':  8 * 1024**3},
            'qwen2.5-14b':   {'gpu': 10 * 1024**3, 'ram': 16 * 1024**3},
            'qwen2.5-32b':   {'gpu': 20 * 1024**3, 'ram': 32 * 1024**3},
            'qwen3-omni-30b':{'gpu': 17 * 1024**3, 'ram': 20 * 1024**3},
            'ultravox-8b':   {'gpu':  6 * 1024**3, 'ram':  8 * 1024**3},
        }

    def optimize_model_allocation(self, model_name: str, task_complexity: str) -> Dict[str, Any]:
        if model_name not in self.model_requirements:
            return {'placement': 'cpu', 'optimization': 'quantized', 'fallback_model': 'qwen2.5-7b'}

        req = self.model_requirements[model_name]
        if req['gpu'] <= self.gpu_memory_limit:
            return {'placement': 'gpu', 'optimization': 'full_precision'}
        if req['ram'] <= self.ram_limit:
            return {'placement': 'hybrid', 'optimization': 'quantized_gpu_cpu'}
        return {'placement': 'cpu', 'optimization': 'quantized', 'fallback_model': 'qwen2.5-7b'}


class LegionResourceManager:
    """Адаптивное управление ресурсами."""

    GPU_LIMIT = 8 * 1024 ** 3
    RAM_LIMIT = 64 * 1024 ** 3

    def __init__(self) -> None:
        self.current_load: Dict[str, int] = {'gpu': 0, 'ram': 0}
        self.task_queue: list = []

    def allocate_resources(self, task_priority: str, model_requirements: Dict[str, int]) -> Dict[str, Any]:
        available_gpu = self.GPU_LIMIT - self.current_load['gpu']
        available_ram = self.RAM_LIMIT - self.current_load['ram']

        if task_priority == 'high':
            if model_requirements.get('gpu', 0) > available_gpu:
                self._interrupt_low_priority_tasks()
            return {'gpu': True, 'ram': True, 'priority_boost': True}

        return {
            'gpu': available_gpu >= model_requirements.get('gpu', 0),
            'ram': available_ram >= model_requirements.get('ram', 0),
            'priority_boost': False,
        }

    def _interrupt_low_priority_tasks(self) -> None:
        self.task_queue = [t for t in self.task_queue if t.get('priority') == 'high']


class LegionDecisionOptimizer:
    """Локальная оптимизация принятия решений."""

    def __init__(self, ollama_base_url: str = "http://localhost:11434") -> None:
        self.ollama_base_url = ollama_base_url
        self.decision_cache: Dict[str, Dict[str, Any]] = {}

    def optimize_decision_for_legion(
        self, decision_context: Dict[str, Any], available_models: List[str]
    ) -> Dict[str, Any]:
        cache_key = str(hash(str(sorted(decision_context.items()))))
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        complexity = self._analyze_task_complexity(decision_context)
        optimal_model = self._select_optimal_model(complexity, available_models)
        result = {'model': optimal_model, 'complexity': complexity}
        self.decision_cache[cache_key] = result
        return result

    def _analyze_task_complexity(self, ctx: Dict[str, Any]) -> str:
        if ctx.get('context_length', 0) > 1000:
            return 'high'
        if ctx.get('options_count', 0) > 5:
            return 'medium'
        return 'low'

    def _select_optimal_model(self, complexity: str, available_models: List[str]) -> str:
        if complexity == 'high' and 'qwen2.5-14b' in available_models:
            return 'qwen2.5-14b'
        return 'qwen2.5-7b'


class LegionOptimizedAgent(SmartDecisionAgent):
    """Оптимизированный агент для системы Legion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {})
        cfg = config or {}

        self.model_distributor = LegionModelDistributor(
            gpu_memory_limit=cfg.get('gpu_memory_limit', 8 * 1024 ** 3),
            ram_limit=cfg.get('ram_limit', 64 * 1024 ** 3),
        )
        self.resource_manager = LegionResourceManager()
        self.decision_optimizer = LegionDecisionOptimizer(
            ollama_base_url=cfg.get('base_url', 'http://localhost:11434')
        )
        self.available_models: List[str] = cfg.get('available_models', ['qwen2.5-7b'])
        logger.info("LegionOptimizedAgent initialized")

    def select_action(self, state: Dict[str, Any], available_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        task_priority = 'high' if state.get('urgent') else ('medium' if state.get('complexity') == 'high' else 'low')
        model_name = self.config.get('model_name', 'qwen2.5-7b')
        model_req = self.model_distributor.model_requirements.get(model_name, {'gpu': 5 * 1024**3, 'ram': 8 * 1024**3})
        resource_allocation = self.resource_manager.allocate_resources(task_priority, model_req)

        optimized = self.decision_optimizer.optimize_decision_for_legion(
            {'state': state, 'task_priority': task_priority},
            self.available_models,
        )

        action = super().select_action(state, available_actions)
        action['legion_optimization'] = {
            'resource_allocation': resource_allocation,
            'model_used': optimized.get('model', model_name),
            'task_priority': task_priority,
        }
        return action

    def cleanup_resources(self) -> None:
        gc.collect()
        logger.info("Resources cleaned up")
```

---

### 3. БАЗОВАЯ ГОЛОСОВАЯ ИНТЕГРАЦИЯ

**Создать файл:** `openmanus_rl/integration/legion_voice_integration.py`
```python
"""
Голосовая интеграция для системы Legion.
"""
import io
import logging
from typing import Dict, Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class LegionSTTIntegration:
    """STT через Faster-Whisper с поддержкой AMD ROCm."""

    def __init__(
        self,
        model_name: str = "faster-whisper-large-v3-turbo",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.whisper_model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import faster_whisper
            self.whisper_model = faster_whisper.WhisperModel(
                self.model_name, device=self.device, compute_type="float16"
            )
            logger.info(f"Loaded {self.model_name} on {self.device}")
        except ImportError:
            logger.warning("faster-whisper not installed — STT unavailable")

    def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        if self.whisper_model is None:
            return ""
        try:
            if isinstance(audio_data, bytes):
                import soundfile as sf
                audio_data, _ = sf.read(io.BytesIO(audio_data))
            segments, _ = self.whisper_model.transcribe(audio_data, beam_size=5)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            return ""

    def enhance_decision_context(
        self, audio_input: Union[bytes, np.ndarray], decision_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            **decision_context,
            'voice_input': self.transcribe_audio(audio_input),
            'input_modality': 'voice',
        }


class LegionTTSIntegration:
    """TTS через Kokoro-82M (до 210× real-time, AMD ROCm via ONNX)."""

    def __init__(self, model_name: str = "kokoro-82m", voice: str = "af_sky") -> None:
        self.model_name = model_name
        self.voice = voice
        self.tts_model = None
        self.voice_cache: Dict[str, bytes] = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            import kokoro
            self.tts_model = kokoro.Kokoro(self.model_name)
            logger.info(f"Loaded {self.model_name} TTS")
        except ImportError:
            logger.warning("kokoro not installed — TTS unavailable")

    def text_to_speech(self, text: str) -> Optional[bytes]:
        if not text or self.tts_model is None:
            return None
        if text in self.voice_cache:
            return self.voice_cache[text]
        try:
            audio = self.tts_model.generate(text, voice=self.voice)
            self.voice_cache[text] = audio
            return audio
        except Exception as exc:
            logger.error(f"TTS error: {exc}")
            return None


class LegionVoicePipeline:
    """Полный голосовой pipeline: STT → decision (Sprint 2) → TTS."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.stt = LegionSTTIntegration(
            model_name=cfg.get('stt_model', 'faster-whisper-large-v3-turbo'),
            device=cfg.get('device', 'cuda'),
        )
        self.tts = LegionTTSIntegration(
            model_name=cfg.get('tts_model', 'kokoro-82m'),
            voice=cfg.get('voice', 'af_sky'),
        )
        logger.info("LegionVoicePipeline initialized")

    def process_voice_interaction(self, audio_input: Union[bytes, np.ndarray]) -> Optional[bytes]:
        text = self.stt.transcribe_audio(audio_input)
        if not text:
            return None
        # Sprint 2: заменить заглушку на реальный decision engine
        reply_text = f"Processed: {text}"
        return self.tts.text_to_speech(reply_text)
```

---

### 4. БАЗОВАЯ OSINT ИНТЕГРАЦИЯ

**Создать файл:** `openmanus_rl/integration/legion_osint_integration.py`
```python
"""
OSINT интеграция для системы Legion через SpiderFoot API.
"""
import logging
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LegionOSINTIntegration:
    """Интеграция с SpiderFoot для обогащения контекста."""

    def __init__(self, spiderfoot_api: str = "http://localhost:5009") -> None:
        self.spiderfoot_api = spiderfoot_api
        self.search_cache: Dict[str, Any] = {}
        self.client = httpx.AsyncClient(timeout=30.0)

    async def enhance_decision_context(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            search_results = await self._search_osint(decision_context)
            return {
                **decision_context,
                'osint_data': search_results,
                'risk_factors': self._extract_risk_factors(search_results),
                'confidence_score': self._calculate_confidence(search_results),
            }
        except Exception as exc:
            logger.error(f"OSINT enhancement error: {exc}")
            return decision_context

    async def _search_osint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        queries = self._generate_queries(context)
        results: Dict[str, Any] = {}
        for query in queries:
            if query in self.search_cache:
                results[query] = self.search_cache[query]
                continue
            try:
                resp = await self.client.post(
                    f"{self.spiderfoot_api}/api/scan",
                    json={"target": query, "modules": ["all"]},
                )
                data = resp.json() if resp.status_code == 200 else {"error": resp.status_code}
                results[query] = data
                self.search_cache[query] = data
            except Exception as exc:
                logger.error(f"OSINT search error for {query}: {exc}")
                results[query] = {"error": str(exc)}
        return results

    def _generate_queries(self, context: Dict[str, Any]) -> List[str]:
        queries: List[str] = []
        for key in ('entities', 'keywords', 'names'):
            queries.extend(context.get(key, []))
        return queries

    def _extract_risk_factors(self, results: Dict[str, Any]) -> List[str]:
        risk_factors: List[str] = []
        for query, data in results.items():
            for item in data.get('data', []):
                if isinstance(item, dict) and item.get('type') in ('sanction', 'blacklist', 'warning', 'risk'):
                    risk_factors.append(f"{query}: {item.get('value', 'Unknown')}")
        return risk_factors

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        total = sum(len(d.get('data', [])) for d in results.values() if isinstance(d.get('data'), list))
        if total == 0:
            return 0.0
        if total < 5:
            return 0.5
        if total < 20:
            return 0.7
        return 0.9

    async def cleanup(self) -> None:
        await self.client.aclose()


class LegionOSINTEnhancedAgent:
    """Агент с OSINT обогащением контекста."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.osint = LegionOSINTIntegration(
            spiderfoot_api=cfg.get('spiderfoot_api', 'http://localhost:5009')
        )
        logger.info("LegionOSINTEnhancedAgent initialized")

    async def select_action(
        self, state: Dict[str, Any], available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            enhanced = await self.osint.enhance_decision_context(state)
            # Sprint 2: подключить base_agent
            return {
                'action': 'wait',
                'explanation': 'OSINT enhanced context ready (Sprint 2 will wire decision engine)',
                'osint_enhanced': True,
                'risk_factors': enhanced.get('risk_factors', []),
                'confidence_score': enhanced.get('confidence_score', 0.0),
            }
        except Exception as exc:
            logger.error(f"OSINT action selection error: {exc}")
            return {'action': 'error', 'explanation': str(exc), 'osint_enhanced': False}

    async def cleanup(self) -> None:
        await self.osint.cleanup()
```

---

### 5. СТАРТОВЫЙ СКРИПТ

**Создать файл:** `scripts/start_legion_decision.py`
```python
"""
Стартовый скрипт для Legion Decision Framework.
"""
import asyncio
import logging
import os

from openmanus_rl.integration.legion_integration import LegionOptimizedAgent
from openmanus_rl.integration.legion_voice_integration import LegionVoicePipeline
from openmanus_rl.integration.legion_osint_integration import LegionOSINTEnhancedAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/legion_decision.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def start_legion_decision() -> None:
    logger.info("Starting Legion Decision Framework")

    for d in ('logs', 'data', 'models'):
        os.makedirs(d, exist_ok=True)

    config = {
        'model_name': os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct'),
        'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'gpu_memory_limit': 8 * 1024 ** 3,
        'ram_limit': 64 * 1024 ** 3,
    }

    agent = LegionOptimizedAgent(config)
    voice_pipeline = LegionVoicePipeline(config)
    osint_agent = LegionOSINTEnhancedAgent(config)

    logger.info("Legion Decision Framework started — Sprint 2 will wire the main loop")

    await osint_agent.cleanup()
    logger.info("Legion Decision Framework stopped")


if __name__ == "__main__":
    asyncio.run(start_legion_decision())
```

---

### 6. ЗАВИСИМОСТИ

**Создать файл:** `requirements-legion.txt`
```
# Core
openai>=1.0.0
httpx>=0.24.0
numpy>=1.24.0
pandas>=2.0.0

# Voice
faster-whisper>=0.9.0
kokoro>=0.7.0
soundfile>=0.12.0

# OSINT
spiderfoot>=3.3.0

# System
psutil>=5.9.0
structlog>=23.0.0
aiofiles>=23.0.0
aiohttp>=3.8.0
```

---

## ИНСТРУКЦИИ ДЛЯ CLAUDE CODE

1. **Создать директории:** `docker/legion-decision/`, `logs/`, `data/`, `models/`
2. **Создать файлы** в точно указанных путях — без переименований
3. **При ошибке:** остановись, опиши проблему и конкретный файл
4. **Проверить результат:** синтаксис Python, YAML, Dockerfile
5. **Сообщить:** список созданных файлов + следующие шаги для тестирования

## ПРИОРИТЕТЫ

| Приоритет | Задача |
|-----------|--------|
| Высокий | Docker окружение, LegionOptimizedAgent, STT + TTS |
| Средний | Rollout скрипт, OSINT интеграция, тесты |
| Низкий | Расширенные voice/OSINT функции, оптимизация |

## ОЖИДАЕМЫЙ РЕЗУЛЬТАТ СПРИНТА 1

- Docker окружение с AMD ROCm + SpiderFoot
- `LegionOptimizedAgent` с управлением ресурсами (GPU/CPU distributor)
- Базовый voice pipeline (Faster-Whisper STT + Kokoro TTS)
- Базовая OSINT интеграция (SpiderFoot async)
- Rollout скрипт для AlfWorld/GAIA/WebShop
- Стартовый скрипт `start_legion_decision.py`

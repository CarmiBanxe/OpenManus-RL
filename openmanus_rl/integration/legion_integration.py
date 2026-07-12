"""
Интеграция фреймворка принятия решений с системой Legion.
"""
import gc
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LegionModelDistributor:
    """Оптимизация распределения моделей между GPU и CPU."""

    def __init__(
        self,
        gpu_memory_limit: int = 8 * 1024 ** 3,
        ram_limit: int = 64 * 1024 ** 3,
    ) -> None:
        self.gpu_memory_limit = gpu_memory_limit
        self.ram_limit = ram_limit
        self.model_requirements: Dict[str, Dict[str, int]] = {
            'qwen2.5-7b':     {'gpu':  5 * 1024 ** 3, 'ram':  8 * 1024 ** 3},
            'qwen2.5-14b':    {'gpu': 10 * 1024 ** 3, 'ram': 16 * 1024 ** 3},
            'qwen2.5-32b':    {'gpu': 20 * 1024 ** 3, 'ram': 32 * 1024 ** 3},
            'qwen3-omni-30b': {'gpu': 17 * 1024 ** 3, 'ram': 20 * 1024 ** 3},
            'ultravox-8b':    {'gpu':  6 * 1024 ** 3, 'ram':  8 * 1024 ** 3},
        }

    def optimize_model_allocation(
        self, model_name: str, task_complexity: str
    ) -> Dict[str, Any]:
        if model_name not in self.model_requirements:
            return {
                'placement': 'cpu',
                'optimization': 'quantized',
                'fallback_model': 'qwen2.5-7b',
            }
        req = self.model_requirements[model_name]
        if req['gpu'] <= self.gpu_memory_limit:
            return {'placement': 'gpu', 'optimization': 'full_precision'}
        if req['ram'] <= self.ram_limit:
            return {'placement': 'hybrid', 'optimization': 'quantized_gpu_cpu'}
        return {
            'placement': 'cpu',
            'optimization': 'quantized',
            'fallback_model': 'qwen2.5-7b',
        }


class LegionResourceManager:
    """Адаптивное управление ресурсами."""

    GPU_LIMIT: int = 8 * 1024 ** 3
    RAM_LIMIT: int = 64 * 1024 ** 3

    def __init__(self) -> None:
        self.current_load: Dict[str, int] = {'gpu': 0, 'ram': 0}
        self.task_queue: List[Dict[str, Any]] = []

    def allocate_resources(
        self, task_priority: str, model_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
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
        self,
        decision_context: Dict[str, Any],
        available_models: List[str],
    ) -> Dict[str, Any]:
        cache_key = str(hash(str(sorted(str(decision_context).split()))))
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        complexity = self._analyze_task_complexity(decision_context)
        optimal_model = self._select_optimal_model(complexity, available_models)
        result: Dict[str, Any] = {'model': optimal_model, 'complexity': complexity}
        self.decision_cache[cache_key] = result
        return result

    def _analyze_task_complexity(self, ctx: Dict[str, Any]) -> str:
        if ctx.get('context_length', 0) > 1000:
            return 'high'
        if ctx.get('options_count', 0) > 5:
            return 'medium'
        return 'low'

    def _select_optimal_model(
        self, complexity: str, available_models: List[str]
    ) -> str:
        if complexity == 'high' and 'qwen2.5-14b' in available_models:
            return 'qwen2.5-14b'
        return 'qwen2.5-7b'


class LegionOptimizedAgent:
    """Оптимизированный агент принятия решений для системы Legion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.model_distributor = LegionModelDistributor(
            gpu_memory_limit=self.config.get('gpu_memory_limit', 8 * 1024 ** 3),
            ram_limit=self.config.get('ram_limit', 64 * 1024 ** 3),
        )
        self.resource_manager = LegionResourceManager()
        self.decision_optimizer = LegionDecisionOptimizer(
            ollama_base_url=self.config.get('base_url', 'http://localhost:11434')
        )
        self.available_models: List[str] = self.config.get(
            'available_models', ['qwen2.5-7b']
        )
        logger.info("LegionOptimizedAgent initialized")

    def select_action(
        self,
        state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        task_priority = (
            'high' if state.get('urgent')
            else ('medium' if state.get('complexity') == 'high' else 'low')
        )
        model_name = self.config.get('model_name', 'qwen2.5-7b')
        model_req = self.model_distributor.model_requirements.get(
            model_name, {'gpu': 5 * 1024 ** 3, 'ram': 8 * 1024 ** 3}
        )
        resource_allocation = self.resource_manager.allocate_resources(
            task_priority, model_req
        )
        optimized = self.decision_optimizer.optimize_decision_for_legion(
            {'state': state, 'task_priority': task_priority},
            self.available_models,
        )

        action: Dict[str, Any] = {
            'action': available_actions[0] if available_actions else 'wait',
            'explanation': 'LegionOptimizedAgent decision',
            'legion_optimization': {
                'resource_allocation': resource_allocation,
                'model_used': optimized.get('model', model_name),
                'task_priority': task_priority,
            },
        }
        return action

    def cleanup_resources(self) -> None:
        gc.collect()
        logger.info("Resources cleaned up")

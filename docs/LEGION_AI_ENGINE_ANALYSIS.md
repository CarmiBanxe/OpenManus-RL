# Анализ AI движка для Legion (небанковский проект)

## Обзор проекта Legion

Legion - это локальная AI система с 8GB VRAM и 64GB RAM, предназначенная для запуска локальных LLM моделей (Qwen3.6-35B-A3B и Gemma-4-12B) через Ollama на localhost:11434.

## Ключевые характеристики системы

### Аппаратные ресурсы
- **GPU:** 8GB VRAM
- **RAM:** 64GB
- **Платформа:** Linux (Legion)
- **Локальный LLM сервер:** Ollama на localhost:11434
- **Установленные модели:** qwen2.5-coder:7b-instruct-q4_K_M и qwen2.5:7b-instruct

### Программная среда
- **OpenManus:** Основной проект в ~/OpenManus
- **Ollama:** Локальный LLM сервер
- **Python окружение:** hf-env
- **Зависимости:** openai, together, gym, gymnasium, beautifulsoup4, selenium

## Потенциальные улучшения для движка принятия решений

### 1. Оптимизация распределения моделей

**Проблема:** Ограниченные ресурсы GPU (8GB) для больших моделей
**Решение:** Интеллектуальное распределение между GPU и CPU

```python
class LegionModelDistributor:
    def __init__(
        self,
        gpu_memory_limit: int = 8 * 1024 * 1024 * 1024,
        ram_limit: int = 64 * 1024 * 1024 * 1024,
    ):
        self.gpu_memory_limit = gpu_memory_limit
        self.ram_limit = ram_limit
        self.model_requirements = {
            'qwen3.6-35b': {'gpu': 12 * 1024**3, 'ram': 20 * 1024**3},
            'gemma-4-12b': {'gpu':  8 * 1024**3, 'ram': 16 * 1024**3},
            'qwen2.5-7b':  {'gpu':  5 * 1024**3, 'ram':  8 * 1024**3},
        }

    def optimize_model_allocation(self, model_name: str, task_complexity: str) -> dict:
        """Оптимизация распределения модели между GPU и CPU"""
        if model_name not in self.model_requirements:
            return {}

        requirements = self.model_requirements[model_name]

        if requirements['gpu'] <= self.gpu_memory_limit:
            return {'placement': 'gpu', 'optimization': 'full_precision'}

        if requirements['ram'] <= self.ram_limit:
            return {'placement': 'hybrid', 'optimization': 'quantized_gpu_cpu'}

        return {
            'placement': 'cpu',
            'optimization': 'quantized',
            'fallback_model': 'qwen2.5-7b',
        }
```

### 2. Адаптивное управление ресурсами

**Применение:** Динамическое распределение ресурсов в зависимости от задачи

```python
class LegionResourceManager:
    def __init__(self):
        self.current_load = {'gpu': 0, 'ram': 0}
        self.task_queue: list = []
        self.resource_history: list = []

    def allocate_resources(self, task_priority: str, model_requirements: dict) -> dict:
        """Выделение ресурсов на основе приоритета задачи"""
        available_gpu = 8 * 1024**3 - self.current_load['gpu']
        available_ram = 64 * 1024**3 - self.current_load['ram']

        if task_priority == 'high':
            if model_requirements['gpu'] > available_gpu:
                self._interrupt_low_priority_tasks()
            return {'gpu': True, 'ram': True, 'priority_boost': True}

        return {
            'gpu': available_gpu >= model_requirements['gpu'],
            'ram': available_ram >= model_requirements['ram'],
            'priority_boost': False,
        }
```

### 3. Локальная оптимизация принятия решений

**Применение:** Улучшение процесса принятия решений с учетом локальных ограничений

```python
class LegionDecisionOptimizer:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_cache: dict = {}
        self.decision_cache: dict = {}

    def optimize_decision_for_legion(
        self, decision_context: dict, available_models: list
    ) -> dict:
        """Оптимизация принятия решений с учетом ресурсов Legion"""
        complexity = self._analyze_task_complexity(decision_context)
        optimal_model = self._select_optimal_model(complexity, available_models)

        cache_key = self._generate_cache_key(decision_context)
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        decision = self._execute_with_optimized_resources(optimal_model, decision_context)
        self.decision_cache[cache_key] = decision
        return decision
```

### 4. Интеграция с локальными данными

**Применение:** Использование локальных данных для улучшения принятия решений

```python
class LegionDataIntegrator:
    def __init__(self, data_path: str = "~/OpenManus/data"):
        self.data_path = data_path
        self.local_knowledge_base: dict = {}
        self.user_preferences: dict = {}

    def enhance_decision_with_local_data(self, decision_context: dict) -> dict:
        """Улучшение решений с использованием локальных данных"""
        local_insights = self._analyze_local_data(decision_context)
        user_context = self._get_user_preferences(decision_context.get('user_id'))

        return {
            **decision_context,
            'local_insights': local_insights,
            'user_preferences': user_context,
            'resource_constraints': self._get_current_resource_status(),
        }
```

## Интеграция в роудмап

### Спринт 1: Улучшение интеграции с OpenManus (дополнения для Legion)
**Добавить:**
- Оптимизация распределения моделей между GPU и CPU
- Адаптивное управление ресурсами для системы Legion
- Интеграция с локальным Ollama сервером

### Спринт 4: Оптимизация производительности (дополнения для Legion)
**Добавить:**
- Специфичная оптимизация для 8GB VRAM
- Кеширование решений для локальной системы
- Управление очередью задач с учетом приоритетов

### Спринт 7: Интеграция с внешними системами (дополнения для Legion)
**Добавить:**
- Интеграция с локальными данными
- Оптимизация работы с локальными файлами
- Улучшение работы с локальными моделями

## Преимущества для Legion

1. **Оптимальное использование ресурсов** за счет интеллектуального распределения
2. **Улучшенная производительность** через адаптивное управление
3. **Персонализация** через учет локальных данных и предпочтений
4. **Стабильность** через эффективное управление очередью задач
5. **Эффективность** через кеширование решений

## Реализация для OpenManus

```python
# В openmanus_rl/integration/legion_integration.py

class LegionOptimizedAgent:
    def __init__(self, config: dict):
        self.model_distributor = LegionModelDistributor()
        self.resource_manager = LegionResourceManager()
        self.decision_optimizer = LegionDecisionOptimizer()
        self.data_integrator = LegionDataIntegrator()

    def select_action(self, state: dict, available_actions: list) -> dict:
        """Оптимизированный выбор действий для системы Legion"""
        enhanced_context = self.data_integrator.enhance_decision_with_local_data(state)

        resource_allocation = self.resource_manager.allocate_resources(
            enhanced_context.get('priority', 'normal'),
            enhanced_context.get('model_requirements', {'gpu': 0, 'ram': 0}),
        )

        decision = self.decision_optimizer.optimize_decision_for_legion(
            enhanced_context,
            resource_allocation.get('available_models', ['qwen2.5-7b']),
        )

        return decision
```

## Следующие шаги

1. Реализовать оптимизацию распределения моделей для 8GB VRAM
2. Добавить адаптивное управление ресурсами
3. Интегрировать локальные данные в процесс принятия решений
4. Оптимизировать работу с очередью задач
5. Добавить кеширование решений для повышения производительности

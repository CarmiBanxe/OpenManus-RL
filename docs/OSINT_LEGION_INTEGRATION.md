# OSINT-стек для усиления движка Legion

## Анализ документа для небанковского применения

### Что из OSINT-стека применимо к Legion

Legion - это локальная AI система с 8GB VRAM и 64GB RAM, предназначенная для запуска локальных LLM моделей. Из документа об OSINT-стеке для AML/KYC можно извлечь несколько компонентов, полезных для усиления движка принятия решений.

### 1. Поисковые механизмы и подходы

#### SpiderFoot - универсальный OSINT-фреймворк
**Применение для Legion:**
- 200+ модулей для поиска информации
- Встроенная интеграция с TOR для поиска в dark web
- YAML-настраиваемый correlation engine
- Экспорт в CSV/JSON/GEXF, SQLite backend
- Web-интерфейс + CLI, поддержка Docker

**Интеграция в движок принятия решений:**
```python
class LegionOSINTIntegration:
    def __init__(self):
        self.spiderfoot_api = "http://localhost:5009"
        self.search_cache = {}
    
    def enhance_decision_context(self, decision_context):
        """Улучшение контекста принятия решений через OSINT"""
        # Поиск релевантной информации
        search_results = self._search_osint(decision_context)
        
        # Обогащение контекста
        enhanced_context = {
            **decision_context,
            'osint_data': search_results,
            'risk_factors': self._extract_risk_factors(search_results),
            'confidence_score': self._calculate_confidence(search_results)
        }
        
        return enhanced_context
    
    def _search_osint(self, context):
        """Поиск информации через SpiderFoot"""
        # Определение поисковых запросов на основе контекста
        queries = self._generate_queries(context)
        
        # Выполнение поиска
        results = {}
        for query in queries:
            if query not in self.search_cache:
                response = requests.post(f"{self.spiderfoot_api}/api", json=query)
                results[query] = response.json()
                self.search_cache[query] = results[query]
            else:
                results[query] = self.search_cache[query]
        
        return results
```

#### Maltego Community Edition - граф-аналитика
**Применение для Legion:**
- Визуализация сетей связей
- 200 Credits/месяц бесплатно
- Transforms для различных источников данных

**Интеграция в движок принятия решений:**
```python
class LegionGraphAnalytics:
    def __init__(self):
        self.maltego_api = "https://localhost:8080"
        self.graph_cache = {}
    
    def analyze_relationships(self, decision_context):
        """Анализ связей для улучшения принятия решений"""
        # Создание графа связей
        graph_data = self._build_relationship_graph(decision_context)
        
        # Выявление ключевых узлов и связей
        key_nodes = self._identify_key_nodes(graph_data)
        
        # Интеграция в контекст принятия решений
        enhanced_context = {
            **decision_context,
            'relationship_graph': graph_data,
            'key_influencers': key_nodes,
            'network_centrality': self._calculate_centrality(graph_data)
        }
        
        return enhanced_context
```

### 2. Технические компоненты для интеграции

#### Docker-контейнеры для локального развертывания
**Применение для Legion:**
- SpiderFoot: `docker run -p 5009:5009 spiderfoot/spiderfoot:latest`
- OpenSanctions yente: Docker-контейнер + ElasticSearch
- Maltego CE: локальное развертывание

#### API-интеграции
**Применение для Legion:**
- OpenSanctions API: 2.2M+ сущностей
- OpenCorporates API: 204M компаний
- GDELT Project: adverse media, 100+ языков

### 3. Сравнение с текущим решением Legion

| Компонент | Текущее решение Legion | OSINT-стек | Преимущества интеграции |
|-----------|------------------------|------------|------------------------|
| Поиск информации | Базовый веб-поиск | SpiderFoot (200+ модулей) | Расширенный поиск, TOR-интеграция |
| Визуализация связей | Отсутствует | Maltego CE | Граф-аналитика связей |
| Источники данных | Ограниченные | OpenSanctions, OpenCorporates, GDELT | Множество источников данных |
| Кеширование | Базовое | SQLite backend | Улучшенное кеширование |
| Анализ рисков | Базовый | Корреляционный движок | Улучшенный анализ рисков |

### 4. Дополнения к движку принятия решений

#### Улучшенный контекст для принятия решений
```python
class EnhancedLegionDecisionEngine:
    def __init__(self):
        self.osint_integration = LegionOSINTIntegration()
        self.graph_analytics = LegionGraphAnalytics()
        self.base_engine = SmartDecisionAgent()
    
    def select_action(self, state, available_actions):
        """Улучшенное принятие решений с OSINT"""
        # Обогащение контекста через OSINT
        osint_enhanced_context = self.osint_integration.enhance_decision_context(state)
        
        # Анализ связей
        graph_enhanced_context = self.graph_analytics.analyze_relationships(osint_enhanced_context)
        
        # Принятие решений с улучшенным контекстом
        action = self.base_engine.select_action(graph_enhanced_context, available_actions)
        
        return action
```

#### Улучшенное управление ресурсами
```python
class LegionResourceManagerWithOSINT:
    def __init__(self):
        self.base_resource_manager = LegionResourceManager()
        self.osint_priority = {
            'spiderfoot': 'medium',
            'maltego': 'low',
            'opencorporates': 'medium',
            'gdelt': 'high'
        }
    
    def allocate_resources_with_osint(self, task_priority, model_requirements, osint_needs):
        """Распределение ресурсов с учетом OSINT-потребностей"""
        # Базовое распределение ресурсов
        base_allocation = self.base_resource_manager.allocate_resources(
            task_priority, model_requirements
        )
        
        # Дополнительное распределение для OSINT
        osint_allocation = {}
        for osint_tool, priority in osint_needs.items():
            if self.osint_priority[osint_tool] == 'high' or task_priority == 'high':
                osint_allocation[osint_tool] = {'enabled': True, 'resources': 'high'}
            else:
                osint_allocation[osint_tool] = {'enabled': True, 'resources': 'low'}
        
        return {**base_allocation, 'osint_allocation': osint_allocation}
```

## Интеграция в роудмап

### Спринт 2: Расширение подходов принятия решений (дополнения)
**Добавить из OSINT-стека:**
- Интеграция SpiderFoot для расширенного поиска информации
- Базовая граф-аналитика связей
- Кеширование OSINT-результатов

### Спринт 4: Оптимизация производительности (дополнения)
**Добавить из OSINT-стека:**
- Оптимизация распределения ресурсов для OSINT-запросов
- Кеширование OSINT-данных в SQLite
- Приоритизация OSINT-источников

### Спринт 6: Специализированные доменные модели (дополнения)
**Добавить из OSINT-стека:**
- Модель для анализа связей и сетей
- Модель для оценки рисков на основе OSINT-данных
- Модель для принятия решений с учетом неструктурированной информации

## Преимущества для Legion

1. **Расширенный поиск информации** через 200+ модулей SpiderFoot
2. **Визуализация связей** через Maltego CE
3. **Улучшенный контекст** для принятия решений
4. **Приоритизация источников** для оптимального использования ресурсов
5. **Кеширование результатов** для повышения производительности

## Реализация для OpenManus

```python
# В openmanus_rl/integration/legion_osint_integration.py

class LegionOSINTEnhancedAgent:
    def __init__(self, config):
        self.base_agent = SmartDecisionAgent(config)
        self.osint_integration = LegionOSINTIntegration()
        self.graph_analytics = LegionGraphAnalytics()
        self.resource_manager = LegionResourceManagerWithOSINT()
    
    def select_action(self, state, available_actions):
        """Улучшенное принятие решений с OSINT"""
        # Распределение ресурсов с учетом OSINT
        osint_needs = self._determine_osint_needs(state)
        resource_allocation = self.resource_manager.allocate_resources_with_osint(
            state.get('priority', 'normal'),
            state.get('model_requirements'),
            osint_needs
        )
        
        # Обогащение контекста через OSINT
        if resource_allocation['osint_allocation'].get('spiderfoot', {}).get('enabled'):
            osint_enhanced_context = self.osint_integration.enhance_decision_context(state)
        else:
            osint_enhanced_context = state
        
        # Анализ связей
        if resource_allocation['osint_allocation'].get('maltego', {}).get('enabled'):
            graph_enhanced_context = self.graph_analytics.analyze_relationships(osint_enhanced_context)
        else:
            graph_enhanced_context = osint_enhanced_context
        
        # Принятие решений с улучшенным контекстом
        action = self.base_agent.select_action(graph_enhanced_context, available_actions)
        
        return action
```

## Следующие шаги

1. Развернуть SpiderFoot в Docker-контейнере
2. Интегрировать SpiderFoot API в движок принятия решений
3. Добавить базовую граф-аналитику связей
4. Оптимизировать распределение ресурсов для OSINT-запросов
5. Реализовать кеширование OSINT-данных

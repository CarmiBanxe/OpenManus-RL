# Голосовой менеджер для Legion: анализ и интеграция

## Анализ применимости для небанковского движка

### Что из голосового менеджера применимо к Legion

Legion - это локальная AI система с 8GB VRAM и 64GB RAM. Из документа о голосовом менеджере BANXE можно извлечь несколько компонентов, полезных для усиления движка принятия решений через голосовой интерфейс.

### 1. Голосовые модели для локального развертывания

#### Qwen3-Omni-30B-A3B - основная рекомендация
**Применение для Legion:**
- Энд-to-енд мультимодальная модель: аудио, текст, изображения
- Thinker–Talker MoE архитектура
- Latency первого пакета: 234ms
- 119 языков для текста, 19 для speech input, 10 для speech output
- Лицензия: Apache 2.0 - бесплатно для коммерческого использования

**Адаптация для Legion:**
```python
class LegionVoiceInterface:
    def __init__(self):
        self.qwen_omni_model = "http://localhost:8082"  # vLLM endpoint
        self.voice_cache = {}

    def process_voice_input(self, audio_data):
        """Обработка голосового ввода для принятия решений"""
        response = requests.post(
            f"{self.qwen_omni_model}/v1/audio/chat",
            files={"audio": audio_data},
            json={"model": "qwen3-omni-30b-a3b"}
        )
        return response.json()

    def voice_decision_query(self, audio_query):
        """Голосовой запрос к движку принятия решений"""
        text_query = self.process_voice_input(audio_query)
        decision = self.decision_engine.select_action(
            text_query,
            self.get_available_actions()
        )
        voice_response = self.text_to_speech(decision['explanation'])
        return voice_response
```

#### Ultravox - легковесная альтернатива
**Применение для Legion:**
- Мультимодальный LLM, понимающий речь без ASR
- Вариант 8B для систем с ограниченными ресурсами
- First token: 150ms, 60 tokens/sec

**Интеграция в движок принятия решений:**
```python
class LegionVoiceDecisionEngine:
    def __init__(self):
        self.ultravox_model = "http://localhost:8083"  # vLLM endpoint
        self.base_engine = SmartDecisionAgent()

    def voice_select_action(self, audio_input, available_actions):
        """Принятие решений через голосовой интерфейс"""
        text_context = self.process_speech_to_text(audio_input)
        decision = self.base_engine.select_action(text_context, available_actions)
        return decision
```

### 2. Компоненты обработки речи

#### Faster-Whisper для STT
**Применение для Legion:**
- До 4× быстрее оригинала
- Поддержка AMD ROCm через форк wyoming-faster-whisper-rocm
- large-v3 — 2–4× realtime на AMD GPU

**Интеграция в движок принятия решений:**
```python
class LegionSTTIntegration:
    def __init__(self):
        self.whisper_model = "faster-whisper-large-v3-turbo"
        self.device = "cuda"  # AMD ROCm

    def transcribe_audio(self, audio_data):
        """Транскрибация аудио для контекста принятия решений"""
        result = self.whisper_model.transcribe(audio_data)
        return result['text']

    def enhance_decision_context(self, audio_input, decision_context):
        """Обогащение контекста через транскрипцию аудио"""
        transcribed_text = self.transcribe_audio(audio_input)
        enhanced_context = {
            **decision_context,
            'voice_input': transcribed_text,
            'input_modality': 'voice'
        }
        return enhanced_context
```

#### Kokoro-82M для TTS
**Применение для Legion:**
- 82M параметров, 54 голоса на 8 языках
- Генерация до 210× real-time на GPU
- AMD ROCm совместимость через ONNX runtime

**Интеграция в движок принятия решений:**
```python
class LegionTTSIntegration:
    def __init__(self):
        self.tts_model = KokoroModel()
        self.voice_cache = {}

    def text_to_speech(self, text):
        """Преобразование текста решения в речь"""
        if text in self.voice_cache:
            return self.voice_cache[text]
        audio_data = self.tts_model.synthesize(text)
        self.voice_cache[text] = audio_data
        return audio_data

    def voice_decision_output(self, decision_result):
        """Голосовой вывод результата принятия решений"""
        explanation_text = decision_result.get('explanation', '')
        return self.text_to_speech(explanation_text)
```

### 3. Оркестрация голосового интерфейса

#### Pipecat - основной фреймворк
**Применение для Legion:**
- Python-библиотека для построения voice pipeline
- Поддержка 40+ AI-провайдеров
- WebRTC и WebSocket транспорт
- Полностью self-hosted деплой

**Интеграция в движок принятия решений:**
```python
class LegionVoicePipeline:
    def __init__(self):
        self.stt = LegionSTTIntegration()
        self.decision_engine = SmartDecisionAgent()
        self.tts = LegionTTSIntegration()

    def process_voice_interaction(self, audio_input):
        """Полный цикл голосового взаимодействия"""
        text_input = self.stt.transcribe_audio(audio_input)
        decision = self.decision_engine.select_action(
            text_input,
            self.get_available_actions()
        )
        audio_output = self.tts.text_to_speech(decision['explanation'])
        return audio_output
```

### 4. Сравнение с текущим решением Legion

| Компонент | Текущее решение Legion | Голосовой менеджер | Преимущества интеграции |
|-----------|------------------------|-------------------|------------------------|
| Ввод данных | Текстовый/клавиатурный | Голосовой (STT) | Естественный интерфейс |
| Вывод данных | Текстовый | Голосовой (TTS) | Удобство использования |
| Модальность | Только текст | Мультимодальность | Расширенные возможности |
| Задержка | Минимальная | 234ms (Qwen3-Omni) | Приемлемо для интерактивности |
| Ресурсы | 8GB VRAM, 64GB RAM | Доп. 2-3GB VRAM | Оптимизированное использование |

### 5. Дополнения к движку принятия решений

#### Мультимодальный контекст для принятия решений
```python
class LegionMultimodalDecisionEngine:
    def __init__(self):
        self.base_engine = SmartDecisionAgent()
        self.voice_pipeline = LegionVoicePipeline()
        self.context_cache = {}

    def select_action_multimodal(self, input_data, input_modality, available_actions):
        """Принятие решений с учетом модальности ввода"""
        if input_modality == 'voice':
            text_context = self.voice_pipeline.stt.transcribe_audio(input_data)
        elif input_modality == 'text':
            text_context = input_data
        elif input_modality == 'image':
            text_context = self.process_image_description(input_data)
        else:
            text_context = str(input_data)

        enhanced_context = {
            'query': text_context,
            'input_modality': input_modality,
            'raw_input': input_data
        }

        decision = self.base_engine.select_action(enhanced_context, available_actions)
        decision['response_modality'] = input_modality
        return decision
```

#### Оптимизация распределения ресурсов для голосовых моделей
```python
class LegionVoiceResourceManager:
    def __init__(self):
        self.base_resource_manager = LegionResourceManager()
        self.voice_models = {
            'qwen3_omni': {'memory': 17_000_000_000, 'priority': 'high'},
            'ultravox':   {'memory':  8_000_000_000, 'priority': 'medium'},
            'whisper':    {'memory':  2_500_000_000, 'priority': 'medium'},
            'kokoro':     {'memory':    300_000_000, 'priority': 'low'}
        }

    def allocate_resources_for_voice(self, voice_models_needed):
        """Распределение ресурсов для голосовых моделей"""
        available_memory = 8 * 1024 * 1024 * 1024  # 8GB VRAM
        total_needed = sum(
            self.voice_models[m]['memory'] for m in voice_models_needed
        )

        if total_needed <= available_memory:
            return {m: {'enabled': True} for m in voice_models_needed}

        prioritized = sorted(
            voice_models_needed,
            key=lambda x: self.voice_models[x]['priority']
        )
        allocation = {}
        used_memory = 0
        for model in prioritized:
            if used_memory + self.voice_models[model]['memory'] <= available_memory:
                allocation[model] = {'enabled': True}
                used_memory += self.voice_models[model]['memory']
            else:
                allocation[model] = {'enabled': False}
        return allocation
```

## Интеграция в роудмап

### Спринт 1: Улучшение интеграции с OpenManus (дополнения)
**Добавить из голосового менеджера:**
- Базовая интеграция Faster-Whisper для распознавания речи
- Простая TTS интеграция через Kokoro-82M
- Тестирование голосового ввода/вывода

### Спринт 2: Расширение подходов принятия решений (дополнения)
**Добавить из голосового менеджера:**
- Мультимодальные входные данные (голос + текст)
- Qwen3-Omni интеграция для end-to-end обработки голоса
- Адаптация движка принятия решений к голосовому интерфейсу

### Спринт 4: Оптимизация производительности (дополнения)
**Добавить из голосового менеджера:**
- Оптимизация распределения ресурсов для голосовых моделей
- Кеширование голосовых запросов и ответов
- Приоритизация голосовых компонентов

### Спринт 7: Интеграция с внешними системами (дополнения)
**Добавить из голосового менеджера:**
- Pipecat интеграция для оркестрации голосового pipeline
- WebRTC транспорт для реального времени
- Веб-интерфейс для голосового взаимодействия

## Преимущества для Legion

1. **Естественный интерфейс** через голосовое взаимодействие
2. **Мультимодальность** для обогащения контекста принятия решений
3. **Удобство использования** для нетехнических пользователей
4. **Реальное время** взаимодействия с задержкой ~234ms
5. **Локальная обработка** без утечки данных

## Реализация для OpenManus

```python
# В openmanus_rl/integration/legion_voice_integration.py

class LegionVoiceEnhancedAgent:
    def __init__(self, config):
        self.base_agent = SmartDecisionAgent(config)
        self.voice_pipeline = LegionVoicePipeline()
        self.multimodal_engine = LegionMultimodalDecisionEngine()
        self.resource_manager = LegionVoiceResourceManager()

    def select_action_with_voice(self, input_data, input_modality, available_actions):
        """Принятие решений с голосовой поддержкой"""
        voice_models_needed = self._determine_voice_models(input_modality)
        resource_allocation = self.resource_manager.allocate_resources_for_voice(
            voice_models_needed
        )

        if not all(m['enabled'] for m in resource_allocation.values()):
            return self.base_agent.select_action(
                self._fallback_text_processing(input_data, input_modality),
                available_actions
            )

        decision = self.multimodal_engine.select_action_multimodal(
            input_data, input_modality, available_actions
        )

        if input_modality == 'voice':
            decision['voice_response'] = self.voice_pipeline.tts.text_to_speech(
                decision['explanation']
            )

        return decision
```

## Следующие шаги

1. Развернуть Faster-Whisper с поддержкой AMD ROCm
2. Интегрировать базовую TTS через Kokoro-82M
3. Добавить голосовой ввод в движок принятия решений
4. Оптимизировать распределение ресурсов для голосовых моделей
5. Реализовать полный голосовой pipeline через Pipecat

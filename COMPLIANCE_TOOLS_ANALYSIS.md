# Анализ compliance-инструментов для ИИ-помощника

## Ключевые выводы для OpenManus RL

### 1. Поисковые и скрейпинговые инструменты
- **Crawl4AI**: LLM-ready веб-краулер (Apache 2.0)
- **Wikidata SPARQL**: 100M+ сущностей под CC0 (публичное достояние)
- **AlgorithmWatch media-monitoring**: скрейпинг новостей + LLM классификация

### 2. Распознавание и обработка документов
- **PassportEye**: OCR для документов (MIT)
- **DeepFace**: распознавание лиц + liveness detection (MIT)
- **Marble-NER**: Entity Name Recognition API (MIT)

### 3. Оркестрация и управление процессами
- **Ballerine**: workflow engine для KYC/KYB (self-hosted free)
- **Jube**: real-time monitoring с ML (AGPLv3, internal free)
- **Marble**: case management + rule builder (Elastic V2, self-hosted free)

### 4. Интеграция с внешними API
- **UK Companies House API**: корпоративные данные (бесплатно)
- **open-kyc**: US registry lookup (npm package)
- **FINOS OpenAML**: crypto/blockchain AML (Apache 2.0)

### 5. Локальная обработка с LLM
- **AMI-Agent**: adverse media scoring через локальный LLM
- **Ollama integration**: использование qwen3:8b для анализа
- **RAG pipelines**: crawl4ai + LLM для обработки документов

Быстрый старт
=============

Все примеры — под РЕАЛЬНЫЙ API: единственная точка входа принятия решения — ``select_action``.

Агент напрямую (in-process)
---------------------------

.. code-block:: python

   import asyncio
   from openmanus_rl.config import load_config
   from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent

   # config — ВТОРОЙ аргумент конструктора (первый — base_agent)
   agent = EnhancedDecisionAgent(config=load_config("development"))

   result = asyncio.run(
       agent.select_action({"text": "What is the risk of BTC?"}, ["buy", "sell", "wait"])
   )
   print(result["action"], result["confidence"])
   # ключи ответа: action, confidence, explanation, osint_enhanced,
   #               remizov_used, source, episode_id, timestamp

API (FastAPI, за JWT-auth, localhost)
-------------------------------------

Секрет и админ-креды — только из env (никаких дефолтов)::

   export OPENMANUS_SECRET_KEY="<32+ bytes>"
   export OPENMANUS_ADMIN_USER="admin"
   export OPENMANUS_ADMIN_PASSWORD="<strong>"
   uvicorn openmanus_rl.api.server:app --host 127.0.0.1 --port 8000

.. code-block:: bash

   # 1) логин -> токен
   TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
     -H 'Content-Type: application/json' \
     -d '{"username":"admin","password":"'"$OPENMANUS_ADMIN_PASSWORD"'"}' | jq -r .access_token)

   # 2) запрос (требует токен; публичного эндпоинта нет)
   curl -s -X POST http://127.0.0.1:8000/query \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"text":"risk of BTC","available_actions":["buy","sell","wait"]}'

Метрики (Prometheus, за auth)
-----------------------------

``GET /metrics`` доступен только с токеном (не публичный)::

   curl -s http://127.0.0.1:8000/metrics -H "Authorization: Bearer $TOKEN"

Резервное копирование
---------------------

.. code-block:: bash

   python scripts/backup.py create                 # -> backups/openmanus_backup_<ts>.tar.gz
   python scripts/backup.py verify backups/<archive>.tar.gz
   python scripts/backup.py restore backups/<archive>.tar.gz /path/to/restore

Валидация спринтов (реальный pytest)
------------------------------------

.. code-block:: bash

   python scripts/validate_sprint.py --sprint all      # все наборы
   python scripts/validate_sprint.py --sprint metrics  # только /metrics
   python scripts/security_validator.py                # security-инварианты

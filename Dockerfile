# OpenManus API — production image (private Legion contour)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Реальный файл зависимостей проекта (НЕ requirements.txt) + API-стек
COPY requirements-legion.txt .
RUN pip install --no-cache-dir -r requirements-legion.txt fastapi "uvicorn[standard]" pyjwt bcrypt

COPY . .

ENV PYTHONPATH=/app
ENV OPENMANUS_CONFIG_FILE=production

EXPOSE 8000

# В контейнере bind 0.0.0.0; наружу выставляется ТОЛЬКО через compose-маппинг на 127.0.0.1.
# Секрет и админ-креды передаются через env (OPENMANUS_SECRET_KEY / OPENMANUS_ADMIN_*).
CMD ["uvicorn", "openmanus_rl.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

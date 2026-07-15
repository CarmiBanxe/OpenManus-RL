# Network Hardening — устранение публичной экспозиции (S-18 §1.2)

**Контекст:** `network_validator.py` обнаружил, что сервисы приватного Legion-контура слушают
`0.0.0.0` (все интерфейсы) вместо `127.0.0.1`, т.е. достижимы из сети:

```
❌ 0.0.0.0:4000  — LiteLLM gateway (fallback evo1/evo2)
❌ 0.0.0.0:3000  — Grafana
```

Красная линия: приватный контур (uncensored-движок + его инфраструктура) **не должен** быть
достижим из сети / из banking-контура. Ниже — как забиндить на localhost. Все действия —
**операторские** (сервисы вне OpenManus-репо).

---

## 1. LiteLLM gateway (порт 4000)

**Причина:** запускается с `--host 0.0.0.0` (или дефолт). Нужно `127.0.0.1`.

### Вариант A — прямой запуск litellm
```bash
# было (экспозиция):
litellm --config litellm-config.yaml --host 0.0.0.0 --port 4000
# стало (localhost-only):
litellm --config litellm-config.yaml --host 127.0.0.1 --port 4000
```

### Вариант B — systemd-юнит
В `ExecStart=` заменить `--host 0.0.0.0` на `--host 127.0.0.1`, затем:
```bash
sudo systemctl daemon-reload && sudo systemctl restart litellm
```

### Вариант C — docker (если LiteLLM в контейнере)
Порт-маппинг только на localhost хоста:
```yaml
ports:
  - "127.0.0.1:4000:4000"   # не "4000:4000" и не "0.0.0.0:4000:4000"
```

### Вариант D — не можешь сменить bind → фаервол
```bash
sudo ufw deny in on <ext_iface> to any port 4000
# либо nftables/iptables: DROP входящих на 4000 не с loopback
```

---

## 2. Grafana (порт 3000)

**Причина:** дефолт Grafana биндит `0.0.0.0:3000`.

### Вариант A — grafana.ini
```ini
[server]
http_addr = 127.0.0.1
http_port = 3000
```
Перезапуск: `sudo systemctl restart grafana-server`.

### Вариант B — env (docker/standalone)
```bash
GF_SERVER_HTTP_ADDR=127.0.0.1
```

### Вариант C — docker port mapping
```yaml
ports:
  - "127.0.0.1:3000:3000"
```

> Доступ к Grafana извне localhost — только через SSH-туннель:
> `ssh -L 3000:127.0.0.1:3000 user@legion` → открывать `http://127.0.0.1:3000` у себя.

---

## 3. Проверка после изменений

```bash
# оба порта должны показывать 127.0.0.1, НЕ 0.0.0.0 / [::]
ss -tlnH | grep -E ':(3000|4000)$'
# ожидаемо:
#   LISTEN 0 128 127.0.0.1:4000 ...
#   LISTEN 0 128 127.0.0.1:3000 ...
```

## 4. Валидация через network_validator

```bash
cd ~/OpenManus && /home/mmber/hf-env/bin/python scripts/network_validator.py
```
**Ожидаемо после исправления:**
```
# Network posture (private Legion contour)
  ℹ tor_socks_9050 = ...   ← мониторинг: проверяет ОТСУТСТВИЕ tor-демона, не включает его
✅ NETWORK ISOLATION OK
```
(exit 0 — публичных слушателей среди наблюдаемых портов больше нет)

> **SECURITY NOTE (2026-07-15, Charter §8):**
> Строка `tor_socks_9050` выше — это **только мониторинговый вывод** network_validator'а,
> который проверяет, что порт 9050 (Tor SOCKS) НЕ прослушивается.
> Tor / onion / anonymizing transports **ЗАПРЕЩЕНЫ** в этом проекте и несовместимы с
> Banxe SAFE-PORT charter §8. Tor-демон не установлен и не должен устанавливаться.

Плюс регрессия логики валидатора:
```bash
/home/mmber/hf-env/bin/python scripts/validate_sprint.py --sprint network   # 7/7 passed
```

---

## Примечание по остальным сервисам
Наши компоненты уже забиндены корректно:
- FastAPI (`server.py`) — дефолт `127.0.0.1` (config `host`); в docker наружу только через compose `127.0.0.1:8000:8000`.
- Streamlit/Gradio — localhost, `share=False`.
- llama-server (:8080/:8081), Ollama (:11434) — в `config.toml` указаны как `127.0.0.1` (проверь фактический bind тем же `ss`).

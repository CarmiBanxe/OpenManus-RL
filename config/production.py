"""Production-конфигурация — тонкая обёртка над единым load_config (без class-vs-dict)."""
from openmanus_rl.config import load_config

CONFIG = load_config("production")

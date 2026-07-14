"""Модуль структурированного логирования для OpenManus (structlog / JSON)."""
import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory


class OpenManusLogger:
    """Структурированный логгер для OpenManus."""

    def __init__(self, name: str = "openmanus", log_level: str = "INFO") -> None:
        self.name = name
        self.log_level = log_level
        self._logger = None
        self._configure_logging()

    def _configure_logging(self) -> None:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(format="%(message)s", stream=sys.stdout,
                            level=getattr(logging, self.log_level.upper()))
        self._logger = structlog.get_logger(self.name)

    def request(self, engine_type: str, model: str, operation: str, request_id: str, **kwargs: Any) -> None:
        self._logger.info("request_started", engine_type=engine_type, model=model,
                          operation=operation, request_id=request_id, **kwargs)

    def response(self, engine_type: str, model: str, operation: str, request_id: str,
                 status: str, duration: float, tokens: int = 0, **kwargs: Any) -> None:
        self._logger.info("request_completed", engine_type=engine_type, model=model,
                          operation=operation, request_id=request_id, status=status,
                          duration=duration, tokens=tokens, **kwargs)

    def error(self, engine_type: str, operation: str, request_id: str,
              error: str, error_type: str = "unknown", **kwargs: Any) -> None:
        self._logger.error("request_failed", engine_type=engine_type, operation=operation,
                          request_id=request_id, error=error, error_type=error_type, **kwargs)

    def fallback(self, engine_type: str, from_model: str, to_model: str,
                 request_id: str, **kwargs: Any) -> None:
        self._logger.warning("fallback_used", engine_type=engine_type, from_model=from_model,
                            to_model=to_model, request_id=request_id, **kwargs)

    def system_event(self, event: str, **kwargs: Any) -> None:
        self._logger.info(event, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, **kwargs)

    def error_log(self, message: str, **kwargs: Any) -> None:
        self._logger.error(message, **kwargs)


_logger = None


def get_logger(name: str = "openmanus", log_level: str = "INFO") -> OpenManusLogger:
    global _logger
    if _logger is None:
        _logger = OpenManusLogger(name, log_level)
    return _logger


def configure_logging(log_level: str = "INFO") -> None:
    global _logger
    if _logger is None:
        _logger = OpenManusLogger("openmanus", log_level)
    else:
        _logger.log_level = log_level
        _logger._configure_logging()

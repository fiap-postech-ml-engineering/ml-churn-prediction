"""Configuração centralizada de logging."""

import json
import logging
import os
from contextvars import ContextVar, Token
from datetime import datetime, timezone

from src.config.settings import (
    ENVIRONMENT,
    LOG_FORMAT,
    LOG_LEVEL,
    LOGS_DIR,
    SERVICE_NAME,
)

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> Token:
    """Salva o request_id no contexto atual para enriquecimento dos logs."""

    return _REQUEST_ID_CTX.set(request_id)


def clear_request_id(token: Token) -> None:
    """Restaura o valor anterior de request_id no contexto."""

    _REQUEST_ID_CTX.reset(token)


def get_request_id() -> str:
    """Retorna o request_id atual do contexto de execução."""

    return _REQUEST_ID_CTX.get()


class RequestContextFilter(logging.Filter):
    """Injeta request_id em todos os registros de log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Formata logs em JSON para observabilidade."""

    def __init__(self, environment: str, service_name: str):
        super().__init__()
        self.environment = environment
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": self.environment,
            "service": self.service_name,
            "request_id": get_request_id(),
        }

        extra_fields = (
            "event",
            "method",
            "path",
            "status_code",
            "latency_ms",
            "client_ip",
            "error",
        )
        for field in extra_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def setup_logging(environment=None, logs_dir=None, service_name=None, log_format=None):
    """Setup centralizado de logging para toda aplicação.

    Args:
        environment: 'development' ou 'production'. Se None, usa ENVIRONMENT.
        logs_dir: Path para diretório de logs. Se None, usa LOGS_DIR.
        service_name: Nome do serviço para logs estruturados.
        log_format: Formato de saída. "json" ou "text".
    """

    env = environment or os.getenv("ENVIRONMENT", ENVIRONMENT)
    log_dir = logs_dir or LOGS_DIR
    app_name = service_name or os.getenv("SERVICE_NAME", SERVICE_NAME)
    output_format = (log_format or os.getenv("LOG_FORMAT", LOG_FORMAT)).lower()

    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    handlers = [logging.StreamHandler()]
    if env == "production":
        handlers.append(logging.FileHandler(log_dir / "app.log", encoding="utf-8"))

    context_filter = RequestContextFilter()
    if output_format == "json":
        formatter = JsonFormatter(environment=env, service_name=app_name)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - request_id=%(request_id)s - %(message)s"
        )

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)

    root_logger.setLevel(log_level)
    for handler in handlers:
        root_logger.addHandler(handler)

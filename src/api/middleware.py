"""Middlewares utilitarios para observabilidade da API."""

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from src.config.logging_config import clear_request_id, set_request_id
from src.config.settings import LATENCY_WARN_MS, REQUEST_ID_HEADER

logger = logging.getLogger(__name__)


def register_observability_middleware(app: FastAPI) -> None:
    """Registra middleware de latencia e correlacao de requisicoes."""

    @app.middleware("http")
    async def latency_middleware(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        token = set_request_id(request_id)
        started_at = perf_counter()
        status_code = 500
        response = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            log_extra = {
                "event": "request.completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "client_ip": request.client.host if request.client else None,
            }

            log_fn = logger.warning if latency_ms >= LATENCY_WARN_MS else logger.info
            log_fn("request.completed", extra=log_extra)

            if response is not None and REQUEST_ID_HEADER not in response.headers:
                response.headers[REQUEST_ID_HEADER] = request_id

            clear_request_id(token)

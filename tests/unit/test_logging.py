import json
import logging

import pytest

from src.config import setup_logging


def test_setup_logging_creates_handlers(tmp_path):
    """Testa se handlers são criados"""
    setup_logging(environment="development", logs_dir=tmp_path)

    logger = logging.getLogger()
    assert len(logger.handlers) > 0, "Nenhum handler configurado"


def test_console_handler_exists(tmp_path):
    """Testa se StreamHandler existe"""
    setup_logging(environment="development", logs_dir=tmp_path)

    logger = logging.getLogger()
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert has_stream, "StreamHandler não encontrado"


def test_file_handler_production(tmp_path):
    """Testa se FileHandler é criado em produção"""
    setup_logging(environment="production", logs_dir=tmp_path)

    logger = logging.getLogger()
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert has_file, "FileHandler não encontrado em produção"


def test_no_file_handler_development(tmp_path):
    """Testa se NÃO cria FileHandler em development"""
    setup_logging(environment="development", logs_dir=tmp_path)

    logger = logging.getLogger()
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert not has_file, "FileHandler não deveria existir em development"


def test_setup_logging_uses_json_formatter_by_default(tmp_path):
    """Testa se o formatter padrão produz payload JSON estruturado."""
    setup_logging(environment="development", logs_dir=tmp_path)

    logger = logging.getLogger()
    stream_handler = next(
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    )

    record = logging.LogRecord(
        name="tests.logging",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="health check",
        args=(),
        exc_info=None,
    )

    payload = json.loads(stream_handler.format(record))

    assert payload["message"] == "health check"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "-"
    assert payload["environment"] == "development"


def test_setup_logging_is_idempotent_without_handler_duplication(tmp_path):
    """Chamadas repetidas não devem acumular handlers no root logger."""
    setup_logging(environment="development", logs_dir=tmp_path)
    logger = logging.getLogger()
    first_handler_count = len(logger.handlers)

    setup_logging(environment="development", logs_dir=tmp_path)
    second_handler_count = len(logger.handlers)

    assert first_handler_count == 1
    assert second_handler_count == 1


def test_setup_logging_is_idempotent_when_switching_environment(tmp_path):
    """Reconfiguração entre ambientes mantém quantidade esperada de handlers."""
    setup_logging(environment="production", logs_dir=tmp_path)
    logger = logging.getLogger()
    assert len(logger.handlers) == 2

    setup_logging(environment="development", logs_dir=tmp_path)
    logger = logging.getLogger()
    assert len(logger.handlers) == 1

    setup_logging(environment="production", logs_dir=tmp_path)
    logger = logging.getLogger()
    assert len(logger.handlers) == 2

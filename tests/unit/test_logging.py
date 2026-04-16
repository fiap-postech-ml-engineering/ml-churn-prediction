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

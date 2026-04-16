"""Configuração centralizada de logging"""

import getpass
import logging
import os

from src.config.settings import LOG_LEVEL, LOGS_DIR


def setup_logging(environment=None, logs_dir=None):
    """Setup centralizado de logging para toda aplicação

    Args:
        environment: 'development' ou 'production'. Se None, lê de ENVIRONMENT env var
        logs_dir: Path para diretório de logs. Se None, usa LOGS_DIR
    """

    env = environment or os.getenv("ENVIRONMENT", "development")
    log_dir = logs_dir or LOGS_DIR

    log_level = getattr(logging, LOG_LEVEL)
    user = getpass.getuser()

    # Limpar handlers anteriores
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Handlers
    handlers = [logging.StreamHandler()]  # Console sempre

    if env == 'production':
        file_handler = logging.FileHandler(log_dir / "app.log")
        handlers.append(file_handler)

    # Configuração
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s - [%(levelname)s] - {user} - %(name)s: %(message)s',
        handlers=handlers,
    )

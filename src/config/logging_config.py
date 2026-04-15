"""Configuração centralizada de logging"""
import logging
import getpass
from src.config.settings import LOG_LEVEL, ENVIRONMENT, LOGS_DIR, DEBUG

def setup_logging():
    """Setup centralizado de logging para toda aplicação"""
    
    log_level = getattr(logging, LOG_LEVEL)
    user = getpass.getuser()
    
    # Handlers
    handlers = [logging.StreamHandler()]  # Console sempre
    
    if ENVIRONMENT == 'production':
        file_handler = logging.FileHandler(LOGS_DIR / "app.log")
        handlers.append(file_handler)
    
    # Configuração
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s - [%(levelname)s] - {user} - %(name)s: %(message)s',
        handlers=handlers
    )
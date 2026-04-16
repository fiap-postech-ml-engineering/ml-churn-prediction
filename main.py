"""
Entry point para rodar a API com uvicorn.

Uso:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from src.api.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    import logging
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Iniciando servidor FastAPI...")
    logger.info("=" * 70)
    logger.info("Documentação disponível em: http://localhost:8000/docs")
    logger.info("Redoc disponível em: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

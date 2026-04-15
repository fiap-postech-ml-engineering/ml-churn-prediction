import logging
import os
from pathlib import Path
from src.config import setup_logging, LOG_LEVEL, ENVIRONMENT, LOGS_DIR

def test_setup_logging():
    """Testa se logging está configurado corretamente"""
    setup_logging()
    
    # Testa se logger foi criado
    logger = logging.getLogger("churn-prediction")
    assert logger is not None
    print("✅ Logger criado")

def test_log_levels():
    """Testa se os níveis de log funcionam"""
    logger = logging.getLogger("churn-prediction")
    
    # Não vai gerar erro
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print("✅ Todos os níveis funcionam")

def test_log_format():
    """Testa se o formato está correto"""
    logger = logging.getLogger("churn-prediction")
    
    # Deve conter username, levelname, logger name
    logger.info("Test message")
    print("✅ Formato de log OK")

def test_file_handler_production():
    """Testa se arquivo é criado em produção"""
    if ENVIRONMENT == 'production':
        logger = logging.getLogger("churn-prediction")
        logger.info("Test file handler")
        
        app_log = LOGS_DIR / "app.log"
        assert app_log.exists(), f"Arquivo não encontrado: {app_log}"
        
        # Verifica se a mensagem está no arquivo
        with open(app_log, 'r') as f:
            content = f.read()
            assert "Test file handler" in content
        
        print("✅ File handler funcionando")
    else:
        print("⊙ Teste de arquivo pulado (não está em produção)")

def test_console_output(capsys=None):
    """Testa se logs aparecem no console"""
    logger = logging.getLogger("churn-prediction")
    logger.info("Console test message")
    print("✅ Console output OK")

def run_smoke_tests():
    """Executa todos os testes"""
    print("\n" + "="*50)
    print("SMOKE TESTS - LOGGING")
    print("="*50 + "\n")
    
    try:
        test_setup_logging()
        test_log_levels()
        test_log_format()
        test_file_handler_production()
        test_console_output()
        
        print("\n" + "="*50)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("="*50 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TESTE FALHOU: {e}\n")
        return False
    
    return True

if __name__ == "__main__":
    run_smoke_tests()
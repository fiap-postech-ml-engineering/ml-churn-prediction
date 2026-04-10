.PHONY: help test test-cov lint lint-fix format format-check check clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make test            - Roda testes com output verboso"
	@echo "  make test-cov        - Roda testes com cobertura (relatório HTML)"
	@echo "  make lint            - Verifica estilo do código com Ruff"
	@echo "  make lint-fix        - Corrige automaticamente issues de linting"
	@echo "  make format          - Formata código com Ruff"
	@echo "  make format-check    - Verifica formatação sem modificar"
	@echo "  make check           - Executa lint, format-check e testes (sequencial)"
	@echo "  make clean           - Remove arquivos temporários"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

format-check:
	ruff format src/ tests/ --check

check: lint format-check test
	@echo "✓ Todos os checks passaram!"

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.DEFAULT_GOAL := help

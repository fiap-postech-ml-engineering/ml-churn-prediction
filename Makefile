.PHONY: help test test-cov lint lint-fix format format-check format-diff format-verbose check clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make test            - Roda testes com output verboso"
	@echo "  make test-cov        - Roda testes com cobertura (relatório HTML)"
	@echo "  make lint            - Verifica estilo do código com Ruff"
	@echo "  make lint-fix        - Corrige automaticamente issues de linting"
	@echo "  make lint-fix-unsafe - Corrige automaticamente issues com unsafe-fixes"
	@echo "  make format          - Verifica formatação sem modificar (Black)"
	@echo "  make format-fix      - Formata código com Black"
	@echo "  make format-diff     - Mostra diferenças de formatação sem modificar"
	@echo "  make format-verbose  - Formata código com output verboso"
	@echo "  make check           - Executa lint, format-check e testes (sequencial)"
	@echo "  make clean           - Remove arquivos temporários"

test:
	python -m pytest tests/ -v --no-cov

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	python -m ruff check src/ tests/

lint-fix:
	python -m ruff check src/ tests/ --fix

lint-fix-unsafe:
	ruff check src/ tests/ --fix --unsafe-fixes

format:
	black --check src/ tests/

format-fix:
	black src/ tests/

format-diff:
	black --diff src/ tests/

format-verbose:
	black -v src/ tests/

check: lint format-check test
	@echo "✓ Todos os checks passaram!"

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.DEFAULT_GOAL := help

init:
	@mkdir -p logs
	@nohup uvicorn main:app --reload --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 & echo $$! > .uvicorn.pid
	@nohup mlflow ui --backend-store-uri ./mlruns --port 8001 > logs/mlflow.log 2>&1 & echo $$! > .mlflow.pid
	@echo "API em background na porta 8000 <localhost:8000> (PID $$(cat .uvicorn.pid))"
	@echo "MLflow em background na porta 8001 <localhost:8001> (PID $$(cat .mlflow.pid))"

stop:
	@kill $$(cat .uvicorn.pid) 2>/dev/null || true
	@kill $$(cat .mlflow.pid) 2>/dev/null || true
	@rm -f .uvicorn.pid .mlflow.pid
	@echo "Servicos finalizados"
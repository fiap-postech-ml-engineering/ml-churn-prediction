import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import register_observability_middleware


def create_test_app() -> FastAPI:
    app = FastAPI()
    register_observability_middleware(app)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def test_middleware_adds_request_id_header() -> None:
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/ping")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


def test_middleware_preserves_incoming_request_id_header() -> None:
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/ping", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"


def test_middleware_logs_latency_fields(caplog) -> None:
    app = create_test_app()
    client = TestClient(app)
    caplog.set_level(logging.INFO)

    client.get("/ping")

    records = [
        record
        for record in caplog.records
        if record.name == "src.api.middleware" and record.getMessage() == "request.completed"
    ]

    assert records
    record = records[-1]
    assert getattr(record, "event", None) == "request.completed"
    assert getattr(record, "method", None) == "GET"
    assert getattr(record, "path", None) == "/ping"
    assert getattr(record, "status_code", None) == 200
    assert isinstance(getattr(record, "latency_ms", None), float)

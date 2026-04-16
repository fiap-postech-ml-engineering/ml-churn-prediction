import pandas as pd
import pytest

from src.data.load_data import load_csv_data


def test_load_csv_data_success_with_semicolon(tmp_path, capsys) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1;col2\n1;2\n3;4\n", encoding="utf-8")

    df = load_csv_data(str(csv_path))

    assert df is not None
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

    captured = capsys.readouterr()
    assert "file loaded" in captured.out
    assert "Dataframe shape" in captured.out


def test_load_csv_data_returns_none_for_missing_file(capsys) -> None:
    df = load_csv_data("arquivo_que_nao_existe.csv")

    assert df is None

    captured = capsys.readouterr()
    assert "File not found" in captured.out


def test_load_csv_data_returns_none_for_parser_error(monkeypatch, capsys) -> None:
    def fake_read_csv(*args, **kwargs):
        raise pd.errors.ParserError("parser error")

    monkeypatch.setattr("src.data.load_data.pd.read_csv", fake_read_csv)

    df = load_csv_data("fake.csv")

    assert df is None

    captured = capsys.readouterr()
    assert "CSV parsing error" in captured.out
    assert 'force a "," separator' in captured.out


def test_load_csv_data_returns_none_for_unexpected_error(monkeypatch, capsys) -> None:
    def fake_read_csv(*args, **kwargs):
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr("src.data.load_data.pd.read_csv", fake_read_csv)

    df = load_csv_data("fake.csv")

    assert df is None

    captured = capsys.readouterr()
    assert "Unexpected error loading" in captured.out
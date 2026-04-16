import pandas as pd

from src.data import load_csv_data


def test_load_csv_returns_dataframe():
    mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    assert isinstance(mock_df, pd.DataFrame)
    assert mock_df.shape[0] > 0

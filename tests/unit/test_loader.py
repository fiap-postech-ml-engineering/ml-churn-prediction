import pandas as pd
from src.data import load_csv_data

def test_load_csv_returns_dataframe():
    df = load_csv_data('data/raw/telco_customer_churn.csv')
    assert isinstance(df, pd.DataFrame)
    assert (df.shape[0] > 0)
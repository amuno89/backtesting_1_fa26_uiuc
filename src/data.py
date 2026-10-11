import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'raw'

def load_stock_data():

    prices = pd.read_csv(DATA_DIR / 'prices.csv', index_col=0, parse_dates=True)
    for col in prices.columns:
        assert prices[col].dtype == np.float64, f"Column {col} is not float64"
    return prices

def load_risk_free_rate():
    prices = load_stock_data()
    fred = pd.read_csv(DATA_DIR / 'DTB3.csv', index_col=0, parse_dates=True, na_values='.')
    rf = fred.iloc[:, 0]
    rf = rf / 100 / 252
    rf = rf.reindex(prices.index).ffill()
    
    assert not rf.isna().any(), "Risk-free rate contains NaN values after reindexing and forward-filling"
    return rf
import pandas as pd
import numpy as np

def ma_crossover(prices, short=50, long=200):
    ma_short = prices.rolling(window=short).mean()
    ma_long = prices.rolling(window=long).mean()
    pos_ma = (ma_short > ma_long).astype(int)
    return pos_ma


def momentum(prices, lookback=252):
    trailing = prices.pct_change(lookback)
    pos_mom = (trailing>0).astype(int)
    return pos_mom

def rsi_mr(prices, period=14, entry=30, exit=50):
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    signal = pd.DataFrame(np.nan, index=rsi.index, columns=rsi.columns)
    signal[rsi < entry] = 1    # entry
    signal[rsi > exit] = 0    # exit
    rsi_pos = signal.ffill().fillna(0).astype(int)
    return rsi_pos
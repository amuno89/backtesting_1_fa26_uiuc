import yfinance as yf
import pandas as pd
import math 
import numpy as np

def buy_hold(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    prices = data['Close'][ticker]
    returns = prices.pct_change()
    equity = (1 + returns).cumprod()
    return equity

def momentumShifter(ticker, start, end):
    data = yf.download(ticker,start=start, end=end)
    df_prices = data[['Close']]
    twelveMonths = df_prices.pct_change(252)
    df_prices['Signal'] = (twelveMonths > 0).astype(int)
    return df_prices['Signal'].to_frame(name= f'{ticker}_Signal')
def GoldenCross(ticker,start,end):
    data = yf.download(ticker, start=start, end=end)
    df_prices = data[['Close']]
 
# Moves the data in a data frame
# along with that it also creates a new colummn so that I'm able to compare the two
    df_prices['MovingAverage50'] = df_prices['Close'].rolling(window = 50).mean()
    df_prices['MovingAverage200'] = df_prices['Close'].rolling(window = 200).mean()
    df_prices['Signal'] = (df_prices["MovingAverage50"] > df_prices['MovingAverage200']).astype(int)
    return df_prices['Signal'].to_frame(name= f'{ticker}_Signal')


def meanRSI(ticker, start, end, window=14, buy_below=30, sell_above=70):
    data = yf.download(ticker, start=start, end=end)
    prices = data['Close'][ticker]

    # 1. RSI: what share of the last 14 days' movement was upward?
    change = prices.diff()                  # today's close minus yesterday's
    gains = change.clip(lower=0)            # keep up moves, down moves become 0
    losses = -change.clip(upper=0)          # keep down moves (as positive numbers), up moves become 0
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rsi = avg_gain / (avg_gain + avg_loss) * 100

    # 2. Signal: buy when oversold, sell when overbought, otherwise keep what you had
    signal = pd.Series(float('nan'), index=prices.index)   # blank = "no new decision today"
    signal.loc[rsi < buy_below] = 1         # RSI under 30 → buy
    signal.loc[rsi > sell_above] = 0        # RSI over 70 → sell
    signal = signal.ffill().fillna(0).astype(int)

    return signal.to_frame(name=f'{ticker}_RSI')
   
def backtest(prices, signal):
    signal = signal.squeeze()
    returns = prices.pct_change()          # daily returns: you know this one[]
    position = signal.shift(1)        # move the signal down one day
    strat_returns = position * returns      # position × return
    equity = (1 + strat_returns).cumprod()   # same as buy_hold
    return equity    


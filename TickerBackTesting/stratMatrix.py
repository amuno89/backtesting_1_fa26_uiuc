from Strat import GoldenCross, meanRSI, momentumShifter, backtest
import yfinance as yf
import pandas as pd
 
tickers = ['SPY', 'QQQ', 'IWM', 'JPM', 'XOM', 'AAPL', 'JNJ', 'NVDA', 'KO']
start = '2016-01-01'
end = '2026-01-01'
all_prices = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
 
# The first 252 days are warmup (momentum needs a full year of history before it can signal).
# I cut the same days from everything so the raw assets and strategies are compared fairly.
warmup = 253
 
# Raw assets, same thing as Phase 2
daily_returns = all_prices.pct_change().iloc[warmup:]
 
# One column per strategy/ticker, holding that strategy's daily return
columns = {}
for ticker in tickers:
    prices = all_prices[ticker].dropna()
    # backtest gives an equity curve, pct_change turns it back into daily returns
    columns[ticker + ' GC'] = backtest(prices, GoldenCross(prices)).pct_change()
    columns[ticker + ' Mom'] = backtest(prices, momentumShifter(prices)).pct_change()
    columns[ticker + ' RSI'] = backtest(prices, meanRSI(prices)).pct_change()
 
strat_returns = pd.DataFrame(columns).iloc[warmup:]
 
# A strategy that never buys has zero variance, which breaks the math later, so drop it
strat_returns = strat_returns.loc[:, strat_returns.std() > 0]
 
if __name__ == '__main__':
    print(strat_returns.shape)
    print(strat_returns.head())
 
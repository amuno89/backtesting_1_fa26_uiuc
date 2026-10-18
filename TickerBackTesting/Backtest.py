from Strat import GoldenCross, meanRSI, momentumShifter, buy_hold, backtest
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from metrics import evalaute
tickers = ['SPY', 'QQQ', 'IWM', 'JPM', 'XOM', 'AAPL', 'JNJ', 'NVDA', 'KO']
start = '2016-01-01'
end = '2026-01-01'
all_prices = yf.download(tickers,start=start,end=end,auto_adjust=True)['Close']
fig, axes = plt.subplots(3,3, figsize=(15,10))
results = []
for i, ticker in enumerate(tickers):
    ax = axes.flat[i]
    prices = all_prices[ticker].dropna()

    # Making all the equity graphs
    gc_sig = GoldenCross(prices)
    ms_sig = momentumShifter(prices)
    rsi_sig = meanRSI(prices)

    gc_eq= backtest(prices,gc_sig)
    ms_eq = backtest(prices,ms_sig)
    rsi_eq = backtest(prices, rsi_sig)
    bh_eq = buy_hold(prices)

    # Plotting 
    ax.plot(gc_eq,label = 'Golden Cross for ' f'{ticker}')
    ax.plot(ms_eq,label = 'MomentumShift for 'f'{ticker}')
    ax.plot(rsi_eq, label = 'MeanRSI for ' f'{ticker}')
    ax.plot(bh_eq, label = "Buy hold for "f'{ticker}')
    ax.legend()
    ax.set_title(f'{ticker}'': Strategies against Buy and Hold')

    results.append(evalaute(gc_eq,ticker,'Golden Cross'))
    results.append(evalaute(ms_eq, ticker, 'Momentum'))
    results.append(evalaute(rsi_eq, ticker, 'Mean Reversion'))
    results.append(evalaute(bh_eq, ticker, 'Buy and Hold'))
plt.tight_layout()
plt.show()
#converts the dictionary to data frame so its easily seen

df_results = pd.DataFrame(results)

#converting to excel sheet to better see
df_results.to_csv('results.csv', index = False)



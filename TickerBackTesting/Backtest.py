from Strat import GoldenCross, meanRSI, momentumShifter, buy_hold, backtest
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from metrics import evalaute
tickers = ['SPY', 'QQQ', 'IWM', 'JPM', 'XOM', 'AAPL', 'JNJ', 'NVDA', 'KO']
start = '2016-01-01'
end = '2026-01-01'
results = []
for ticker in tickers:
    plt.figure()
    prices = yf.download(ticker,start=start,end=end)['Close'][ticker]

    # Making all the equity graphs
    gc_sig = GoldenCross(ticker,start,end)
    ms_sig = momentumShifter(ticker,start,end)
    rsi_sig = meanRSI(ticker,start,end)

    gc_eq= backtest(prices,gc_sig)
    ms_eq = backtest(prices,ms_sig)
    rsi_eq = backtest(prices, rsi_sig)
    bh_eq = buy_hold(ticker,start,end)

    # Plotting 
    plt.plot(gc_eq,label = 'Golden Cross for ' f'{ticker}')
    plt.plot(ms_eq,label = 'MomentumShift for 'f'{ticker}')
    plt.plot(rsi_eq, label = 'MeanRSI for ' f'{ticker}')
    plt.plot(bh_eq, label = "Buy hold for "f'{ticker}')
    plt.legend()
    plt.title(f'{ticker}'': Strategies against Buy and Hold')

    results.append(evalaute(gc_eq,ticker,'Golden Cross'))
    results.append(evalaute(ms_eq, ticker, 'Momentum'))
    results.append(evalaute(rsi_eq, ticker, 'Mean Reversion'))
    results.append(evalaute(bh_eq, ticker, 'Buy and Hold'))

plt.show()
#converts the dictionary to data frame so its easily seen

df_results = pd.DataFrame(results)

#converting to excel sheet to better see
df_results.to_csv('results.csv', index = False)



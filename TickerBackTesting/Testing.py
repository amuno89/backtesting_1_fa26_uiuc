import pandas as pd
import numpy as np
import math


test = pd.Series([1.0, 1.1, 0.99, 1.2, 1.08])
#okay lets start with some shared varibles
def evaulaute(equity):
    final_val = equity.iloc[-1]
    daily_returns = equity.pct_change().dropna()
    # First metric , Total return
    Total_return = final_val - 1
    # Seccond metric, annualized return, still pretty easy
    annual_return = final_val ** (1 / (len(equity) / 252)) - 1
    # Third Metric annualized Volatility
    annual_vol = np.std(daily_returns) * math.sqrt(252)
    # fourth metric, sharpe. We just plug and chug the other two.
    sharpe = (annual_return - 0) / annual_vol
    # Fifth Metric, Max Drawdown need a series then find the min there
    drawdown = (equity / equity.cummax()) -1
    max_drawdown = drawdown.min()
    # Finally the winrate. For this one
    # I got to sort equity into two series, one that is just market days and one where its above 0
    in_market_days = daily_returns[daily_returns != 0]
    winning_days = in_market_days[daily_returns > 0]
    win_rate = len(winning_days)/len(in_market_days)
    # then now we have to print everything at the end
    print(f'Total Return {Total_return}')
    print(f'Annual Return {annual_return}')
    print(f'Annual Volatility {annual_vol}')
    print(f'Sharpe {sharpe}')
    print(f'Max drawdown {max_drawdown}')
    print(f'Win Rate {win_rate}')

evaulaute(test)

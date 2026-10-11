def backtest(prices, positions, rf=0):
    returns = prices.pct_change()
    strat_returns = positions.shift(1)*returns
    strat_returns += (1 - positions.shift(1)).mul(rf, axis=0)
    return strat_returns
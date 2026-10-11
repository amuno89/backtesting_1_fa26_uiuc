import pandas as pd
import numpy as np

def calculate_performance_metrics(returns, rf=0, positions=None):
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    n = len(returns)
    if n == 0:
        return {}
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return)**(252/n) - 1
    ann_vol = returns.std() * np.sqrt(252)
    excess = returns - rf
    ann_sharpe_ratio = (excess.mean() / excess.std()) * np.sqrt(252)
    mdd = (equity / equity.cummax() - 1).min()
    if positions is not None:
        in_market = (positions.shift(1) == 1).reindex(returns.index, fill_value=False)
        active_returns = returns[in_market]
        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else np.nan
    else:
        win_rate = returns[returns > 0].count() / n
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'annualized_sharpe_ratio': ann_sharpe_ratio,
        'max_drawdown': mdd,
        'win_rate': win_rate
    }
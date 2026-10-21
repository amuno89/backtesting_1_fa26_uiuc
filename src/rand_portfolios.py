import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import load_stock_data, load_risk_free_rate
from scipy.optimize import minimize

start, end = '2016-01-01', '2026-01-01'
prices = load_stock_data()
rf = load_risk_free_rate()

returns = prices.pct_change().loc[start:end]

mean_returns = returns.mean() * 252
annual_cov_matrix = returns.cov() * 252
rf_annual = rf.loc[start:end].mean() * 252

mu = mean_returns.values
cov = annual_cov_matrix.values

np.random.seed(42)
num_portfolios = 10000
n_assets = len(mean_returns)

results = []
for _ in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(n_assets))
    weights /= np.sum(weights)
    
    portfolio_return = np.dot(weights, mu)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe_ratio = (portfolio_return - rf_annual) / portfolio_volatility
    
    results.append({
        'Return': portfolio_return,
        'Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratio,
        **{f'Weight_{ticker}': weight for ticker, weight in zip(prices.columns, weights)}
    })
portfolio_df = pd.DataFrame(results)

def portfolio_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))

def portfolio_stats(w):
    port_return = np.dot(w, mu)
    port_volatility = portfolio_vol(w)
    sharpe_ratio = (port_return - rf_annual) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def negative_sharpe_ratio(w):
    return -portfolio_stats(w)[2]

bounds = [(0, 1)] * n_assets
w0 = np.ones(n_assets) / n_assets
sum_to_one_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

min_var = minimize(portfolio_vol, w0, method='SLSQP', bounds=bounds, constraints=sum_to_one_constraint)
min_var_return = np.dot(min_var.x, mu)

max_sharpe = minimize(negative_sharpe_ratio, w0, method='SLSQP', bounds=bounds, constraints=sum_to_one_constraint)

equal_weights = np.ones(n_assets) / n_assets

spy_weight = np.zeros(n_assets)
spy_weight[mean_returns.index.get_loc('SPY')] = 1.0

key = {
    'Min Variance': min_var.x,
    'Max Sharpe Ratio': max_sharpe.x,
    'Equal Weight': equal_weights,
    'SPY Only': spy_weight
}

targets = np.linspace(min_var_return, mu.max(), 100)

fronteir_volatility = []
for target in targets:
    constraints = [
        sum_to_one_constraint,
        {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target}
    ]
    res = minimize(portfolio_vol, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    fronteir_volatility.append(res.fun if res.success else np.nan)


fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')

sc = ax.scatter(portfolio_df['Volatility'], portfolio_df['Return'],
                c=portfolio_df['Sharpe Ratio'], label='Random Portfolios', cmap='viridis', s=8, alpha=0.6)
fig.colorbar(sc, ax=ax, label='Sharpe Ratio')
markers = {'Min Variance': 'D', 'Max Sharpe Ratio': '*', 'Equal Weight': 's', 'SPY Only': 'P'}
for name, w in key.items():
    ret, vol, _ = portfolio_stats(w)
    ax.scatter(vol, ret, marker=markers[name], s=200,
               edgecolor='black', zorder=5, label=name)
ax.set_xlabel('Annualized Volatility')
ax.set_ylabel('Annualized Return')
ax.set_title(f'{num_portfolios:,} Random Portfolios, 2016–2025')
ax.plot(fronteir_volatility, targets, color='blue', linestyle='--', linewidth=2, label='Efficient Frontier')
asset_vol = np.sqrt(np.diag(cov))
ax.scatter(asset_vol, mu, marker='x', color='red')
for t, x, y in zip(mean_returns.index, asset_vol, mu):
    ax.annotate(t, (x, y), textcoords='offset points', xytext=(5, 5))
ax.legend(loc='upper left')
rows = []
for name, w in key.items():
    ret, vol, sharpe = portfolio_stats(w)
    rows.append({'Portfolio': name, 'Return': ret, 'Volatility': vol, 'Sharpe': sharpe,
                 **dict(zip(prices.columns, w))})
key_df = pd.DataFrame(rows).set_index('Portfolio')
print(key_df.round(3))
key_df.to_csv('results/tables/phase2_key_portfolios.csv')

plt.savefig('results/figures/random_portfolios.png', dpi=150)
plt.show()
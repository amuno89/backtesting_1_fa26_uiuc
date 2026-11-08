
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from stratMatrix import daily_returns, strat_returns
from metrics import evalaute
np.random.seed(42)
 
# Same method as portfolio.py, wrapped in a function so I can run it on raw assets AND strategies
def build(returns):
    mu = (returns.mean() * 252).values     # annual return of each column
    cov = (returns.cov() * 252).values     # annual covariance matrix
    n = len(mu)
 
    # Random portfolios. Dirichlet with 0.3 makes some weights big and some tiny.
    # With 27 columns, plain random/sum weights all average out to about equal weight and the cloud is one blob
    w = np.random.dirichlet(np.ones(n) * 0.3, 3000)
    rand_rets = w @ mu
    rand_vols = np.sqrt(((w @ cov) * w).sum(axis=1))    # w @ cov @ w for every row at once
 
    # Optimizer: weights between 0 and 1 and adding to 1
    bounds = [(0, 1)] * n
    guess = np.ones(n) / n
    adds_to_one = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
    vol = lambda x: np.sqrt(x @ cov @ x)
    # Max Sharpe: minimizing the negative Sharpe is the same as maximizing it
    best = minimize(lambda x: -(x @ mu) / vol(x), guess, method='SLSQP', bounds=bounds, constraints=[adds_to_one])
 
    # Frontier: for each target return, find the lowest risk that still hits it
    targets = np.linspace(mu.min(), mu.max(), 50)
    front_vols = []
    for t in targets:
        rules = [adds_to_one, {'type': 'eq', 'fun': lambda x: x @ mu - t}]
        r = minimize(lambda x: x @ cov @ x, guess, method='SLSQP', bounds=bounds, constraints=rules)
        front_vols.append(np.sqrt(r.fun) if r.success else np.nan)
 
    return {'w': best.x, 'rand_vols': rand_vols, 'rand_rets': rand_rets,
            'front_vols': front_vols, 'targets': targets, 'mu': mu, 'vol': vol(best.x)}
 
raw = build(daily_returns)
strat = build(strat_returns)
 
# The max Sharpe NUMBER: the optimizer already found the best weights, this is the Sharpe those weights get.
# It uses the simple average yearly return (same as Phase 2). The table further down compounds the returns
# instead, so its Sharpe comes out a bit lower. Use this number for "max Sharpe", and the table to compare portfolios.
print('Max Sharpe, raw assets (Phase 2):', round(raw['w'] @ raw['mu'] / raw['vol'], 3))
print('Max Sharpe, strategies (Phase 3):', round(strat['w'] @ strat['mu'] / strat['vol'], 3))
 
# Frontier plots, same style as Phase 2
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for a, res, title in [(ax[0], raw, 'Raw assets (Phase 2)'), (ax[1], strat, 'Strategies (Phase 3)')]:
    a.scatter(res['rand_vols'], res['rand_rets'], s=5)
    a.plot(res['front_vols'], res['targets'], color='black', label='Efficient Frontier')
    a.scatter(res['vol'], res['w'] @ res['mu'], marker='*', s=300, color='red', label='Max Sharpe')
    a.set_xlabel('Risk')
    a.set_ylabel('Return')
    a.set_title(title)
    a.legend()
plt.show()
 
# The best Phase 3 portfolio = the red star = max Sharpe. These are its weights, as percents of the money
weights = pd.Series(strat['w'], index=strat_returns.columns)
weights.to_csv('phase3_weights.csv')           # full list saved so you can open it in Excel
shown = weights[weights > 0.005].sort_values(ascending=False)
print((shown * 100).round(1).to_string())
print('Total shown:', round(shown.sum() * 100, 1), '% (the rest is tiny weights under 0.5%)')
 
# Quick check on how diversified it really is: the same weights grouped by strategy type and by ticker
print((weights.groupby(lambda c: c.split()[1]).sum() * 100).round(1).to_string())
print((weights.groupby(lambda c: c.split()[0]).sum() * 100).round(1).sort_values(ascending=False).to_string())
 
# Head to head. Each portfolio becomes one daily return series -> equity curve -> Phase 1 metrics
best_single = strat_returns.columns[np.argmax(strat['mu'] / (strat_returns.std() * np.sqrt(252)))]
portfolios = {
    'Best single strategy (' + best_single + ')': strat_returns[best_single],
    'Phase 2 raw asset portfolio': daily_returns @ raw['w'],
    'Phase 3 strategy portfolio': strat_returns @ strat['w'],
}
results = []
fig, ax = plt.subplots()
for name, r in portfolios.items():
    equity = (1 + r).cumprod()
    ax.plot(equity, label=name)
    results.append(evalaute(equity, 'Portfolio', name))
ax.legend()
ax.set_title('Growth of $1')
table = pd.DataFrame(results).drop(columns='Ticker').round(3)
print(table.to_string())    # to_string so pandas can't hide the Sharpe column behind "..."
print('Highest Sharpe (in-sample):', table.loc[table['Sharpe'].idxmax(), 'Strategy'])
plt.show()
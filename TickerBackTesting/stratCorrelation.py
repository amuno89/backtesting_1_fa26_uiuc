import numpy as np
import matplotlib.pyplot as plt
from stratMatrix import daily_returns, strat_returns
 
raw_corr = daily_returns.corr()
strat_corr = strat_returns.corr()
 
# Average correlation, ignoring the diagonal (every asset is 1.0 with itself, which would inflate it)
def avg_corr(c):
    n = len(c)
    return (c.values.sum() - n) / (n * n - n)
 
print('Average raw asset correlation:', round(avg_corr(raw_corr), 2))
print('Average strategy correlation: ', round(avg_corr(strat_corr), 2))
 
# Most and least correlated strategy pairs.
# triu keeps only the top triangle so each pair shows up once and nothing is paired with itself
pairs = strat_corr.where(np.triu(np.ones(strat_corr.shape), k=1) == 1).stack().dropna().sort_values()
print('Least correlated pair:', pairs.index[0], round(pairs.iloc[0], 2))
print('Most correlated pair: ', pairs.index[-1], round(pairs.iloc[-1], 2))
 
# Two heatmaps side by side. vmin/vmax fix the color scale to -1..1 so both charts use the same colors
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
for a, corr, title in [(ax[0], raw_corr, 'Raw assets (Phase 2)'), (ax[1], strat_corr, 'Strategies (Phase 3)')]:
    im = a.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    a.set_xticks(range(len(corr)), labels=corr.columns, rotation=90, fontsize=7)
    a.set_yticks(range(len(corr)), labels=corr.columns, fontsize=7)
    a.set_title(title)
    fig.colorbar(im, ax=a)
plt.tight_layout()
plt.show()
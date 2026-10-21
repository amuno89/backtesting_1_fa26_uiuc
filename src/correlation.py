import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from data import load_stock_data

start= '2016-01-01'
end= '2026-01-01'

prices = load_stock_data()
buy_and_hold_returns = prices.pct_change().loc[start:end]

corr = buy_and_hold_returns.corr()

fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
             linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of Buy-and-Hold Returns (2016-2025)', fontsize=16)
plt.savefig('results/figures/correlation_matrix.png', dpi=300)
plt.show()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
pairs = corr.where(mask).stack().dropna().sort_values(ascending=False)
print("Top 5 pairs of assets with the highest correlation:")
print(pairs.head(5))
print("\nTop 5 pairs of assets with the lowest correlation:")
print(pairs.tail(5))

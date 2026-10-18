import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
## Sets every run to be the same
np.random.seed(42)

tickers = ['SPY', 'QQQ', 'IWM', 'JPM', 'XOM', 'AAPL', 'JNJ', 'NVDA', 'KO']
start = '2016-01-01'
end = '2026-01-01'
all_prices = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']

daily_returns = all_prices.pct_change()

annual_vol = daily_returns.std()* np.sqrt(252)
annual_return = daily_returns.mean() * 252
sharpe = (annual_return -0)/annual_vol

## Time to make a dictionary holding all 3 values so I can put it in one table. Then I make it into one table with pd.DataFrame
stats = pd.DataFrame({
    'Annual Volatility' : annual_vol,
    'Annual Return' : annual_return,
    'Sharpe' : sharpe
})


#print(stats.round(3))

## Time to do correlation. Pandas already has a correlation function so Ill just call on that. Its .corr
## Let me break this code down peice by piece. Daily returns is my original series with all the tickers and closing prices. .corr calculates the corelation
## numer_only= true only takes the value that have  numbers.

correlation = daily_returns.corr(numeric_only=True)
##  time to make the covariance matrix with .cov. Adds everything in the list so no worries and the * 252 is to just match the annual return
covariance = daily_returns.cov() * 252
# need this function for the markorwitz
def portfolio_vol(w):
    return np.sqrt(w @ covariance @ w)

## time to make graph
#fig just means figure like the window, and ax means chart. plt subplot is just combining the two.
#Make sure subplot has an s at the end. Very imperative. 
fig,ax = plt.subplots()
# Im show makes the image show into colored squares. the bigger the number the brighter the square. Oh thats why the heat map.
im = ax.imshow(correlation)
# Sets the name of the number to each ticker name
ax.set_xticks(range(9),labels=correlation.columns)
ax.set_yticks(range(9),labels=correlation.columns)
#color each of teh tickers to shwo that they mean
fig.colorbar(im)
#plt.show()
# Time to do a for loop to loop this for 3 thousand times
# First I also have to establish the return list so the scatterplot works
port_returns =[]
port_vols = []
port_weights =[]
port_sharpes =[]
# now the loop

for i in range(3000):

    # Weight. The weight is just how much each is invested in each portfolio for each ticker
    # cum sum is the cummalitive while sum is what I need here. They are different because cum sum is the running total. ALso to make sure everything is even by 100, I need to divide by the total number
    # the nine in random craetes the 9 array list of varaibles
    rand_nine = np.random.random(9)
    weight = rand_nine/rand_nine.sum()
    
    #time to assign an weight to each ticker. I would multiply each tickers annual return to the weight.
    port_return = (weight * annual_return).sum()
    #@ is like the multiplication of the cumlative sum. then its multipliying evetrything by the covariance matrix?
    port_vol= np.sqrt(weight @ covariance @ weight)
    # sharpe calculation = return-0/vol
    port_sharpe = (port_return - 0)/port_vol
    port_vols.append(port_vol)
    port_returns.append(port_return)
    port_weights.append(weight)
    port_sharpes.append(port_sharpe)
    
   # print(weight.sum()) This was only a check
    #print(port_return)
   # print(port_val) 
    



## prints the lowest risk weight and the highest sharpe weight
# first save the value
min_riskidx = np.argmin(port_vols)
max_sharpeidx = np.argmax(port_sharpes)

## I legit have no clue what this mean. REVIEW
target_returns = np.linspace(annual_return.min(), annual_return.max(), 50)
frontier_vols = []
bounds = [(0, 1)] * 9                         # each weight between 0% and 100%
start = np.ones(9) / 9                        # first guess: equal weight

for target in target_returns:
    rules = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},                          # weights add to 1
        {'type': 'eq', 'fun': lambda w: (w * annual_return).sum() - target},   # hits the target return
    ]
    result = minimize(portfolio_vol, start, method='SLSQP', bounds=bounds, constraints=rules)
    frontier_vols.append(result.fun)

#the equal weight portfolio
eq_weight = np.ones(9)/9
eq_return = (eq_weight * annual_return).sum()
eq_vol = np.sqrt(eq_weight @ covariance @ eq_weight)
eq_sharpe = (eq_return-0)/eq_vol

port_comp = pd.DataFrame({
    'Return': [port_returns[min_riskidx], port_returns[max_sharpeidx], eq_return, stats.loc['SPY', 'Annual Return']],
    'Risk':   [port_vols[min_riskidx],port_vols[max_sharpeidx], eq_vol, stats.loc['SPY', 'Annual Volatility'] ],
    'Sharpe': [port_sharpes[min_riskidx],port_sharpes[max_sharpeidx], eq_sharpe, stats.loc['SPY','Sharpe']],
}, index=['Min Risk', 'Max Sharpe', 'Equal Weight', 'SPY'])
print(port_comp.round(3))

## now to print the scatter plot. use fig and ax again
fig,ax = plt.subplots()
ax.scatter(port_vols,port_returns)
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
ax.set_title('Risk vs Return')
ax.scatter(port_vols[max_sharpeidx], port_returns[max_sharpeidx],
           marker='*', s=300, color='red', label='Max Sharpe')
ax.scatter(port_vols[min_riskidx], port_returns[min_riskidx],
           marker='*', s=300, color='green', label='Min Variance')
ax.scatter(eq_vol, eq_return,
           marker='*', s=300, color='orange', label='Equal Weight')
ax.scatter(stats.loc['SPY', 'Annual Volatility'], stats.loc['SPY', 'Annual Return'],
           marker='*', s=300, color='black', label='SPY')
ax.plot(frontier_vols, target_returns, color='black', label='Efficient Frontier')

ax.legend()
plt.show()

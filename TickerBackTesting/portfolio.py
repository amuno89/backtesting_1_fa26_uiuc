import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    'Annual Volality' : annual_vol,
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
    port_vols.append(port_vol)
    port_returns.append(port_return)
   
   # print(weight.sum()) This was only a check
    #print(port_return)
   # print(port_val) 
    
## now to print the scatter plot. use fig and ax again
fig,ax = plt.subplots()
ax.scatter(port_vols,port_returns)
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
ax.set_title('Risk vs Return')
plt.show()


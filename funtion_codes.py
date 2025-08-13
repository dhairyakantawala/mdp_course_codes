from pylab import mpl, plt
plt.style.use(plt.style.available[11])
mpl.rcParams['font.family'] = 'serif'
from tvDatafeed import TvDatafeed, Interval
username = ''
password = ''

tvl = TvDatafeed(username, password)
import numpy as np
import pandas as pd



def t_matrix_maker(ticker):
    data=tvl.get_hist(ticker, 'NSE', interval=Interval.in_1_hour, n_bars=300000, fut_contract=None, extended_session=False)
    data['returns'] = (data['close'] - data['close'].shift(1))/data['close'].shift(1)
    returns = data['returns']
    returns = returns.dropna()
    returns = returns.round(2)
    returns = returns.to_list()
    zeros = np.zeros((int(100*(max(returns) - min(returns))) + 1, int(100*(max(returns) - min(returns))) + 1))
    t_matrix = pd.DataFrame(zeros, [i for i in range(int(100*min(returns)), int(100*max(returns)) + 1)], [i for i in range(int(100*min(returns)), int(100*max(returns)) + 1)])
    
    for i in range(len(returns)-1):
        curr = int (100*returns[i])
        next = int (100*returns[i+1])
        t_matrix.loc[curr, next] += 1

    sums = t_matrix.sum()
    t_matrix = t_matrix.T
    to_drop = []
    for i in t_matrix.columns :
        if sums[i] == 0:
            to_drop.append(i)
    t_matrix = t_matrix.drop(columns=to_drop)
    t_matrix = t_matrix.T
    t_matrix = t_matrix.drop(columns=to_drop)

    sums = t_matrix.sum()
    t_matrix = t_matrix.T
    for i in t_matrix.columns:
        t_matrix[i] = t_matrix[i]/sums[i]
    t_matrix = t_matrix.T
    return t_matrix
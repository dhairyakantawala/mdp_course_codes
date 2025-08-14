from pylab import mpl, plt
plt.style.use(plt.style.available[11])
mpl.rcParams['font.family'] = 'serif'
from tvDatafeed import TvDatafeed, Interval
username = ''
password = ''

tvl = TvDatafeed(username, password)
import numpy as np
import pandas as pd



def t_matrix_maker(ticker, round_value = 2):
    num_round = 10**round_value
    data=tvl.get_hist(ticker, 'NSE', interval=Interval.in_1_hour, n_bars=300000, fut_contract=None, extended_session=False)
    data['returns'] = (data['close'] - data['close'].shift(1))/data['close'].shift(1)
    returns = data['returns']
    returns = returns.dropna()
    returns = returns.round(round_value)
    returns = returns.to_list()
    zeros = np.zeros((int(num_round*(max(returns) - min(returns))) + 1, int(num_round*(max(returns) - min(returns))) + 1))
    t_matrix = pd.DataFrame(zeros, [i for i in range(int(num_round*min(returns)), int(num_round*max(returns)) + 1)], [i for i in range(int(num_round*min(returns)), int(num_round*max(returns)) + 1)])
    
    for i in range(len(returns)-1):
        curr = int (num_round*returns[i])
        next = int (num_round*returns[i+1])
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

def expected_returns_maker(t_matrix, round_value = 2):
    exp_returns = {}
    for i in t_matrix.index:
        exp_r = 0
        for col in t_matrix.columns:
            exp_r += int(col)*(t_matrix.loc[i, col])
        exp_returns[i] = float(exp_r)/(10**round_value)
        exp_r = 0
    return exp_returns

def reward(state, action, exp_returns):
    returns, weights = state
    return (weights + action)*exp_returns[returns] - 0.003*abs(action)

def terminal_reward(state, exp_returns):
    _, weights = state
    return reward(state, -weights, exp_returns)

def t_prob(next_state, prev_state, action, t_matrix):
    if action == -2:
        prev_r, _ = prev_state
        next_r, _ = next_state
        return float(t_matrix.loc[prev_r, next_r])
    
    elif action == -1:
        prev_r, prev_w = prev_state
        next_r, next_w = next_state
        if (next_w - prev_w)!=-1:
            return 0.0
        else:
            return float(t_matrix.loc[prev_r, next_r])
        
    elif action == 0:
        prev_r, prev_w = prev_state
        next_r, next_w = next_state
        if (next_w - prev_w)!=0:
            return 0.0
        else:
            return float(t_matrix.loc[prev_r, next_r])

    elif action == 1:
        prev_r, prev_w = prev_state
        next_r, next_w = next_state
        if (next_w - prev_w)!=1:
            return 0.0
        else:
            return float(t_matrix.loc[prev_r, next_r])

    elif action == 2:
        prev_r, _ = prev_state
        next_r, _ = next_state
        return float(t_matrix.loc[prev_r, next_r])
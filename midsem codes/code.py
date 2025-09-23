# Author: Dhairya Kantawala 23b3321
# Description: This code was soley written by me, the only help I have taken is from my own IE 708 class notes which are also availible on my website (dhairyakantawala.github.io/notes) and only for the matplotlib part for my own understanding I have used claude for like 2-3 lines on how to generate graphs, but that too was for my own understanding of how the norm convergence looks
import math
import pandas as pd
import matplotlib.pyplot as plt

class questionMDP:
    def __init__(self, N, lmbd, R):
        self.N = N
        self.lmbd = lmbd
        self.R = R

        self.g = {}
        for i in range(1, N+1):
            self.g[i] = math.log10(i)

        self.C = {}
        for i in range(1, N+1):
            self.C[i] = sum(1 /(1 + abs(i - j)) for j in range(1, N+1))

        self.Q = {}
        for i in range(1, N+1):
            self.Q[i] = {}
            for j in range(1, N+1):
                self.Q[i][j] = 1 / ((1 + abs(i - j)) * (self.C[i]))
    
    def belman_operator(self, v_n):
        v_n_1 = {}
        d = {}
        for i in range(1, self.N + 1):
            action_nothing = -1*(self.g[i]) + self.lmbd * (sum([self.Q[i][j] * v_n[j] for j in range(1, self.N +1)]))
            action_reset = -1*(self.g[1] + self.R) + self.lmbd * v_n[1]
            if action_nothing > action_reset:
                d[i] = 'nothing'
                v_n_1[i] = action_nothing
            else :
                d[i] = 'reset'
                v_n_1[i] = action_reset

        return v_n_1, d

    def norm(self, v, u):
        return max([abs(v[i] - u[i]) for i in range(1, self.N + 1)])
        
    def value_iteration(self, epsilon, save_csv=False, plot_norm=False, verbos=False):
        norm_values = []
        n = 0
        v_n = {i: 0 for i in range(1, self.N + 1)}
        v_n_1, d_n_1 = self.belman_operator(v_n)
        df = pd.DataFrame(columns=[f'v^{n}'])
        df['v^0'] = v_n
        while self.norm(v_n, v_n_1) >= (epsilon*(1 - self.lmbd))/(2*self.lmbd):
            norm_values.append(self.norm(v_n, v_n_1))
            n+=1
            v_n = v_n_1
            v_n_1, d_n_1 = self.belman_operator(v_n)
            df = pd.concat([df, pd.Series(v_n, name=f'v^{n}')], axis=1)
        df[f'v^{n+1}'] = v_n_1
        if save_csv:
            df.to_csv('v_i.csv')
        if plot_norm:
            plt.figure(figsize=(16, 6))
            plt.plot([i+1 for i in range(len(norm_values))], norm_values)
            plt.xlabel('iteration')
            plt.ylabel('norm')
            plt.title('norm vs iteration')
            plt.show()
        if verbos:
            print(f"stopped value iteration with total iterations: {n}")
        return v_n_1, d_n_1
    
    def gs_operator(self, v_n):
        v_n_1 = {}
        d = {}
        for i in range(1, self.N + 1):
            action_nothing = -1*(self.g[i])
            action_reset = -1*(self.g[1] + self.R) 
            for j in range(1, i):
                action_nothing += self.lmbd * self.Q[i][j] * v_n_1[j]
            for j in range(i, self.N + 1):
                action_nothing += self.lmbd * self.Q[i][j] * v_n[j]
        
            if(i == 1):
                action_reset += self.lmbd * v_n[1]
            else:
                action_reset += self.lmbd * v_n_1[1]
            
            if action_nothing > action_reset:
                d[i] = 'nothing'
                v_n_1[i] = action_nothing
            else :
                d[i] = 'reset'
                v_n_1[i] = action_reset
        return v_n_1, d

    def gauss_seidel_iteration(self, epsilon, save_csv=False, plot_norm=False, verbos=False):
        norm_values = []
        n = 0
        v_n = {i: 0 for i in range(1, self.N + 1)}
        v_n_1, d_n_1 = self.gs_operator(v_n)
        df = pd.DataFrame(columns=[f'v^{n}'])
        df['v^0'] = v_n
        while self.norm(v_n, v_n_1) >= (epsilon*(1 - self.lmbd))/(2*self.lmbd):
            norm_values.append(self.norm(v_n, v_n_1))
            n+=1
            v_n = v_n_1
            v_n_1, d_n_1 = self.gs_operator(v_n)
            df = pd.concat([df, pd.Series(v_n, name=f'v^{n}')], axis=1)
        df[f'v^{n+1}'] = v_n_1
        if save_csv:
            df.to_csv('gsv_i.csv')
        if plot_norm:
            plt.figure(figsize=(16, 6))
            plt.plot([i+1 for i in range(len(norm_values))], norm_values)
            plt.xlabel('iteration')
            plt.ylabel('norm')
            plt.title('norm vs iteration')
            plt.show()
        if verbos:
            print(f"stopped gs value iteration with total iterations: {n}")
        return v_n_1, d_n_1

    def calculate_aarc(self):
        v_star, _ = self.gauss_seidel_iteration(epsilon=1e-300)
        n = 0
        v_0 = {i: 0 for i in range(1, self.N + 1)} 
        v_n = {i: 0 for i in range(1, self.N + 1)}
        v_n_1, d_n_1 = self.gs_operator(v_n)
        deno = self.norm(v_star, v_0)
        epsilon = 1e-12
        while True:
            n += 1
            v_n = v_n_1
            v_n_1, d_n_1 = self.gs_operator(v_n)
            curr_norm = self.norm(v_n, v_star)
            next_norm = self.norm(v_n_1, v_star)
            curr_value = (curr_norm / deno) ** (1 / n)
            next_value = (next_norm / deno) ** (1 / (n + 1))
            margin = 0.001
            if abs(curr_value - next_value) < margin:
                return f'{curr_value} ± {margin}'


mdp = questionMDP(N=100, lmbd=0.9, R=10)

print("value iteration:")
v_final, d_final = mdp.value_iteration(epsilon=0.001, save_csv=True, plot_norm=False, verbos=True)
print("saved v_i.csv")

print("\n\nGauss-Seidel value iteration:")
v_final, d_final = mdp.gauss_seidel_iteration(epsilon=0.001, save_csv=True, plot_norm=False, verbos=True)
print("saved gsv_i.csv")
print(mdp.calculate_aarc(), end=" is the AARC value")
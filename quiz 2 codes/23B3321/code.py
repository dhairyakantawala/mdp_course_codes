import numpy as np
import math

from tabulate import tabulate


class AgoOfInformation:
    def __init__(self, L, eta, sigma, q):
        self.L = L
        self.eta = eta
        self.sigma = sigma
        self.q = q

        def f(i, j):
            if i < self.L and j < self.L:
                return math.exp(-self.eta * i - j)
            else:
                return 0.0

        self.rewards = np.zeros((self.L+1, self.L+1, 3))
        for i in range(self.L+1):
            for j in range(self.L+1):
                self.rewards[i, j, 0] = f(i, j)
                self.rewards[i, j, 1] = f(i, j) - self.sigma
                self.rewards[i, j, 2] = f(i, j) - self.sigma

        self.transitions = np.zeros((self.L+1, self.L+1, self.L+1, self.L+1, 3))
        for i in range(self.L+1):
            for j in range(self.L+1):
                if i < self.L and j < self.L:
                    self.transitions[i+1, j+1, i, j, 0] = 1.0
                    self.transitions[i+1, j+1, i, j, 1] = 1.0 - self.q
                    self.transitions[i+1, j+1, i, j, 2] = 1.0 - self.q
                    self.transitions[i+1, 0, i, j, 2] = self.q
                    self.transitions[0, j+1, i, j, 1] = self.q

                if i == self.L and j < self.L:
                    self.transitions[i, j+1, i, j, 0] = 1.0
                    self.transitions[i, j+1, i, j, 1] = 1.0 - self.q
                    self.transitions[i, j+1, i, j, 2] = 1.0 - self.q
                    self.transitions[i, 0, i, j, 2] = self.q
                    self.transitions[0, j+1, i, j, 1] = self.q

                if j == self.L and i < self.L:
                    self.transitions[i+1, j, i, j, 0] = 1.0
                    self.transitions[i+1, j, i, j, 1] = 1.0 - self.q
                    self.transitions[i+1, j, i, j, 2] = 1.0 - self.q
                    self.transitions[i+1, 0, i, j, 2] = self.q
                    self.transitions[0, j, i, j, 1] = self.q

                if i == self.L and j == self.L:
                    self.transitions[i, j, i, j, 0] = 1.0
                    self.transitions[i, j, i, j, 1] = 1.0 - self.q
                    self.transitions[i, j, i, j, 2] = 1.0 - self.q
                    self.transitions[i, 0, i, j, 2] = self.q
                    self.transitions[0, j, i, j, 1] = self.q

    def optimal_policy_iteration(self, lam):
        S = (self.L+1)*(self.L+1)
        idx = lambda i,j: i*(self.L+1)+j

        d = np.zeros((self.L+1, self.L+1), dtype=int)

        while True:
            P = np.zeros((S, S))
            R = np.zeros(S)
            for i in range(self.L+1):
                for j in range(self.L+1):
                    s = idx(i,j)
                    a = d[i,j]
                    R[s] = self.rewards[i,j,a]
                    for ni in range(self.L+1):
                        for nj in range(self.L+1):
                            ns = idx(ni,nj)
                            P[s, ns] = self.transitions[ni,nj,i,j,a]

            V_flat = np.linalg.solve(np.eye(S) - lam*P, R)
            v = V_flat.reshape((self.L+1, self.L+1))

            d_new = np.zeros_like(d)
            for i in range(self.L+1):
                for j in range(self.L+1):
                    Q = np.zeros(3)
                    for a in range(3):
                        q_val = self.rewards[i,j,a]
                        for ni in range(self.L+1):
                            for nj in range(self.L+1):
                                q_val += lam*self.transitions[ni,nj,i,j,a]*v[ni,nj]
                        Q[a] = q_val
                    d_new[i,j] = np.argmax(Q)

            if np.array_equal(d, d_new):
                break
            d = d_new
        return d, v

    def policy_metrics(self, d):
        S = (self.L + 1) * (self.L + 1)
        idx = lambda i, j: i * (self.L + 1) + j

        P = np.zeros((S, S))
        R_vec = np.zeros(S)
        for i in range(self.L + 1):
            for j in range(self.L + 1):
                s = idx(i, j)
                a = int(d[i, j])
                R_vec[s] = self.rewards[i, j, a]
                for ni in range(self.L + 1):
                    for nj in range(self.L + 1):
                        ns = idx(ni, nj)
                        P[s, ns] = self.transitions[ni, nj, i, j, a]

        A = P.T - np.eye(S)
        A[-1, :] = 1.0
        b = np.zeros(S)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            pi = np.ones(S) / S
            for _ in range(10000):
                pi_new = pi @ P
                if np.linalg.norm(pi_new - pi, 1) < 1e-12:
                    break
                pi = pi_new
            pi = pi / pi.sum()

        pi = pi.astype(float)
        pi /= pi.sum()

        Pstar = np.ones((S, 1)) @ pi.reshape(1, S)

        g = pi.dot(R_vec)

        I = np.eye(S)
        M = I - P + Pstar
        try:
            Z = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Z = np.linalg.solve(M, I)

        D = Z - Pstar
        ones = np.ones(S)
        h = D.dot(R_vec - g * ones)

        h -= ones * (pi.dot(h))
        h = h.reshape((S,))
        S = (self.L + 1)
        h_grid = h.reshape((S, S))
        return g, h_grid

    def B(self, g, h_grid):
        S = self.L + 1
        idx = lambda i, j: i * S + j
        h_flat = h_grid.reshape(S * S)

        B_grid = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                Q = np.zeros(3)
                for a in range(3):
                    val = 0.0
                    for ni in range(S):
                        for nj in range(S):
                            ns = idx(ni, nj)
                            val += self.transitions[ni, nj, i, j, a] * h_flat[ns]
                    Q[a] = self.rewards[i, j, a] - g + val - h_flat[idx(i, j)]
                B_grid[i, j] = np.max(Q)

        return B_grid

def pretty_matrix(matrix, fmt="{:.4f}"):
    rows = [[fmt.format(x) for x in row] for row in matrix]
    return tabulate(rows, tablefmt="fancy_grid")

def pretty_policy(d):
    symbols = {0: "A0", 1: "A1", 2: "A2"}
    rows = [[symbols[x] for x in row] for row in d]
    return tabulate(rows, tablefmt="fancy_grid")

def pretty_matrix(matrix, title, row_labels, col_labels, fmt="{:.4f}"):
    table = []
    for i, row in enumerate(matrix):
        table.append([row_labels[i]] + [fmt.format(x) for x in row])
    headers = ["x1 \\ x2"] + col_labels
    print(f"\n{title}")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

def pretty_policy(d, title, row_labels, col_labels):
    table = []
    for i, row in enumerate(d):
        table.append([row_labels[i]] + [str(x) for x in row])
    headers = ["x1 \\ x2"] + col_labels
    print(f"\n{title}")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    try:
        L = 3
        eta = float(input("η (eta): "))
        sigma = float(input("σ (sigma): "))
        q = float(input("q: "))
    except Exception as e:
        print("invalid input")
        exit(1)

    aoi = AgoOfInformation(L, eta, sigma, q)

    lambdas = [0.9, 0.95, 0.99, 0.999]
    row_labels = [f"{i}" for i in range(L+1)]
    col_labels = [f"{j}" for j in range(L+1)]

    print("\n" + "="*70)
    print("              OPTIMAL POLICIES AND VALUE FUNCTIONS")
    print("="*70 + "\n")

    results = {}
    for lam in lambdas:
        d, v = aoi.optimal_policy_iteration(lam)
        results[lam] = (d, v)

        print(f"\nλ = {lam}")
        print("-"*70)
        pretty_policy(d, "optimal policy (actions 0, 1, 2):", row_labels, col_labels)
        pretty_matrix(v, "value function (v):", row_labels, col_labels)
        print("-"*70)

    d_star, _ = results[0.999]
    g, h = aoi.policy_metrics(d_star)
    B = aoi.B(g, h)

    print("\n" + "="*70)
    print("          AVERAGE-REWARD METRICS FOR λ = 0.999 POLICY")
    print("="*70)
    print(f"Gain (g): {g:.6f}\n")

    pretty_matrix(h, "Bias Function (h):", row_labels, col_labels)
    pretty_matrix(B, "B(g,h):", row_labels, col_labels)

    max_B = np.max(np.abs(B))
    print(f"\nMax |B(g,h)| = {max_B:.6e}")
    if max_B < 1e-6:
        print("Optimality condition satisfied: B(g, h) ≈ 0\n")
    else:
        print("B(g, h)!= 0, optimality conditon not satisfied\n")
import numpy as np
import time

def make_asset_graph(n, tot_assets, integ):
    A_ib = np.random.choice([0, 1], size=(n,n), p=[1 - integ, integ])

    for i in range(n):
        A_ib[i,i] = 0

    for bank in np.where([sum(A_ib[i, 0:n]) == 0 for i in range(n)])[0]:
        if np.size(bank) == 0:
            break
        possible_links = [x for x in range(n) if x != bank]
        A_ib[bank, np.random.choice(possible_links, size = 1)] = 1

    A_ib = tot_assets / np.transpose([sum(A_ib[i, 0:n]) for i in range(n)] * np.transpose(A_ib))
    A_ib = np.where(A_ib == np.inf, 0, A_ib)
    return A_ib

np.sum(A_ib, 1) # sum assets

def is_solvent(bank, A_ib, D, A_M, phi, q):
    (1-phi) * np.sum(A_ib, 1)[bank] + q * np.array(A_M)[bank] - np.sum(A_ib, 0)[bank] - np.array(D)[bank] > 0

def degree(bank, A_ib, direction):
    n = np.size(A_ib, 1)
    if direction == "in":
        return np.sum(A_ib[bank, 0:n] != 0, axis=1)
    else:
        return np.sum(A_ib[0:n, bank] != 0, axis = 0)

def capital_buffer(bank, A_ib, D, A_M):
    return np.sum(A_ib, 1)[bank] + np.array(A_M)[bank] - np.sum(A_ib,0)[bank] - np.array(D)[bank]

# np.sum(A_ib, 0) to liab

def contagion(n, init_b, exp_A_M, exp_A_ib, buffer_rate, link_p):
    A_M = np.random.normal(exp_A,1, n)
    A_ib = make_asset_graph(n, exp_A_ib, link_p)
    D = [exp_A_ib] * n + A_M[0:n] * (1 - buffer_rate) - np.sum(A_ib, 0)[0:n]

    A_ib[0:n, np.random.choice(range(n), 1)] = 0
    n_default = [0]
    n_default = np.append(n_default, sum(capital_buffer(range(n), A_ib, D, A_M) < 0))

    while n_default[-1] != n_default[-2]:
        A_ib[0:n, capital_buffer(range(n), A_ib, D, A_M) < 0] = 0
        n_default = np.append(n_default, sum(capital_buffer(range(n), A_ib, D, A_M) < 0))
    
    return n_default

contagion(1000, 1,80,20,0.04,0.01)

start = time.time()
np.mean([contagion(1000, 1, 80, 20, 0.05, 0.01)[-1] for _ in range(100)])
end = time.time()
print(end - start)

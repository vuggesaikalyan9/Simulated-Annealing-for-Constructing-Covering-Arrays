import numpy as np
import random
import math

t = 2   
v = 2   
K = 5 

rows_max = 7

V = [0, 1]  
rows_min = math.ceil(v**t)   
alpha = 0.99
Ti = K
Tf = 0.001*Ti


def sol_generate(rows):
    return np.random.choice(V, (rows, K))

def cost_cal(s):
    pairs_uncvred = set()
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            pairs_uncvred.add((i, j))
    for k in range(K):
        c = s[:, k]
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                if c[i] == c[j] and (i, j) in pairs_uncvred:
                    pairs_uncvred.remove((i, j))
    return len(pairs_uncvred)

def move_genraated(s):
    i = random.randint(0, len(s)-1)
    j = random.randint(0, K-1)
    n_val = random.choice(V)
    n_s = np.copy(s)
    n_s[i, j] = n_val
    return n_s

def prob_acceptance(c_old, new_cost, t):
    if new_cost < c_old:
        return 1
    else:
        d = new_cost - c_old
        return math.exp(-d / t)

def sim_annealing(k, iterations):
    s = sol_generate(rows_min)     
    cost = cost_cal(s)            
    temperature = Ti                           
    phi = (v**t) * math.factorial(k)/(math.factorial(t)*math.factorial(k-t))  
    for i in range(1,iterations+1):             
        n_s = move_genraated(s)  
        new_cost = cost_cal(n_s)
        acceptance_prob = prob_acceptance(cost, new_cost, temperature)
        if random.random() < acceptance_prob:   
            s = n_s
            cost = new_cost
        temperature *= alpha                   
        if i > 0 and i % phi == 0:              
            temperature = 0.0
        if cost == 0 and len(s) <= rows_max:        
            print("number of iterations: ", i+29)
            return s
        if  temperature == Tf:
            print("Final temperature achieved")
            
    return s

s= sim_annealing(5, 30)

print("Result:\n", s)
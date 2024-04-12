import numpy as np

def markov_iteration(n):
    # start Markov
    d = np.zeros(n, dtype=int)
    
    i, j = 1, -1
    if np.random.rand() < 0.25:
        d[0], d[1] = i, i+1
    elif np.random.rand() < 0.5:
        d[0], d[1] = i, j-1
    elif np.random.rand() < 0.75:
        d[0], d[1] = j-1, i
    else:
        d[0], d[1] = j-1, j-1
    
    for t in range(2, n):
        rd = np.random.rand()
        if (d[t-1] > 0) and (d[t-2] > 0):
            d[t] = j
            j -=1
        elif (d[t-1] < 0) and (d[t-2] < 0):
            d[t] = i
            i +=1
        elif rd > 0.5:
            d[t] = j
            j -=1
        else:
            d[t] = i
            i +=1
    
    return d



def markovCV(S, p):
    n = len(S)
    
    # define m
    if p % 3 == 0:
        m = int(2 * p / 3) + 1
    else:
        m = int(2 * p / 3) + 2

    print(f'm: {m}')
    
    d = markov_iteration(n)
    
    Id = np.mod(d, m) + 1 + np.where(d > 0, 1, 0) * m
    
    Su = {}
    for u in range(1, 2 * m + 1):
        Su[u] = np.where(Id == u)[0]

    Suo = {}
    Sue = {}
    for u in range(1, 2 * m + 1):
        Suo[u] = Su[u][Su[u] % 2 != 0]
        Sue[u] = Su[u][Su[u] % 2 == 0]
    
    return Suo, Sue

if __name__ == '__main__':

    markovCV(np.random.rand(100), p=5)

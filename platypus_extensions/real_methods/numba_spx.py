import numpy as np
from numba import njit

"""(In development)"""

@njit
def _single_spx(parent_floats, min_value, max_value):
    k = len(parent_floats)
    if k <= 1:
        return parent_floats[0]
    
    expansion = (k + 1)**(0.5)
    g = sum(parent_floats) / k
    x = np.zeros(k, np.float32)
    for i, p in enumerate(parent_floats):
        x[i] = g + (expansion * (p - g))
        
    c = 0.0
    for i in range(1,k):
        r = np.random.uniform(0.0,1.0) ** (1.0 / i)
        c = r * (x[i-1] - x[i] + c)
        
    return np.float32(max(min_value, min(max_value, x[-1] + c)))

@njit
def _multi_spx(parent_floats, min_value, max_value, noffspring):
    output = np.zeros(noffspring, np.float32)
    k = len(parent_floats)
    if k <= 1:
        for i in range(noffspring):
            output[i] = parent_floats[0]
        return output
    
    if noffspring == 1:
        output[0] = _single_spx(parent_floats, min_value, max_value)
        return output
    
    expansion = (k + 1)**(0.5)
    g = sum(parent_floats) / k
    x = np.zeros(k, np.float32)
    for i, p in enumerate(parent_floats):
        x[i] = g + (expansion * (p - g))
        
    output = np.zeros(noffspring, np.float32)
    for i in range(noffspring):
        c = 0
        for j in range(1, k):
            r = np.random.uniform(0.0, 1.0) ** (1.0 / i)
            c = r * (x[i-1] - x[i] + c)
                
        offspring_float = max(min_value, min(x[-1] + c, max_value))
        output[i] = offspring_float
            
    return output
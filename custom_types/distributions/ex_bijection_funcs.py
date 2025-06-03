import numpy as np
from numba import njit
from functools import partial

# @njit
def _half_life_return(x, ub, A, T): # try guvectorize
    return ub + A*(2.0**(x/T))

# @njit
def _half_life_inverse(y, ub, B, m): # try guvectorize
    if y >= ub:
        return ub
    return B * np.log(m*(ub - y)) #store B*log(m)

def half_life_bijection(y_min, y_max, T, m, max_y_separation):
    """
    (Note: The smallest distance between x values to produce unique y values is ~ `1e-8`)

    Args:
        y_min (_type_): _description_
        y_max (_type_): _description_
        T (_type_): _description_
        m (_type_): _description_
        last_y_separation (_type_): _description_

    Raises:
        ValueError: If `m` <= 0 
        ValueError: If `T` == 0
        ValueError: If `y_min >= y_max` or `y_min + last_y_separation >= y_max`

    Returns:
        tuple[functools.partial, functools.partial, float, int]
        - 1. forward function
        - 2. inverse function
        - 3. maximum `x` value
        - 4. minimum 'x' separation
        - 5. maximum number of points
    """    
    if not m > 0:
        raise ValueError("'m' must be > 0")
    if T == 0:
        raise ValueError("'T' cannot be 0'")
    if y_min >= y_max or 0 < max_y_separation + y_min >= y_max:
        raise ValueError(f"y_min must be greater than than y max, and max_y_separationn > 0 and < y_max - y_min. \n Got y_min {y_min}, y_max {y_max}, {max_y_separation}")
    
    m = np.float64(m)
    T = np.float64(T)
    ub = np.float64(1.0 / m) + np.float64(y_max)
    B = T / np.float64(np.log(2))
    
    penultimate_x = B * (np.log(m*(ub - (y_min + max_y_separation))))
    max_x = B * (np.log(m * (ub - y_min)))
    max_x_separation = max_x - penultimate_x
    max_points = int(max_x / max_x_separation) + 1
    A = np.float64(-1.0) / m
    
    return (partial(_half_life_return, ub = ub, A = A, T = T), 
            partial(_half_life_inverse, ub = ub, B = B, m = m), 
            max_x, 
            max_x_separation,
            max_points)
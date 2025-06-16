import numpy as np
from numba.np.random.random_methods import buffered_bounded_lemire_uint32
from numba import (
    njit, guvectorize, 
    float32, float64, boolean,
    prange, typeof)

@njit
def real_PM(x, lb, ub, distribution_index):
    """Polynomial mutation of a single float

    Args:
        x (_type_): A value to mutate
        lb (_type_): lower_bound
        ub (_type_): lower_bound
        distribution_index (_type_): Controls spread of 

    Returns:
        float: The mutated float
    """
    u = np.random.uniform()
    dx = ub - lb
    if u < 0.5:
        bl = (x - lb) / dx
        b = 2.0*u + (1.0 - 2.0*u)*pow(1.0 - bl, distribution_index + 1.0)
        delta = pow(b, 1.0 / (distribution_index+ 1.0)) - 1.0
    else:
        bu = (ub - x) / dx
        b = 2.0*(1.0 - u) + 2.0*(u - 0.5)*pow(1.0 - bu, distribution_index+ 1.0)
        delta = 1.0 - pow(b, 1.0 / (distribution_index+ 1.0))
        
    x = x + delta*dx
    return max(lb, min(x, ub))



import numpy as np
from numba import njit
from ..utils import _float32_min_max_norm

@njit
def real_mutation(x, lb, ub, distrib_idx):
    u = np.random.uniform()
    dx = ub - lb
    if u < 0.5:
        bl = (x - lb) / dx
        b = 2.0*u + (1.0 - 2.0*u)*pow(1.0 - bl, distrib_idx+ 1.0)
        delta = pow(b, 1.0 / (distrib_idx+ 1.0)) - 1.0
    else:
        bu = (ub - x) / dx
        b = 2.0*(1.0 - u) + 2.0*(u - 0.5)*pow(1.0 - bu, distrib_idx + 1.0)
        delta = 1.0 - pow(b, 1.0 / (distrib_idx + 1.0))
        
    x = x + delta*dx
    return max(lb, min(x, ub))

@njit("float32(float32, float32, float32, float32, float32, float32, boolean)")
def _real_evolve(min_val, max_val, p1, p2, p3, step_size, normalize_initial = True):
    y = 0
    clipped_y = 0.0
    if normalize_initial:
        v1 = _float32_min_max_norm(min_val, max_val, p1, True)
        v2 = _float32_min_max_norm(min_val, max_val, p2, True)
        v3 = _float32_min_max_norm(min_val, max_val, p3, True)  
        y = v3 + step_size*(v1 - v2)
        clipped_y = max(0.0, min(y, 1.0))
    else:
        y = p3 + step_size*(p1 - p2)
        clipped_y = max(min_val, min(y, max_val))
    return clipped_y if not normalize_initial else _float32_min_max_norm(min_val, max_val, clipped_y, False)

def differential_evolve(min_value, max_value, p1, p2, p3, step_size, normalize_initial = True):
    original_type = type(p1)
    return original_type(_real_evolve(
        np.float32(min_value), np.float32(max_value), 
        np.float32(p1), np.float32(p2), np.float32(p3), 
        np.float32(step_size), normalize_initial))
    
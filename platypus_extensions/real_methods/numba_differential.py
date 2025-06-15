import numpy as np
from numba.np.random.new_random_methods import buffered_bounded_lemire_uint32
from numba import (
    njit, guvectorize, 
    float32, float64, boolean,
    prange, typeof, optional)
from ..utils import _min_max_norm_convert

bitgen = np.random.SFC64(7)
gen    = np.random.Generator(bitgen)
BIT_GEN_TYPE = typeof(gen.bit_generator)

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

real_evolve_sig = [
    float32(float32, float32, float32, float32, float32, float32, boolean),
    float64(float64, float64, float64, float64, float64, float32, boolean),
    float64(float64, float64, float64, float64, float64, float64, boolean),
]
@njit(real_evolve_sig)
def differential_evolve(min_val, max_val, p1, p2, p3, step_size, normalize_initial):
    y = 0
    clipped_y = 0.0
    if normalize_initial:
        v1 = _min_max_norm_convert(min_val, max_val, p1, True)
        v2 = _min_max_norm_convert(min_val, max_val, p2, True)
        v3 = _min_max_norm_convert(min_val, max_val, p3, True)  
        y = v3 + step_size*(v1 - v2)
        clipped_y = max(0.0, min(y, 1.0))
    else:
        y = p3 + step_size*(p1 - p2)
        clipped_y = max(min_val, min(y, max_val))
    return clipped_y if not normalize_initial else _min_max_norm_convert(min_val, max_val, clipped_y, False)
 
_vector_evolve_sig = [
    (float32[:], float32[:], float32[:], float32[:,:], float32,float32[:]),
    (float32[:], float32[:], float32[:], float32[:,:], float32, float32[:]),
    (float64[:], float64[:], float64[:], float32[:,:], float32, float64[:]),
    (float64[:], float64[:], float64[:], float64[:,:], float64, float64[:]),
]
@guvectorize(_vector_evolve_sig, '(x),(x),(x),(x,y),()->(x)', nopython = True)
def gu_differential_evolve(
    p1, p2, p3, 
    bounds, 
    step_size, 
    out):
    """ 
    Creates a evolved array of offspring floats from 3 (same-length) arrays of parent floats
    
    Args:
        p1 (np.ndarray): An array of parent floats
        p2 (np.ndarray): An array of parent floats
        p3 (np.ndarray): An array of parent floats
        bounds (np.ndarray): The min and max value of each float. Must be of shape `(len(offspring), 2)`.
            `bounds[i, 0]` is lower bound of float `i`, `bounds[i, 1]` is upper bound of float `i`,
        step_size (float):
        out (optional, np.ndarray): An array to write the results to. Must be the same length as the parent arrays.
    """    
    
    nvars = len(p1)
    for i in prange(nvars):
        e = p3[i] + step_size*(p1[i] - p2[i])
        out[i] = max(bounds[i,0], min(e, bounds[i,1]))

vector_evolve_sig_prob = [
    (float32[:], float32[:], float32[:], float32[:], float32[:,:], float32, float32),
    (float32[:], float32[:], float32[:], float32[:], float32[:,:], float32, float32),
    (float64[:], float64[:], float64[:], float64[:], float32[:,:], float32, float32),
    (float64[:], float64[:], float64[:], float64[:], float64[:,:], float64, float64),
]
@guvectorize(vector_evolve_sig_prob, '(x),(x),(x),(x),(x,y),(),()',
             nopython = True, cache = True, writable_args = (0,))
def differential_evolve_with_probability(
    offspring,
    p1, p2, p3,
    bounds,
    step_size, crossover_rate):
    """
    Differential evolution of a range of offspring floats. Alters offspring array in-place.
    
    Applies probability gate to each float. At least one variable is guaranteed to evolve
    
    Fastest when there is < ~1000 floats

    Args:
        offspring (np.ndarray): An array of floats. Floats that pass probability gate will be evolved
        p1 (np.ndarray): An array of parent floats. Must be same length as `offspring`
        p2 (np.ndarray): An array of parent floats. Must be same length as `offspring`
        p3 (np.ndarray): An array of parent floats. Must be same length as `offspring`
        bounds (np.ndarray): The min and max value of each float. Must be of shape `(len(offspring), 2)`.
            `bounds[i, 0]` is lower bound of float `i`, `bounds[i, 1]` is upper bound of float `i`,
        step_size (float):
        crossover_rate (float):
    """    

    nvars = bounds.shape[0]
    for i in range(nvars):
        if np.random.rand() < crossover_rate:
            d = p3[i] + (p1[i] - p2[i])*step_size
            offspring[i] = max(bounds[i, 0], min(d, bounds[i, 1]))

    irand = np.random.randint(nvars)
    d = p3[irand] + (p1[irand] - p2[irand])*step_size
    offspring[irand] = max(bounds[i, 0], min(d, bounds[i, 1]))

@njit(cache = True)
def DE_with_probability(
    bit_gen: np.random.BitGenerator, 
    offspring,
    p1, p2, p3,
    bounds,
    step_size, crossover_rate):
    """Similar to `differential_evolve_with_probability()`.
    
    Instead of using 'np.random.rand()` for the probability gate, directly uses Numba's random integer methods
    with an input `np.random.BitGenerator'
    
    When the number of variables > ~1000, faster than `differential_evolve_with_probability()`.
    """
    
    if crossover_rate <= 0:
        return
    if crossover_rate >= 1:
        gu_differential_evolve(p1, p2, p3, bounds, np.float32(step_size), offspring)
        return
    
    UB = 1048576
    new_cxr = np.uint32(max(1, min(UB - 1, np.uint32(crossover_rate * UB))))
    nvars = len(offspring)
    for i in range(nvars):
        if buffered_bounded_lemire_uint32(bit_gen, UB) < new_cxr:
            d = p3[i] + (p1[i] - p2[i])*step_size
            offspring[i] = max(
                bounds[i, 0], 
                min(d, bounds[i, 1])
            )
    
    irand = np.random.randint(nvars)
    d = p3[irand] + (p1[irand] - p2[irand])*step_size
    offspring[irand] = max(bounds[irand, 0], min(d, bounds[irand, 1]))

@njit(cache = True)
def _inner_DE(
    bit_gen: np.random.BitGenerator,
    offspring,
    p1, p2, p3,
    bounds,
    step_size, 
    UB, new_crx,
    start, end
):
    for i in range(start, end):
        if buffered_bounded_lemire_uint32(bit_gen, UB) < new_crx:
            d = p3[i] + (p1[i] - p2[i])*step_size
            offspring[i] = max(
                bounds[i, 0], 
                min(d, bounds[i, 1])
            )

@njit(cache = True, parallel = True, looplift = True)
def parallel_DE_with_probability(
    bit_gen: list[np.random.BitGenerator],
    offspring,
    p1, p2, p3,
    bounds,
    step_size, crossover_rate):
    """(Experimental)
    
    Takes a Numba `typed.List` of bit generators (spawns) and does differential evolution in parallel batches.
    
    When the number of variables is very large (> ~100,000), significant speed ups. 
    
    Usually only around 5 - 10 bit generator spawns is optimal (more spawns for more variables)"""
    
    if crossover_rate <= 0:
        return
    if crossover_rate >= 1:
        gu_differential_evolve(p1, p2, p3, bounds, np.float32(step_size), offspring)
        return 
    
    UB = 1048576
    new_cxr = (np.uint32( max(1, min(UB - 1, np.uint32(crossover_rate * UB))) ) )
    nvars = len(offspring)
    nbit_gens = len(bit_gen)
    chunk_size = (nvars + nbit_gens - 1) // nbit_gens 
    
    for b in prange(nbit_gens):
        start = b * chunk_size
        if start >= nvars:
            continue
        
        end = min(nvars, start + chunk_size)
        _inner_DE(
            bit_gen[b], 
            offspring, 
            p1, p2, p3, 
            bounds,
            step_size, 
            UB, 
            new_cxr, 
            start, end)
    
    irand = np.random.randint(nvars)
    d = p3[irand] + (p1[irand] - p2[irand])*step_size
    offspring[irand] = max(bounds[irand, 0], min(d, bounds[irand, 1]))
    


# vectorized_evolve_sig2 = [
#     float32(float32, float32, float32, float32, float32, float32),
#     float32(float64, float64, float32, float32, float32, float32),
#     float64(float32, float32, float64, float64, float64, float32),
#     float64(float64, float64, float64, float64, float64, float64)
# ]
# @vectorize(vectorized_evolve_sig2, nopython = True)
# def vectorized_DE(
#     min_value, max_value, 
#     p1, p2, p3, step_size):
    
#     e = p3 + (p1 - p2)*step_size
#     return max(min_value, min(e, max_value))

# @vectorize(vectorized_evolve_sig2, nopython = True)
# def normal_vectorized_DE(
#     min_value, max_value, 
#     p1, p2, p3, step_size):
#     if min_value >= max_value:
#         return max_value
    
#     d = max_value - min_value
#     v1 = (p1 - min_value) / d
#     v2 = (p2 - min_value) / d
#     v3 = (p3 - min_value) / d
#     e = max(0, min(v3 + step_size*(v1 - v2), 1))
#     return e * d + min_value

# @njit(cache = True)
# def get_bit_mask(bit_gen, nvars, crossover_rate):
#     nbytes = (nvars + 7) // 8
#     if crossover_rate <= 0:
#         return np.zeros(nbytes, dtype=np.uint8)
    
#     UB = 1048576
#     new_crx = (
#         UB if crossover_rate >= 1 else 
#         np.uint32(
#             max(1, min(UB - 1, np.uint32(crossover_rate * UB)))
#         ))
    
#     mask = np.zeros(nbytes, dtype=np.uint8)
#     for i in range(nvars):
#         if buffered_bounded_lemire_uint32(bit_gen, UB) < new_crx:
#             byte = i // 8
#             bit_idx = i % 8
#             mask[byte] |= np.uint8(1 << bit_idx)
#     return mask

# vector_evolve_sig2 = [
#     (float32[:,:], float32[:], float32[:], float32[:], float32[:], float32),
#     (float64[:,:], float32[:], float32[:], float32[:], float32[:], float32),
#     (float32[:,:], float64[:], float64[:], float64[:], float64[:], float32),
#     (float64[:,:], float64[:], float64[:], float64[:], float64[:], float64),
# ]
# @guvectorize(vector_evolve_sig2, '(x,y),(x),(x),(x),(x),()', writable_args = (1,), nopython = True)
# def gu_differential_evolve2(
#     bounds,
#     offspring, p1, p2, p3, 
#     step_size):
#     """(in place)"""
    
#     nvars = len(p1)
#     for i in range(nvars):
#         e = p3[i] + step_size*(p1[i] - p2[i])
#         offspring[i] = max(bounds[i,0], min(e, bounds[i,1]))


    


    

import numpy as np
from sys import float_info
from collections import namedtuple
from numba import jit, njit, guvectorize, typeof, types, prange, uint32, float32, float64, boolean

"""
Two methods for PCX using static typing:
    - normalized_1d_pcx() -> one variable case
    - normalized_2d_pcx() -> multi-variable case (i.e. usual case)

"""

EPSILON = float_info.epsilon
_sig_tuple = namedtuple('signatures', ['valid32','valid64','invalid32','invalid64', 'cgs32', 'cgs64', 'offspring32', 'offspring64'] )
S = _sig_tuple(
    valid32 = types.Tuple((float32[:, :], float32, uint32))(float32[:, :], uint32, uint32, float32[:], float32[:], float32[:, :]),
    valid64 = types.Tuple((float64[:, :], float64, uint32))(float64[:, :], uint32, uint32, float64[:], float64[:], float64[:, :]),
    invalid32 = types.Tuple((float32[:, :], float32, uint32))(float32[:, :], uint32, uint32, float32[:], float32[:, :]),
    invalid64 = types.Tuple((float64[:, :], float64, uint32))(float64[:, :], uint32, uint32, float64[:], float64[:, :]),
    cgs32 = types.Tuple((float32[:, :], float32, uint32))(float32[:, :], uint32, uint32, float32[:]),
    cgs64 = types.Tuple((float64[:, :], float64, uint32))(float64[:, :], uint32, uint32, float64[:]),
    offspring32= float32[:, :](float32[:, :], uint32, uint32, uint32, float32[:], float32[:], float32[:, :], float32, float32),
    offspring64= float64[:, :](float64[:, :], uint32, uint32, uint32, float64[:], float64[:], float64[:, :], float64, float64)
)

@guvectorize(
    [(float32[:], float32[:], float32[:], uint32[:]),
     (float64[:], float64[:], float64[:], uint32[:])], 
    '(n),(n)->(n),()',
    nopython = True)
def _vectorized_subtract(u, v, out, out_nzero):
    """
    Element wise `out[i]` = `u[i]` - `v[i]`
    and counts the number of `abs(u[i]` - `v[i]`) < eps
    
    gu_vectorized: only need `u` and `v` when signature active
    """    
    k = len(u)
    # EPS = np.finfo(u.dtype).eps
    nzero = 0
    for i in range(k):
        z_i = u[i] - v[i]
        if abs(z_i) < EPSILON:
            nzero += 1
        out[i] = z_i
    out_nzero[0] = nzero

@guvectorize(
    [(float32[:,:], float32[:]),
     (float64[:,:], float64[:])], 
    '(p,n)->(n)',
    nopython = True)
def _find_g(
    parent_vars: np.ndarray,
    out):
    """Takes parent variables (2D array 1 row of variables per parent), 
    
    Finds the mean value of each variable. 
    gu_vectorized = only parents_vars required"""
    
    k = parent_vars.shape[0]
    n = parent_vars.shape[1]
    flt_k = np.float32(k)
    for var in range(n):
        g_var = 0.0
        for parent_idx in range(k):
            g_var += parent_vars[parent_idx,var]
        out[var] = g_var / flt_k

@njit(
    [types.Tuple((float32, uint32))(float32[:], uint32, float64), 
    types.Tuple((float64, uint32))(float64[:], uint32, float64)], 
    parallel = True, cache = True
)
def _sum_and_count(new_basis, nvars, eps):
    e_sum = 0.0
    count = 0
    # EPS = np.finfo(np.float32).eps
    for j in prange(nvars):
        b = new_basis[j]
        if b < eps:
            count += 1
        e_sum += b*b
    return e_sum, count

@njit([S.invalid32, S.invalid64])
def _invalid_e0_gs(
    parent_vars: np.ndarray, 
    nparents: np.uint32, 
    nvars: np.uint32,
    g: np.ndarray,
    e_eta: np.ndarray,
):
    """
    Special case where parent basis vector is a 0 vector
        (To avoid divide by 0 and all 0 offspring)

    Returns:
       tuple: (basis vectors, D, number of non zero vectors)
    """    
    D = 0
    num_non_zero = 0
    zero_chain = np.uint8(0)
    first_valid_idx = 0
    
    for i in range(nparents-1):
        d     = np.empty(nvars, dtype = parent_vars.dtype)
        d_nzero = np.empty(1, dtype = np.uint32)
        _vectorized_subtract(parent_vars[i], g, d, d_nzero)
        if d_nzero == nvars:
            continue
        
        e_sum = 0.0
        zero_count = 0
        temp_basis = np.zeros(nvars, parent_vars.dtype)
        # A valid previous basis has not been found yet 
        # (no quotient with reference dot product)
        if num_non_zero == 0: 
            for j in range(nvars):
                d_sub = d[j]
                temp_basis[j] = d_sub 
                e_sum += d_sub * d_sub   
        # A valid previous basis vector has been found
        else:
            for var in range(nvars): # make a copy of first basis vector
                temp_basis[var] = e_eta[first_valid_idx, var]
            
            for prev in range(first_valid_idx, num_non_zero):
                prev_basis = np.ascontiguousarray(e_eta[prev])
                sub = np.dot(temp_basis, prev_basis)
                temp_basis -= sub * prev_basis
            
            e_sum, zero_count = _sum_and_count(temp_basis, nvars, EPSILON)
        
        # Create final basis vector w/ magnitude
        if zero_count == nvars or e_sum < EPSILON:
            zero_chain += 1
            if zero_chain > 2:
                break
        else:
            zero_chain = 0
            if num_non_zero == 0:
                first_valid_idx = i
            e_magnitude = np.sqrt(e_sum)
            D += e_magnitude
            e_eta[num_non_zero] = temp_basis / e_magnitude
            num_non_zero += 1
    
    return e_eta, D, num_non_zero


@guvectorize(
    [(float32[:], float32[:], float32, uint32, float32[:], float32[:]), 
    (float64[:], float64[:], float64, uint32, float64[:], float64[:])],
    '(x),(x),(),()->(x),()', nopython = True, cache = True
)
def _no_valid_basis(d, e0_hat, dot_d_e0, nvars, out_basis, out_sum):
    e_sum = 0.0
    for j in range(nvars):
        d_sub = d[j] - (dot_d_e0 * e0_hat[j])
        out_basis[j] = d_sub
        e_sum += d_sub * d_sub
    out_sum[0] = e_sum

@njit([(S.valid32), (S.valid64)])
def _batch_gs(
    parent_vars: np.ndarray, 
    nparent: np.uint32, 
    nvars: np.uint32,
    g: np.ndarray,
    e0: np.ndarray, 
    e_eta: np.ndarray,
):
    
    e0 = np.ascontiguousarray(e0)
    dot_reference = np.dot(e0, e0)
    if dot_reference < EPSILON:
        return _invalid_e0_gs(parent_vars, nparent, nvars, g, e_eta)
    
    e0_hat = e0 / dot_reference
    e_eta = np.ascontiguousarray(e_eta)
    num_non_zero = np.uint32(0)
    D = 0
    EPS = np.finfo(np.float32).eps
    
    for i in range(nparent -1):
        d  = np.empty(nvars, parent_vars.dtype)
        d_nzero = np.empty(1,np.uint32)
        _vectorized_subtract(parent_vars[i], g, d, d_nzero)
        if d_nzero[0] == nvars:
            continue
 
        e_sum = 0.0
        dot_d_first = np.dot(d, e0)
        temp = np.ascontiguousarray(d - (dot_d_first * e0_hat))
        if num_non_zero == 0: 
            s = np.empty(1,parent_vars.dtype)
            _no_valid_basis(d, e0_hat, dot_d_first, nvars, temp, s)
            e_sum = s[0]
        else:
            B = e_eta[:num_non_zero]
            coeffs = B.dot(temp) 
            temp -= coeffs @ B
            e_sum = temp.dot(temp)
            
        if e_sum < EPS:
            break
        norm = np.sqrt(e_sum)
        next = D + norm
        D = next
        e_eta[num_non_zero] = temp / norm
        num_non_zero += 1
        if norm / next < 5e-5:
            break
    
    return e_eta, D, num_non_zero
    

@njit([S.valid32,S.valid64])
def _modified_gs(
    parent_vars: np.ndarray, 
    nparent: np.uint32, 
    nvars: np.uint32,
    g: np.ndarray,
    e0: np.ndarray, 
    e_eta: np.ndarray):
    """
    k = num parents
    n = num variables per parent
    g = mean vector

    Returns: (updated basis vectors, if basis vectors are non-zero, D, num_non_zero vectors)
    """    
    # construct matrix for all basis vectors

    e0 = np.ascontiguousarray(e0)
    dot_reference = np.dot(e0, e0)
    if dot_reference < EPSILON:
        return _invalid_e0_gs(parent_vars, nparent, nvars, g, e_eta)
    
    e0_hat = e0 / dot_reference
    e_eta = np.ascontiguousarray(e_eta)
    num_non_zero = 0
    D = 0
    zero_chain = np.uint8(0)
    
    for i in range(nparent-1):
        d      = np.empty(nvars, parent_vars.dtype)
        d_nzero = np.empty(1,np.uint32)
        _vectorized_subtract(parent_vars[i], g, d, d_nzero)
        if d_nzero[0] == nvars:
            continue
        
        e_sum = 0.0
        dot_d_first = np.dot(d, e0)
        temp_basis = np.ascontiguousarray(d - (dot_d_first * e0_hat))

        # A valid previous basis has not been found yet
        if num_non_zero == 0: 
            t = np.empty(1,parent_vars.dtype)
            _no_valid_basis(d, e0_hat, dot_d_first, nvars, temp_basis, t)
            e_sum = t[0]
        else:
            for prev in range(num_non_zero):
                temp_basis -= temp_basis.dot(e_eta[prev]) * e_eta[prev]
            e_sum = temp_basis.dot(temp_basis)

        # Create final basis vector w/ magnitude
        if  e_sum < EPSILON:
            zero_chain += 1
            if zero_chain > 2:
                break
        else:
            zero_chain = 0
            e_magnitude = np.sqrt(e_sum)
            D += e_magnitude 
            e_eta[num_non_zero] = temp_basis / e_magnitude
            num_non_zero += 1
        
    return e_eta, D, num_non_zero


@njit([S.cgs32,S.cgs64])
def _classic_gs(
    parent_vars,
    nparents,
    nvars,
    g):
           
    d_reference = parent_vars[-1] - g
    d_rest  = parent_vars[:-1] - g  
    
    e_eta = np.empty((nvars, nparents), parent_vars.dtype) #transposed
    e_eta[:, 0] = d_reference
    e_eta[:, 1:] = d_rest.T
    
    Q, R = np.linalg.qr(e_eta)    
    r = np.abs(np.diag(R))      
    D = r[1:].sum()
    
    non_zero_indices = np.where(r > EPSILON)[0]
    n_non_zero = len(non_zero_indices)
    first_included = non_zero_indices[0] == 0 
    if n_non_zero == 0 or (n_non_zero == 1 and first_included):
        return np.zeros((1,nvars), parent_vars.dtype), D, 0
    
    out = np.empty((n_non_zero - 1, nvars), parent_vars.dtype)
    start = 1 if non_zero_indices[0] == 0 else 0
    for i in range(start, n_non_zero):
        col = non_zero_indices[i]
        out[i-start,:] = Q[:, col]
    return out, D, n_non_zero


@njit([S.offspring32, S.offspring64])
def _find_offspring_pcx(
    parent_vars:np.ndarray, 
    noffspring: np.uint32,
    k: np.uint32, 
    n: np.uint32,
    g: np.ndarray,
    e0: np.ndarray, 
    e_eta: np.ndarray,
    eta = np.float64,
    zeta = np.float64):
    """
    expects parent vars to be normalized to range [0,1]
    
    k = num parents
    n = num variables per parent
    g = mean vector
    e0 reference basis vector
    """    
    D = 0
    num_non_zero = np.uint32(0)
    if max(k,n) < 100 or n <= 10:
        e_eta, D, num_non_zero = _modified_gs(parent_vars, k, n, g, e0, e_eta)
    elif k == n:
        e_eta, D, num_non_zero = _classic_gs(parent_vars, k, n, g)
    else:
        e_eta, D, num_non_zero = _batch_gs( parent_vars, k, n, g, e0, e_eta)
        
    D /= (k - 1.0)
    output = np.zeros((noffspring, n), parent_vars.dtype)
    for offspring in range(noffspring): #zeta perturbation
        zeta_perturbation = np.random.normal(0.0, zeta) 
        output[offspring] = parent_vars[-1] + zeta_perturbation * e0

    if D >= EPSILON and num_non_zero > 0:
        eta_sum = np.sum(e_eta[:num_non_zero], 0)
        for offspring in range(noffspring): #eta perturbation
            eta_D = np.random.normal(0.0,eta) * D
            output[offspring] += eta_D * eta_sum
            
    np.clip(output, 0.0, 1.0, output)
    return output

@njit([float32[:](float32[:], types.uint32, float32, float32, boolean),
      float64[:](float64[:], types.uint32, float64, float64, boolean)])
def normalized_1d_pcx(
    parent_vars: np.ndarray, 
    noffspring: np.uint32, 
    eta: np.float32,
    zeta: np.float32,
    randomize = True):
    """
    PCX where each parent has 1 variable. Updated so that both eta and zeta are still used
    
    Expects all parent_vars elements to be normalized to [0,1] prior to input
    
    There should be more than 1 parent
    
   
    """   
    k = len(parent_vars)
    if k == 0:
        return np.zeros(noffspring, parent_vars.dtype)
    if k == 1:
        return np.full(noffspring, parent_vars[0], parent_vars.dtype)
    
    g = sum(parent_vars) / k
    d = 0.0 # mean abs deviation
    d_sign = 0.0
    e0 = 0.0
    reference = parent_vars[-1]
    if not randomize:
        for i in range(k - 1):
            deviation = parent_vars[i] - g
            d_sign += deviation
            d += abs(deviation)
        if k > 2:
            d /= (k - 1)
        e0 = parent_vars[-1] - g
        if d_sign == 0:
            d_sign = -1.0 if np.random.randint(2) else 1.0
        else:
            d_sign = -1.0 if d_sign < 0 else 1.0
    offspring = np.empty(noffspring, parent_vars.dtype)
    
    for i in range(noffspring):
        if randomize:
            rand_idx = np.random.randint(k)
            reference = parent_vars[rand_idx]
            e0 = reference - g
            d = 0.0 
            for j in range(k):
                if j != reference:
                    deviation = parent_vars[i] - g
                    d_sign += deviation
                    d += abs(deviation)
            if k > 2: 
                d /= (k - 1)
            if d_sign == 0:
                d_sign = -1.0 if np.random.randint(2) else 1.0
            else:
                d_sign = -1.0 if d_sign < 0 else 1.0
                
        perturb_zeta = np.random.normal(0.0, zeta) * e0 # Bias for reference parent value
        perturb_eta = np.random.normal(0.0, eta) * d * d_sign # Bias for exploration      
        offspring_val = reference + perturb_zeta + perturb_eta
        offspring[i] = max(0.0, min(offspring_val, 1.0))
        
    return offspring


def normalized_2d_pcx(
    parent_vars: np.ndarray, 
    noffspring, 
    eta: np.float32,
    zeta: np.float32,
    randomize = True):
    """
    General parent-centric crossover
    
    Expects all parent_vars elements to be normalized to [0,1] prior to input

    Expects parent_vars to be a 2D npndarray of type float32
    
    There should be more than 1 parent
    
    """    
    noffspring = np.uint8(noffspring)
    nparents = np.uint8(parent_vars.shape[0])
    nvars = np.uint32(parent_vars.shape[1])
    dtype = parent_vars.dtype
    if nparents == 0:
        return np.zeros((noffspring, nvars), dtype)
    
    output = np.empty((noffspring, nvars), dtype)
    if nparents == 1: 
        parent_row = parent_vars[0]
        for i in range(noffspring):
            output[i] = parent_row
        return output
    
    assert eta > 0 and zeta > 0, f"eta and zeta must be greater than 0, got {eta} and {zeta} respectively"
    if nvars == 1: 
        all_vars = parent_vars[:,0]
        output[:,0] = normalized_1d_pcx(all_vars, noffspring, eta, zeta, randomize) #TODO: See if can reshape, and if it is faster
        return output
    
    # mean of each var
    # g = np.empty(nvars, np.float32)
    g = _find_g(parent_vars)
    if noffspring == 1 or not randomize:
        if randomize:
            rand_parent = np.random.randint(nparents)
            parent_vars[[rand_parent, -1]] = parent_vars[[-1, rand_parent]]

        e0, _ = _vectorized_subtract(parent_vars[-1], g)
        e_eta = np.zeros((nparents-1, nvars), dtype)
        return _find_offspring_pcx(parent_vars, noffspring, nparents, nvars, g, e0, e_eta, eta, zeta)
    
    # Remember configuration of reference parents
    count_dict = {}
    for rand_p in (np.random.randint(nparents) for _ in range(noffspring)):
        new_count = count_dict.get(rand_p, 0) + 1
        count_dict[rand_p] = new_count
    
    curr_row = 0
    for parent_idx, count in count_dict.items():
        
        parent_vars[[parent_idx, -1]] = parent_vars[[-1, parent_idx]]
        e0, _=  _vectorized_subtract(parent_vars[-1], g)
        e_eta = np.zeros((nparents-1,nvars), dtype, order = 'C')
        
        config_offspring = _find_offspring_pcx(parent_vars, count, nparents, nvars, g, e0, e_eta, eta, zeta)
        output[curr_row:curr_row+count, :] = config_offspring
        curr_row += count

    return output


# @guvectorize(
#     [(float32[:,:], float32[:], uint32, float32[:]), 
#      (float64[:,:], float32[:], uint32, float64[:])], 
#     '(x,y),(x),()->(y)', nopython = True, cache = True)
# def _safe_mat_mult(prev_bases, coeffs, nvars, temp_basis):
#     nbasis = len(coeffs)
#     for b in range(nbasis):
#         for n in range(nvars):
#             temp_basis[n] -= coeffs[b] * prev_bases[b,n]

# def _pcx_orig(p_matrix: list):
#     k = len(p_matrix)
#     nvars = len(p_matrix[0])
#     g = [sum([p_matrix[i][j] for i in range(k)]) / k for j in range(nvars)] # mean of each var
#     D = 0.0
#     # basis vectors defined by parents
#     e_eta = []
#     prev = subtract(p_matrix[k-1], g)
#     e_eta.append(prev) 
#     for i in range(k-1): 
#         d = subtract(p_matrix[i], g)
#         if not is_zero(d):
#             e = orthogonalize(d, e_eta)
#             if not is_zero(e):
#                 D += magnitude(e) #sqrt(dot(e,e))
#                 e_eta.append(normalize(e))# put in [0,1] range
#     return e_eta, D
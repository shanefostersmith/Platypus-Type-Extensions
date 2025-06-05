import numpy as np
from numba import njit, vectorize, guvectorize, types, float32, boolean
from numba import types

"""
Two methods for PCX using static typing:
    - normalized_1d_pcx() -> one variable case
    - normalized_2d_pcx() -> multi-variable case (i.e. usual case)

"""
ortho_ret_type = types.Tuple((
    float32[:, :], 
    boolean[:], 
    float32, 
    types.uint8
))

ortho_valid_sig = ortho_ret_type(
    float32[:, :], 
    types.uint8, 
    types.uint32, 
    float32[:], 
    float32[:], 
    float32[:, :], 
    boolean[:], 
    float32
)
ortho_invalid_sig = ortho_ret_type(
    float32[:, :], 
    types.uint8, 
    types.uint32,
    float32[:], 
    float32[:, :], 
    boolean[:]
)

@guvectorize(
    [(float32[:], float32[:], float32[:], types.uint32[:])], 
    '(n),(n)->(n),()',
    nopython = True)
def _vectorized_subtract(u, v, out, out_nzero):
    """
    Element wise `out[i]` = `u[i]` - `v[i]`
    and counts the number of `abs(u[i]` - `v[i]`) < eps
    
    gu_vectorized: only need `u` and `v` when signature active
    """    
    k = len(u)
    EPS = np.finfo(np.float32).tiny
    nzero = 0
    for i in range(k):
        z_i = u[i] - v[i]
        if abs(z_i) < EPS:
            nzero += 1
            
        out[i] = z_i
    out_nzero[0] = nzero

@guvectorize(
    [(float32[:,:], float32[:])], 
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
        
# @njit(ortho_invalid_sig)
def _invalid_e0_orthogonalize(
    parent_vars: np.ndarray, 
    k: np.uint8, 
    n: np.uint32,
    g: np.ndarray,
    e_eta: np.ndarray,
    is_all_zero: np.ndarray,
):
    """
    Special case where parent basis vector is a 0 vector
        (To avoid divide by 0 and all 0 offspring)

    Returns:
       tuple: (basis vectors, whether basis vectors are 0, D, number of non zero vectors )
    """    
    
    EPS = np.finfo(np.float32).tiny
    
    # construct matrix for all basis vectors
    D = np.float32(0)
    num_non_zero = np.uint8(0)
    first_valid_idx = 0
    for i in range(k-1):
        # d = np.zeros(n , np.float32)
        # d_all_zero = np.zeros(1 , np.uint32)
        d, d_nzero = _vectorized_subtract(parent_vars[i], g)
        if d_nzero == n:
            is_all_zero[i] = True
            continue
        
        e_sum = 0.0
        temp_new_d = np.zeros(n, np.float32)
        
        # A valid previous basis has not been found yet 
        # (no quotient with reference dot product)
        if num_non_zero == 0: 
            for j in range(n):
                d_sub = d[j]
                if not abs(d_sub) < EPS:
                    temp_new_d[j] = d_sub 
                    e_sum += d_sub * d_sub
                    
        # A valid previous basis vector has been found
        else:
            for var in range(n): # make a copy of first basis vector
                temp_new_d[var] = e_eta[first_valid_idx, var]
            
            for prev in range(first_valid_idx + 1, i):
                if is_all_zero[prev]:
                    continue
                prev_basis = np.ascontiguousarray(e_eta[prev])
                dot2 = np.dot(prev_basis, prev_basis)
                if abs(dot2) < EPS:
                    continue
                dot1 = np.dot(temp_new_d, prev_basis)
                quotient = dot1 / dot2
                for j in range(n):
                    new_d = temp_new_d[j] - quotient*prev_basis[j]
                    temp_new_d[j] = new_d
                    
            # Find dot product of new basis vector
            for j in range(n):
                curr_d = temp_new_d[j] 
                e_sum += curr_d * curr_d   
        
        # Create final basis vector w/ magnitude
        e_magnitude = np.sqrt(e_sum)
        if e_magnitude < EPS:
            is_all_zero[i] = True
        else:
            if num_non_zero == 0:
                first_valid_idx = i
            num_non_zero += np.int32(1)
            D += e_magnitude 
            for j in range(n):
                d_magnitude = (1.0 / e_magnitude) * temp_new_d[j]
                e_eta[i,j] = d_magnitude
    
    return e_eta, is_all_zero, D, num_non_zero

# @njit(ortho_valid_sig)
def _valid_e0_orthogonalize(
    parent_vars: np.ndarray, 
    nparent: np.uint8, 
    n: np.uint32,
    g: np.ndarray,
    e0: np.ndarray, 
    e_eta: np.ndarray,
    is_all_zero: np.ndarray,
    dot_reference: np.float32):
    """
    k = num parents
    n = num variables per parent
    g = mean vector

    Returns: (updated basis vectors, if basis vectors are non-zero, D, num_non_zero vectors)
    """    
    
    EPS = np.finfo(np.float32).tiny
    
    # construct matrix for all basis vectors
    D = np.float32(0)
    num_non_zero = np.uint8(0)
    e0 = np.ascontiguousarray(e0)
    for i in range(nparent-1):
        # d = np.ascontiguousarray(np.zeros(n , np.float32))
        # d_all_zero = np.zeros(1 , np.uint32)
        d, d_nzero = _vectorized_subtract(parent_vars[i], g)
        if d_nzero == n:
            is_all_zero[i] = True
            continue
        
        e_sum = 0.0
        temp_new_d = np.zeros(n, np.float32)
        dot_d_first = np.dot(d, e0)
        quotient_first = dot_d_first / dot_reference
        print(f"dot_d_first {dot_d_first }, dot_d_referece {dot_reference}")
        
        # A valid previous basis has not been found yet
        if num_non_zero == 0: 
            for j in range(n):
                d_sub = d[j] - (quotient_first * e0[j])
                if not abs(d_sub) < EPS:
                    temp_new_d[j] = d_sub
                    e_sum += d_sub * d_sub
        else:
            # First basis vector is always the e0 vector
            for j in range(n): 
                d_sub = d[j] - (quotient_first * e0[j])
                if not abs(d_sub) < EPS:
                    temp_new_d[j] = d_sub
            
            # Include all previous basis vectors that are non-zero
            for prev in range(i):
                if is_all_zero[prev]:
                    continue
                prev_basis = np.ascontiguousarray(e_eta[prev])
                dot2 = np.dot(prev_basis, prev_basis)
                if abs(dot2) < EPS:
                    continue
                dot1 = np.dot(temp_new_d, prev_basis)
                quotient = dot1 / dot2
                for j in range(n):
                    new_d = temp_new_d[j] - quotient*prev_basis[j]
                    temp_new_d[j] = new_d
            
            # Find dot product of new basis vector
            for j in range(n):
                curr_d = temp_new_d[j] 
                e_sum += curr_d * curr_d
        
        # Create final basis vector w/ magnitude
        e_magnitude = np.sqrt(e_sum)
        if e_magnitude < EPS:
            is_all_zero[i] = True
        else:
            num_non_zero += 1
            D += e_magnitude 
            for j in range(n):
                d_magnitude = (1.0 / e_magnitude) * temp_new_d[j]
                e_eta[i,j] = d_magnitude

    return e_eta, is_all_zero, D, num_non_zero


# @njit("float32[:, :](float32[:, :], uint8, uint8, uint32, float32[:], float32[:], float32[:, :], float32, float32)")
def _orthogonalize_pcx(
    parent_vars:np.ndarray, 
    noffspring: np.uint8,
    k: np.uint8, 
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
    EPS = np.finfo(np.float32).tiny
    D = np.float32(0)
    is_all_zero = np.zeros(k-1, np.bool_)
    num_non_zero = np.uint8(0)
    e0 = np.ascontiguousarray(e0)
    dot_reference = np.dot(e0, e0)
    if abs(dot_reference) < EPS: # Using quotient for e0 is invalid
        e_eta, is_all_zero, D, num_non_zero = _invalid_e0_orthogonalize(
            parent_vars, k, n, g, e_eta, is_all_zero
        )
    else:
        e_eta, is_all_zero, D, num_non_zero = _valid_e0_orthogonalize(
            parent_vars, k, n, g, e0, e_eta, is_all_zero, dot_reference)
    
    # Find all offspring variables
    D /= np.float32(k - 1)
    output = np.zeros((noffspring, n), np.float32)
    use_eta = D >= EPS and num_non_zero > 0
    for offspring in range(noffspring):
        
        # Apply zeta pertubation
        reference_copy = np.empty(n, np.float32)
        zeta_perturbation = np.random.normal(0.0, zeta) 
        for var in range(n): #TODO vectorize
            reference_copy[var] = parent_vars[-1, var]
            reference_copy[var] += zeta_perturbation * e0[var]
            
        # Apply eta pertubation
        if use_eta:
            eta_D = np.random.normal(0.0,eta) * D
            for i in range(k-1):
                if is_all_zero[i]:
                    continue
                for var in range(n):
                    new_offspring_var = reference_copy[var] + (eta_D * e_eta[i,var])
                    reference_copy[var] = new_offspring_var
 
        # Clip output variables
        np.clip(reference_copy, 0.0, 1.0)
        # for var in range(n):
        #     new_offspring_var = reference_copy[var]
        #     reference_copy[var] = max(0.0, min(new_offspring_var,1.0))  
             
        output[offspring] = reference_copy
            
    return output

# @njit("float32[:](float32[:], uint8, float32, float32, boolean)")
def normalized_1d_pcx(
    parent_vars: np.ndarray, 
    noffspring: np.uint8, 
    eta: np.float32,
    zeta: np.float32,
    randomize = True):
    """
    PCX where each parent has 1 variable. Updated so that both eta and zeta are still used
    
    Expects all parent_vars elements to be normalized to [0,1] prior to input
    
    There should be more than 1 parent
    
   
    """   
    # print(f"1d pcx: {parent_vars}") 
    k = len(parent_vars)
    if k == 0:
        return np.zeros(noffspring, np.float32)
    if k == 1:
        return np.full(noffspring, parent_vars[0], np.float32)
    
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
            d /= np.float32(k - 1)
        e0 = parent_vars[-1] - g
        if d_sign == 0:
            d_sign = -1.0 if np.random.randint(2) else 1.0
        else:
            d_sign = -1.0 if d_sign < 0 else 1.0
    offspring = np.empty(noffspring, dtype=np.float32)
    
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
                d /= np.float32(k - 1)
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
    if nparents == 0:
        return np.zeros((noffspring, nvars), np.float32)
    
    output = np.empty((noffspring, nvars), np.float32)
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
        # e0 = np.zeros(nvars, np.float32, order = 'C')
        # temp_out = np.zeros(1, np.uint32)
        e0, _ = _vectorized_subtract(parent_vars[-1], g)
        e_eta = np.zeros((nparents-1, nvars), np.float32)
        return _orthogonalize_pcx(parent_vars, noffspring, nparents, nvars, g, e0, e_eta, eta, zeta)
    
    # Remember configuration of reference parents
    count_dict = {}
    for rand_p in (np.random.randint(nparents) for _ in range(noffspring)):
        new_count = count_dict.get(rand_p, 0) + 1
        count_dict[rand_p] = new_count
    
    curr_row = 0
    for parent_idx, count in count_dict.items():
        # print(f"count = {count}, parent_idx = {parent_idx}, curr_row = {curr_row}")
        parent_vars[[parent_idx, -1]] = parent_vars[[-1, parent_idx]]
        # e0 = np.zeros(nvars, np.float32, order = 'C')
        # temp_out = np.zeros(1, np.uint32)
        e0, _=  _vectorized_subtract(parent_vars[-1], g)
        e_eta = np.zeros((nparents-1,nvars), np.float32, order = 'C')
        config_offspring = _orthogonalize_pcx(parent_vars, count, nparents, nvars, g, e0, e_eta, eta, zeta)
        output[curr_row:curr_row+count, :] = config_offspring
        curr_row += count

    return output
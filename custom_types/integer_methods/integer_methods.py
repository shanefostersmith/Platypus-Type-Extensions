import numpy as np
from numba import njit, boolean, float64, types, intp

@njit(types.Tuple((boolean[:],boolean))(boolean[:], float64))
def int_mutation(bits: np.ndarray, probability = 0.0):
    """Bit flip mutation (in-place)
    
    If len(bits) == 0, will do nothing.

    Args:
        bits (np.ndarray): 1d numpy array of np.bool_ elements (the gray encoding of an integer)
        probability (float | numpy.float64): The probability an individual bit will be flipped. Defaults to 0.0
            If probability <= 0.0, will default to `1 / max(2, len(bits))`
            
    Returns:
        (np.ndarray, bool): The mutated gray encoding (not a new array), and a bool indicating if a bit was flipped
    """    
    nbits = bits.shape[0]
    if nbits == 0:
        return bits, False
    if probability <= 0:
        probability = 1.0 / max(2.0, nbits)
    did_mutation = False
    for i in range(nbits):
        if np.random.uniform() <= probability:
            temp = bits[i]
            bits[i] = ~temp
            did_mutation = True
    return bits, did_mutation 

@njit
def _cut_points(num_bits, num_parents):
    """ Segment a 'num_bits' sized array into 'num_parents' number of segments.
    
    If num_parents > num_bits, then num_bits number of segments are returned
    
    
    - """
    if num_parents >= num_bits:
        return np.ones(num_bits, np.uint16)
        
    base = num_bits // num_parents
    remainder = num_bits % num_parents
    differences = np.empty(num_parents, dtype=np.uint16) # bits per segments
    
    # Only one segment is base
    if remainder == num_parents - 1: 
        base += 1
        for i in range(num_parents):
            differences[i] = base
        rand_reduce_idx = np.random.randint(num_parents)
        differences[rand_reduce_idx] -= 1 
    else:
        for i in range(num_parents):
            differences[i] = base
        if remainder > 0: 
            if remainder == 1: # Only one segment is base + 1
                rand_increase_idx = np.random.randint(num_parents)
                differences[rand_increase_idx] += 1
            else:
                # Shuffle segment indices, and extend random segments by 1
                indices = np.empty(num_parents, dtype=np.uint16) 
                for i in range(num_parents):
                    indices[i] = i
                    
                for i in range(num_parents-1, 0, -1):
                    j = np.random.randint(0, i + 1)
                    temp = indices[i]
                    indices[i] = indices[j]
                    indices[j] = temp
                
                for k in range(remainder): # extend random segments
                    rand_extended_idx = indices[k]
                    differences[rand_extended_idx] += 1      
    return differences

@njit(boolean[:](boolean[:,:], types.uint16[:], intp, intp, intp))
def _execute_crossover(
        parent_bits: np.ndarray, 
        segment_sizes, 
        num_parents,
        num_bits,
        num_segments):
    
    idx1 = 0
    crossover_bits = np.zeros(num_bits, np.bool_)
    for j in range(num_segments):
        rand_parent = np.random.randint(num_parents)
        idx2 = idx1 + segment_sizes[j]
        crossover_bits[idx1:idx2] = parent_bits[rand_parent][idx1:idx2]
        idx1 = idx2
    
    return crossover_bits
    
@njit(boolean[:](boolean[:,:]))
def int_cross_over(parent_bits: np.ndarray):
    """Integer crossover given any number of parents and their bits
        Produces 1 offspring  

    There should be > 1 parent (otherwise copy of parent is returned)
    
    Args:
        parent_bits (np.ndarray): 2D numpy matrix of np.bool_ elements

    Returns:
        np.ndarray: 1D array of np.bool_ elements
    """

    num_parents = parent_bits.shape[0]
    num_bits = parent_bits.shape[1]
    assert(num_parents > 0 and num_bits > 0), "num parents and num bits must be greater than 0"
    if num_parents == 1:
        return np.copy(parent_bits[0])
    
    segment_sizes = _cut_points(num_bits, num_parents)
    num_segments = len(segment_sizes)
    return _execute_crossover(parent_bits, segment_sizes, num_parents, num_bits, num_segments)

@njit(boolean[:,:](boolean[:,:], intp))
def multi_int_crossover(parent_bits: np.ndarray, noffspring: int):
    """Integer crossover given any number of parents and their bits
        Can produce multiple offspring 
    
    There should be > 1 parent (otherwise copy of parent is returned)

    Args:
        parent_bits (np.ndarray): 2D numpy array of np.bool_ elements (must be that type)
        noffspring (int): number of offspring arrays of bits to produce 

    Returns:
        np.ndarray:2D numpy array of np.bool_ elements. One row per offspring
    """    
    assert(noffspring > 0), "noffspring must be greater than 0"
    num_parents = parent_bits.shape[0]
    num_bits = parent_bits.shape[1]

    crossover_bits = np.empty((noffspring, num_bits), np.bool_)
    if num_parents <= 1 or num_bits == 0:
        if num_parents == 0 or num_bits == 0:
            return crossover_bits
        r = np.copy(parent_bits)
        for i in range(noffspring):
            crossover_bits[i] = r
        return crossover_bits
    
    if noffspring == 1:
        crossover_bits[0] = int_cross_over(parent_bits)
        return crossover_bits

    segment_sizes = _cut_points(num_bits, num_parents)
    num_segments = len(segment_sizes)
    for i in range(noffspring):
        crossover_bits[i] = _execute_crossover(parent_bits, segment_sizes, num_parents, num_bits, num_segments)
    return crossover_bits

@njit(boolean[:](boolean[:,:], boolean[:]))
def single_binary_swap(parent_bits: np.ndarray, offspring_bits: np.ndarray):
    """Given offspring bits and parents bits, choose a random parent and 
    randomly set offspring bits with those parent bits

    Similar to HUX, but only changing the offspring bits (in-place)
    
    Args:
        parent_bits (np.ndarray): 2D numpy array of np.bool_ elements
        offspring_bits (np.ndarray): 1D numpy array of np.bool_ elements 
    
    Raises:
        ValueError: If the number of columns of 'parent_bits' != length of 'offspring_bits'

    Returns:
        np.ndarray: 1D numpy array of np.bool_ elements
    """    
    num_parents = parent_bits.shape[0]
    num_bits = parent_bits.shape[1]
    if len(offspring_bits) != num_bits:
        raise ValueError(f"Got {num_bits} parent bits and {len(offspring_bits)} offspring bits, these must be the same ")
    if num_parents == 0 or num_bits == 0:
        return offspring_bits
    
    rand_parent_idx = 0 if num_parents == 1 else np.random.randint(num_parents)
    rand_parent = parent_bits[rand_parent_idx]
    swap_probability = 1.0 / max(1.0, (num_bits - 1))
    for i in range(num_bits):
        if np.random.uniform() < swap_probability:
            offspring_bits[i] = rand_parent[i]
    return offspring_bits
        
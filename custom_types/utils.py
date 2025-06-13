import numpy as np
from numba import njit, vectorize, guvectorize, float32, float64, boolean
from numbers import Integral
from math import floor

def clip(val, lower_bound, upper_bound):
    return max(lower_bound, min(val, upper_bound))

@njit([
    float32(float32, float32, float32, boolean),
    float32(float64, float64, float32, boolean),
    float64(float32, float32, float64, boolean),
    float64(float64, float64, float64, boolean),
])
def _min_max_norm_convert(min_val, max_val, curr_val, to_norm):
    if to_norm:
        return 1.0 if max_val == min_val else (curr_val - min_val) / (max_val - min_val)
    return curr_val * (max_val - min_val) + min_val

def vectorized_to_norm(ranges: np.ndarray, values: np.ndarray, in_place = False):
    """Convert a 1D or 2D array of values to 0-1 range
    
    This function assumes there are no zero-width ranges

    Args:
        ranges (np.ndarray): 2D array where the number of columns = 2 
            (column 0 contains lower bounds and column 1 contains upper bounds)
        values (np.ndarray): 1D or 2D array of floats
        
            If 1D, must be true that:
            ``len(values) == ranges.shape[0]``
            
            If 2D, must be true that: ``values.shape[1] == ranges.shape[0]``
    Returns:
        np.ndarray: (same shape as input values)
    """    
    mins = ranges[:, 0]
    denoms= ranges[:, 1] - mins 
    if not in_place:
        normalized = np.empty_like(values, dtype=np.float64)
        np.subtract(values, mins, out=normalized)
        np.divide(normalized, denoms, out=normalized)
        return normalized
    else:
        np.subtract(values, mins, out=values)
        np.divide(values, denoms, out=values)
        return values

def vectorized_from_norm(ranges: np.ndarray, values: np.ndarray, dtype: type = None, in_place = False):
    """Convert a 1D or 2D array of values from 0-1 range to their original scale
    
    This function assumes there are no zero-width ranges

    Args:
        ranges (np.ndarray): 2D numpy array where number of columns = 2 
            (column 0 contains lower bounds and column 1 contains upper bounds)
        values (np.ndarray): 1D or 2D array of floats
            If 1D, must be true that:
            ``len(values) == ranges.shape[0]``
            
            If 2D, must be true that: ``values.shape[1] == ranges.shape[0]``
            
        dtype (type): Should be a numpy float type

    Returns:
        np.ndarray: 1D array of floats
    """    
    mins = ranges[:, 0]
    diffs = ranges[:, 1] - mins 
    if not in_place:
        dtype = dtype or np.float32
        original_scale = np.empty_like(values, dtype=dtype)
        np.multiply(values, diffs, out=original_scale)
        np.add(original_scale, mins, out=original_scale)
        return original_scale
    else:
        np.multiply(values, diffs, out=values)
        np.add(values, mins, out=values)
        return values


gu_scale_2Dsig = [
    (float32[:,:], float32[:,:], float32[:,:]), 
    (float64[:,:], float64[:,:], float64[:,:]),
    (float64[:,:], float32[:,:], float32[:,:]),
    (float32[:,:], float64[:,:], float64[:,:]),
    (float64[:,:], float64[:,:], float64[:,:])
]
gu_scale_1Dsig = [
    (float32[:,:], float32[:], float32[:]), 
    (float64[:,:], float64[:], float64[:]),
    (float64[:,:], float32[:], float32[:]),
    (float32[:,:], float64[:], float64[:]),
    (float64[:,:], float64[:], float64[:])
]
gu_descale_2Dsig = [
    (float32[:,:], float32[:,:]), 
    (float64[:,:], float32[:,:]),
    (float32[:,:], float64[:,:]),
    (float64[:,:], float64[:,:])
]
gu_descale_1Dsig = [
    (float32[:,:], float32[:]), 
    (float64[:,:], float32[:]),
    (float32[:,:], float64[:]),
    (float64[:,:], float64[:])
]

@guvectorize(gu_scale_1Dsig, '(r,c),(r)->(r)', nopython = True, cache = True)
def gu_normalize2D_1D(ranges, values, out): 
    """Normalize an array of values.
    It is assumed each element of 'values' 
    has a minimum and maximum value given by the corresponding row in 'ranges'
        - i.e each value `values[i]` has a lower bound `ranges[i,0]` and an upper bound `ranges[i,1]`
        
    Expects the shape of ranges `(r,2)` and shape of values `(r,)`"""
    nrows = ranges.shape[0]
    for r in range(nrows):
        if ranges[r,0] == ranges[r,1]:
            out[r] = 1
        else:
            out[r] = (values[r] - ranges[r,0]) / (ranges[r,1] - ranges[r,0])

@guvectorize(gu_descale_1Dsig, '(r,c),(r)', 
             nopython = True, cache = True,
             writable_args = (1,))
def gu_denormalize2D_1D(ranges, values): 
    """*In-place* conversion of normalized values (in [0,1] range) to original scale
    
    Expects the shape of ranges `(r,2)` and shape of values `(r,)` (see gu_normalize2D_1D)"""
    nrows = ranges.shape[0]
    for r in range(nrows):
        d = ranges[r,1] - ranges[r,0]
        values[r] = (values[r] * d) + ranges[r,0]

@guvectorize(gu_scale_2Dsig, '(r,c),(p,r)->(p,r)', nopython = True, cache = True)
def gu_normalize2D_2D(ranges, values, out): 
    """Normalize a matrix of values.
    It is assumed each column of 'values' 
    has a minimum and maximum value given by the corresponding row in 'ranges'
        - i.e each value in column `i` has a lower bound `ranges[i,0]` and an upper bound `ranges[i,1]`
        
    Expects the shape of ranges `(r,2)` and shape of values `(p,r)`"""
    nvars = ranges.shape[0]
    nvectors = values.shape[0]
    for v in range(nvars):
        if ranges[v,0] == ranges[v,1]:
            for r in range(nvectors):
                out[r,v] = 1
        else:
            lb = ranges[v,0]
            d = ranges[v,1] - lb
            for r in range(nvectors):
                out[r,v] = (values[r, v] - lb) / d
            
@guvectorize(gu_descale_2Dsig, '(r,c),(p,r)', 
             nopython = True, cache = True,
             writable_args = (1,))
def gu_denormalize2D_2D(ranges, values):
    """*In-place* conversion of normalized values (in [0,1] range) to original scale
    
    Expects the shape of ranges `(r,2)` and shape of values `(p,r)` (see gu_normalize2D_2D)"""
    nvars = ranges.shape[0]
    nvectors = values.shape[0]
    for v in range(nvars):
        lb = ranges[v,0]
        d = ranges[v,1] - lb
        for r in range(nvectors):
            values[r, v] = (values[r,v] * d) + lb

def int_to_gray_encoding(value: int, min_value: int, max_value: int, nbits: int | None = None) -> np.ndarray:
    """
    Combines all the steps of converting an integer into a gray encoding. 
    
    Will clip 'value' if it is smaller than 'min_value'
    
    If min_value == max_value, will return ( [False] (as ndarray), 1 )

    Args:
        value (int): A integer value (should be true that min_value <= value <= max_value)
        
        min_value (int): Lower bound on value
        
        max_value (int): Upper bound on value
        
        nbits (int | None): Optionally provide 'nbits'
            - If not provided, nbits will be calculated

    Raises:
        ValueError: If min value > max_value

    Returns:
        tuple[ndarray, int]: 
        - ndarray: An array of numpy bool_ (the grey encoding), 
        - int: the number of bits that represent the 'value' as a binary string
            (minimum of 1)
    """    

    if min_value > max_value:
        raise ValueError(f"min_value {min_value} > than max_value {max_value}")
    if min_value == max_value:
        return np.zeros(1, np.bool_), 1
    if value < min_value:
        value = min_value
    if not nbits:
        nbits = _nbits_encode(min_value, max_value)
    grey_encoding = _bin2gray(_int2bin(value, min_value, nbits, np.zeros(nbits, np.bool_)))
    return grey_encoding

def gray_encoding_to_int(min_value: Integral, max_value: Integral, gray_encoding: np.ndarray, dtype: type = None):
    """Combines all the steps of converting an gray encoding back into an integer

    Same as Platypus encoding, but internally uses statically typed functions and ndarrays
    
    If gray encoding is of length 0, will return the min_value

    Args:
        min_value (Integral): Lower bound for output. Should be an integer type
        max_value (Integral): Upper bound for output. Should be an integer type
        gray_encoding (np.ndarray): The grey encoded numpy array of a value. Must have dtype = bool_
        dtype (_type_, optional): The output integer type to cast the result to. Defaults to None.
            If None, will cast the output to whatever type the min_value is. Should be a valid integer type if not None

    Raises:
        AssertionError: If gray_encoding is not a ndarray of type bool_
    """    
    
    assert isinstance(gray_encoding, np.ndarray) and gray_encoding.dtype == np.bool_, "'gray_encoding' must be an ndarray of bool_"
    if dtype is None:
        dtype = type(min_value)
    if not gray_encoding.shape[0]:
        return dtype(min_value)
    value = _bin2int(_gray2bin(gray_encoding))
    if value > max_value - min_value:
        value -= max_value - min_value
    return dtype(min_value + value)
 
def _nbits_encode(min_value: int, max_value: int) -> int:
    """Find nbits for grey encoding. 
    
    - Assumes min_value < max_value and are integers
     """
    print(f"min = {min_value}, max = {max_value}")
    return int(np.log2(max_value - min_value)) + 1

@njit
def _int2bin(value, min_value, nbits, bit_arr) -> np.ndarray:
    """(internal, statically typed integer to bins)
    
    Assumes inputs are all integers (python or numpy) that do not exceed 64 bit range
    
    Assumes value >= min_value and len(bit_arr) == nbits
    
    'bit_arr' is a preallocated numpy array of bool_
    
    Returns:
        np.ndarray: 1D array of numpy bool_
    """
    n = value - min_value
    idx = nbits - 1
    while n > 0 and idx >= 0:
        bit_arr[idx] = (n & 1) == 1
        n >>= 1
        idx -= 1
    return bit_arr

@njit("boolean[:](boolean[:])")
def _bin2gray(bits: np.ndarray) -> np.ndarray:
    """(internal, statically typed bit string to gray encoding)"""
    n = bits.shape[0]
    gray = np.empty(n, np.bool_)
    if n > 0:
        gray[0] = bits[0]
        for j in range(1,n):
            gray[j] = bits[j-1] ^ bits[j]
    return gray

@njit("boolean[:](boolean[:])") 
def _gray2bin(bits: np.ndarray) -> np.ndarray: 
    """(internal, statically typed gray encoding to bit string)"""
    n = bits.shape[0]
    b = np.empty(n, np.bool_)
    if n > 0:
        b[0] = bits[0]
        for i in range(1, n):
            b[i] = b[i-1] ^ bits[i]
    return b

@njit("intp(boolean[:])")
def _bin2int(bits: np.ndarray) -> int:
    """(internal, statically typed bit string to int)"""
    i = 0
    for b in bits:
        i <<= 1
        i += b
    return i 

@njit
def _int_to_gray(value, min_value, max_value):
    """(internal: for use inside an njit func)"""
    if min_value == max_value:
        return np.zeros(1, np.bool_), 1
    if value < min_value:
        value = min_value
    nbits = floor(np.log2(max_value - min_value)) + 1
    grey_encoding =  np.zeros(nbits, np.bool_)
    return _bin2gray(_int2bin(value, min_value, nbits, grey_encoding)), nbits
  
@njit  
def _gray_to_int(min_value, max_value, gray_encoding):
    """(internal: assumes len(gray_encoding) > 0, for use inside njit func)"""
    value = _bin2int(_gray2bin(gray_encoding))
    if value > max_value - min_value:
        value -= max_value - min_value
    return min_value + value


@vectorize(
    [float32(float32, float32, float32), 
    float64(float64, float64, float64)], 
    nopython = True, cache = True)
def _vector_normalize1D(lb, ub, value):
    if lb == ub:
        return 1
    else:
        return (value - lb) / (ub - lb)

@vectorize(
    [float32(float32, float32, float32), 
    float64(float64, float64, float64)], 
    nopython = True, cache = True)
def _vector_denormalize1D(lb, ub, value):
    return value * (ub - lb) + lb


# Future work: Implement this faster to nbits function (input = max_value - min_value)
# @njit('int_(uint32)') #stackoverflow implementation
# def _popcount(v):
#     v = v - ((v >> 1) & 0x55555555)
#     v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
#     c = np.uint32((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24
#     return c


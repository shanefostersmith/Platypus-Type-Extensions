import pytest
import numpy as np
import custom_types.real_methods.numba_pcx as pcx


SPEED_TEST_DIM = [(3,3), (10, 100), (250,250), (10000,10), (10,10000)]
# SPEED_TEST = [(3,3), (100000, 2), (10000, 10), (2000, 50), (1000, 100), (500, 500), (100, 990), (50,2000), (10, 10000)]

def _numpy_pcx(X: np.ndarray, npar, nvars):
    g = pcx._find_g(X)  
    return pcx._classic_gs(X, npar, nvars, g.astype(X.dtype))
   
def _mod_pcx(X: np.ndarray, npar, nvars):
    
    g = pcx._find_g(X)
    e0, _ = pcx._vectorized_subtract(X[-1], g)
    e_eta = np.zeros((npar-1, nvars), X.dtype, order = 'C')
    e_eta, D, num_non_zero = pcx._modified_gs(X, np.uint32(npar), np.uint32(nvars), g.astype(X.dtype), e0.astype(X.dtype), e_eta)
    return e_eta, D, num_non_zero

def _batch_pcx(X: np.ndarray, npar, nvars):
    g = pcx._find_g(X)
    e0, _ = pcx._vectorized_subtract(X[-1], g)
    e_eta2 = np.zeros((npar-1, nvars), X.dtype, order = 'C')
    e_eta2, D, num_non_zero2 = pcx._batch_gs(X, np.uint32(npar), np.uint32(nvars), g.astype(X.dtype), e0.astype(X.dtype), e_eta2)
    return e_eta2, D, num_non_zero2

@pytest.fixture(params = [np.float32, np.float64], ids = lambda v: f"dtype={v}")
def float_type(request):
    return request.param

@pytest.fixture(params = [(2,1),(1,2),(4,3),(3,4),(100,101)], ids=lambda v: f"rows={v[0]},cols={v[1]}")
def matrix_dim(request):
    return request.param

@pytest.fixture(params = SPEED_TEST_DIM, ids=lambda v: f"rows={v[0]},cols={v[1]}")
def matrix_dim_speed(request):
    return request.param

@pytest.fixture(params=[True,False],ids=lambda v: f"zero_reference={v}")
def zero_reference(request):
    "whether the reference (last) row is all zeros"
    return request.param

@pytest.fixture
def p_matrix(matrix_dim, float_type, zero_reference):
    bit_gen = np.random.PCG64(123)
    rows, cols = matrix_dim
    
    rng = np.random.default_rng(bit_gen)
    parent_matrix = rng.random((rows,cols), float_type)
    if zero_reference:
        parent_matrix[-1] = np.zeros(cols, float_type)
    return parent_matrix

@pytest.fixture( 
    params = [(5, 100.0), (100, 1e10), (int(1e4), 1e10), (int(1e5), 1e5), (int(2.5e5), 1.1)], 
    ids = lambda v: f"ranges=vector_length: {v[0]}, max_number {v[1]}" )
def ranges_type(request):
    return request.param

@pytest.fixture(
    params = [(5, 100.0), (1000, 1.0e4), (int(1e4), 1.0e10), (int(5e4), 5.0), (int(2e5), 5.0)], 
    ids = lambda v: f"ranges=vector_length: {v[0]}, max_number {v[1]}" #, (int(1e5), 1e5)
)
def np32_ranges_DE(request):
    vector_length, max_val = request.param
    matrix = np.zeros((vector_length, 2), dtype = np.float32)
    rng = np.random.default_rng(0)
    half_max = max_val / 2
    matrix[:,1] = (half_max * rng.random(size = vector_length, dtype = np.float32)) + half_max
    orig = np.full(vector_length, half_max / 2, dtype = np.float32)
    p1 = np.full(vector_length, half_max / 3, dtype = np.float32)
    p2 = np.full(vector_length, half_max / 4, dtype = np.float32)
    p3 = np.full(vector_length, half_max / 2.5, dtype = np.float32)
    return matrix, orig, p1, p2, p3


import numpy as np
import pytest
import time
from math import ceil
from platypus._math import is_zero, magnitude, orthogonalize, subtract, normalize, dot, multiply
from custom_types.real_methods.numba_pcx import (
    _find_g, _vectorized_subtract, _batch_gs, _modified_gs, _classic_gs)

ZETA = 0.1
SPEED = [(3,3), (10, 100), (250,250), (10000,10), (10,10000)]
SPEED2 = [(3,3), (100000, 2), (10000, 10), (2000, 50), (1000, 100), (500, 500), (100, 990), (50,2000), (10, 10000)]
def project(u, vs): # vs == previous eta
    dot1 = dot(u, vs)
    dot2 = dot(vs, vs)
    quotient = dot1/dot2
    print(f"dot1 {dot1}, ndot2 {dot2} \n q: {quotient}") #all scalar
    return multiply(dot(u, vs) / dot(vs, vs), vs)


@pytest.fixture(params= SPEED2,  #[(10,5)],
                ids = lambda v: f"rows={v[0]}, cols={v[1]}") #
def p_matrix(request):
    bit_gen = np.random.PCG64(123)
    rows, cols = request.param
    out = []
    rng = np.random.default_rng(bit_gen)
    for _ in range(rows):
        one_row = rng.random(cols, np.float32) * 1000.0
        one_row = list(map(float, one_row.tolist()))
        out.append(one_row)
    return out


def _pcx_orig(p_matrix: list):
    k = len(p_matrix)
    nvars = len(p_matrix[0])
    g = [sum([p_matrix[i][j] for i in range(k)]) / k for j in range(nvars)] # mean of each var
    D = 0.0
    # basis vectors defined by parents
    e_eta = []
    prev = subtract(p_matrix[k-1], g)
    e_eta.append(prev) 
    for i in range(k-1): 
        d = subtract(p_matrix[i], g)
        if not is_zero(d):
            e = orthogonalize(d, e_eta)
            if not is_zero(e):
                D += magnitude(e) #sqrt(dot(e,e))
                e_eta.append(normalize(e))# put in [0,1] range
    return e_eta, D

def _numpy_pcx(X: np.ndarray, npar, nvars):
    g = _find_g(X)  
    return _classic_gs(X, npar, nvars, g)
   
def _mod_pcx(X: np.ndarray, npar, nvars):
    
    g = _find_g(X)
    e0, _ = _vectorized_subtract(X[-1], g)
    e_eta = np.zeros((npar-1, nvars), np.float32, order = 'C')
    e_eta, D, num_non_zero = _modified_gs(X, np.uint32(npar), np.uint32(nvars), g, e0, e_eta)
    return e_eta, D, num_non_zero

def _batch_pcx(X: np.ndarray, npar, nvars):
    g = _find_g(X)
    e0, _ = _vectorized_subtract(X[-1], g)
    e_eta2 = np.zeros((npar-1, nvars), np.float32, order = 'C')
    e_eta2, D, num_non_zero2 = _batch_gs(X, np.uint32(npar), np.uint32(nvars), g, e0, e_eta2)
    return e_eta2, D, num_non_zero2

def test_pcx_2(p_matrix):
    
    print("NEW TEST")
    X = np.vstack(p_matrix, dtype = np.float32) 
    print("shape X: {X.shape}")
    nparents = np.uint32(X.shape[0])
    nvars = np.uint32(X.shape[1])
    orig_eta, D_orig = _pcx_orig(p_matrix)
    print(f"ORIG_PCX: nvectors {len(orig_eta)}, D = {D_orig}\n")
    for i, eta in enumerate(orig_eta):
        if i > 0:
            print(f" e{i}: {eta}")
    # print("----------------- \n")

    print("NP PCX: \n")
    np_eta, D, nvect1 = _numpy_pcx(X, nparents, nvars)
    # print(np_eta[1:,:])
    print(np_eta)
    print(f"num_vectors: {nvect1}, D: {D}")
    print("----------------- \n")
    
    new_eta, D2, nvect2 = _mod_pcx(X, nparents, nvars)
    print(f"NEW PCX: num_vectors: {nvect2}, new_D = {D2} \n")
    print(new_eta)
    print("----------------- \n")
    
    batch_eta, D3, nvect3 = _batch_pcx(X, nparents, nvars)
    print(f"BATCH PCX: num_vectors: {nvect3}, new_D = {D3} \n")
    print(batch_eta)
    
    # percent_diff = 0.0 if D_orig == new_D else abs(D_orig - new_D)/((D_orig + new_D)/2.0) * 100.0
    # print(f"PERCENT DIFF MAG: {percent_diff}")
    
    
def test_speed_pcx(p_matrix):
    X = np.vstack(p_matrix, dtype = np.float32) 
    LOOPS = 50
    nparents = np.uint32(X.shape[0])
    nvars = np.uint32(X.shape[1])
    if X.shape[0] == 3:
        _batch_pcx(X, nparents, nvars)
        _numpy_pcx(X, nparents, nvars)
        _mod_pcx(X, nparents, nvars)
    else:
        print(f"\n NEW TEST: shape {X.shape}\n")
        
        np_eta, D1,_ = _numpy_pcx(X, nparents, nvars)
        nvectors_1 = np_eta.shape[0]
        n_start = time.perf_counter()
        for i in range(LOOPS):
            _numpy_pcx(X, nparents, nvars)
        n_end = time.perf_counter()
        
        _, D2, num_non_zero = _mod_pcx(X, nparents, nvars)
        new_start = time.perf_counter()
        for i in range(LOOPS):
            _mod_pcx(X, nparents, nvars)
        new_end = time.perf_counter()
        
        _, D3, num_non_zero2 = _batch_pcx(X, nparents, nvars)
        batch_start = time.perf_counter()
        for i in range(LOOPS):
            _batch_pcx(X, nparents, nvars)
        batch_end = time.perf_counter()
        
        
        # print(f"ORIG: v = {nvectors_0}, time = {o_end - o_start}")
        print(f"NUMPY: v = {nvectors_1}, time = {n_end - n_start}, D: {D1}")
        print(f"NEW: v = {num_non_zero}, time = {new_end - new_start}, D: {D2}")
        print(f"BATCH: v = {num_non_zero2}, time = {batch_end - batch_start}, D: {D3}")


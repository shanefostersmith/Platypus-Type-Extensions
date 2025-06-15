import pytest
import numpy as np
import time
from numba.typed.typedlist import List
from tests.conftest import np64_ranges, np64_2D_ranges
from tests.test_real_int.conftest import np32_ranges_DE, matrix_dim_speed, _batch_pcx, _mod_pcx, _numpy_pcx
from platypus_extensions.real_methods.numba_differential import *
from platypus_extensions.utils import (
    gu_normalize2D_1D, gu_normalize2D_2D,
    gu_denormalize2D_1D, gu_denormalize2D_2D,
    _vector_normalize1D, vectorized_to_norm)

def test_norm_speed(np64_ranges):
    matrix, values = np64_ranges
    nvars = len(values)
    LOOPS = 1000
    if nvars == 5: #compile gu
        # _ = vectorized_to_norm(matrix, values)
        # _ = gu_normalize2D_1D(matrix, values)
        # _ = parallel_normalized1d(matrix, values)
        print(f"values: {values}\n")
        normalized =  gu_normalize2D_1D(matrix, values)
        print(f"normalized: {normalized}\n")
        _ = gu_denormalize2D_1D(matrix, normalized)
        np.testing.assert_almost_equal(normalized, values)
    else:
        print(f"nvars = {nvars}")
        start_normal = time.perf_counter()
        for i in range(LOOPS):
            _ =  gu_normalize2D_1D(matrix, values)
        end_time = time.perf_counter()
        print(f"VECTORIZED: {end_time - start_normal}")

        start_gu = time.perf_counter()
        for i in range(LOOPS):
            _ = _vector_normalize1D(matrix[:,0], matrix[:,1], values)
        end_gu = time.perf_counter()
        print(f"NORMAL: {end_gu - start_gu}")

def test_norm_speed_2d(np64_2D_ranges):
    matrix, values = np64_2D_ranges
    LOOPS = 1000
    if values.shape[0] == 5: #compile gu
        print(f"values: {values}\n")
        out = gu_normalize2D_2D(matrix, values)
        print(f"normalized: {out}\n")
        gu_denormalize2D_2D(matrix, out)
        np.testing.assert_almost_equal(out, values)
        print(f"compiled: after conversion {out} \n")
    else:
        print(f"SHAPE: {values.shape}")
        start_gu = time.perf_counter()
        for i in range(LOOPS):
            _ = gu_normalize2D_2D(matrix, values)
        end_gu = time.perf_counter()
        print(f"VECTORIZED 2D: shape: {end_gu - start_gu}")
        
        start_normal = time.perf_counter()
        for i in range(LOOPS):
            _ = vectorized_to_norm(matrix, values)
    
        end_time = time.perf_counter()
        print(f"NORMAL 2D: {end_time - start_normal}\n")


def test_differential_evolve(np32_ranges_DE): # test gu
    matrix, orig, p1, p2, p3 = np32_ranges_DE
    step_size = 0.25
    crossover_rate = 0.5
    
    LOOPS = 500
    nvars = matrix.shape[0]
    seed = np.random.SeedSequence(1234)
    bit_gen = np.random.SFC64(seed)
    
    if nvars == 5:
        DE_with_probability(bit_gen, orig, p1, p2, p3, matrix, step_size, crossover_rate)
        differential_evolve_with_probability(orig, p1, p2, p3, matrix, step_size, crossover_rate)
    else:
        print(f"nvars: {nvars}")
        start_normal = time.perf_counter()
        for i in range(LOOPS):
            differential_evolve_with_probability(orig, p1, p2, p3, matrix, step_size, crossover_rate)
        end_time = time.perf_counter()
        print(f"NORMAL: {end_time - start_normal}")
        
        start_gu = time.perf_counter()
        for i in range(LOOPS):
            DE_with_probability(bit_gen, orig, p1, p2, p3, matrix, step_size, crossover_rate)
        end_gu = time.perf_counter()
        print(f"BIT_GEN {end_gu - start_gu}\n")

# RNG = np.random.Generator(np.random.PCG64(7))
@pytest.mark.parametrize(
    'crossover_rate',
    [0.25, 0.5, 0.95]
)
def test_differential_evolve2(crossover_rate, np32_ranges_DE): # test gu
    matrix, orig, p1, p2, p3 = np32_ranges_DE
    step_size = 0.25
    
    NGENS = 5
    LOOPS = 500
    nvars = matrix.shape[0]
    seed = np.random.SeedSequence(1234)
    bit_gen = np.random.SFC64(seed)

    if nvars == 5:
        # print(f"\nbefore: {orig}")
        # DE_with_probability(bit_gen, orig, p1, p2, p3, matrix, step_size, crossover_rate)
        gens = bit_gen.spawn(NGENS)
        pg_list = List.empty_list(typeof(bit_gen), NGENS)
        for j in range(NGENS):
            pg_list.append(gens[j])
        parallel_DE_with_probability(pg_list, orig, p1, p2, p3, matrix, step_size, crossover_rate)
        # print(f"after2: {orig}")

    else:
        print(f"nvars: {nvars}, crossover_rate {crossover_rate}")
        start_gu = time.perf_counter()
        for i in range(LOOPS):
            DE_with_probability(bit_gen, orig, p1, p2, p3, matrix, step_size, crossover_rate)

        end_gu = time.perf_counter()
        print(f"BIT_GEM {end_gu - start_gu}")

        start_normal = time.perf_counter()
        for i in range(LOOPS):
            gens = bit_gen.spawn(NGENS)
            pg_list = List.empty_list(BIT_GEN_TYPE, NGENS)
            for j in range(NGENS):
                pg_list.append(gens[j])
            parallel_DE_with_probability(pg_list, orig, p1, p2, p3, matrix, step_size, crossover_rate)
        end_time = time.perf_counter()
        print(f"PARALLEL: {end_time - start_normal}\n")
        
        
@pytest.mark.parametrize('nspawns', [2, 5, 10] )
def test_differential_evolve3(nspawns, np32_ranges_DE): # test gu
    matrix, orig, p1, p2, p3 = np32_ranges_DE
    step_size = 0.25
    crossover_rate = 0.5
    
    LOOPS = 500
    nvars = matrix.shape[0]
    seed = np.random.SeedSequence(1234)
    bit_gen = np.random.SFC64(seed)
    
    if nvars == 5:
        gens = bit_gen.spawn(nspawns)
        pg_list = List.empty_list(typeof(bit_gen), nspawns)
        for j in range(nspawns):
            pg_list.append(gens[j])
        parallel_DE_with_probability(pg_list, orig, p1, p2, p3, matrix, step_size, crossover_rate)
    else:
        print(f"nvars: {nvars}, nspawns {nspawns}")
        start_normal = time.perf_counter()
        for i in range(LOOPS):
            gens = bit_gen.spawn(nspawns)
            pg_list = List.empty_list(typeof(bit_gen), nspawns)
            for j in range(nspawns):
                pg_list.append(gens[j])
            parallel_DE_with_probability(pg_list, orig, p1, p2, p3, matrix, step_size, crossover_rate)
         
        end_time = time.perf_counter()
        print(f"SFC: {end_time - start_normal}\n")
        
def test_pcx_speed(matrix_dim_speed):
    nparents, nvars = matrix_dim_speed 
    bit_gen = np.random.PCG64(123)
    rng = np.random.default_rng(bit_gen)
    X = rng.random((nparents, nvars), np.float32)
    LOOPS = 50
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
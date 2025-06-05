import pytest
import numpy as np
import custom_types.real_methods.numba_pcx as pcx
# import timeit as timer
import time
from inspect import unwrap
from tests.conftest import np_normalized_vector32, np_normalized_matrix32
from custom_types.utils import (
    gu_normalize2D_1D, gu_normalize2D_2D,
    gu_denormalize2D_1D, gu_denormalize2D_2D,
    vector_normalize1D, vectorized_to_norm)
from custom_types.real_methods.numba_pcx import normalized_1d_pcx, normalized_2d_pcx
from pytest_mock import MockerFixture

class TestNormalizedPCX:
    
    @pytest.mark.parametrize(
        "nrows,ncols,zero_reference,zero_column",
        [
            (2, 5, False, False),
            (2, 2, True, False),
            (2, 2, False, True)
        ],
        ids = ["rand", "zero_ref", "zero_col"],
        indirect=["nrows", "ncols", "zero_reference", "zero_column"],
    )
    def test_run_no_error(
        self, 
        nrows, ncols, zero_reference, zero_column, 
        np_normalized_matrix32):
        
        normalized_2d_pcx(
            parent_vars = np_normalized_matrix32, 
            noffspring = 1,
            eta = np.float32(0.1),
            zeta = np.float32(0.1),
            randomize=False
        )
        normalized_2d_pcx(
            parent_vars = np_normalized_matrix32, 
            noffspring = 1,
            eta = np.float32(0.1),
            zeta = np.float32(0.1),
            randomize=True
        )
        
    @pytest.mark.parametrize(
        "nrows, ncols, zero_reference, zero_column",
        [(2, 5, False, False)],
        ids = ["valid_e0"],
        indirect=["nrows", "ncols", "zero_reference", "zero_column"],
    )
    def test_valid_call(self, 
        nrows, ncols, zero_reference, zero_column, 
        np_normalized_matrix32, mocker: MockerFixture):
        
        # remove from njit decorators
        orig_orth = pcx._orthogonalize_pcx.py_func
        orig_valid = pcx._valid_e0_orthogonalize.py_func
        orig_invalid = pcx._invalid_e0_orthogonalize.py_func
        mocker.patch.object(pcx, "_invalid_e0_orthogonalize", orig_invalid)
        mocker.patch.object(pcx, "_valid_e0_orthogonalize", orig_valid)
        mocker.patch.object(pcx, "_orthogonalize_pcx",      orig_orth)
        
        
        spy = mocker.spy(pcx, "_valid_e0_orthogonalize")
        spy2 = mocker.spy(pcx, "_invalid_e0_orthogonalize")
        
        pcx.normalized_2d_pcx(
            parent_vars = np_normalized_matrix32, 
            noffspring = 1,
            eta = np.float32(0.1),
            zeta = np.float32(0.1),
            randomize=False
        ) 
        spy.assert_called()
        spy2.assert_not_called()
    

    def test_invalid_call(self,mocker: MockerFixture):
        
        # remove from njit decorators
        orig_orth = pcx._orthogonalize_pcx.py_func
        orig_valid = pcx._valid_e0_orthogonalize.py_func
        orig_invalid = pcx._invalid_e0_orthogonalize.py_func
        mocker.patch.object(pcx, "_invalid_e0_orthogonalize", orig_invalid)
        mocker.patch.object(pcx, "_valid_e0_orthogonalize", orig_valid)
        mocker.patch.object(pcx, "_orthogonalize_pcx",      orig_orth)

        spy = mocker.spy(pcx, "_valid_e0_orthogonalize")
        spy2 = mocker.spy(pcx, "_invalid_e0_orthogonalize")
        matrix = np.zeros((10,4), np.float32)
        
        offspring = pcx.normalized_2d_pcx(
            parent_vars = matrix, 
            noffspring = 8,
            eta = np.float32(0.1),
            zeta = np.float32(0.1),
            randomize=True
        ) 
        print(offspring)
        spy2.assert_called()
        spy.assert_not_called()
    
    @pytest.mark.parametrize(
        "vector_size,vector_kind",
        [
            (10, "random"),
            (10, "zeros"),
        ],
        ids=["10-random", "10-zeros"],
        indirect=["vector_size", "vector_kind"],
    )   
    def test_1d_call(self, np_normalized_vector32, noffspring, mocker: MockerFixture):
        
        # remove from njit decorators
        orig_orth = pcx._orthogonalize_pcx.py_func
        orig_1D = pcx.normalized_1d_pcx.py_func
        mocker.patch.object(pcx, "normalized_1d_pcx", orig_1D)
        mocker.patch.object(pcx, "_orthogonalize_pcx", orig_orth)
        
        spy = mocker.spy(pcx, "normalized_1d_pcx")
        spy2 = mocker.spy(pcx, "_orthogonalize_pcx")
        
        matrix = np.empty((len(np_normalized_vector32), 1))
        
        matrix[:,0] = np_normalized_vector32
        # print(f"noffpring: {noffspring}, before: {matrix}")
        offspring = pcx.normalized_2d_pcx(matrix, noffspring, np.float32(0.1), np.float32(0.1), True)
        # print(f'after {offspring}')
        
        spy.assert_called()
        spy2.assert_not_called()

def test_norm_speed(np64_ranges):
    matrix, values = np64_ranges
    nvars = len(values)
    LOOPS = 10000
    if nvars == 5: #compile gu
        # _ = vectorized_to_norm(matrix, values)
        # _ = gu_normalize2D_1D(matrix, values)
        # _ = parallel_normalized1d(matrix, values)
        print(f"values: {values}\n")
        normalized =  gu_normalize2D_1D(matrix, values)
        print(f"normalized: {normalized}\n")
        any_return = gu_denormalize2D_1D(matrix, normalized)
        np.testing.assert_almost_equal(normalized, values)
        print(f"compiled: after conversion {normalized} \n, any_return? {any_return}")
        # _ = vector_normalize1D(matrix[:,0], matrix[:,1], values)
    else:
        start_normal = time.perf_counter()
        early_time = None
        mid_time = None
        for i in range(LOOPS):
            # _ = gu_normalize2D_1D(matrix, values)
            _ =  gu_normalize2D_1D(matrix, values)
            if i == 500:
                early_time = time.perf_counter()
            elif i == 3750:
                mid_time = time.perf_counter()
            
            
        end_time = time.perf_counter()
        print(f"GU_VECTORIZED nvars = {nvars}")
        print(f"1. {early_time - start_normal}")
        print(f"2. {mid_time - start_normal}")
        print(f"Final. {end_time - start_normal}")
        print(f"500 -> end: {end_time - early_time}\n")
        
        start_gu = time.perf_counter()
        early_gu = None
        mid_gu = None
        for i in range(LOOPS):
            # _ = parallel_normalized1d(matrix, values)
            _ = vector_normalize1D(matrix[:,0], matrix[:,1], values)
            if i == 1000:
                early_gu = time.perf_counter()
            elif i == 3750:
                mid_gu = time.perf_counter()
        end_gu = time.perf_counter()
        print(f"PARALLEL_VECTORIZED: nvars = {nvars}")
        print(f"1. {early_gu - start_gu}")
        print(f"2. {mid_gu - start_gu}")
        print(f"Final. {end_gu - start_gu}")
        print(f"500 -> end: {end_gu - early_gu}\n")
        
        
def test_norm_speed_2d(np64_2D_ranges):
    matrix, values = np64_2D_ranges
    LOOPS = 6000
    if values.shape[0] == 5: #compile gu
        print(f"values: {values}\n")
        out = gu_normalize2D_2D(matrix, values)
        print(f"normalized: {out}\n")
        gu_denormalize2D_2D(matrix, out)
        np.testing.assert_almost_equal(out, values)
        print(f"compiled: after conversion {out} \n")
    else:
        start_gu = time.perf_counter()
        early_gu = None
        mid_gu = None
        for i in range(LOOPS):
            _ = gu_normalize2D_2D(matrix, values)
            if i == 1000:
                early_gu = time.perf_counter()
            elif i == 3750:
                mid_gu = time.perf_counter()
        end_gu = time.perf_counter()
        print(f"GU 2D: shape: {values.shape}")
        print(f"1. {early_gu - start_gu}")
        print(f"2. {mid_gu - start_gu}")
        print(f"Final. {end_gu - start_gu}")
        print(f"500 -> end: {end_gu - early_gu}\n")
        
        
        start_normal = time.perf_counter()
        early_time = None
        mid_time = None
        for i in range(LOOPS):
            _ = vectorized_to_norm(matrix, values)
            if i == 500:
                early_time = time.perf_counter()
            elif i == 3750:
                mid_time = time.perf_counter()
            
        end_time = time.perf_counter()
        print(f"NORMAL 2D: ")
        print(f"1. {early_time - start_normal}")
        print(f"2. {mid_time - start_normal}")
        print(f"Final. {end_time - start_normal}")
        print(f"500 -> end: {end_time - early_time}\n")
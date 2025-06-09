import pytest
import numpy as np
import custom_types.real_methods.numba_pcx as pcx
from tests.conftest import np_normalized_vector32, np_normalized_matrix32
from custom_types.real_methods.numba_pcx import (normalized_1d_pcx, normalized_2d_pcx)
from pytest_mock import MockerFixture
from platypus._math import orthogonalize, subtract, add


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


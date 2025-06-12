import pytest
import numpy as np
import custom_types.real_methods.numba_pcx as pcx
from tests.conftest import noffspring
from tests.test_real_int.conftest import p_matrix, _batch_pcx, _mod_pcx, _numpy_pcx
# from pytest_mock import MockerFixture

class TestNormalizedPCX:
    
    def test_gs_smoke(self,p_matrix):
        
        nparents = np.uint32(p_matrix.shape[0])
        nvars = np.uint32(p_matrix.shape[1])
        
        print("NP PCX: \n")
        np_eta, D, nvect1 = _numpy_pcx(p_matrix, nparents, nvars)
        # print(np_eta[1:,:])
        # print(np_eta)
        print(f"num_vectors: {nvect1}, D: {D}")
        print("----------------- \n")
        
        new_eta, D2, nvect2 = _mod_pcx(p_matrix, nparents, nvars)
        print(f"NEW PCX: num_vectors: {nvect2}, new_D = {D2} \n")
        # print(new_eta)
        print("----------------- \n")
        
        batch_eta, D3, nvect3 = _batch_pcx(p_matrix, nparents, nvars)
        print(f"BATCH PCX: num_vectors: {nvect3}, new_D = {D3} \n")
        # print(batch_eta)
    
    @pytest.mark.parametrize('randomize', [True, False])
    def test_main_smoke(self,randomize, p_matrix, noffspring):
        
        out = pcx.normalized_2d_pcx(
            parent_vars=p_matrix, 
            noffspring=noffspring, 
            eta = 0.25, zeta = 0.25,
            randomize=randomize
        )
        assert out.shape[0] == noffspring
        assert out.shape[1] == p_matrix.shape[1]
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_zeros_empty(self): #mocker: MockerFixture):
        
        # remove from njit decorators
        # orig_invalid = pcx._invalid_e0_gs.py_func
        # mocker.patch.object(pcx, "_invalid_e0_gs", orig_invalid)
        # spy = mocker.spy(pcx, "_invalid_e0_gs")
        zero_matrix = np.zeros((10,4), np.float32)
        empty_matrix = np.zeros((0,4), np.float32)
        _ = pcx.normalized_2d_pcx(
            parent_vars = zero_matrix, 
            noffspring = 8,
            eta = np.float32(0.1),
            zeta = np.float32(0.1),
            randomize=True
        ) 
        _ = pcx.normalized_2d_pcx(
            parent_vars = empty_matrix, 
            noffspring = 8,
            eta = np.float32(0.1),
            zeta = np.float32(0.1),
            randomize=True
        ) 
        
    
 

